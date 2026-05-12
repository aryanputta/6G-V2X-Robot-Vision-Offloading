#!/usr/bin/env python3
"""
6G V2X Video Offload Demo
==========================
Reads frames from a video file or webcam, sends each frame to the
6G edge server (WebSocket), receives real YOLOv8 detections, and
applies an IoU-based multi-object tracker locally.

Outputs:
  • demo_output.mp4  — annotated video (detections + tracking trails + RTT HUD)
  • Terminal stats   — live FPS / RTT / detection count

Usage:
    # Generate test video first (if no webcam):
    python demos/generate_test_video.py

    # Run with test video:
    python demos/video_offload_demo.py --input assets/test_scene.mp4

    # Run with webcam:
    python demos/video_offload_demo.py --webcam

    # Compare 4G vs 5G vs 6G:
    python demos/video_offload_demo.py --input assets/test_scene.mp4 --network 4G

    # Headless (no display), just save MP4:
    python demos/video_offload_demo.py --input assets/test_scene.mp4 --headless
"""

import argparse
import asyncio
import base64
import json
import math
import os
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.sim import NetworkSimulator, NETWORK_PROFILES
from shared_state import NetworkGen


# ──────────────────────────────────────────────────────────────────────────────
# IoU Multi-Object Tracker (runs on the robot — no GPU needed)
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 51, 255,  86), ( 51, 255, 189),
    ( 51, 167, 255), (157,  51, 255), (255,  51, 201), (255, 255,  51),
    ( 51, 255, 255), (200, 100, 200),
]


def _iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]);  y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid: int, bbox: List[float], label: str, conf: float):
        self.id    = tid
        self.bbox  = bbox
        self.label = label
        self.conf  = conf
        self.age   = 0
        self.color = _PALETTE[tid % len(_PALETTE)]
        self.trail: deque = deque(maxlen=25)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.trail.append((cx, cy))

    def update(self, bbox: List[float], label: str, conf: float):
        self.bbox  = bbox
        self.label = label
        self.conf  = conf
        self.age   = 0
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.trail.append((cx, cy))


class IouTracker:
    """
    Lightweight IoU-based multi-object tracker.
    Associates detections across frames without a GPU.
    Runs entirely on the robot (Raspberry Pi viable).
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 6):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[dict]) -> List[Track]:
        """Match new detections to existing tracks; return active tracks."""
        unmatched_tracks = set(self._tracks.keys())
        matched_det_idx  = set()

        for i, det in enumerate(detections):
            bbox  = det.get("bbox", [])
            label = det.get("class", "object")
            conf  = det.get("confidence", 0.0)
            if len(bbox) != 4:
                continue

            best_tid, best_iou = None, self.iou_threshold
            for tid in list(unmatched_tracks):
                iou = _iou(bbox, self._tracks[tid].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update(bbox, label, conf)
                unmatched_tracks.discard(best_tid)
                matched_det_idx.add(i)
            else:
                self._tracks[self._next_id] = Track(self._next_id, bbox, label, conf)
                matched_det_idx.add(i)
                self._next_id += 1

        # age out unmatched tracks
        for tid in list(unmatched_tracks):
            self._tracks[tid].age += 1
            if self._tracks[tid].age > self.max_age:
                del self._tracks[tid]

        return list(self._tracks.values())


# ──────────────────────────────────────────────────────────────────────────────
# Frame annotator
# ──────────────────────────────────────────────────────────────────────────────

def _draw_tracks(frame: np.ndarray, tracks: List[Track],
                 W: int, H: int, alpha: float = 0.85):
    overlay = frame.copy()
    for tk in tracks:
        x1 = int(tk.bbox[0] * W);  y1 = int(tk.bbox[1] * H)
        x2 = int(tk.bbox[2] * W);  y2 = int(tk.bbox[3] * H)
        c  = tk.color

        # translucent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        overlay = frame.copy()

        # border
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        # tracking trail (normalized → pixel)
        trail_pts = [
            (int(px * W), int(py * H))
            for px, py in tk.trail
        ]
        for j in range(1, len(trail_pts)):
            fade = int(255 * j / len(trail_pts))
            cv2.line(frame, trail_pts[j - 1], trail_pts[j],
                     (min(c[0], fade), min(c[1], fade), min(c[2], fade)), 2)

        # label tag
        tag = f"#{tk.id} {tk.label} {tk.conf:.2f}"
        tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tag_y = max(th + 4, y1)
        cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 4, tag_y), c, -1)
        cv2.putText(frame, tag, (x1 + 2, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def _draw_hud(frame: np.ndarray, rtt_ms: float, fps: float,
              n_tracks: int, frame_id: int, network_gen: str,
              processing_ms: float):
    H, W = frame.shape[:2]

    # top bar
    cv2.rectangle(frame, (0, 0), (W, 60), (15, 15, 15), -1)

    gen_colors = {"4G": (50, 50, 220), "5G": (50, 180, 220), "6G": (50, 220, 80)}
    bar_color  = gen_colors.get(network_gen, (200, 200, 200))

    cv2.putText(frame, f"6G V2X EDGE OFFLOAD  [{network_gen}]",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                bar_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"RTT {rtt_ms:.1f}ms  |  GPU {processing_ms:.1f}ms  |  {fps:.0f}fps  |  {n_tracks} tracked",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)

    # RTT bar (bottom)
    bar_h = 6
    max_rtt = 60.0
    bar_len = int(min(rtt_ms / max_rtt, 1.0) * W)
    cv2.rectangle(frame, (0, H - bar_h), (W, H), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, H - bar_h), (bar_len, H), bar_color, -1)

    # frame counter (bottom-right)
    cv2.putText(frame, f"frame {frame_id:05d}",
                (W - 110, H - bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Async edge-offload pipeline
# ──────────────────────────────────────────────────────────────────────────────

class OffloadPipeline:
    """
    Sends frames to the 6G edge server and collects detections.
    Wraps the async WebSocket calls so they can be called from sync code.
    """

    def __init__(self, server_uri: str, network_gen: str):
        self.server_uri = server_uri
        self.network_gen = network_gen
        self.net_sim = NetworkSimulator(network_gen)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._lock = threading.Lock()
        self._frame_id = 0
        self.last_rtt_ms = 0.0
        self.last_proc_ms = 0.0

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def set_network(self, gen: str):
        self.network_gen = gen
        self.net_sim.set_network(gen)

    def detect(self, jpeg_bytes: bytes, timeout: float = 15.0) -> Tuple[List[dict], float, float]:
        """
        Synchronously send a JPEG frame, simulate 6G uplink, wait for detections.
        Returns (detections, rtt_ms, server_processing_ms).
        """
        fut = self._submit(self._async_detect(jpeg_bytes))
        try:
            return fut.result(timeout=timeout)
        except Exception as e:
            print(f"  [warn] detect failed: {e}")
            return [], 0.0, 0.0

    async def _async_detect(self, jpeg_bytes: bytes) -> Tuple[List[dict], float, float]:
        with self._lock:
            self._frame_id += 1
            fid = self._frame_id

        # simulate 6G uplink delay
        uplink_ms = await self.net_sim.simulate_uplink(len(jpeg_bytes))

        msg = json.dumps({
            "msg_id":         "VISION",
            "frame_id":       fid,
            "robot_id":       "robot-001",
            "timestamp":      time.time(),
            "frame_jpeg_b64": base64.b64encode(jpeg_bytes).decode(),
        })

        async with __import__("websockets").connect(self.server_uri) as ws:
            t_send = time.perf_counter()
            await ws.send(msg)
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)

        rtt_ms = (time.perf_counter() - t_send) * 1000

        resp = json.loads(raw)
        proc_ms = resp.get("server_processing_ms", 0.0)

        # simulate downlink delay
        await self.net_sim.simulate_downlink(len(raw.encode()))

        self.last_rtt_ms  = rtt_ms
        self.last_proc_ms = proc_ms
        return resp.get("detections", []), rtt_ms, proc_ms


# ──────────────────────────────────────────────────────────────────────────────
# Edge server (runs in-process for the demo)
# ──────────────────────────────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the TCP port is accepting connections (or timeout)."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def _start_edge_server(port: int = 8766):
    """Launch the edge server in a background thread; block until ready."""
    import asyncio as _asyncio
    from edge_server.server import EdgeServer

    loop = _asyncio.new_event_loop()

    def _run():
        import logging
        logging.getLogger("websockets").setLevel(logging.CRITICAL)
        _asyncio.set_event_loop(loop)
        server = EdgeServer(host="localhost", port=port,
                            network_gen=NetworkGen.SIX_G,
                            vision_backend="yolov8")
        loop.run_until_complete(server.run())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # Wait until the port is actually accepting connections (handles slow YOLO init)
    if not _wait_for_port("localhost", port, timeout=30.0):
        print(f"[ERROR] Edge server failed to start on port {port}")
        sys.exit(1)
    return loop


# ──────────────────────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(input_path: Optional[str], webcam: bool,
             network_gen: str, output_path: str,
             headless: bool, max_frames: Optional[int],
             server_port: int = 8766):

    # ── start edge server ────────────────────────────────────────────────
    print(f"Starting 6G edge server on port {server_port}...")
    _start_edge_server(server_port)

    # ── open video source ────────────────────────────────────────────────
    if webcam:
        cap = cv2.VideoCapture(0)
        source_label = "Webcam"
    elif input_path and Path(input_path).exists():
        cap = cv2.VideoCapture(input_path)
        source_label = Path(input_path).name
    else:
        print(f"[ERROR] Video source not found: {input_path}")
        print("        Run: python demos/generate_test_video.py  to create a test video.")
        sys.exit(1)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Source: {source_label}  {W}x{H} @ {fps:.0f}fps")

    # ── video writer ─────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # ── init pipeline and tracker ────────────────────────────────────────
    pipeline = OffloadPipeline(f"ws://localhost:{server_port}", network_gen)
    pipeline.start()
    tracker  = IouTracker()

    rtt_history: deque = deque(maxlen=30)
    frame_idx = 0
    t_start = time.time()
    print(f"\nNetwork: {network_gen}  |  Output: {out_path}\n")
    print(f"{'Frame':>6}  {'RTT':>8}  {'GPU':>7}  {'FPS':>6}  {'Tracks':>7}  {'Detections'}")
    print("─" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            frame_idx += 1

            # encode frame as JPEG for transmission
            ok, jpeg_buf = cv2.imencode(".jpg", frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            jpeg_bytes = jpeg_buf.tobytes()

            # offload to edge server
            dets, rtt_ms, proc_ms = pipeline.detect(jpeg_bytes)
            rtt_history.append(rtt_ms)

            # local tracking (IoU — no GPU)
            tracks = tracker.update(dets)

            # annotate frame
            frame = _draw_tracks(frame, tracks, W, H)
            frame = _draw_hud(frame, rtt_ms, fps, len(tracks),
                              frame_idx, network_gen, proc_ms)

            writer.write(frame)

            if not headless:
                cv2.imshow("6G V2X Edge Offload Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("1"):
                    network_gen = NetworkGen.FOUR_G
                    pipeline.set_network(network_gen)
                elif key == ord("2"):
                    network_gen = NetworkGen.FIVE_G
                    pipeline.set_network(network_gen)
                elif key == ord("3"):
                    network_gen = NetworkGen.SIX_G
                    pipeline.set_network(network_gen)

            # terminal stats
            elapsed = time.time() - t_start
            effective_fps = frame_idx / elapsed if elapsed > 0 else 0
            det_names = ", ".join(d.get("class", "?") for d in dets[:3])
            if frame_idx % 5 == 0 or frame_idx <= 3:
                print(f"{frame_idx:>6}  {rtt_ms:>7.1f}ms  {proc_ms:>6.1f}ms  "
                      f"{effective_fps:>5.1f}  {len(tracks):>6}  {det_names}")

    finally:
        cap.release()
        writer.release()
        if not headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_rtt = sum(rtt_history) / len(rtt_history) if rtt_history else 0
    print(f"\n{'─'*60}")
    print(f"Done.  {frame_idx} frames in {elapsed:.1f}s  ({frame_idx/elapsed:.1f}fps effective)")
    print(f"Avg RTT: {avg_rtt:.1f}ms  |  Output: {out_path}")
    print(f"Play:    ffplay {out_path}")


def main():
    ap = argparse.ArgumentParser(description="6G V2X Video Offload Demo")
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--input",     default="assets/test_scene.mp4",
                     help="Input video file (default: assets/test_scene.mp4)")
    grp.add_argument("--webcam",    action="store_true",
                     help="Use webcam instead of video file")
    ap.add_argument("--network",    default="6G", choices=["4G", "5G", "6G"],
                    help="Simulated network generation")
    ap.add_argument("--output",     default="assets/demo_output.mp4")
    ap.add_argument("--headless",   action="store_true",
                    help="No display window — save to file only")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--port",       type=int, default=8766)
    args = ap.parse_args()

    run_demo(
        input_path  = None if args.webcam else args.input,
        webcam      = args.webcam,
        network_gen = args.network,
        output_path = args.output,
        headless    = args.headless,
        max_frames  = args.max_frames,
        server_port = args.port,
    )


if __name__ == "__main__":
    main()
