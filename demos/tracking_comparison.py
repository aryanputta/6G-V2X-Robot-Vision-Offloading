#!/usr/bin/env python3
"""
Tracking Comparison Demo
=========================
Three-panel side-by-side video comparing:
  LEFT  — Raw input frame (what the robot "sees")
  CENTER — Local YOLOv8 on-robot (needs expensive hardware)
  RIGHT  — 6G Edge Offload + IoU tracking (Raspberry Pi viable)

Saves assets/comparison_output.mp4

Usage:
    python demos/tracking_comparison.py
    python demos/tracking_comparison.py --input assets/test_scene.mp4
    python demos/tracking_comparison.py --input assets/test_scene.mp4 --headless
"""

import argparse
import asyncio
import base64
import json
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.sim import NetworkSimulator
from shared_state import NetworkGen


# ──────────────────────────────────────────────────────────────────────────────
# Re-use tracker from video_offload_demo
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (255, 56, 56), (255, 157, 51), (51, 255, 86), (51, 255, 189),
    (51, 167, 255), (157, 51, 255), (255, 51, 201), (255, 255, 51),
]


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, bbox, label, conf):
        self.id    = tid; self.bbox = bbox
        self.label = label; self.conf = conf
        self.age   = 0
        self.color = _PALETTE[tid % len(_PALETTE)]
        self.trail = deque(maxlen=20)
        self.trail.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))

    def update(self, bbox, label, conf):
        self.bbox = bbox; self.label = label; self.conf = conf; self.age = 0
        self.trail.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))


class IouTracker:
    def __init__(self, iou_thr=0.25, max_age=6):
        self.iou_thr = iou_thr; self.max_age = max_age
        self._tracks: Dict[int, Track] = {}; self._nid = 1

    def update(self, dets):
        unmatched = set(self._tracks)
        for det in dets:
            bbox = det.get("bbox", [])
            if len(bbox) != 4: continue
            best_tid, best_iou = None, self.iou_thr
            for tid in list(unmatched):
                v = _iou(bbox, self._tracks[tid].bbox)
                if v > best_iou: best_iou = v; best_tid = tid
            if best_tid:
                self._tracks[best_tid].update(bbox, det["class"], det["confidence"])
                unmatched.discard(best_tid)
            else:
                self._tracks[self._nid] = Track(self._nid, bbox, det["class"], det["confidence"])
                self._nid += 1
        for tid in list(unmatched):
            self._tracks[tid].age += 1
            if self._tracks[tid].age > self.max_age: del self._tracks[tid]
        return list(self._tracks.values())


# ──────────────────────────────────────────────────────────────────────────────
# Local YOLOv8 (runs on the robot — simulates expensive onboard GPU)
# ──────────────────────────────────────────────────────────────────────────────

class LocalYOLO:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self._tracker = IouTracker()

    def infer(self, frame: np.ndarray) -> Tuple[List[dict], float]:
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        for box in results.boxes:
            xyxyn = box.xyxyn[0].tolist()
            dets.append({
                "class":      results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox":       [round(v, 3) for v in xyxyn],
            })
        tracks = self._tracker.update(dets)
        return tracks, ms


# ──────────────────────────────────────────────────────────────────────────────
# Edge offload (simulates 6G over WebSocket)
# ──────────────────────────────────────────────────────────────────────────────

class EdgeOffload:
    def __init__(self, server_uri: str, network_gen: str = NetworkGen.SIX_G):
        self.server_uri  = server_uri
        self.network_gen = network_gen
        self.net_sim     = NetworkSimulator(network_gen)
        self._tracker    = IouTracker()
        self._loop       = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def infer(self, jpeg_bytes: bytes) -> Tuple[List[Track], float, float]:
        fut = asyncio.run_coroutine_threadsafe(self._async_infer(jpeg_bytes), self._loop)
        try:
            return fut.result(timeout=15.0)
        except Exception as e:
            print(f"  [warn] edge infer failed: {e}")
            return [], 0.0, 0.0

    async def _async_infer(self, jpeg_bytes: bytes) -> Tuple[List[Track], float, float]:
        uplink_ms = await self.net_sim.simulate_uplink(len(jpeg_bytes))
        msg = json.dumps({
            "msg_id":         "VISION",
            "frame_id":       int(time.time() * 1000),
            "robot_id":       "robot-001",
            "timestamp":      time.time(),
            "frame_jpeg_b64": base64.b64encode(jpeg_bytes).decode(),
        })
        async with __import__("websockets").connect(self.server_uri) as ws:
            t0 = time.perf_counter()
            await ws.send(msg)
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        rtt_ms  = (time.perf_counter() - t0) * 1000
        await self.net_sim.simulate_downlink(len(raw.encode()))
        resp     = json.loads(raw)
        proc_ms  = resp.get("server_processing_ms", 0.0)
        tracks   = self._tracker.update(resp.get("detections", []))
        return tracks, rtt_ms, proc_ms


# ──────────────────────────────────────────────────────────────────────────────
# Panel renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_panel(frame: np.ndarray, tracks: List[Track],
                  title: str, stats: str, color: Tuple) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]

    for tk in tracks:
        x1 = int(tk.bbox[0]*W); y1 = int(tk.bbox[1]*H)
        x2 = int(tk.bbox[2]*W); y2 = int(tk.bbox[3]*H)
        c  = tk.color
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

        # trail
        pts = [(int(px*W), int(py*H)) for px, py in tk.trail]
        for j in range(1, len(pts)):
            cv2.line(out, pts[j-1], pts[j], c, 2)

        # label
        tag = f"#{tk.id} {tk.label}"
        tw  = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        ty  = max(16, y1)
        cv2.rectangle(out, (x1, ty-14), (x1+tw+4, ty), c, -1)
        cv2.putText(out, tag, (x1+2, ty-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # panel header
    cv2.rectangle(out, (0, 0), (W, 44), (20, 20, 20), -1)
    cv2.putText(out, title, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(out, stats, (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # border
    cv2.rectangle(out, (0, 0), (W-1, H-1), color, 2)
    return out


def _render_raw_panel(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    cv2.rectangle(out, (0, 0), (W, 44), (20, 20, 20), -1)
    cv2.putText(out, "RAW INPUT", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(out, "Robot camera feed — no processing",
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)
    cv2.rectangle(out, (0, 0), (W-1, H-1), (100, 100, 100), 2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def _start_edge_server(port: int = 8767):
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
    threading.Thread(target=_run, daemon=True).start()
    if not _wait_for_port("localhost", port, timeout=30.0):
        print(f"[ERROR] Edge server failed to start on port {port}")
        sys.exit(1)


def run(input_path: str, output_path: str, headless: bool,
        max_frames: Optional[int], port: int = 8767, network_gen: str = "6G"):

    print("Starting edge server...")
    _start_edge_server(port)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {input_path}")
        print("Run: python demos/generate_test_video.py  to create a test video.")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    PW  = W   # panel width = original width
    # output is 3 panels wide
    out_w = PW * 3
    out_h = H

    print(f"Input: {input_path}  {W}x{H} @ {FPS:.0f}fps")
    print(f"Output: {output_path}  ({out_w}x{out_h})")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (out_w, out_h))

    print("Loading YOLOv8n (local inference)...")
    local_yolo  = LocalYOLO()
    edge_offload = EdgeOffload(f"ws://localhost:{port}", network_gen)

    frame_idx = 0
    rtt_history: deque = deque(maxlen=30)
    t_start = time.time()

    print("\nRunning comparison... (press Q to stop if display is shown)\n")
    print(f"{'Frame':>6}  {'LocalGPU':>9}  {'EdgeRTT':>9}  {'EdgeGPU':>9}")
    print("─" * 45)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break
        frame_idx += 1

        # resize to panel size
        panel_frame = cv2.resize(frame, (PW, H))

        # ── local inference (simulates onboard GPU) ──────────────────────
        local_tracks, local_ms = local_yolo.infer(panel_frame)

        # ── edge offload (simulates 6G) ───────────────────────────────────
        ok, jpeg_buf = cv2.imencode(".jpg", panel_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        edge_tracks, rtt_ms, proc_ms = edge_offload.infer(jpeg_buf.tobytes()) if ok else ([], 0, 0)
        rtt_history.append(rtt_ms)

        # ── build 3-panel composite ───────────────────────────────────────
        p_raw   = _render_raw_panel(panel_frame)
        p_local = _render_panel(panel_frame, local_tracks,
                                "LOCAL YOLOv8  (onboard GPU)",
                                f"Inference {local_ms:.0f}ms  |  {len(local_tracks)} tracks  [expensive hw]",
                                (50, 50, 220))
        p_edge  = _render_panel(panel_frame, edge_tracks,
                                f"6G EDGE OFFLOAD  [{network_gen}]",
                                f"RTT {rtt_ms:.0f}ms  GPU {proc_ms:.0f}ms  |  {len(edge_tracks)} tracks  [Pi viable]",
                                (50, 220, 80))

        composite = np.concatenate([p_raw, p_local, p_edge], axis=1)
        writer.write(composite)

        if not headless:
            display = cv2.resize(composite, (min(out_w, 1440), int(out_h * min(out_w, 1440) / out_w)))
            cv2.imshow("Tracking Comparison: Raw | Local YOLO | 6G Edge", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 5 == 0 or frame_idx <= 3:
            print(f"{frame_idx:>6}  {local_ms:>8.1f}ms  {rtt_ms:>8.1f}ms  {proc_ms:>8.1f}ms")

    cap.release()
    writer.release()
    if not headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_rtt = sum(rtt_history) / len(rtt_history) if rtt_history else 0
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"\n{'─'*45}")
    print(f"Done.  {frame_idx} frames in {elapsed:.1f}s")
    print(f"Avg 6G RTT: {avg_rtt:.1f}ms  |  Output: {out_path} ({size_mb:.1f} MB)")
    print(f"Play: ffplay {out_path}")


def main():
    ap = argparse.ArgumentParser(description="3-panel tracking comparison demo")
    ap.add_argument("--input",      default="assets/test_scene.mp4")
    ap.add_argument("--output",     default="assets/comparison_output.mp4")
    ap.add_argument("--network",    default="6G", choices=["4G", "5G", "6G"])
    ap.add_argument("--headless",   action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--port",       type=int, default=8767)
    args = ap.parse_args()

    run(
        input_path  = args.input,
        output_path = args.output,
        headless    = args.headless,
        max_frames  = args.max_frames,
        port        = args.port,
        network_gen = args.network,
    )


if __name__ == "__main__":
    main()
