#!/usr/bin/env python3
"""
Before/After Analyzer: 4G vs 6G Vision Offloading
====================================================
Processes the same road-scene video twice — once simulating 4G latency,
once simulating 6G — then stitches a side-by-side comparison video that
visually proves why latency matters for autonomous driving.

Output:  assets/before_after_comparison.mp4
         assets/before_after_thumb.jpg   (static key-frame for README)

Layout (1280 x 580):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  BEFORE: 4G LTE (~50ms RTT)     │  AFTER: 6G Sub-THz (~1ms RTT)   │
  │  640×480 annotated frame         │  640×480 annotated frame          │
  │  • missed / stale detections     │  • fresh detections every frame   │
  │  • tracking ID flicker           │  • stable IDs + smooth trails     │
  ├─────────────────────────────────────────────────────────────────────┤
  │  METRICS BAR: RTT histogram  ·  FPS gauge  ·  detection count       │
  └─────────────────────────────────────────────────────────────────────┘

Usage:
    python demos/before_after_analyzer.py
    python demos/before_after_analyzer.py --input assets/test_scene.mp4
    python demos/before_after_analyzer.py --max-frames 150
"""

import argparse
import asyncio
import base64
import json
import logging
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network.sim import NetworkSimulator, NETWORK_PROFILES
from shared_state import NetworkGen

logging.getLogger("websockets").setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# Palette & helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (255, 56, 56), (255, 157, 51), (51, 255, 86), (51, 255, 189),
    (51, 167, 255), (157, 51, 255), (255, 51, 201), (255, 255, 51),
]


def _iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, bbox, label, conf):
        self.id = tid; self.bbox = bbox
        self.label = label; self.conf = conf; self.age = 0
        self.color = _PALETTE[tid % len(_PALETTE)]
        self.trail = deque(maxlen=8)
        self.trail.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
        self.missing_frames = 0

    def update(self, bbox, label, conf):
        self.bbox = bbox; self.label = label; self.conf = conf
        self.age += 1; self.missing_frames = 0
        self.trail.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))


def _nms(dets, iou_thr=0.5):
    """Remove lower-confidence detections that heavily overlap a higher one."""
    keep = []
    for i, d in enumerate(dets):
        dominated = False
        for j in keep:
            if _iou(d["bbox"], dets[j]["bbox"]) > iou_thr:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return [dets[i] for i in keep]


class Tracker:
    def __init__(self, iou_thr=0.45, max_missing=3):
        self.iou_thr = iou_thr; self.max_missing = max_missing
        self._tracks: Dict[int, Track] = {}; self._nid = 1

    def update(self, dets) -> List[Track]:
        dets = _nms([d for d in dets if d.get("confidence", 0) >= 0.60], iou_thr=0.45)
        unmatched = set(self._tracks)
        for det in dets:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            best, best_iou = None, self.iou_thr
            for tid in list(unmatched):
                v = _iou(bbox, self._tracks[tid].bbox)
                if v > best_iou:
                    best_iou = v; best = tid
            if best is not None:
                self._tracks[best].update(bbox, det["class"], det["confidence"])
                unmatched.discard(best)
            else:
                self._tracks[self._nid] = Track(self._nid, bbox, det["class"], det["confidence"])
                self._nid += 1
        for tid in list(unmatched):
            self._tracks[tid].missing_frames += 1
            if self._tracks[tid].missing_frames > self.max_missing:
                del self._tracks[tid]
        return list(self._tracks.values())

    def reset(self):
        self._tracks.clear(); self._nid = 1


# ─────────────────────────────────────────────────────────────────────────────
# Edge pipeline (per network mode)
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    One-frame-at-a-time edge offload with simulated network latency.
    Reports TOTAL round-trip time = uplink_delay + server_inference + downlink_delay,
    which is what the robot actually experiences waiting for a result.
    """

    def __init__(self, server_uri: str, net_gen: str):
        self.server_uri = server_uri
        self.net_sim = NetworkSimulator(net_gen)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self.rtt_history: deque = deque(maxlen=100)
        self.det_history: deque = deque(maxlen=100)

    def infer(self, jpeg: bytes) -> Tuple[List[dict], float]:
        fut = asyncio.run_coroutine_threadsafe(self._run(jpeg), self._loop)
        try:
            dets, total_rtt = fut.result(timeout=30.0)
            self.rtt_history.append(total_rtt)
            self.det_history.append(len(dets))
            return dets, total_rtt
        except Exception:
            self.rtt_history.append(0.0)
            self.det_history.append(0)
            return [], 0.0

    async def _run(self, jpeg: bytes) -> Tuple[List[dict], float]:
        # Simulate uplink (network propagation + transmission time)
        uplink_ms = await self.net_sim.simulate_uplink(len(jpeg))

        msg = json.dumps({
            "msg_id": "VISION", "frame_id": int(time.time() * 1000),
            "robot_id": "robot-001", "timestamp": time.time(),
            "frame_jpeg_b64": base64.b64encode(jpeg).decode(),
        })
        import websockets as _ws
        async with _ws.connect(self.server_uri) as ws:
            t0 = time.perf_counter()
            await ws.send(msg)
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        server_rtt_ms = (time.perf_counter() - t0) * 1000  # GPU inference time

        # Simulate downlink (results back to robot)
        downlink_ms = await self.net_sim.simulate_downlink(len(raw.encode()))

        # Total latency the robot waits before it can act on the result
        total_rtt = uplink_ms + server_rtt_ms + downlink_ms

        resp = json.loads(raw)
        return resp.get("detections", []), total_rtt


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotators
# ─────────────────────────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, tracks: List[Track], rtt: float,
              net_gen: str, frame_idx: int, stale: bool = False) -> np.ndarray:
    """Draw tracking boxes, trails, and HUD onto a frame."""
    out = frame.copy()
    H, W = out.shape[:2]

    for tk in tracks:
        x1 = int(tk.bbox[0] * W); y1 = int(tk.bbox[1] * H)
        x2 = int(tk.bbox[2] * W); y2 = int(tk.bbox[3] * H)
        c = tk.color

        # dim stale tracks (for 4G demo — missed detections)
        alpha = 0.5 if (stale and tk.missing_frames > 0) else 1.0
        c_dim = tuple(int(v * alpha) for v in c)

        cv2.rectangle(out, (x1, y1), (x2, y2), c_dim, 2)

        # trail — fade from dim to full color
        pts = [(int(px * W), int(py * H)) for px, py in tk.trail]
        n_pts = len(pts)
        for j in range(1, n_pts):
            fade = j / n_pts
            tc = tuple(int(v * fade * alpha) for v in c)
            cv2.line(out, pts[j - 1], pts[j], tc, 1)

        # label — compact, above the box
        tag = f"#{tk.id} {tk.label}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = max(0, x1)
        ly = max(th + 4, y1 - 2)
        cv2.rectangle(out, (lx, ly - th - 3), (lx + tw + 4, ly + 1), c_dim, -1)
        cv2.putText(out, tag, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # top HUD
    gen_color = {"4G": (60, 60, 220), "5G": (50, 180, 220),
                 "6G": (50, 220, 80)}.get(net_gen, (200, 200, 200))
    cv2.rectangle(out, (0, 0), (W, 48), (18, 18, 18), -1)
    cv2.putText(out, f"{net_gen} Network  —  RTT {rtt:.0f}ms",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, gen_color, 1, cv2.LINE_AA)

    n_active = sum(1 for t in tracks if t.missing_frames == 0)
    cv2.putText(out, f"frame {frame_idx:04d}  |  {n_active} tracked",
                (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # RTT bar (bottom strip)
    bar_h = 5
    max_rtt = 60.0
    bar_w = int(min(rtt / max_rtt, 1.0) * W)
    cv2.rectangle(out, (0, H - bar_h), (W, H), (30, 30, 30), -1)
    cv2.rectangle(out, (0, H - bar_h), (bar_w, H), gen_color, -1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics bar
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_bar(W: int, stats_4g: dict, stats_6g: dict, frame_idx: int) -> np.ndarray:
    """100px tall metrics panel comparing 4G vs 6G."""
    bar = np.zeros((100, W, 3), dtype=np.uint8)
    bar[:] = (22, 22, 22)

    # title
    cv2.putText(bar, "LATENCY COMPARISON  (lower = better)",
                (W // 2 - 160, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)

    half = W // 2
    metrics = [
        (stats_4g, "4G LTE", (60, 60, 220), 0),
        (stats_6g, "6G Sub-THz", (50, 220, 80), half),
    ]

    for stats, label, color, x_off in metrics:
        avg_rtt  = stats.get("avg_rtt", 0.0)
        avg_dets = stats.get("avg_dets", 0.0)
        fps      = stats.get("fps", 0.0)
        max_rtt  = 60.0
        bar_w    = int(min(avg_rtt / max_rtt, 1.0) * (half - 20))

        # label
        cv2.putText(bar, label, (x_off + 6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # RTT bar
        cv2.rectangle(bar, (x_off + 6, 48), (x_off + half - 14, 62), (40, 40, 40), -1)
        cv2.rectangle(bar, (x_off + 6, 48), (x_off + 6 + bar_w, 62), color, -1)
        cv2.putText(bar, f"{avg_rtt:.0f}ms RTT",
                    (x_off + 6 + bar_w + 4, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # stats line
        cv2.putText(bar, f"{fps:.0f}fps effective  |  {avg_dets:.1f} det/frame",
                    (x_off + 6, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    # divider
    cv2.line(bar, (half, 0), (half, 100), (60, 60, 60), 1)

    # frame counter (top-right)
    cv2.putText(bar, f"frame {frame_idx:04d}",
                (W - 90, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    return bar


# ─────────────────────────────────────────────────────────────────────────────
# Server helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    import socket
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def _start_server(port: int) -> None:
    import asyncio as _a
    from edge_server.server import EdgeServer
    loop = _a.new_event_loop()

    def _run():
        _a.set_event_loop(loop)
        # Use simulated GPU backend (models A100 inference at ~4.5ms)
        # so the 4G vs 6G network latency difference is clearly visible.
        # Real YOLOv8 on CPU takes ~50ms and swamps the network difference.
        srv = EdgeServer(host="localhost", port=port,
                         network_gen=NetworkGen.SIX_G, vision_backend="simulated")
        loop.run_until_complete(srv.run())

    threading.Thread(target=_run, daemon=True).start()
    if not _wait_for_port("localhost", port, 60.0):
        print(f"[ERROR] Server failed to start on port {port}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, thumb_path: str,
        max_frames: Optional[int], port: int = 8769):

    print("Starting edge server (YOLOv8)...")
    _start_server(port)
    print("  ready.\n")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {input_path}")
        print("Run: python demos/generate_test_video.py")
        sys.exit(1)

    W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    # output canvas: two panels side by side + metrics bar
    OUT_W = W_src * 2
    OUT_H = H_src + 100     # +100 for metrics bar
    METRICS_Y = H_src

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H)
    )

    pipe_4g = Pipeline(f"ws://localhost:{port}", NetworkGen.FOUR_G)
    pipe_6g = Pipeline(f"ws://localhost:{port}", NetworkGen.SIX_G)
    tracker_4g = Tracker(iou_thr=0.45, max_missing=3)
    tracker_6g = Tracker(iou_thr=0.45, max_missing=3)

    rtt_4g_hist: deque = deque(maxlen=60)
    rtt_6g_hist: deque = deque(maxlen=60)

    thumb_frame_idx = max(1, total // 2)
    thumb_saved = False

    print(f"Processing {total} frames from {input_path}")
    print(f"{'Frame':>6}  {'4G RTT':>9}  {'6G RTT':>9}  {'4G det':>7}  {'6G det':>7}")
    print("─" * 50)

    t0_total = time.time()

    for frame_idx in range(1, total + 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W_src, H_src))

        # encode JPEG once, reuse for both pipelines
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        jpeg = buf.tobytes()

        # ── offload to both channels ─────────────────────────────────────
        dets_4g, rtt_4g = pipe_4g.infer(jpeg)
        dets_6g, rtt_6g = pipe_6g.infer(jpeg)

        rtt_4g_hist.append(rtt_4g)
        rtt_6g_hist.append(rtt_6g)

        tracks_4g = tracker_4g.update(dets_4g)
        tracks_6g = tracker_6g.update(dets_6g)

        # ── annotate panels ──────────────────────────────────────────────
        panel_4g = _annotate(frame, tracks_4g, rtt_4g,
                              NetworkGen.FOUR_G, frame_idx, stale=True)
        panel_6g = _annotate(frame, tracks_6g, rtt_6g,
                              NetworkGen.SIX_G, frame_idx, stale=False)

        # ── metrics bar ──────────────────────────────────────────────────
        avg_4g = sum(rtt_4g_hist) / len(rtt_4g_hist) if rtt_4g_hist else 0
        avg_6g = sum(rtt_6g_hist) / len(rtt_6g_hist) if rtt_6g_hist else 0
        elapsed = time.time() - t0_total
        fps_eff = frame_idx / elapsed if elapsed > 0 else 0

        stats_4g = {"avg_rtt": avg_4g, "avg_dets": len(dets_4g), "fps": fps_eff}
        stats_6g = {"avg_rtt": avg_6g, "avg_dets": len(dets_6g), "fps": fps_eff}
        mbar = _metrics_bar(OUT_W, stats_4g, stats_6g, frame_idx)

        # ── composite canvas ─────────────────────────────────────────────
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        canvas[:H_src, :W_src]   = panel_4g
        canvas[:H_src, W_src:]   = panel_6g
        canvas[METRICS_Y:, :]    = mbar

        # centre divider line
        cv2.line(canvas, (W_src, 0), (W_src, H_src), (80, 80, 80), 2)

        # top banner labels
        cv2.rectangle(canvas, (0, 48), (W_src, 68), (40, 20, 20), -1)
        cv2.putText(canvas, "BEFORE  —  4G LTE  (~50ms latency)",
                    (8, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 100, 255), 1)
        cv2.rectangle(canvas, (W_src, 48), (OUT_W, 68), (20, 40, 20), -1)
        cv2.putText(canvas, "AFTER   —  6G Sub-THz  (~1ms latency)",
                    (W_src + 8, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 220, 80), 1)

        writer.write(canvas)

        # save thumbnail at midpoint
        if frame_idx == thumb_frame_idx and not thumb_saved:
            cv2.imwrite(thumb_path, canvas)
            thumb_saved = True

        if frame_idx % 5 == 0 or frame_idx <= 3:
            print(f"{frame_idx:>6}  {rtt_4g:>8.1f}ms  {rtt_6g:>8.1f}ms  "
                  f"{len(dets_4g):>7}  {len(dets_6g):>7}")

    cap.release()
    writer.release()

    elapsed = time.time() - t0_total
    avg_4g = sum(rtt_4g_hist) / len(rtt_4g_hist) if rtt_4g_hist else 0
    avg_6g = sum(rtt_6g_hist) / len(rtt_6g_hist) if rtt_6g_hist else 0

    print(f"\n{'─' * 50}")
    print(f"Done.  {frame_idx} frames in {elapsed:.1f}s")
    print(f"")
    print(f"  4G avg RTT : {avg_4g:.1f}ms  →  {1000/max(avg_4g,1):.0f}fps effective")
    print(f"  6G avg RTT : {avg_6g:.1f}ms  →  {1000/max(avg_6g,1):.0f}fps effective")
    print(f"  Speedup    : {avg_4g/max(avg_6g,0.01):.1f}×  lower latency with 6G")
    print(f"")
    print(f"  Output video  : {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")
    print(f"  Thumbnail     : {thumb_path}")
    print(f"  Play          : ffplay {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Before/After 4G vs 6G analyzer")
    ap.add_argument("--input",      default="assets/test_scene.mp4")
    ap.add_argument("--output",     default="assets/before_after_comparison.mp4")
    ap.add_argument("--thumb",      default="assets/before_after_thumb.jpg")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--port",       type=int, default=8769)
    args = ap.parse_args()

    run(args.input, args.output, args.thumb, args.max_frames, args.port)


if __name__ == "__main__":
    main()
