#!/usr/bin/env python3
"""
Generate a synthetic road-scene test video for the 6G demo.

Produces assets/test_scene.mp4  (640x480, 30fps, ~10s)
with moving cars, a pedestrian, a stop sign, and traffic light
so demos work without a real camera.

Usage:
    python demos/generate_test_video.py
    python demos/generate_test_video.py --output my_scene.mp4 --duration 15
"""

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Scene object definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneActor:
    label: str
    x: float          # center-x (0..1)
    y: float          # center-y (0..1)
    w: float          # width  (0..1)
    h: float          # height (0..1)
    vx: float = 0.0   # velocity per frame
    vy: float = 0.0
    color: Tuple[int, int, int] = (100, 100, 200)  # BGR
    static: bool = False
    shadow: bool = True

    # internal
    trail: List[Tuple[int, int]] = field(default_factory=list)

    def move(self, W: int, H: int):
        if self.static:
            return
        self.x += self.vx
        self.y += self.vy
        if self.x < -self.w:
            self.x = 1.0 + self.w
        if self.x > 1.0 + self.w:
            self.x = -self.w
        cx = int(self.x * W)
        cy = int(self.y * H)
        self.trail.append((cx, cy))
        if len(self.trail) > 30:
            self.trail.pop(0)

    def bbox_px(self, W: int, H: int) -> Tuple[int, int, int, int]:
        x1 = int((self.x - self.w / 2) * W)
        y1 = int((self.y - self.h / 2) * H)
        x2 = int((self.x + self.w / 2) * W)
        y2 = int((self.y + self.h / 2) * H)
        return (
            max(0, x1), max(0, y1),
            min(W - 1, x2), min(H - 1, y2),
        )


def _make_scene(W: int, H: int) -> List[SceneActor]:
    return [
        # --- moving cars ---
        SceneActor("car",   x=0.15, y=0.68, w=0.18, h=0.10,
                   vx=0.004, color=(220, 80, 50)),
        SceneActor("car",   x=0.55, y=0.75, w=0.16, h=0.10,
                   vx=-0.003, color=(50, 120, 220)),
        SceneActor("car",   x=0.80, y=0.65, w=0.14, h=0.09,
                   vx=0.0025, color=(60, 180, 60)),
        SceneActor("truck", x=0.35, y=0.72, w=0.24, h=0.13,
                   vx=-0.002, color=(80, 80, 80)),
        # --- pedestrian ---
        SceneActor("person", x=0.62, y=0.60, w=0.04, h=0.12,
                   vx=0.001, color=(30, 200, 200)),
        # --- static infrastructure ---
        SceneActor("stop_sign",     x=0.88, y=0.54, w=0.05, h=0.06,
                   static=True, color=(0, 0, 200)),
        SceneActor("traffic_light", x=0.10, y=0.50, w=0.03, h=0.08,
                   static=True, color=(0, 200, 0)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────────────

def _sky(frame: np.ndarray, H: int, W: int, t: float):
    """Animated sky with moving clouds."""
    horizon = H // 2 - 20
    for row in range(horizon):
        alpha = row / horizon
        r = int(180 * (1 - alpha) + 100 * alpha)
        g = int(210 * (1 - alpha) + 150 * alpha)
        b = int(255)
        frame[row, :] = (b, g, r)

    # simple cloud ellipses
    cloud_x = int((math.sin(t * 0.1) * 0.3 + 0.5) * W)
    cv2.ellipse(frame, (cloud_x, horizon // 2), (80, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, (cloud_x + 60, horizon // 2 - 10), (50, 15), 0, 0, 360, (245, 245, 245), -1)


def _road(frame: np.ndarray, H: int, W: int):
    """Draw road with perspective lane markings."""
    horizon = H // 2 - 20
    road_color = np.array([60, 62, 65], dtype=np.uint8)

    # road surface
    frame[horizon:, :] = road_color

    # kerb
    frame[horizon:horizon + 4, :] = [200, 200, 200]

    # perspective vanishing-point lane lines
    vp_x, vp_y = W // 2, horizon
    for lane_x_bot in [W // 4, W // 2, 3 * W // 4]:
        cv2.line(frame, (vp_x, vp_y), (lane_x_bot, H), (220, 210, 50), 2)

    # dashed center line
    for seg in range(8):
        y1 = horizon + seg * 40
        y2 = y1 + 18
        if y2 > H:
            break
        # perspective x shift
        t1 = (y1 - horizon) / (H - horizon)
        t2 = (y2 - horizon) / (H - horizon)
        x1 = int(vp_x + (W // 2 - vp_x) * t1 * 0.5)
        x2 = int(vp_x + (W // 2 - vp_x) * t2 * 0.5)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def _draw_actor(frame: np.ndarray, actor: SceneActor, W: int, H: int, frame_idx: int):
    x1, y1, x2, y2 = actor.bbox_px(W, H)
    if x1 >= x2 or y1 >= y2:
        return

    bgr = actor.color

    # shadow
    if actor.shadow:
        sh = int((y2 - y1) * 0.12)
        cv2.ellipse(frame, ((x1 + x2) // 2, y2 + sh // 2),
                    ((x2 - x1) // 2, sh), 0, 0, 360, (30, 30, 30), -1)

    # body
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)

    # highlight edge
    cv2.rectangle(frame, (x1, y1), (x2, y2), (min(bgr[0] + 60, 255),
                                               min(bgr[1] + 60, 255),
                                               min(bgr[2] + 60, 255)), 1)

    # simple windscreen for cars/trucks
    if actor.label in ("car", "truck"):
        ws_x1 = x1 + (x2 - x1) // 4
        ws_y1 = y1 + (y2 - y1) // 5
        ws_x2 = x2 - (x2 - x1) // 4
        ws_y2 = y1 + (y2 - y1) // 2
        cv2.rectangle(frame, (ws_x1, ws_y1), (ws_x2, ws_y2), (180, 220, 240), -1)
        # windscreen glare
        cv2.line(frame, (ws_x1 + 2, ws_y1 + 2), (ws_x1 + 8, ws_y2 - 2), (220, 240, 255), 1)

    # wheels for vehicles
    if actor.label in ("car", "truck"):
        wr = max(3, (y2 - y1) // 5)
        wq = (x2 - x1) // 5
        for wx in [x1 + wq, x2 - wq]:
            cv2.circle(frame, (wx, y2), wr, (20, 20, 20), -1)
            cv2.circle(frame, (wx, y2), max(1, wr - 2), (60, 60, 60), -1)

    # traffic-light lenses
    if actor.label == "traffic_light":
        lx = (x1 + x2) // 2
        spacing = (y2 - y1) // 3
        cv2.circle(frame, (lx, y1 + spacing // 2), 4, (0, 0, 220), -1)
        cv2.circle(frame, (lx, y1 + spacing + spacing // 2), 4, (0, 180, 220), -1)
        g_pulse = int(128 + 127 * math.sin(frame_idx * 0.1))
        cv2.circle(frame, (lx, y1 + 2 * spacing + spacing // 2), 4, (0, g_pulse, 0), -1)

    # stop-sign octagon
    if actor.label == "stop_sign":
        cx_s, cy_s = (x1 + x2) // 2, (y1 + y2) // 2
        r = (x2 - x1) // 2
        pts = np.array([
            [cx_s + int(r * math.cos(math.radians(i * 45 + 22.5))),
             cy_s + int(r * math.sin(math.radians(i * 45 + 22.5)))]
            for i in range(8)
        ], np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 200))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)
        cv2.putText(frame, "STOP", (cx_s - 11, cy_s + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # label tag
    label_y = max(0, y1 - 5)
    cv2.rectangle(frame, (x1, label_y - 14), (x1 + len(actor.label) * 7 + 4, label_y),
                  bgr, -1)
    cv2.putText(frame, actor.label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def _hud(frame: np.ndarray, frame_idx: int, fps: float, n_objects: int):
    """Heads-Up Display overlay."""
    cv2.rectangle(frame, (0, 0), (310, 52), (20, 20, 20), -1)
    cv2.putText(frame, f"SYNTHETIC SCENE  |  frame {frame_idx:04d}",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Rutgers Makerspace Robot-001",
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)
    cv2.putText(frame, f"{fps:.0f}fps  |  {n_objects} objects",
                (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────

def generate(output_path: str, duration_s: float = 10.0,
             fps: int = 30, W: int = 640, H: int = 480) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (W, H))

    actors = _make_scene(W, H)
    total_frames = int(duration_s * fps)

    print(f"Generating {total_frames} frames → {out}")
    t0 = time.time()

    for i in range(total_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        t = i / fps

        _sky(frame, H, W, t)
        _road(frame, H, W)

        for actor in actors:
            actor.move(W, H)

        # draw back-to-front (simple depth sort by y)
        for actor in sorted(actors, key=lambda a: a.y):
            _draw_actor(frame, actor, W, H, i)

        _hud(frame, i, fps, len(actors))

        writer.write(frame)

        if i % fps == 0:
            elapsed = time.time() - t0
            pct = i / total_frames * 100
            eta = elapsed / max(i, 1) * (total_frames - i)
            print(f"  {pct:5.1f}%  frame {i}/{total_frames}  ETA {eta:.0f}s")

    writer.release()
    size_mb = out.stat().st_size / 1e6
    print(f"\nSaved: {out}  ({size_mb:.1f} MB,  {total_frames} frames,  {duration_s:.0f}s)")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic road-scene test video")
    ap.add_argument("--output",   default="assets/test_scene.mp4")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds")
    ap.add_argument("--fps",      type=int,   default=30)
    ap.add_argument("--width",    type=int,   default=640)
    ap.add_argument("--height",   type=int,   default=480)
    args = ap.parse_args()
    generate(args.output, args.duration, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
