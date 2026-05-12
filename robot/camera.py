import base64
import io
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


@dataclass
class SceneObject:
    label: str
    x: float        # normalized center x [0, 1]
    y: float        # normalized center y [0, 1]
    w: float        # normalized width
    h: float        # normalized height
    vx: float       # x velocity per frame
    vy: float       # y velocity per frame
    color: Tuple[int, int, int]


_TEMPLATES = [
    {"label": "car",        "color": (30,  120, 200), "w": 0.15, "h": 0.10},
    {"label": "car",        "color": (200,  60,  60), "w": 0.14, "h": 0.09},
    {"label": "car",        "color": (60,  180,  60), "w": 0.13, "h": 0.09},
    {"label": "truck",      "color": (100, 100, 100), "w": 0.22, "h": 0.14},
    {"label": "person",     "color": (240, 180,  80), "w": 0.04, "h": 0.13},
    {"label": "person",     "color": (180, 120, 240), "w": 0.04, "h": 0.12},
    {"label": "bicycle",    "color": (255, 165,   0), "w": 0.05, "h": 0.10},
    {"label": "stop_sign",  "color": (220,  40,  40), "w": 0.05, "h": 0.06},
    {"label": "traffic_light", "color": (0, 200, 0),  "w": 0.03, "h": 0.07},
]


class CameraSimulator:
    """
    Generates synthetic 640x480 road-scene JPEG frames.
    Objects move with realistic kinematics; used as ground-truth
    for evaluating edge-server detection accuracy.
    """

    WIDTH  = 640
    HEIGHT = 480

    def __init__(self, jpeg_quality: int = 85):
        self.jpeg_quality = jpeg_quality
        self.frame_count   = 0
        self._t0           = time.time()
        self._objects: List[SceneObject] = []
        self._init_scene()

    # ------------------------------------------------------------------ #
    def _init_scene(self):
        n = random.randint(4, 7)
        templates = random.sample(_TEMPLATES, min(n, len(_TEMPLATES)))
        for t in templates:
            self._objects.append(SceneObject(
                label=t["label"],
                x=random.uniform(0.05, 0.95),
                y=random.uniform(0.55, 0.88),
                w=t["w"] * random.uniform(0.8, 1.3),
                h=t["h"] * random.uniform(0.8, 1.3),
                vx=random.choice([-1, 1]) * random.uniform(0.001, 0.004),
                vy=random.uniform(-0.0005, 0.0005),
                color=t["color"],
            ))

    def _update_objects(self):
        for o in self._objects:
            o.x = (o.x + o.vx) % 1.0
            o.y = max(0.52, min(0.91, o.y + o.vy))

    # ------------------------------------------------------------------ #
    def _render_pil(self) -> bytes:
        img  = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(img)

        # Sky gradient (approximate with two rectangles)
        draw.rectangle([0, 0, self.WIDTH, self.HEIGHT // 2 - 20], fill=(100, 160, 220))
        draw.rectangle([0, self.HEIGHT // 2 - 20, self.WIDTH, self.HEIGHT // 2], fill=(140, 180, 210))

        # Road surface
        draw.rectangle([0, self.HEIGHT // 2, self.WIDTH, self.HEIGHT], fill=(70, 70, 72))

        # Lane markings (perspective-style)
        cx = self.WIDTH // 2
        hy = self.HEIGHT // 2
        for offset in [-200, -100, 0, 100, 200]:
            bx = cx + offset
            draw.line([(cx, hy), (bx, self.HEIGHT)], fill=(230, 220, 50), width=2)

        # Dashed center line
        for seg in range(6):
            y_start = hy + seg * 40
            y_end   = y_start + 20
            x_start = cx + (y_start - hy) * 0
            x_end   = cx + (y_end - hy) * 0
            draw.line([(cx, y_start), (cx, min(y_end, self.HEIGHT))], fill=(255, 255, 255), width=2)

        # Scene objects
        for o in self._objects:
            x1 = int((o.x - o.w / 2) * self.WIDTH)
            y1 = int((o.y - o.h / 2) * self.HEIGHT)
            x2 = int((o.x + o.w / 2) * self.WIDTH)
            y2 = int((o.y + o.h / 2) * self.HEIGHT)
            draw.rectangle([x1, y1, x2, y2], fill=o.color, outline=(255, 255, 255), width=2)
            draw.text((max(0, x1 + 2), max(0, y1 + 2)), o.label, fill=(255, 255, 255))

        # HUD overlay
        ts = f"Frame #{self.frame_count:05d}  |  {time.strftime('%H:%M:%S')}"
        draw.text((6, 5),  ts,                         fill=(255, 255,   0))
        draw.text((6, 20), "Robot-001 | Rutgers MXS",  fill=(200, 200, 200))
        draw.text((6, 35), f"Obj: {len(self._objects)}", fill=(200, 200, 200))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def _render_numpy_fallback(self) -> bytes:
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:self.HEIGHT // 2] = [100, 160, 220]
        frame[self.HEIGHT // 2:] = [70, 70, 72]
        for o in self._objects:
            x1 = max(0, int((o.x - o.w / 2) * self.WIDTH))
            y1 = max(0, int((o.y - o.h / 2) * self.HEIGHT))
            x2 = min(self.WIDTH  - 1, int((o.x + o.w / 2) * self.WIDTH))
            y2 = min(self.HEIGHT - 1, int((o.y + o.h / 2) * self.HEIGHT))
            frame[y1:y2, x1:x2] = list(o.color)
        # return raw data as stand-in (no JPEG encoder without PIL/cv2)
        size = int(self.WIDTH * self.HEIGHT * 0.12)
        header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        payload = frame.tobytes()[:size]
        return header + payload + b'\xff\xd9'

    # ------------------------------------------------------------------ #
    def capture(self) -> Tuple[bytes, List[SceneObject]]:
        """Return (jpeg_bytes, ground_truth_objects)."""
        self._update_objects()
        self.frame_count += 1
        jpeg = self._render_pil() if _HAS_PIL else self._render_numpy_fallback()
        return jpeg, list(self._objects)

    @property
    def ground_truth(self) -> List[SceneObject]:
        return list(self._objects)
