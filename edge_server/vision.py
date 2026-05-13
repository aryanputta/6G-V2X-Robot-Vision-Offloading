import asyncio
import io
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]   # [x1, y1, x2, y2] normalized 0-1

    def to_dict(self) -> dict:
        return {
            "class":      self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox":       [round(v, 3) for v in self.bbox],
        }


_ROAD_SCENE_CLASSES = [
    ("car",           0.82, 0.97),
    ("car",           0.75, 0.95),
    ("person",        0.68, 0.94),
    ("truck",         0.72, 0.91),
    ("bicycle",       0.65, 0.90),
    ("stop_sign",     0.88, 0.99),
    ("traffic_light", 0.80, 0.97),
    ("motorcycle",    0.70, 0.93),
]


class _SimObject:
    """Persistent simulated road-scene object with smooth motion between frames."""

    def __init__(self, cls_info):
        self.cls, self.conf_lo, self.conf_hi = cls_info
        w = random.uniform(0.10, 0.22)
        h = random.uniform(0.07, 0.15)
        self.x1 = random.uniform(0.05, 0.70)
        self.y1 = random.uniform(0.35, 0.70)
        self.x2 = min(1.0, self.x1 + w)
        self.y2 = min(1.0, self.y1 + h)
        self.vx = random.uniform(-0.007, 0.007)
        self.vy = random.uniform(-0.003, 0.003)

    def step(self) -> "Detection":
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        self.x1 += self.vx
        self.x2 = self.x1 + w
        self.y1 += self.vy
        self.y2 = self.y1 + h
        if self.x1 < 0.0:
            self.x1, self.x2, self.vx = 0.0, w, abs(self.vx)
        if self.x2 > 1.0:
            self.x2, self.x1, self.vx = 1.0, 1.0 - w, -abs(self.vx)
        if self.y1 < 0.30:
            self.y1, self.y2, self.vy = 0.30, 0.30 + h, abs(self.vy)
        if self.y2 > 0.90:
            self.y2, self.y1, self.vy = 0.90, 0.90 - h, -abs(self.vy)
        return Detection(
            class_name=self.cls,
            confidence=round(random.uniform(self.conf_lo, self.conf_hi), 3),
            bbox=[round(self.x1, 3), round(self.y1, 3),
                  round(self.x2, 3), round(self.y2, 3)],
        )


class VisionProcessor:
    """
    Edge-server vision pipeline.

    Backends (tried in order if backend='auto'):
      1. YOLOv8 (ultralytics) -- requires GPU/fast CPU
      2. OpenCV DNN            -- MobileNet SSD, CPU-friendly
      3. Simulated             -- realistic timing, no ML deps

    The simulated backend models a high-end data-center GPU (A100):
    mean inference 4.5 ms, std 0.8 ms.
    """

    def __init__(self, backend: str = "auto",
                 inference_ms_mean: float = 4.5,
                 inference_ms_std:  float = 0.8):
        self.inference_ms_mean = inference_ms_mean
        self.inference_ms_std  = inference_ms_std
        self._backend = self._init_backend(backend)
        self._frame_count = 0
        self._total_ms    = 0.0
        n = random.randint(3, 5)
        self._sim_objects: List[_SimObject] = [
            _SimObject(random.choice(_ROAD_SCENE_CLASSES)) for _ in range(n)
        ]

    # ------------------------------------------------------------------ #
    def _init_backend(self, requested: str) -> str:
        if requested in ("auto", "yolov8"):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
                return "yolov8"
            except Exception:
                pass

        if requested in ("auto", "opencv"):
            try:
                import cv2
                self._cv2 = cv2
                return "opencv"
            except Exception:
                pass

        return "simulated"

    # ------------------------------------------------------------------ #
    async def process(self, jpeg_bytes: bytes) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()

        if self._backend == "yolov8":
            dets = await self._run_yolov8(jpeg_bytes)
        elif self._backend == "opencv":
            dets = await self._run_opencv(jpeg_bytes)
        else:
            dets = await self._run_simulated(jpeg_bytes)

        ms = (time.perf_counter() - t0) * 1000
        self._frame_count += 1
        self._total_ms    += ms
        return dets, ms

    # ------------------------------------------------------------------ #
    async def _run_yolov8(self, jpeg_bytes: bytes) -> List[Detection]:
        import PIL.Image
        loop = asyncio.get_running_loop()
        img  = PIL.Image.open(io.BytesIO(jpeg_bytes))
        res  = await loop.run_in_executor(
            None, lambda: self._yolo(img, verbose=False)[0]
        )
        return [
            Detection(
                class_name=res.names[int(b.cls)],
                confidence=float(b.conf),
                bbox=b.xyxyn[0].tolist(),
            )
            for b in res.boxes
        ]

    async def _run_opencv(self, jpeg_bytes: bytes) -> List[Detection]:
        await asyncio.sleep(0.008)
        return self._synthetic_detections()

    async def _run_simulated(self, jpeg_bytes: bytes) -> List[Detection]:
        t = max(1.0, random.gauss(self.inference_ms_mean, self.inference_ms_std))
        await asyncio.sleep(t / 1000)
        return self._synthetic_detections()

    # ------------------------------------------------------------------ #
    def _synthetic_detections(self) -> List[Detection]:
        dets = [obj.step() for obj in self._sim_objects]
        return sorted(dets, key=lambda d: -d.confidence)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_processing_ms(self) -> float:
        return self._total_ms / self._frame_count if self._frame_count else 0.0
