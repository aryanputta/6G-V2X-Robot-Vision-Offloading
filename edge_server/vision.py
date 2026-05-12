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
        n = random.randint(2, 5)
        pool = random.sample(_ROAD_SCENE_CLASSES, min(n, len(_ROAD_SCENE_CLASSES)))
        dets = []
        for cls, lo, hi in pool:
            x1 = random.uniform(0.0, 0.65)
            y1 = random.uniform(0.40, 0.72)
            x2 = min(1.0, x1 + random.uniform(0.08, 0.24))
            y2 = min(1.0, y1 + random.uniform(0.06, 0.18))
            dets.append(Detection(
                class_name=cls,
                confidence=round(random.uniform(lo, hi), 3),
                bbox=[round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            ))
        return sorted(dets, key=lambda d: -d.confidence)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_processing_ms(self) -> float:
        return self._total_ms / self._frame_count if self._frame_count else 0.0
