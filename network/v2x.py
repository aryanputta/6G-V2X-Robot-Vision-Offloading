import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional


class MsgType:
    BSM = "BSM"
    SPAT = "SPAT"
    SPAT_REQUEST = "SPAT_REQUEST"
    VISION = "VISION"
    DETECTION = "DETECTION"
    MAP = "MAP"


@dataclass
class BSM:
    """SAE J2735 Basic Safety Message — periodic 10Hz broadcast."""
    msg_id: str = MsgType.BSM
    msg_count: int = 0
    vehicle_id: str = ""
    timestamp: float = field(default_factory=time.time)
    lat: float = 40.5008     # Rutgers University lat
    lon: float = -74.4474    # Rutgers University lon
    elev_m: float = 10.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    accel_mps2: float = 0.0
    brake_status: str = "none"
    width_m: float = 0.5
    length_m: float = 0.8

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "BSM":
        return cls(**json.loads(s))


@dataclass
class SPATMessage:
    """Signal Phase and Timing message from traffic infrastructure RSU."""
    msg_id: str = MsgType.SPAT
    intersection_id: str = "RU-BUSCH-01"
    timestamp: float = field(default_factory=time.time)
    phase: str = "GREEN"
    time_remaining_s: float = 30.0
    next_phase: str = "YELLOW"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class VisionOffloadRequest:
    """Custom V2X: robot → edge server, carries JPEG frame for inference."""
    msg_id: str = MsgType.VISION
    frame_id: int = 0
    robot_id: str = ""
    timestamp: float = field(default_factory=time.time)
    frame_jpeg_b64: str = ""
    frame_width: int = 640
    frame_height: int = 480
    priority: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "VisionOffloadRequest":
        return cls(**json.loads(s))


@dataclass
class VisionOffloadResponse:
    """Custom V2X: edge server → robot, carries detection results."""
    msg_id: str = MsgType.DETECTION
    frame_id: int = 0
    robot_id: str = ""
    timestamp: float = field(default_factory=time.time)
    server_processing_ms: float = 0.0
    detections: List[Dict] = field(default_factory=list)
    network_gen: str = "6G"
    edge_server_id: str = "edge-01"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "VisionOffloadResponse":
        return cls(**json.loads(s))


class V2XHub:
    """
    Simulates a Roadside Unit (RSU) at Rutgers Busch Campus intersection.
    Broadcasts SPAT (traffic light timing) at 10Hz.
    """

    _PHASES = ["GREEN", "YELLOW", "RED"]
    _DURATIONS = {"GREEN": 30.0, "YELLOW": 5.0, "RED": 25.0}

    def __init__(self, intersection_id: str = "RU-BUSCH-01"):
        self.intersection_id = intersection_id
        self._phase_idx = 0
        self._phase_start = time.time()

    def get_spat(self) -> SPATMessage:
        now = time.time()
        current = self._PHASES[self._phase_idx]
        duration = self._DURATIONS[current]
        elapsed = now - self._phase_start

        if elapsed >= duration:
            self._phase_idx = (self._phase_idx + 1) % len(self._PHASES)
            self._phase_start = now
            elapsed = 0.0
            current = self._PHASES[self._phase_idx]
            duration = self._DURATIONS[current]

        next_phase = self._PHASES[(self._phase_idx + 1) % len(self._PHASES)]
        return SPATMessage(
            intersection_id=self.intersection_id,
            phase=current,
            time_remaining_s=round(duration - elapsed, 1),
            next_phase=next_phase,
        )
