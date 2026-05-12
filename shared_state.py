import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Deque


@dataclass
class FrameMetrics:
    frame_id: int
    rtt_ms: float
    server_processing_ms: float
    uplink_ms: float
    downlink_ms: float
    n_detections: int
    frame_size_bytes: int
    timestamp: float = field(default_factory=time.time)


class NetworkGen:
    FOUR_G = "4G"
    FIVE_G = "5G"
    SIX_G = "6G"

    ALL = ["4G", "5G", "6G"]


class SharedState:
    """Thread-safe shared state between asyncio backend and rich dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self.network_gen: str = NetworkGen.SIX_G
        self.rtt_history: Deque[float] = deque(maxlen=60)
        self.current_detections: List[Dict] = []
        self.recent_metrics: Deque[FrameMetrics] = deque(maxlen=200)
        self.v2x_log: Deque[str] = deque(maxlen=10)
        self.spat_state: Dict = {"phase": "GREEN", "time_remaining_s": 30.0, "next_phase": "YELLOW"}
        self.frame_count: int = 0
        self.bytes_transferred: int = 0
        self.packets_dropped: int = 0
        self.server_ready: bool = False
        self._running: bool = True
        self._network_change_requested: Optional[str] = None
        self._start_time: float = time.time()

    def update_metrics(self, m: FrameMetrics):
        with self._lock:
            self.rtt_history.append(m.rtt_ms)
            self.frame_count += 1
            self.bytes_transferred += m.frame_size_bytes
            self.recent_metrics.append(m)

    def update_detections(self, detections: List[Dict]):
        with self._lock:
            self.current_detections = list(detections)

    def add_v2x_log(self, msg: str):
        with self._lock:
            self.v2x_log.append(msg)

    def update_spat(self, spat: Dict):
        with self._lock:
            self.spat_state = dict(spat)

    def request_network_change(self, gen: str):
        with self._lock:
            self._network_change_requested = gen

    def consume_network_change(self) -> Optional[str]:
        with self._lock:
            c = self._network_change_requested
            self._network_change_requested = None
            return c

    def set_network(self, gen: str):
        with self._lock:
            self.network_gen = gen

    def mark_packet_dropped(self):
        with self._lock:
            self.packets_dropped += 1

    @property
    def avg_rtt_ms(self) -> float:
        with self._lock:
            rtts = list(self.rtt_history)
        return sum(rtts) / len(rtts) if rtts else 0.0

    @property
    def p95_rtt_ms(self) -> float:
        with self._lock:
            rtts = sorted(self.rtt_history)
        if not rtts:
            return 0.0
        idx = int(len(rtts) * 0.95)
        return rtts[min(idx, len(rtts) - 1)]

    @property
    def current_fps(self) -> float:
        with self._lock:
            metrics = list(self.recent_metrics)
        if len(metrics) < 2:
            return 0.0
        dt = metrics[-1].timestamp - metrics[0].timestamp
        return len(metrics) / dt if dt > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
