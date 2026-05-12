import asyncio
import random
import time
from dataclasses import dataclass
from typing import Dict

from shared_state import NetworkGen


@dataclass
class NetworkProfile:
    name: str
    generation: str
    one_way_latency_ms: float
    bandwidth_gbps: float
    jitter_ms: float
    packet_loss_pct: float
    description: str


NETWORK_PROFILES: Dict[str, NetworkProfile] = {
    NetworkGen.FOUR_G: NetworkProfile(
        name="4G LTE",
        generation=NetworkGen.FOUR_G,
        one_way_latency_ms=25.0,
        bandwidth_gbps=0.1,
        jitter_ms=5.0,
        packet_loss_pct=0.1,
        description="Legacy LTE — ~50ms RTT, 100 Mbps peak",
    ),
    NetworkGen.FIVE_G: NetworkProfile(
        name="5G NR mmWave",
        generation=NetworkGen.FIVE_G,
        one_way_latency_ms=4.0,
        bandwidth_gbps=1.0,
        jitter_ms=0.5,
        packet_loss_pct=0.01,
        description="5G mmWave — ~8ms RTT, 1 Gbps peak",
    ),
    NetworkGen.SIX_G: NetworkProfile(
        name="6G Sub-THz",
        generation=NetworkGen.SIX_G,
        one_way_latency_ms=0.5,
        bandwidth_gbps=100.0,
        jitter_ms=0.05,
        packet_loss_pct=0.001,
        description="6G Sub-THz — ~1ms RTT, 100 Gbps peak",
    ),
}


class NetworkSimulator:
    """
    Simulates 4G/5G/6G channel characteristics.

    Modeled parameters:
      - Propagation latency (Rayleigh-faded with jitter)
      - Transmission delay based on link bandwidth
      - Probabilistic packet loss
    """

    def __init__(self, generation: str = NetworkGen.SIX_G):
        self.generation = generation
        self.profile = NETWORK_PROFILES[generation]
        self._bytes_uplink: int = 0
        self._bytes_downlink: int = 0
        self._packets_dropped: int = 0
        self._start_time: float = time.time()

    def set_network(self, generation: str):
        self.generation = generation
        self.profile = NETWORK_PROFILES[generation]

    def _one_way_delay_ms(self, data_bytes: int) -> float:
        latency = self.profile.one_way_latency_ms
        jitter = random.gauss(0, self.profile.jitter_ms * 0.33)
        latency = max(0.05, latency + jitter)
        tx_ms = (data_bytes * 8) / (self.profile.bandwidth_gbps * 1e9) * 1000
        return latency + tx_ms

    async def simulate_uplink(self, data_bytes: int) -> float:
        delay = self._one_way_delay_ms(data_bytes)
        self._bytes_uplink += data_bytes
        await asyncio.sleep(delay / 1000)
        return delay

    async def simulate_downlink(self, data_bytes: int) -> float:
        delay = self._one_way_delay_ms(data_bytes)
        self._bytes_downlink += data_bytes
        await asyncio.sleep(delay / 1000)
        return delay

    def should_drop_packet(self) -> bool:
        if random.random() < (self.profile.packet_loss_pct / 100):
            self._packets_dropped += 1
            return True
        return False

    @property
    def expected_rtt_ms(self) -> float:
        return self.profile.one_way_latency_ms * 2

    @property
    def max_theoretical_fps(self) -> float:
        return 1000 / max(0.1, self.expected_rtt_ms)

    @property
    def throughput_mbps(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed < 0.01:
            return 0.0
        total_bytes = self._bytes_uplink + self._bytes_downlink
        return total_bytes * 8 / elapsed / 1e6

    def get_stats(self) -> dict:
        return {
            "generation": self.generation,
            "name": self.profile.name,
            "expected_rtt_ms": self.expected_rtt_ms,
            "max_fps": self.max_theoretical_fps,
            "bandwidth_gbps": self.profile.bandwidth_gbps,
            "throughput_mbps": self.throughput_mbps,
            "packets_dropped": self._packets_dropped,
        }
