import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from network.sim import NetworkSimulator, NETWORK_PROFILES
from shared_state import NetworkGen


@pytest.mark.asyncio
async def test_6g_rtt_under_5ms():
    sim = NetworkSimulator(NetworkGen.SIX_G)
    total = 0.0
    for _ in range(20):
        total += await sim.simulate_uplink(50_000)
        total += await sim.simulate_downlink(1_000)
    assert total / 20 < 5.0


@pytest.mark.asyncio
async def test_4g_rtt_over_30ms():
    sim = NetworkSimulator(NetworkGen.FOUR_G)
    total = 0.0
    for _ in range(10):
        total += await sim.simulate_uplink(50_000)
        total += await sim.simulate_downlink(1_000)
    assert total / 10 > 30.0


@pytest.mark.asyncio
async def test_6g_faster_than_4g():
    async def measure(gen):
        sim = NetworkSimulator(gen)
        t = 0.0
        for _ in range(5):
            t += await sim.simulate_uplink(50_000) + await sim.simulate_downlink(1_000)
        return t / 5

    assert await measure(NetworkGen.SIX_G) < await measure(NetworkGen.FOUR_G)


def test_all_profiles_exist():
    for g in [NetworkGen.FOUR_G, NetworkGen.FIVE_G, NetworkGen.SIX_G]:
        assert g in NETWORK_PROFILES
        p = NETWORK_PROFILES[g]
        assert p.one_way_latency_ms > 0
        assert p.bandwidth_gbps > 0


def test_max_fps_ordering():
    sims = [NetworkSimulator(g) for g in [NetworkGen.FOUR_G, NetworkGen.FIVE_G, NetworkGen.SIX_G]]
    fps = [s.max_theoretical_fps for s in sims]
    assert fps[0] < fps[1] < fps[2]


def test_set_network():
    sim = NetworkSimulator(NetworkGen.SIX_G)
    sim.set_network(NetworkGen.FOUR_G)
    assert sim.generation == NetworkGen.FOUR_G
    assert sim.profile.one_way_latency_ms == NETWORK_PROFILES[NetworkGen.FOUR_G].one_way_latency_ms
