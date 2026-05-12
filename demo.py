#!/usr/bin/env python3
"""
6G V2X Robot Vision Offloading Demo
====================================
Demonstrates how 6G's ultra-low latency (~1ms) enables a Rutgers Makerspace
robot to offload heavy vision inference to an MEC edge server in real time --
a key enabling technology for Tesla-grade Autopilot on resource-constrained
hardware.

Usage:
    python demo.py               # full terminal dashboard
    python demo.py --benchmark   # run latency benchmark only
    python demo.py --network 4G  # start in 4G mode

Press [1] 4G  [2] 5G  [3] 6G  [b] Benchmark  [q] Quit
"""

import argparse
import asyncio
import sys
import threading
import time

from shared_state import SharedState, NetworkGen, FrameMetrics
from network.sim import NETWORK_PROFILES
from edge_server.server import EdgeServer
from robot.client import RobotClient
from dashboard.terminal import TerminalDashboard


# ---------------------------------------------------------------------------
# Asyncio backend (runs in a dedicated thread)
# ---------------------------------------------------------------------------

def _make_metrics_cb(state: SharedState):
    def callback(m_or_event, extra=None):
        if isinstance(m_or_event, FrameMetrics):
            m = m_or_event
            state.update_metrics(m)
            state.update_detections(extra or [])
            color = "green" if m.rtt_ms < 5 else "yellow" if m.rtt_ms < 20 else "red"
            state.add_v2x_log(
                f"-> Frame #{m.frame_id:05d}  RTT [{color}]{m.rtt_ms:.1f}ms[/{color}]"
                f"  GPU {m.server_processing_ms:.1f}ms  |  {m.n_detections} obj"
            )
        elif m_or_event == "spat" and extra:
            state.update_spat(extra)
            phase = extra.get("phase", "?")
            rem   = extra.get("time_remaining_s", 0)
            state.add_v2x_log(f"<- SPAT: {phase} ({rem:.0f}s remaining)")
    return callback


def _make_server_cb(state: SharedState):
    def callback(event: str, data: dict):
        if event == "bsm":
            vid   = data.get("vehicle_id", "?")
            speed = data.get("speed_mps", 0)
            state.add_v2x_log(f"<- BSM  {vid}  spd={speed:.2f}m/s")
    return callback


def run_backend(state: SharedState, loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)

    server = EdgeServer(
        network_gen     = state.network_gen,
        state_callback  = _make_server_cb(state),
    )
    client = RobotClient(
        network_gen      = state.network_gen,
        metrics_callback = _make_metrics_cb(state),
    )

    async def _backend():
        server_task = asyncio.create_task(server.run())
        await asyncio.sleep(0.4)           # let server bind
        state.server_ready = True
        client_task = asyncio.create_task(client.run())

        while state.is_running:
            change = state.consume_network_change()
            if change:
                server.set_network(change)
                client.set_network(change)
                state.set_network(change)
                p = NETWORK_PROFILES[change]
                state.add_v2x_log(
                    f"Network switched -> {p.name}  "
                    f"(RTT ~{p.one_way_latency_ms * 2:.0f}ms)"
                )
            await asyncio.sleep(0.05)

        client_task.cancel()
        server_task.cancel()

    loop.run_until_complete(_backend())


# ---------------------------------------------------------------------------
# Keyboard input thread
# ---------------------------------------------------------------------------

def run_keyboard(state: SharedState, auto_cycle: bool = True):
    try:
        import readchar
        _kb_loop(state)
    except Exception:
        if auto_cycle:
            _auto_cycle(state)


def _kb_loop(state: SharedState):
    import readchar
    while state.is_running:
        try:
            ch = readchar.readchar()
        except Exception:
            break
        if ch == "1":
            state.request_network_change(NetworkGen.FOUR_G)
        elif ch == "2":
            state.request_network_change(NetworkGen.FIVE_G)
        elif ch == "3":
            state.request_network_change(NetworkGen.SIX_G)
        elif ch in ("b", "B"):
            _run_quick_benchmark(state)
        elif ch in ("q", "Q", "\x03"):
            state.stop()


def _auto_cycle(state: SharedState):
    """Automatically cycle through modes so the demo tells its own story."""
    schedule = [
        (NetworkGen.SIX_G,  10),
        (NetworkGen.FOUR_G, 10),
        (NetworkGen.FIVE_G, 10),
        (NetworkGen.SIX_G,  999),
    ]
    for gen, duration in schedule:
        if not state.is_running:
            return
        state.request_network_change(gen)
        for _ in range(duration * 10):
            if not state.is_running:
                return
            time.sleep(0.1)


def _run_quick_benchmark(state: SharedState):
    """Switch through modes collecting samples; log results."""
    state.add_v2x_log("== Benchmark starting (30s) ==")
    for gen, dur in [(NetworkGen.FOUR_G, 8), (NetworkGen.FIVE_G, 8), (NetworkGen.SIX_G, 8)]:
        state.request_network_change(gen)
        time.sleep(dur)
        avg = state.avg_rtt_ms
        p95 = state.p95_rtt_ms
        fps = state.current_fps
        p   = NETWORK_PROFILES[gen]
        state.add_v2x_log(
            f"[{gen}] RTT={avg:.1f}ms P95={p95:.1f}ms FPS={fps:.0f}"
        )
    state.request_network_change(NetworkGen.SIX_G)
    state.add_v2x_log("== Benchmark complete -- back to 6G ==")


# ---------------------------------------------------------------------------
# Benchmark-only mode (no UI)
# ---------------------------------------------------------------------------

def run_benchmark_only():
    """Run headless benchmark and print comparison table."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print("\n[bold cyan]6G V2X Vision Offload -- Latency Benchmark[/bold cyan]\n")

    state  = SharedState()
    loop   = asyncio.new_event_loop()
    thread = threading.Thread(target=run_backend, args=(state, loop), daemon=True)
    thread.start()

    # wait for server
    for _ in range(30):
        if state.server_ready:
            break
        time.sleep(0.1)

    results = {}
    for gen in [NetworkGen.FOUR_G, NetworkGen.FIVE_G, NetworkGen.SIX_G]:
        state.request_network_change(gen)
        state.rtt_history.clear()
        time.sleep(0.5)   # let it settle
        with state._lock:
            state.rtt_history.clear()
        console.print(f"Measuring [bold]{gen}[/bold]...", end=" ")
        time.sleep(8)
        avg = state.avg_rtt_ms
        p95 = state.p95_rtt_ms
        fps = state.current_fps
        results[gen] = (avg, p95, fps)
        console.print(f"RTT={avg:.1f}ms  P95={p95:.1f}ms  FPS={fps:.0f}")

    state.stop()

    tbl = Table(title="\nBenchmark Results", box=box.DOUBLE_EDGE)
    tbl.add_column("Network",    style="bold", justify="left")
    tbl.add_column("Avg RTT",    justify="right")
    tbl.add_column("P95 RTT",    justify="right")
    tbl.add_column("Eff. FPS",   justify="right")
    tbl.add_column("Autopilot?", justify="center")

    viability = {
        NetworkGen.FOUR_G: ("[red]X < 30fps[/red]",   "red"),
        NetworkGen.FIVE_G: ("[yellow]~ marginal[/yellow]", "yellow"),
        NetworkGen.SIX_G:  ("[bright_green]✓ >100fps[/bright_green]", "bright_green"),
    }

    for gen in [NetworkGen.FOUR_G, NetworkGen.FIVE_G, NetworkGen.SIX_G]:
        avg, p95, fps = results[gen]
        viable, color = viability[gen]
        tbl.add_row(
            NETWORK_PROFILES[gen].name,
            f"[{color}]{avg:.1f}ms[/{color}]",
            f"[{color}]{p95:.1f}ms[/{color}]",
            f"[{color}]{fps:.0f}[/{color}]",
            viable,
        )

    console.print(tbl)
    console.print(
        "\n[dim]6G enables real-time edge inference at >500fps -- "
        "making vision offload viable for L4 autonomy.[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="6G V2X Robot Vision Offloading Demo")
    parser.add_argument("--benchmark",  action="store_true", help="Run headless benchmark")
    parser.add_argument("--network",    default="6G", choices=["4G", "5G", "6G"],
                        help="Starting network mode")
    parser.add_argument("--no-autocycle", action="store_true",
                        help="Disable automatic mode cycling (keyboard only)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_only()
        return

    state = SharedState()
    state.network_gen = args.network

    loop   = asyncio.new_event_loop()
    thread = threading.Thread(target=run_backend, args=(state, loop), daemon=True)
    thread.start()

    # wait for server to bind
    for _ in range(40):
        if state.server_ready:
            break
        time.sleep(0.1)

    kb_thread = threading.Thread(
        target=run_keyboard,
        args=(state, not args.no_autocycle),
        daemon=True,
    )
    kb_thread.start()

    dashboard = TerminalDashboard(state)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()

    print("\n\033[1mDemo ended.\033[0m")
    print(f"  Frames processed : {state.frame_count:,}")
    print(f"  Data transferred : {state.bytes_transferred / 1e6:.1f} MB")
    print(f"  Final avg RTT    : {state.avg_rtt_ms:.1f} ms")


if __name__ == "__main__":
    main()
