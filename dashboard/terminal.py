import time
from typing import List

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from network.sim import NETWORK_PROFILES
from shared_state import NetworkGen, SharedState


_GEN_COLOR = {NetworkGen.FOUR_G: "red", NetworkGen.FIVE_G: "yellow", NetworkGen.SIX_G: "bright_green"}
_GEN_EMOJI = {NetworkGen.FOUR_G: "●", NetworkGen.FIVE_G: "●", NetworkGen.SIX_G: "●"}
_RTT_REFERENCE = {NetworkGen.FOUR_G: 50.0, NetworkGen.FIVE_G: 8.5, NetworkGen.SIX_G: 1.1}


def _sparkline(values: List[float], width: int = 20) -> str:
    if not values:
        return "─" * width
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return bars[0] * width
    recent = list(values)[-width:]
    result = ""
    for v in recent:
        idx = int((v - lo) / (hi - lo) * (len(bars) - 1))
        result += bars[idx]
    return result.ljust(width, bars[0])


class TerminalDashboard:
    """Rich Live terminal dashboard for the 6G demo."""

    TITLE = "6G V2X Robot Vision Offloading  ·  Rutgers Makerspace"

    def __init__(self, state: SharedState):
        self.state   = state
        self.console = Console()

    # ------------------------------------------------------------------ #
    def _header(self) -> Panel:
        t = Text(self.TITLE, justify="center", style="bold white on blue")
        return Panel(t, style="blue", padding=(0, 0))

    def _network_panel(self) -> Panel:
        s   = self.state
        gen = s.network_gen
        tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        tbl.add_column("dot",  width=2)
        tbl.add_column("name", ratio=1)
        tbl.add_column("info", ratio=2)

        for g in NetworkGen.ALL:
            p       = NETWORK_PROFILES[g]
            active  = g == gen
            color   = _GEN_COLOR[g]
            dot     = Text("◉" if active else "○", style=f"bold {color}")
            name    = Text(p.name, style=f"bold {color}" if active else "dim")
            info    = Text(p.description if active else f"{p.one_way_latency_ms:.0f}ms OWL",
                           style="white" if active else "dim")
            tbl.add_row(dot, name, info)

        border = _GEN_COLOR[gen]
        return Panel(tbl, title="[bold]Network Mode[/bold]",
                     border_style=border, padding=(0, 1))

    def _metrics_panel(self) -> Panel:
        s        = self.state
        avg_rtt  = s.avg_rtt_ms
        p95_rtt  = s.p95_rtt_ms
        fps      = s.current_fps
        gen      = s.network_gen

        rtt_color = "bright_green" if avg_rtt < 3 else "yellow" if avg_rtt < 15 else "red"
        fps_color = "bright_green" if fps > 60  else "yellow"   if fps > 20  else "red"

        tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        tbl.add_column("key",   style="dim", width=16)
        tbl.add_column("value", ratio=1)

        with s._lock:
            rtts = list(s.rtt_history)

        tbl.add_row("Avg RTT",     Text(f"{avg_rtt:.1f} ms",    style=f"bold {rtt_color}"))
        tbl.add_row("P95 RTT",     Text(f"{p95_rtt:.1f} ms",    style=rtt_color))
        tbl.add_row("Throughput",  Text(f"{fps:.0f} fps",        style=f"bold {fps_color}"))
        tbl.add_row("Frames",      Text(f"{s.frame_count:,}"))
        tbl.add_row("Data TX",     Text(f"{s.bytes_transferred / 1e6:.1f} MB"))
        tbl.add_row("Dropped",     Text(f"{s.packets_dropped}",
                                        style="red" if s.packets_dropped else "green"))
        tbl.add_row("Uptime",      Text(f"{s.uptime_seconds:.0f}s"))
        tbl.add_row("Latency ▁▇", Text(_sparkline(rtts), style=rtt_color))

        return Panel(tbl, title="[bold]Live Metrics[/bold]",
                     border_style="green", padding=(0, 1))

    def _detections_panel(self) -> Panel:
        with self.state._lock:
            dets = list(self.state.current_detections)

        tbl = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1), expand=True)
        tbl.add_column("Class",      style="bold", ratio=2)
        tbl.add_column("Confidence", ratio=3)
        tbl.add_column("BBox",       ratio=3, style="dim")

        for d in dets[:7]:
            conf    = d.get("confidence", 0.0)
            bar_n   = int(conf * 12)
            bar     = f"[bright_green]{'█' * bar_n}[/bright_green][dim]{'░' * (12 - bar_n)}[/dim]"
            bbox    = d.get("bbox", [])
            bbox_s  = f"[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]" if len(bbox) == 4 else "?"
            tbl.add_row(d.get("class", "?"), f"{bar} {conf:.3f}", bbox_s)

        if not dets:
            tbl.add_row("[dim]— awaiting detections —[/dim]", "", "")

        return Panel(tbl, title="[bold]Edge Server Detections[/bold]",
                     border_style="yellow", padding=(0, 1))

    def _v2x_panel(self) -> Panel:
        with self.state._lock:
            log  = list(self.state.v2x_log)
            spat = dict(self.state.spat_state)

        txt = Text()

        phase = spat.get("phase", "?")
        rem   = spat.get("time_remaining_s", 0)
        nxt   = spat.get("next_phase", "?")
        phase_color = {"GREEN": "bright_green", "YELLOW": "yellow", "RED": "red"}.get(phase, "white")
        txt.append(f"RSU SPAT: ", style="dim")
        txt.append(f"{phase}", style=f"bold {phase_color}")
        txt.append(f" ({rem:.0f}s -> {nxt})\n\n", style="dim")

        for line in log[-7:]:
            txt.append(line + "\n", style="dim")

        return Panel(txt, title="[bold]V2X Message Log[/bold]",
                     border_style="magenta", padding=(0, 1))

    def _comparison_panel(self) -> Panel:
        gen    = self.state.network_gen
        actual = self.state.avg_rtt_ms
        WIDTH  = 38
        txt    = Text()

        txt.append("Network Latency Comparison\n\n", style="bold")

        for g in NetworkGen.ALL:
            ref   = _RTT_REFERENCE[g]
            color = _GEN_COLOR[g]
            is_cur = g == gen
            bar_n  = max(1, int((ref / 55.0) * WIDTH))
            bar    = "█" * bar_n + "░" * (WIDTH - bar_n)
            label  = f" {g:3s} "
            active_marker = " <- ACTIVE" if is_cur else ""

            txt.append(label, style=f"bold {color}" if is_cur else "dim")
            txt.append(bar,   style=color if is_cur else "dim " + color)
            txt.append(f" {ref:.0f}ms{active_marker}\n",
                       style=f"bold {color}" if is_cur else "dim")

        if actual > 0:
            txt.append(f"\nMeasured RTT: {actual:.1f}ms  ", style="bold white")
            eff_fps = 1000 / max(0.1, actual)
            fps_color = "bright_green" if eff_fps > 60 else "yellow" if eff_fps > 20 else "red"
            txt.append(f"-> {eff_fps:.0f} fps effective\n", style=f"bold {fps_color}")

        txt.append("\n[1] 4G   [2] 5G   [3] 6G   [b] Benchmark   [q] Quit",
                   style="dim")

        return Panel(txt, title="[bold]Why 6G Matters for Autopilot[/bold]",
                     border_style="cyan", padding=(0, 1))

    # ------------------------------------------------------------------ #
    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="row1",    ratio=5),
            Layout(name="row2",    ratio=6),
            Layout(name="footer",  ratio=5),
        )
        layout["row1"].split_row(Layout(name="network"), Layout(name="metrics"))
        layout["row2"].split_row(Layout(name="detections"), Layout(name="v2x"))

        layout["header"].update(self._header())
        layout["network"].update(self._network_panel())
        layout["metrics"].update(self._metrics_panel())
        layout["detections"].update(self._detections_panel())
        layout["v2x"].update(self._v2x_panel())
        layout["footer"].update(self._comparison_panel())
        return layout

    def run(self):
        with Live(self._build_layout(), refresh_per_second=8,
                  screen=True, console=self.console) as live:
            while self.state.is_running:
                live.update(self._build_layout())
                time.sleep(0.12)
