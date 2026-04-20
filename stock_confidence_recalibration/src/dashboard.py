"""
Terminal Dashboard
==================
Rich-based real-time terminal visualization of the recalibration pipeline.
"""

import time
import sys
import os
from collections import deque
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import box
    import rich.style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from orchestrator import PipelineResult


console = Console()


def confidence_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.3f}"


def regime_color(regime: str) -> str:
    colors = {"bullish": "green", "bearish": "red", "sideways": "yellow"}
    return colors.get(regime, "white")


def vol_color(vol_label: str) -> str:
    colors = {"HIGH_VOL": "red", "LOW_VOL": "cyan", "NORMAL_VOL": "white"}
    return colors.get(vol_label, "white")


def ece_color(ece: float) -> str:
    if ece > 0.15:
        return "red"
    elif ece > 0.08:
        return "yellow"
    return "green"


def drift_symbol(direction: str) -> str:
    if direction == "OVERCONFIDENT":
        return "↑ OVER"
    elif direction == "UNDERCONFIDENT":
        return "↓ UNDER"
    elif direction == "WORSENING":
        return "⚠ WORSE"
    elif direction == "IMPROVING":
        return "✓ IMPRV"
    return "→ STABLE"


def build_live_table(results: List[PipelineResult], tick: int) -> Table:
    """Build the main live status table."""
    table = Table(
        title=f"[bold cyan]⚡ Online Stock Confidence Recalibration  |  Tick #{tick}[/bold cyan]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
        border_style="bright_blue",
        expand=True
    )

    table.add_column("Symbol",   style="bold white",  width=7)
    table.add_column("Regime",   width=9)
    table.add_column("Vol",      width=10)
    table.add_column("Raw Conf", width=12, justify="right")
    table.add_column("Final Conf", width=22)
    table.add_column("Δ Conf",   width=9,  justify="right")
    table.add_column("ECE",      width=8,  justify="right")
    table.add_column("Miscal",   width=7,  justify="center")
    table.add_column("Drift",    width=10)

    seen = set()
    for r in reversed(results):
        if r.symbol in seen:
            continue
        seen.add(r.symbol)

        regime_str = Text(r.regime.upper()[:7], style=regime_color(r.regime))
        vol_str    = Text(r.vol_label[:10],      style=vol_color(r.vol_label))
        raw_str    = f"{r.raw_confidence:.4f}"
        delta      = r.final_confidence - r.raw_confidence
        delta_str  = Text(
            f"{delta:+.4f}",
            style="green" if delta > 0 else ("red" if delta < 0 else "white")
        )
        ece_str    = Text(f"{r.ece:.4f}", style=ece_color(r.ece))
        miscal_str = Text("⚠ YES" if r.miscalibrated else "✓ NO",
                          style="red bold" if r.miscalibrated else "green")

        # Colored confidence bar
        fc   = r.final_confidence
        col  = "green" if fc > 0.65 else ("red" if fc < 0.45 else "yellow")
        bars = int(fc * 16)
        conf_bar = Text(
            f"{'█'*bars}{'░'*(16-bars)} {fc:.3f}",
            style=col
        )

        table.add_row(
            r.symbol, regime_str, vol_str, raw_str,
            conf_bar, delta_str, ece_str, miscal_str,
            drift_symbol(r.drift_direction)
        )

    return table


def build_stage_table(results: List[PipelineResult]) -> Table:
    """Show the per-stage confidence evolution for the latest tick."""
    table = Table(
        title="[bold yellow]Recalibration Pipeline Stages[/bold yellow]",
        box=box.SIMPLE_HEAD,
        header_style="bold yellow",
        expand=True
    )
    table.add_column("Symbol",    width=7,  style="bold white")
    table.add_column("Raw",       width=8,  justify="right")
    table.add_column("VolAdj",    width=8,  justify="right")
    table.add_column("TimDecay",  width=9,  justify="right")
    table.add_column("RegimeCal", width=10, justify="right")
    table.add_column("Smoothed",  width=9,  justify="right")
    table.add_column("Feedback",  width=9,  justify="right")
    table.add_column("Normalized",width=10, justify="right")

    seen = set()
    for r in reversed(results):
        if r.symbol in seen or not r.stage_confidences:
            continue
        seen.add(r.symbol)
        s = r.stage_confidences

        def fc(key):
            v = s.get(key, 0)
            col = "green" if v > 0.65 else ("red" if v < 0.45 else "yellow")
            return Text(f"{v:.3f}", style=col)

        table.add_row(
            r.symbol,
            fc("raw"), fc("vol_adj"), fc("time_decay"),
            fc("regime_cal"), fc("smoothed"),
            fc("feedback"), fc("normalized")
        )

    return table


def build_benchmark_table(benchmark_data: List[Dict]) -> Table:
    """Compare raw vs recalibrated confidence metrics."""
    table = Table(
        title="[bold green]📊 Benchmark: Raw vs Recalibrated Confidence[/bold green]",
        box=box.SIMPLE_HEAD,
        header_style="bold green",
        expand=True
    )
    table.add_column("Symbol",     width=7,  style="bold white")
    table.add_column("Raw ECE",    width=9,  justify="right")
    table.add_column("Cal ECE",    width=9,  justify="right")
    table.add_column("ECE Δ",      width=9,  justify="right")
    table.add_column("Raw Brier",  width=10, justify="right")
    table.add_column("Cal Brier",  width=10, justify="right")
    table.add_column("Brier Δ",    width=9,  justify="right")
    table.add_column("Sharpness Δ",width=11, justify="right")
    table.add_column("N",          width=5,  justify="right")

    for d in benchmark_data:
        raw = d.get("raw", {})
        cal = d.get("calibrated", {})
        ece_imp   = d.get("ece_improvement", 0)
        brier_imp = d.get("brier_improvement", 0)
        sharp_chg = d.get("sharpness_change", 0)

        ece_col   = "green" if ece_imp > 0 else "red"
        brier_col = "green" if brier_imp > 0 else "red"

        table.add_row(
            d.get("symbol", ""),
            f"{raw.get('ece', 0):.4f}",
            f"{cal.get('ece', 0):.4f}",
            Text(f"{ece_imp:+.4f}", style=ece_col),
            f"{raw.get('brier', 0):.4f}",
            f"{cal.get('brier', 0):.4f}",
            Text(f"{brier_imp:+.4f}", style=brier_col),
            Text(f"{sharp_chg:+.4f}",
                 style="cyan" if sharp_chg > 0 else "white"),
            str(raw.get("n", 0))
        )

    return table


def build_accuracy_table(accuracy_data: Dict) -> Table:
    table = Table(
        title="[bold magenta]🎯 Per-Asset Accuracy & Reliability[/bold magenta]",
        box=box.SIMPLE_HEAD,
        header_style="bold magenta",
        expand=True
    )
    table.add_column("Symbol",      width=7,  style="bold white")
    table.add_column("Predictions", width=12, justify="right")
    table.add_column("Accuracy",    width=10, justify="right")
    table.add_column("ECE",         width=8,  justify="right")
    table.add_column("ECE Trend",   width=10)
    table.add_column("Reliability", width=12, justify="right")

    for sym, data in sorted(accuracy_data.items()):
        acc  = data.get("accuracy", 0)
        ece  = data.get("ece", 0)
        rel  = data.get("reliability", 0)
        trend = data.get("ece_trend", "STABLE")

        acc_col  = "green" if acc > 0.55 else ("red" if acc < 0.45 else "yellow")
        rel_col  = "green" if rel > 0.8 else ("red" if rel < 0.6 else "yellow")
        trend_col = "green" if trend == "IMPROVING" else (
            "red" if trend == "WORSENING" else "white")

        table.add_row(
            sym,
            str(data.get("total", 0)),
            Text(f"{acc:.4f}", style=acc_col),
            Text(f"{ece:.4f}", style=ece_color(ece)),
            Text(trend, style=trend_col),
            Text(f"{rel:.4f}", style=rel_col)
        )

    return table


def print_final_report(pipeline, total_ticks: int, elapsed: float):
    """Print comprehensive final report."""
    console.rule("[bold cyan]FINAL RECALIBRATION REPORT[/bold cyan]")

    console.print(f"\n[bold]Run Statistics:[/bold]")
    console.print(f"  Total Ticks Processed: [cyan]{total_ticks}[/cyan]")
    console.print(f"  Total Runtime:         [cyan]{elapsed:.1f}s[/cyan]")
    console.print(f"  Ticks/second:          [cyan]{total_ticks/elapsed:.1f}[/cyan]")

    console.print()
    console.print(build_benchmark_table(pipeline.get_benchmark_report()))
    console.print()
    console.print(build_accuracy_table(pipeline.get_accuracy_table()))

    # Drift report
    console.print("\n[bold yellow]Confidence Drift Summary:[/bold yellow]")
    for sym, drift in pipeline.get_drift_report().items():
        if drift["drift"] != 0:
            col = "red" if drift["alert"] else "green"
            console.print(
                f"  {sym:6s} | drift={drift['drift']:+.4f} | "
                f"[{col}]{drift['direction']}[/{col}] | "
                f"alert=[{'red' if drift['alert'] else 'green'}]"
                f"{'YES' if drift['alert'] else 'NO'}[/{'red' if drift['alert'] else 'green'}]"
            )

    # Miscalibration alerts
    alerts = pipeline.miscal_detect.alert_log
    if alerts:
        console.print(f"\n[bold red]Miscalibration Alerts ({len(alerts)} total):[/bold red]")
        for alert in alerts[-5:]:
            console.print(
                f"  [red]⚠[/red] {alert['symbol']} | ECE={alert['ece']:.4f} | "
                f"Severity={alert['severity']}"
            )

    console.rule()
    console.print("\n[bold green]✓ Recalibration pipeline complete.[/bold green]")
    console.print("[dim]Results saved to output/ directory.[/dim]\n")
