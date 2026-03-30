#!/usr/bin/env python3
"""
Online Stock Confidence Recalibration — Main Entry Point
=========================================================
Run this script to start the real-time recalibration pipeline.

Usage:
    python main.py [--ticks N] [--speed S] [--no-live]

Options:
    --ticks N      Total number of ticks to simulate (default: 400)
    --speed S      Ticks per second in live mode (default: 8)
    --no-live      Skip live display, just run and report
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from simulator import StockSimulator
from orchestrator import OnlineRecalibrationPipeline
from report_generator import generate_html_report

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.columns import Columns
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    print("Note: Install 'rich' for live dashboard display.")


def parse_args():
    p = argparse.ArgumentParser(description="Online Stock Confidence Recalibration")
    p.add_argument("--ticks",   type=int,   default=400,  help="Number of ticks (default 400)")
    p.add_argument("--speed",   type=float, default=8.0,  help="Ticks/sec in live mode (default 8)")
    p.add_argument("--no-live", action="store_true",       help="Disable live display")
    return p.parse_args()


def run_pipeline(args):
    simulator = StockSimulator(seed=42)
    pipeline  = OnlineRecalibrationPipeline()
    results   = []
    start_ts  = time.time()
    tick_ts   = start_ts

    if RICH and not args.no_live:
        from dashboard import build_live_table, build_stage_table, print_final_report
        from rich.layout import Layout

        console.print(Panel(
            "[bold cyan]Online Stock Confidence Recalibration Engine[/bold cyan]\n"
            "[dim]9 modules: volatility adjustment · time decay · regime calibration ·\n"
            "adaptive smoothing · miscalibration detection · feedback correction ·\n"
            "drift tracking · normalization · benchmarking[/dim]",
            border_style="bright_blue"
        ))
        time.sleep(0.5)

        refresh_rate = min(args.speed, 4)
        with Live(console=console, refresh_per_second=refresh_rate, screen=False) as live:
            for i in range(args.ticks):
                tick_ts += 1.0 / args.speed
                ticks = simulator.generate_tick(timestamp=tick_ts)

                for tick in ticks:
                    raw_conf, pred_dir = simulator.generate_base_prediction(
                        tick.symbol, tick)
                    result = pipeline.process_tick(tick, raw_conf, pred_dir)
                    results.append(result)

                if i % 2 == 0:
                    from rich.console import Group
                    live_tbl   = build_live_table(results, i + 1)
                    stage_tbl  = build_stage_table(results)
                    live.update(Group(live_tbl, stage_tbl))

                delay = 1.0 / args.speed
                time.sleep(max(0, delay - 0.001))

        elapsed = time.time() - start_ts
        print_final_report(pipeline, args.ticks, elapsed)

    else:
        # Non-live mode: fast batch run with progress prints
        print(f"Running {args.ticks} ticks across {len(StockSimulator.SYMBOLS)} assets...")
        for i in range(args.ticks):
            tick_ts += 1.0
            ticks = simulator.generate_tick(timestamp=tick_ts)
            for tick in ticks:
                raw_conf, pred_dir = simulator.generate_base_prediction(tick.symbol, tick)
                result = pipeline.process_tick(tick, raw_conf, pred_dir)
                results.append(result)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{args.ticks} ticks...")

        elapsed = time.time() - start_ts
        print(f"\nCompleted {args.ticks} ticks in {elapsed:.2f}s")

        # Print quick summary
        bench = pipeline.get_benchmark_report()
        print("\n--- Benchmark Summary ---")
        print(f"{'Symbol':<8} {'RawECE':>8} {'CalECE':>8} {'ECE_Δ':>8} {'RawBrier':>10} {'CalBrier':>10}")
        for d in bench:
            raw = d.get("raw", {})
            cal = d.get("calibrated", {})
            imp = d.get("ece_improvement", 0)
            print(f"{d['symbol']:<8} {raw.get('ece',0):>8.4f} {cal.get('ece',0):>8.4f} "
                  f"{imp:>+8.4f} {raw.get('brier',0):>10.4f} {cal.get('brier',0):>10.4f}")

    return pipeline, elapsed


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    print("\n" + "="*60)
    print("  ONLINE STOCK CONFIDENCE RECALIBRATION SYSTEM")
    print("="*60)
    print(f"  Assets:  {', '.join(StockSimulator.SYMBOLS)}")
    print(f"  Ticks:   {args.ticks}")
    print(f"  Modules: 9 (vol · decay · regime · smooth · detect ·")
    print(f"              feedback · drift · normalize · benchmark)")
    print("="*60 + "\n")

    pipeline, elapsed = run_pipeline(args)

    # Generate HTML report
    report_path = os.path.join("output", "recalibration_report.html")
    generate_html_report(pipeline, args.ticks, elapsed, report_path)
    print(f"\n[Report saved] → {os.path.abspath(report_path)}")
    print("Open this file in your browser to view the interactive report.\n")


if __name__ == "__main__":
    main()
