"""
MODULE 03 — Adaptive Market Regime Detection
Uses Hidden Markov Models + volatility clustering to detect:
  - Low Volatility (Stable)
  - Medium Volatility (Trending)
  - High Volatility (Stressed)
  - Crisis Regime
Real data via yfinance.
"""
import json, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from data_fetcher import DataFetcher
from volatility_calculator import VolatilityCalculator
from hmm_detector import HMMRegimeDetector
from regime_classifier import RegimeClassifier
from logger import get_logger

console = Console()
logger = get_logger("regime")

REGIME_COLORS = {
    "LOW_VOL": "green",
    "MEDIUM_VOL": "cyan",
    "HIGH_VOL": "yellow",
    "CRISIS": "red bold",
    "UNKNOWN": "white",
}
REGIME_NAMES = {
    0: "LOW_VOL",
    1: "MEDIUM_VOL",
    2: "HIGH_VOL",
    3: "CRISIS",
}


def load_config():
    try:
        with open("config.json") as f: return json.load(f)
    except: return {}


def build_table(results: list, summary: dict) -> Table:
    table = Table(title="[bold cyan]Market Regime Detection Engine[/bold cyan]",
                  show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Symbol", style="cyan", min_width=8)
    table.add_column("Last Price", min_width=10)
    table.add_column("1D Return", min_width=10)
    table.add_column("Realized Vol (21d)", min_width=18)
    table.add_column("VIX Proxy", min_width=10)
    table.add_column("HMM Regime", min_width=14)
    table.add_column("Confidence", min_width=12)
    table.add_column("Trend", min_width=10)

    for r in results:
        regime = r.get("regime", "UNKNOWN")
        color = REGIME_COLORS.get(regime, "white")
        ret = r.get("daily_return", 0)
        ret_color = "green" if ret > 0 else "red"
        table.add_row(
            r.get("symbol", ""),
            f"${r.get('price', 0):,.2f}",
            f"[{ret_color}]{ret:+.2%}[/{ret_color}]",
            f"{r.get('realized_vol', 0):.4f}",
            f"{r.get('vix_proxy', 0):.2f}",
            f"[{color}]{regime}[/{color}]",
            f"{r.get('confidence', 0):.1%}",
            r.get("trend", ""),
        )

    table.add_section()
    agg_regime = summary.get("market_regime", "UNKNOWN")
    agg_color = REGIME_COLORS.get(agg_regime, "white")
    table.add_row(
        "[bold]MARKET[/bold]", "", "",
        f"{summary.get('avg_vol', 0):.4f}", "",
        f"[bold {agg_color}]{agg_regime}[/bold {agg_color}]",
        f"{summary.get('confidence', 0):.1%}",
        summary.get("recommendation", ""),
    )
    return table


def main():
    config = load_config()
    symbols = config.get("symbols", ["SPY", "QQQ", "IWM", "GLD", "TLT", "VIX"])
    refresh = config.get("refresh_seconds", 60)

    console.print(Panel(
        "[bold green]Multimodal Financial Intelligence System[/bold green]\n"
        "[cyan]Module 03: Adaptive Market Regime Detection[/cyan]\n"
        f"Symbols: {', '.join(symbols)} | Refresh: {refresh}s",
        expand=False))

    fetcher = DataFetcher(symbols)
    vol_calc = VolatilityCalculator()
    hmm = HMMRegimeDetector(n_regimes=4)
    classifier = RegimeClassifier()

    logger.info("Fetching historical data to train HMM...")
    history = fetcher.fetch_history(period="1y", interval="1d")
    if history is not None:
        hmm.fit(history)
        logger.info("HMM trained on 1-year historical data")

    with Live(console=console, refresh_per_second=0.3) as live:
        while True:
            try:
                snapshot = fetcher.fetch_snapshot()
                results = []
                for sym, data in snapshot.items():
                    vol = vol_calc.realized_vol(data["returns"])
                    vix_proxy = vol * np.sqrt(252) * 100
                    regime_id, conf = hmm.predict_latest(data["returns"])
                    regime = REGIME_NAMES.get(regime_id, "UNKNOWN")
                    classified = classifier.classify(vol, data["returns"])
                    trend = classifier.trend(data["returns"])
                    results.append({
                        "symbol": sym,
                        "price": data["price"],
                        "daily_return": data["returns"][-1] if len(data["returns"]) else 0,
                        "realized_vol": vol,
                        "vix_proxy": vix_proxy,
                        "regime": classified,
                        "confidence": conf,
                        "trend": trend,
                    })

                summary = classifier.market_summary(results)
                live.update(build_table(results, summary))
                logger.info(f"Market regime: {summary.get('market_regime')} "
                            f"(confidence {summary.get('confidence', 0):.1%})")
                time.sleep(refresh)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]"); break
            except Exception as e:
                logger.error(f"Error: {e}"); time.sleep(10)


if __name__ == "__main__":
    main()
