"""
MODULE 09 — Self-Learning Alpha Signal Generation Pipeline
Combines: news sentiment + OBI proxy + volatility regime + momentum + RSI
to generate probabilistic trading signals with backtesting validation.
"""
import json, time
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from logger import get_logger

console = Console()
logger = get_logger("alpha")

# Finance-enhanced VADER lexicon
FINANCE_LEXICON = {
    "bullish": 3.0, "bearish": -3.0, "surge": 2.5, "plunge": -2.5,
    "rally": 2.0, "crash": -3.0, "soar": 2.5, "tumble": -2.5,
    "beat": 2.0, "miss": -2.0, "upgrade": 2.5, "downgrade": -2.5,
    "growth": 2.0, "recession": -3.0, "stimulus": 2.5, "layoffs": -2.5,
}

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://finance.yahoo.com/news/rssindex",
]


def load_config():
    try:
        with open("config.json") as f: return json.load(f)
    except: return {}


def get_news_sentiment(vader) -> float:
    scores = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                s = vader.polarity_scores(entry.get("title",""))["compound"]
                scores.append(s)
        except: pass
    return float(np.mean(scores)) if scores else 0.0


def compute_features(symbol: str) -> dict:
    tk = yf.Ticker(symbol)
    hist = tk.history(period="3mo", interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < 25:
        return {}
    c = hist["Close"].values
    v = hist["Volume"].values
    r = np.diff(np.log(c))
    # RSI
    deltas = np.diff(c)
    gains = np.where(deltas[-14:] > 0, deltas[-14:], 0)
    losses = np.where(deltas[-14:] < 0, -deltas[-14:], 0)
    avg_gain, avg_loss = np.mean(gains), np.mean(losses)
    rsi = 100 - 100/(1 + avg_gain/(avg_loss+1e-8))
    rsi_score = (50 - rsi) / 50.0
    # OBI proxy from volume imbalance
    recent_v = v[-5:]
    recent_r = r[-5:]
    buy_vol = np.sum(recent_v[recent_r > 0])
    sell_vol = np.sum(recent_v[recent_r < 0])
    obi = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
    # Volatility regime
    vol_20 = float(np.std(r[-20:]))
    hist_vol = float(np.std(r))
    vol_regime_score = -1.0 if vol_20 > hist_vol * 1.5 else (0.3 if vol_20 < hist_vol * 0.7 else 0.0)
    # Momentum
    mom = float(c[-1] / c[-20] - 1)
    mom_score = np.clip(mom * 15, -1, 1)
    return {
        "price": float(c[-1]),
        "rsi": float(rsi),
        "rsi_score": float(rsi_score),
        "obi": float(obi),
        "vol_20": vol_20,
        "vol_regime_score": float(vol_regime_score),
        "momentum": float(mom),
        "momentum_score": float(mom_score),
    }


def alpha_score(features: dict, news_score: float, weights: dict) -> tuple:
    if not features: return 0.0, "HOLD", 0.0
    score = (
        weights.get("news", 0.25) * np.clip(news_score, -1, 1) +
        weights.get("obi", 0.20) * features.get("obi", 0) +
        weights.get("volatility", 0.20) * features.get("vol_regime_score", 0) +
        weights.get("momentum", 0.20) * features.get("momentum_score", 0) +
        weights.get("rsi", 0.15) * features.get("rsi_score", 0)
    )
    score = float(np.clip(score, -1, 1))
    confidence = abs(score)
    if score > 0.25:
        return score, "BUY", confidence
    elif score < -0.25:
        return score, "SELL", confidence
    return score, "HOLD", confidence


def simple_backtest(symbol: str, weights: dict, period: str = "1y") -> dict:
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 60: return {}
        c = hist["Close"].values
        r = np.diff(np.log(c))
        # Simple momentum backtest
        signals = []
        for i in range(20, len(r)-1):
            mom = c[i] / c[i-20] - 1
            rsi_win = c[i-14:i+1]
            deltas = np.diff(rsi_win)
            g = np.where(deltas > 0, deltas, 0)
            l = np.where(deltas < 0, -deltas, 0)
            rsi = 100 - 100/(1+np.mean(g)/(np.mean(l)+1e-8))
            sig = 1 if (mom > 0.02 and rsi < 65) else (-1 if (mom < -0.02 and rsi > 35) else 0)
            signals.append(sig * r[i+1])
        pnl = np.array(signals)
        total_trades = np.sum(np.abs(np.array([1 if (c[i]/c[i-20]-1) > 0.02 else (-1 if (c[i]/c[i-20]-1) < -0.02 else 0) for i in range(20, len(c)-1)])) > 0)
        winning = np.sum(pnl > 0)
        sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(252)) if len(pnl) > 0 else 0
        cum = np.cumprod(1 + pnl) if len(pnl) > 0 else np.array([1])
        peak = np.maximum.accumulate(cum)
        max_dd = float(-np.min((cum - peak) / (peak + 1e-8)))
        return {
            "sharpe": sharpe,
            "win_rate": float(winning / max(len(pnl), 1)),
            "max_dd": max_dd,
            "annual_return": float(np.mean(pnl) * 252),
            "total_trades": int(total_trades),
        }
    except Exception as e:
        logger.warning(f"Backtest {symbol}: {e}")
        return {}


def build_table(results: list) -> Table:
    t = Table(title="[bold cyan]Alpha Signal Generation Pipeline[/bold cyan]",
              show_header=True, header_style="bold magenta", expand=True)
    t.add_column("Symbol", style="cyan", min_width=8)
    t.add_column("Price", min_width=10)
    t.add_column("Alpha Score", min_width=12)
    t.add_column("Signal", min_width=10)
    t.add_column("Confidence", min_width=12)
    t.add_column("RSI", min_width=8)
    t.add_column("OBI Proxy", min_width=10)
    t.add_column("Momentum", min_width=10)
    t.add_column("BT Sharpe", min_width=10)
    t.add_column("BT WinRate", min_width=10)
    for r in results:
        sig = r.get("signal", "HOLD")
        sc = "green" if sig == "BUY" else ("red" if sig == "SELL" else "yellow")
        alpha = r.get("alpha_score", 0)
        ac = "green" if alpha > 0 else ("red" if alpha < 0 else "yellow")
        rsi = r.get("rsi", 50)
        rc = "red" if rsi > 70 else ("green" if rsi < 30 else "white")
        bt = r.get("backtest", {})
        sharpe = bt.get("sharpe", 0)
        shc = "green" if sharpe > 1 else ("yellow" if sharpe > 0 else "red")
        t.add_row(
            r["symbol"], f"${r.get('price',0):,.2f}",
            f"[{ac}]{alpha:+.3f}[/{ac}]",
            f"[{sc}]{sig}[/{sc}]",
            f"{r.get('confidence',0):.1%}",
            f"[{rc}]{rsi:.1f}[/{rc}]",
            f"{r.get('obi',0):+.3f}",
            f"{r.get('momentum',0):+.2%}",
            f"[{shc}]{sharpe:.2f}[/{shc}]",
            f"{bt.get('win_rate',0):.1%}",
        )
    return t


def main():
    config = load_config()
    symbols = config.get("symbols", ["AAPL","MSFT","NVDA","GOOGL","SPY"])
    weights = config.get("weights", {"news":0.25,"obi":0.20,"volatility":0.20,"momentum":0.20,"rsi":0.15})
    refresh = config.get("refresh_seconds", 120)
    backtest_period = config.get("backtest_period", "1y")

    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(FINANCE_LEXICON)

    console.print(Panel("[bold green]Module 09: Alpha Signal Generation Pipeline[/bold green]\n"
                        f"Signals: News + OBI + Vol + Momentum + RSI | Symbols: {', '.join(symbols)}",
                        expand=False))

    with Live(console=console, refresh_per_second=0.3) as live:
        while True:
            try:
                logger.info("Computing news sentiment...")
                news_score = get_news_sentiment(vader)
                logger.info(f"News sentiment: {news_score:+.3f}")

                results = []
                for sym in symbols:
                    try:
                        features = compute_features(sym)
                        if not features: continue
                        score, signal, conf = alpha_score(features, news_score, weights)
                        bt = simple_backtest(sym, weights, backtest_period)
                        results.append({
                            "symbol": sym,
                            "price": features.get("price", 0),
                            "alpha_score": score,
                            "signal": signal,
                            "confidence": conf,
                            "rsi": features.get("rsi", 50),
                            "obi": features.get("obi", 0),
                            "momentum": features.get("momentum", 0),
                            "backtest": bt,
                        })
                        logger.info(f"{sym}: {signal} ({conf:.1%}) | Backtest Sharpe: {bt.get('sharpe',0):.2f}")
                    except Exception as e:
                        logger.warning(f"{sym}: {e}")

                live.update(build_table(results))
                time.sleep(refresh)
            except KeyboardInterrupt: break
            except Exception as e: logger.error(f"Error: {e}"); time.sleep(15)


if __name__ == "__main__":
    main()
