"""
data_generator.py — Synthetic stock data with realistic technical indicators
Generates both the initial historical dataset and streaming incremental batches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
import config

np.random.seed(config.SEED)


# ─── Technical Indicator Helpers ──────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100, avg_g / (avg_l + 1e-9))
    return 100 - (100 / (1 + rs))


def _macd(close: np.ndarray, fast=12, slow=26) -> np.ndarray:
    s = pd.Series(close)
    return (s.ewm(span=fast).mean() - s.ewm(span=slow).mean()).values


def _bollinger(close: np.ndarray, window=20) -> Tuple[np.ndarray, np.ndarray]:
    s  = pd.Series(close)
    ma = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().fillna(0)
    return (ma + 2 * sd).values, (ma - 2 * sd).values


def _atr(high, low, close, period=14) -> np.ndarray:
    tr = np.maximum(high - low,
         np.maximum(abs(high - np.roll(close, 1)),
                    abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


# ─── Core OHLCV Simulator ────────────────────────────────────────────────────

def _simulate_ohlcv(n_rows: int,
                    start_price: float = 100.0,
                    mu: float = 0.0002,
                    sigma: float = 0.015,
                    regime_change_prob: float = 0.01) -> pd.DataFrame:
    """Geometric Brownian Motion with occasional regime changes."""
    prices = [start_price]
    mu_cur, sigma_cur = mu, sigma
    for _ in range(n_rows - 1):
        if np.random.rand() < regime_change_prob:          # drift / vol shock
            mu_cur    = np.random.uniform(-0.001, 0.002)
            sigma_cur = np.random.uniform(0.008, 0.04)
        ret = np.random.normal(mu_cur, sigma_cur)
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)
    noise  = np.abs(np.random.normal(0, 0.003, n_rows))
    high   = prices * (1 + noise)
    low    = prices * (1 - noise)
    open_  = prices * (1 + np.random.normal(0, 0.002, n_rows))
    vol    = np.random.lognormal(15, 1, n_rows)

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  prices,
        "volume": vol,
    })


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c          = df["close"].values
    h, l       = df["high"].values, df["low"].values
    bb_up, bb_lo = _bollinger(c)
    df["rsi"]      = _rsi(c)
    df["macd"]     = _macd(c)
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_lo
    df["atr"]      = _atr(h, l, c)
    return df.dropna().reset_index(drop=True)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_historical(ticker: str) -> pd.DataFrame:
    """
    Generate N rows of historical OHLCV + indicators for one ticker.
    Adds a 'ticker' column and a 'target' (next-close normalised return).
    """
    np.random.seed(config.SEED + hash(ticker) % 1000)
    start = np.random.uniform(50, 400)
    df    = _simulate_ohlcv(config.INITIAL_SAMPLES + 50,
                             start_price=start)
    df    = _add_indicators(df)
    df    = df.iloc[:config.INITIAL_SAMPLES].copy()
    df["ticker"] = ticker
    df["target"] = df["close"].pct_change(config.PRED_HORIZON).shift(-config.PRED_HORIZON)
    df.dropna(inplace=True)
    return df[config.FEATURE_COLS + ["ticker", "target"]].reset_index(drop=True)


def generate_all_historical() -> pd.DataFrame:
    """Concatenate historical data for all configured tickers."""
    frames = [generate_historical(t) for t in config.TICKERS]
    return pd.concat(frames, ignore_index=True)


def streaming_batches(seed_offset: int = 999) -> Iterator[pd.DataFrame]:
    """
    Yield STREAM_BATCHES incremental data batches (with possible drift).
    Every 10th batch introduces a volatility regime change to test drift detection.
    """
    rng = np.random.default_rng(config.SEED + seed_offset)
    for batch_idx in range(config.STREAM_BATCHES):
        frames = []
        for ticker in config.TICKERS:
            mu    = rng.uniform(-0.0005, 0.001)
            sigma = (0.03 if (batch_idx % 10 == 9) else 0.015)   # regime shock
            n     = config.STREAM_BATCH_SZ + 50
            start = rng.uniform(50, 400)
            df    = _simulate_ohlcv(n, start_price=float(start),
                                    mu=float(mu), sigma=float(sigma))
            df    = _add_indicators(df)
            df    = df.iloc[:config.STREAM_BATCH_SZ].copy()
            df["ticker"] = ticker
            df["target"] = df["close"].pct_change(config.PRED_HORIZON).shift(-config.PRED_HORIZON)
            df.dropna(inplace=True)
            frames.append(df[config.FEATURE_COLS + ["ticker", "target"]])
        batch = pd.concat(frames, ignore_index=True)
        yield batch_idx, batch


def build_sequences(df: pd.DataFrame,
                    seq_len: int = config.SEQUENCE_LEN
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat DataFrame to (X, y) 3-D sequences.
    X shape: (N, seq_len, n_features)
    y shape: (N,)
    """
    feat  = df[config.FEATURE_COLS].values.astype(np.float32)
    tgt   = df["target"].values.astype(np.float32)
    # normalise features per window
    mean  = feat.mean(axis=0, keepdims=True)
    std   = feat.std(axis=0, keepdims=True) + 1e-8
    feat  = (feat - mean) / std

    xs, ys = [], []
    for i in range(seq_len, len(feat)):
        xs.append(feat[i - seq_len: i])
        ys.append(tgt[i])
    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    df = generate_all_historical()
    print(f"Historical data shape : {df.shape}")
    print(df.head())

    for idx, batch in streaming_batches():
        print(f"Batch {idx:02d} shape: {batch.shape}")
        if idx == 3:
            break
