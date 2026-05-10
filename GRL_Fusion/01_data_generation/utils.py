#!/usr/bin/env python3
"""
GRL-Fusion - Shared Utility Functions
Common helpers used across all subfolders of the GRL-Fusion module.
"""

import numpy as np
import pandas as pd
import json
import os
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "GRL-Fusion", level: str = "INFO") -> logging.Logger:
    """Configure and return a logger for GRL-Fusion module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt     = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = get_logger("GRL-Fusion")

# ─────────────────────────────────────────────────────────────────────────────
# FILE I/O UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str, indent: int = 2) -> str:
    """Save data to JSON file with error handling."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug(f"Saved JSON → {path}")
    return path


def load_json(path: str) -> Any:
    """Load JSON file with error handling."""
    if not os.path.exists(path):
        logger.warning(f"JSON file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame to CSV."""
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)
    logger.debug(f"Saved CSV → {path} ({len(df)} rows)")
    return path


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load CSV to DataFrame."""
    if not os.path.exists(path):
        logger.warning(f"CSV file not found: {path}")
        return None
    return pd.read_csv(path)

# ─────────────────────────────────────────────────────────────────────────────
# MATH / ARRAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e     = np.exp(x - x_max)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return X / norms


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute z-scores along an axis."""
    mean = x.mean(axis=axis, keepdims=True)
    std  = x.std(axis=axis, keepdims=True) + 1e-10
    return (x - mean) / std


def clip_outliers(x: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Clip values beyond n_sigma standard deviations."""
    mu, sigma = x.mean(), x.std()
    return np.clip(x, mu - n_sigma * sigma, mu + n_sigma * sigma)


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute sample correlation matrix from observation matrix X (N x F)."""
    return np.corrcoef(X.T)


def adjacency_from_correlation(corr: np.ndarray,
                                threshold: float = 0.5,
                                symmetric: bool = True) -> np.ndarray:
    """Threshold correlation matrix to binary adjacency."""
    A = (np.abs(corr) >= threshold).astype(float)
    np.fill_diagonal(A, 0)
    if symmetric:
        A = np.maximum(A, A.T)
    return A

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric adjacency normalization: D^-1/2 (A+I) D^-1/2."""
    A_hat = A + np.eye(A.shape[0])
    d     = A_hat.sum(axis=1)
    d_inv = np.where(d > 0, d ** -0.5, 0.0)
    D     = np.diag(d_inv)
    return D @ A_hat @ D


def random_walk_normalize(A: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency: D^-1 A (random-walk normalization)."""
    d     = A.sum(axis=1) + 1e-10
    D_inv = np.diag(1.0 / d)
    return D_inv @ A


def compute_graph_stats(A: np.ndarray) -> dict:
    """Compute basic graph statistics."""
    n_nodes = A.shape[0]
    n_edges = int(A.sum()) // 2 if np.allclose(A, A.T) else int(A.sum())
    degrees = A.sum(axis=1)
    return {
        "n_nodes":       n_nodes,
        "n_edges":       n_edges,
        "density":       float(A.mean()),
        "avg_degree":    float(degrees.mean()),
        "max_degree":    float(degrees.max()),
        "min_degree":    float(degrees.min()),
        "degree_std":    float(degrees.std()),
        "is_symmetric":  bool(np.allclose(A, A.T)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    return np.log(prices[1:] / (prices[:-1] + 1e-10))


def rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling annualised volatility."""
    vol = np.full_like(returns, np.nan)
    for i in range(window - 1, len(returns)):
        vol[i] = returns[i - window + 1: i + 1].std() * np.sqrt(252)
    return vol


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio."""
    excess = returns - risk_free / 252
    return float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(252))


def drawdown(prices: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute drawdown series and maximum drawdown."""
    running_max = np.maximum.accumulate(prices)
    dd          = (prices - running_max) / (running_max + 1e-10)
    return dd, float(dd.min())

# ─────────────────────────────────────────────────────────────────────────────
# MANIPULATION DETECTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def detect_price_spikes(prices: np.ndarray,
                         returns: np.ndarray,
                         threshold_sigma: float = 3.0) -> np.ndarray:
    """Flag timesteps where price return exceeds threshold_sigma std devs."""
    z = zscore(returns)
    return (np.abs(z) > threshold_sigma).astype(int)


def detect_volume_spikes(volumes: np.ndarray,
                          threshold_sigma: float = 2.5) -> np.ndarray:
    """Flag timesteps with unusually high volume."""
    z = zscore(volumes)
    return (z > threshold_sigma).astype(int)


def detect_sentiment_bursts(sentiment: np.ndarray,
                              window: int = 10,
                              threshold: float = 0.5) -> np.ndarray:
    """Flag windows with rapid sentiment change."""
    flags = np.zeros(len(sentiment), dtype=int)
    for i in range(window, len(sentiment)):
        window_sent = sentiment[i - window: i]
        if window_sent.mean() > threshold or window_sent.mean() < -threshold:
            flags[i] = 1
    return flags


def coordination_score(timestamps: List[str], window_minutes: int = 5) -> float:
    """
    Estimate coordination among posts by measuring temporal clustering.
    High score = posts arrived in tight time windows (coordinated campaign).
    """
    if len(timestamps) < 2:
        return 0.0
    from datetime import datetime
    times = []
    for ts in timestamps:
        try:
            t = datetime.fromisoformat(ts.replace("Z", ""))
            times.append(t.timestamp())
        except (ValueError, AttributeError):
            pass
    if len(times) < 2:
        return 0.0
    times.sort()
    gaps = np.diff(times) / 60  # in minutes
    tight_gaps = (gaps <= window_minutes).mean()
    return float(tight_gaps)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKSUM / HASH
# ─────────────────────────────────────────────────────────────────────────────

def file_md5(path: str) -> Optional[str]:
    """Compute MD5 checksum of a file."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dict_hash(d: dict) -> str:
    """Compute a stable hash for a dictionary (for caching)."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    print(f"GRL-Fusion utils loaded.")
    A = (np.random.rand(5, 5) < 0.4).astype(float)
    np.fill_diagonal(A, 0)
    stats = compute_graph_stats(A)
    print(f"Graph stats: {stats}")
    prices = np.cumprod(1 + np.random.randn(100) * 0.01) * 100
    dd, mdd = drawdown(prices)
    print(f"Max drawdown: {mdd:.2%}")
