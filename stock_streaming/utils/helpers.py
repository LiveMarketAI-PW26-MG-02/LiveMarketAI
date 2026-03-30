"""
helpers.py — General-purpose utility functions and decorators used across
the stock streaming classification system.
"""

import os
import time
import math
import json
import hashlib
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context manager and callable decorator for latency measurement."""

    def __init__(self, name: str = ""):
        self.name     = name
        self.elapsed  = 0.0
        self._start   = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = (time.perf_counter() - self._start) * 1000  # ms

    @property
    def ms(self) -> float:
        return self.elapsed

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.3f} ms"


def timed(fn: F) -> F:
    """Decorator that prints execution time of a function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        print(f"[timed] {fn.__qualname__} took {ms:.3f} ms")
        return result
    return wrapper  # type: ignore


# ---------------------------------------------------------------------------
# Rate limiter / throttle
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter.
    Useful for ensuring the streaming inference loop does not exceed
    a maximum number of predictions per second.
    """

    def __init__(self, max_rate: float, burst: int = 1):
        self._rate     = max_rate        # tokens per second
        self._burst    = burst
        self._tokens   = float(burst)
        self._last_ts  = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self, timeout: float = 1.0) -> bool:
        """Block until a token is available or timeout is reached."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_ts
                self._tokens = min(self._burst,
                                   self._tokens + elapsed * self._rate)
                self._last_ts = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.001, 1.0 / self._rate))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile of a list (p in 0–100)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(math.ceil(p / 100 * len(sorted_v))) - 1)
    return sorted_v[idx]


def ewma(values: List[float], alpha: float = 0.1) -> List[float]:
    """Compute the exponential weighted moving average of a list."""
    result = []
    prev   = values[0] if values else 0.0
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        result.append(prev)
    return result


def zscore_list(values: List[float]) -> List[float]:
    """Standardise a list to zero mean, unit variance."""
    if not values:
        return []
    mean = sum(values) / len(values)
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) + 1e-9
    return [(v - mean) / std for v in values]


def running_mean(values: List[float]) -> List[float]:
    """Cumulative running mean."""
    result, s = [], 0.0
    for i, v in enumerate(values, 1):
        s += v
        result.append(s / i)
    return result


# ---------------------------------------------------------------------------
# Configuration / file helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it doesn't exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, filepath: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(filepath)))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def fingerprint(obj: Any) -> str:
    """Return an 8-char MD5 hex digest of the JSON representation of obj."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.md5(raw).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def label_to_direction(label: str) -> int:
    """Maps classification label to +1 (bullish), 0 (neutral), -1 (bearish)."""
    mapping = {
        "STRONG_BUY":  +1,
        "BUY":         +1,
        "HOLD":         0,
        "SELL":        -1,
        "STRONG_SELL": -1,
    }
    return mapping.get(label, 0)


def majority_label(labels: List[str]) -> str:
    """Return the most frequent label in the list."""
    if not labels:
        return "HOLD"
    counts: Dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)


def label_agreement(labels: List[str]) -> float:
    """Fraction of the list occupied by the majority label (0–1)."""
    if not labels:
        return 0.0
    dominant_count = max(labels.count(l) for l in set(labels))
    return dominant_count / len(labels)


# ---------------------------------------------------------------------------
# Progress bar (console only)
# ---------------------------------------------------------------------------

def progress_bar(current: int, total: int, width: int = 40,
                 prefix: str = "") -> str:
    """Return an ASCII progress bar string (no newline)."""
    frac   = min(1.0, current / max(1, total))
    filled = int(frac * width)
    bar    = "#" * filled + "-" * (width - filled)
    pct    = f"{frac * 100:5.1f}%"
    return f"\r{prefix}[{bar}] {pct} ({current}/{total})"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Timer
    with Timer("sleep_test") as t:
        time.sleep(0.05)
    print(t)

    # Stats helpers
    vals = [1.0, 2.0, 3.0, 10.0, 4.0, 5.0]
    print(f"p90={percentile(vals, 90):.2f}  zscore={zscore_list(vals)}")

    # Label utilities
    labels_sample = ["BUY", "BUY", "HOLD", "BUY", "SELL"]
    print(f"majority={majority_label(labels_sample)}  agreement={label_agreement(labels_sample):.2f}")

    # Progress bar
    for i in range(0, 11):
        print(progress_bar(i, 10, prefix="Processing: "), end="", flush=True)
        time.sleep(0.05)
    print()

    # Fingerprint
    print(f"fingerprint={{a:1,b:2}} → {fingerprint({'a': 1, 'b': 2})}")
