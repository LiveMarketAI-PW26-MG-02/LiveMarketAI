"""
data_store.py — Thread-safe in-memory store for streaming predictions,
feature snapshots, and window metadata.  Acts as the shared state layer
accessible by the pipeline, evaluation, and visualisation modules.
"""

import threading
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple


class RingBuffer:
    """Fixed-capacity thread-safe ring buffer."""

    def __init__(self, capacity: int):
        self._buf:  deque      = deque(maxlen=capacity)
        self._lock: threading.Lock = threading.Lock()

    def append(self, item: Any) -> None:
        with self._lock:
            self._buf.append(item)

    def latest(self, n: int = 1) -> List[Any]:
        with self._lock:
            buf = list(self._buf)
        return buf[-n:] if n <= len(buf) else buf

    def all(self) -> List[Any]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


class PredictionStore:
    """Stores streaming PredictionEvent objects."""

    def __init__(self, capacity: int = 5000):
        self._buf = RingBuffer(capacity)

    def push(self, event) -> None:
        self._buf.append(event)

    def latest_n(self, n: int) -> List:
        return self._buf.latest(n)

    def all(self) -> List:
        return self._buf.all()

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._buf.all():
            counts[e.label] = counts.get(e.label, 0) + 1
        return counts

    def mean_confidence(self) -> float:
        events = self._buf.all()
        if not events:
            return 0.0
        return sum(e.confidence for e in events) / len(events)

    def mean_latency_ms(self) -> float:
        events = self._buf.all()
        if not events:
            return 0.0
        return sum(e.latency_ms for e in events) / len(events)

    def __len__(self) -> int:
        return len(self._buf)


class FeatureStore:
    """Stores the most recent feature snapshots for debugging."""

    def __init__(self, capacity: int = 1000):
        self._buf = RingBuffer(capacity)

    def push(self, tick_index: int, features: Dict[str, float]) -> None:
        self._buf.append({"tick": tick_index, "features": features})

    def latest(self, n: int = 10) -> List[Dict]:
        return self._buf.latest(n)

    def feature_series(self, key: str, n: int = 200) -> List[float]:
        """Extract a time series for a single feature name."""
        return [r["features"].get(key, 0.0) for r in self._buf.latest(n)]


class WindowMetaStore:
    """Stores window lifecycle events (create, resize, expire)."""

    def __init__(self, capacity: int = 500):
        self._events = RingBuffer(capacity)
        self._resize_count = 0
        self._lock = threading.Lock()

    def record_window(self, window_id: int, window_size: int,
                      timestamp: float, event_type: str = "fire") -> None:
        self._events.append({
            "window_id":   window_id,
            "window_size": window_size,
            "timestamp":   timestamp,
            "type":        event_type,
        })

    def record_resize(self, old_size: int, new_size: int, reason: str) -> None:
        with self._lock:
            self._resize_count += 1
        self._events.append({
            "type":       "resize",
            "old_size":   old_size,
            "new_size":   new_size,
            "reason":     reason,
            "resize_no":  self._resize_count,
        })

    def resize_history(self) -> List[Dict]:
        return [e for e in self._events.all() if e.get("type") == "resize"]

    def window_size_series(self) -> List[int]:
        return [e["window_size"] for e in self._events.all()
                if e.get("type") == "fire"]

    @property
    def total_resizes(self) -> int:
        with self._lock:
            return self._resize_count


class DataStore:
    """
    Composite store providing unified access to predictions, features,
    and window metadata.  Singleton-friendly: instantiate once and pass
    by reference to all pipeline components.
    """

    def __init__(self, pred_capacity: int  = 5000,
                 feat_capacity: int  = 1000,
                 window_capacity: int = 500):
        self.predictions = PredictionStore(capacity=pred_capacity)
        self.features    = FeatureStore(capacity=feat_capacity)
        self.windows     = WindowMetaStore(capacity=window_capacity)
        self._start_time = None
        self._lock       = threading.Lock()

    def start(self) -> None:
        import time
        with self._lock:
            self._start_time = time.time()

    def elapsed_seconds(self) -> float:
        import time
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    def snapshot(self) -> Dict:
        """High-level runtime summary."""
        return {
            "elapsed_s":         round(self.elapsed_seconds(), 1),
            "total_predictions": len(self.predictions),
            "mean_confidence":   round(self.predictions.mean_confidence(), 4),
            "mean_latency_ms":   round(self.predictions.mean_latency_ms(), 4),
            "total_resizes":     self.windows.total_resizes,
            "label_counts":      self.predictions.label_counts(),
        }

    def reset(self) -> None:
        self.predictions._buf.clear()
        self.features._buf.clear()
        self.windows._events.clear()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Simulate pushing fake events
    class FakeEvent:
        def __init__(self, i):
            self.label      = ["BUY","HOLD","SELL"][i % 3]
            self.confidence = 0.5 + (i % 5) * 0.08
            self.latency_ms = 1.0 + i * 0.01

    ds = DataStore()
    ds.start()

    for i in range(50):
        ds.predictions.push(FakeEvent(i))
        ds.features.push(i, {"volatility": 0.001 * i, "rsi": 50 + i * 0.2})
        if i % 10 == 0:
            ds.windows.record_window(i, 60, time.time())

    ds.windows.record_resize(60, 40, "high_volatility")

    snap = ds.snapshot()
    for k, v in snap.items():
        print(f"  {k:<25}: {v}")

    print("\nLatest 3 features:")
    for rec in ds.features.latest(3):
        print(" ", rec)
