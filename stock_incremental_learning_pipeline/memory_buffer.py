"""
memory_buffer.py — Requirement 2
Stock Memory Buffer System
--------------------------
Retains a compact, representative subset of historical stock data using three
strategies:
  • reservoir  — random sampling ensuring uniform coverage of the data stream
  • fifo        — sliding window of the most recent samples
  • priority    — keeps samples with highest prediction error (hard examples)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import config


class StockMemoryBuffer:
    """
    Maintains a fixed-size buffer of (X, y) training pairs drawn from the
    entire stock data stream.  All strategies expose the same API:

        buffer.add(X_batch, y_batch)  → add new observations
        buffer.sample(n)              → draw n samples for rehearsal
        buffer.size                   → current number of stored samples
    """

    def __init__(self,
                 capacity: int = config.BUFFER_SIZE,
                 strategy: str = config.BUFFER_STRATEGY):
        assert strategy in {"reservoir", "fifo", "priority"}, \
            f"Unknown strategy: {strategy}"
        self.capacity = capacity
        self.strategy = strategy
        self._n_seen  = 0          # total samples seen so far (for reservoir)

        # Storage: numpy arrays grown up to capacity
        self._X: Optional[np.ndarray] = None   # (capacity, seq_len, features)
        self._y: Optional[np.ndarray] = None   # (capacity,)
        self._errors: Optional[np.ndarray] = None   # prediction errors (priority)
        self._ptr = 0              # circular write pointer (fifo / reservoir)

        print(f"[MemoryBuffer] capacity={capacity}, strategy='{strategy}'")

    # ─── Public API ──────────────────────────────────────────────────────────

    def add(self,
            X_batch: np.ndarray,
            y_batch: np.ndarray,
            errors: Optional[np.ndarray] = None) -> None:
        """
        Ingest a new mini-batch of sequences.

        Parameters
        ----------
        X_batch : (B, seq_len, features)
        y_batch : (B,)
        errors  : (B,) — prediction errors; required for 'priority' strategy
        """
        n = len(y_batch)
        if self._X is None:
            # First call — allocate storage
            seq_len  = X_batch.shape[1]
            n_feat   = X_batch.shape[2]
            self._X  = np.zeros((self.capacity, seq_len, n_feat), dtype=np.float32)
            self._y  = np.zeros(self.capacity,  dtype=np.float32)
            self._errors = np.zeros(self.capacity, dtype=np.float32)

        if self.strategy == "reservoir":
            self._reservoir_add(X_batch, y_batch)
        elif self.strategy == "fifo":
            self._fifo_add(X_batch, y_batch)
        elif self.strategy == "priority":
            err = errors if errors is not None else np.abs(y_batch)
            self._priority_add(X_batch, y_batch, err)

        self._n_seen += n

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw up to `n` samples from the buffer (without replacement if possible).
        Returns (X, y) arrays.
        """
        filled = min(self._n_seen, self.capacity)
        if filled == 0:
            raise RuntimeError("Buffer is empty — call add() first.")
        n = min(n, filled)
        idx = np.random.choice(filled, size=n, replace=False)
        return self._X[idx], self._y[idx]

    @property
    def size(self) -> int:
        return min(self._n_seen, self.capacity)

    def summary(self) -> str:
        return (f"MemoryBuffer | strategy={self.strategy} | "
                f"stored={self.size}/{self.capacity} | total_seen={self._n_seen}")

    # ─── Strategy Implementations ────────────────────────────────────────────

    def _reservoir_add(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Classic reservoir sampling: every sample has equal probability of
        being retained regardless of when it arrived.
        """
        for i in range(len(y)):
            t = self._n_seen + i
            if t < self.capacity:
                self._X[t]      = X[i]
                self._y[t]      = y[i]
                self._errors[t] = 0.0
            else:
                j = np.random.randint(0, t + 1)
                if j < self.capacity:
                    self._X[j]      = X[i]
                    self._y[j]      = y[i]
                    self._errors[j] = 0.0

    def _fifo_add(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sliding window — always keep the most recent `capacity` samples."""
        for i in range(len(y)):
            pos            = self._ptr % self.capacity
            self._X[pos]   = X[i]
            self._y[pos]   = y[i]
            self._ptr     += 1

    def _priority_add(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       errors: np.ndarray) -> None:
        """
        Replace the lowest-error stored sample when the buffer is full and the
        incoming sample has a higher error (hard-example mining).
        """
        for i in range(len(y)):
            filled = min(self._n_seen + i, self.capacity)
            if filled < self.capacity:
                self._X[filled]      = X[i]
                self._y[filled]      = y[i]
                self._errors[filled] = errors[i]
            else:
                min_idx = int(np.argmin(self._errors[:self.capacity]))
                if errors[i] > self._errors[min_idx]:
                    self._X[min_idx]      = X[i]
                    self._y[min_idx]      = y[i]
                    self._errors[min_idx] = errors[i]


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    buf = StockMemoryBuffer(capacity=200, strategy="reservoir")

    for step in range(20):
        Xb = rng.random((30, 20, 10)).astype(np.float32)
        yb = rng.random(30).astype(np.float32)
        buf.add(Xb, yb)

    Xs, ys = buf.sample(50)
    print(buf.summary())
    print(f"Sampled X shape: {Xs.shape}, y shape: {ys.shape}")
