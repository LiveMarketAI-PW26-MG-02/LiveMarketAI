"""
window_manager.py — Coordinates the lifecycle of the sliding tick window:
creation, overlap, resizing, and expiry. Acts as the single gatekeeper
between the raw tick stream and the feature/classification pipeline.
"""

import time
from collections import deque
from typing import List, Optional, Tuple
from core.data_generator import StockTick
from core.config import CFG


class WindowSnapshot:
    """Immutable snapshot of a completed or in-flight window."""

    def __init__(self, ticks: List[StockTick], window_id: int,
                 start_ts: float, end_ts: float, window_size: int):
        self.ticks = ticks
        self.window_id = window_id
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.window_size = window_size
        self.duration_ms = (end_ts - start_ts) * 1000.0

    def __len__(self) -> int:
        return len(self.ticks)

    def prices(self) -> List[float]:
        return [t.price for t in self.ticks]

    def volumes(self) -> List[int]:
        return [t.volume for t in self.ticks]


class WindowManager:
    """
    Manages a circular buffer of ticks and produces WindowSnapshots
    according to the configured overlap policy.

    Sliding rule:
        - A new prediction window starts every (1 - overlap_fraction) * window_size ticks.
        - The window contains the last `window_size` ticks at the moment it fires.
    """

    def __init__(self, window_size: Optional[int] = None,
                 overlap_fraction: Optional[float] = None):
        ws  = window_size      or CFG.window.base_window_size
        ovr = overlap_fraction or CFG.window.overlap_fraction
        self._ws  = ws
        self._ovr = ovr
        self._stride = max(1, int(ws * (1.0 - ovr)))
        self._buf: deque = deque(maxlen=ws)
        self._ticks_since_last_window = 0
        self._window_id = 0
        self._total_ticks = 0
        self._last_resize_tick = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, tick: StockTick) -> Optional[WindowSnapshot]:
        """
        Ingest one tick. Returns a WindowSnapshot when the stride condition
        is met and the buffer has enough ticks; otherwise returns None.
        """
        self._buf.append(tick)
        self._total_ticks += 1
        self._ticks_since_last_window += 1

        if (len(self._buf) >= self._ws and
                self._ticks_since_last_window >= self._stride):
            snapshot = self._take_snapshot()
            self._ticks_since_last_window = 0
            return snapshot
        return None

    def resize(self, new_window_size: int) -> bool:
        """
        Resize the window. Respects cooldown to avoid thrashing.
        Returns True if the resize was applied.
        """
        cfg = CFG.window
        cooldown_ok = (self._total_ticks - self._last_resize_tick) >= cfg.resize_cooldown_ticks
        new_ws = max(cfg.min_window_size, min(cfg.max_window_size, new_window_size))
        if not cooldown_ok or new_ws == self._ws:
            return False

        self._ws = new_ws
        self._stride = max(1, int(new_ws * (1.0 - self._ovr)))
        new_buf: deque = deque(maxlen=new_ws)
        for t in self._buf:
            new_buf.append(t)
        self._buf = new_buf
        self._last_resize_tick = self._total_ticks
        return True

    @property
    def current_window_size(self) -> int:
        return self._ws

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def buffer_fill_pct(self) -> float:
        return len(self._buf) / self._ws * 100.0

    def peek_buffer(self) -> List[StockTick]:
        return list(self._buf)

    def overlap_ticks(self) -> int:
        """Number of ticks shared between consecutive windows."""
        return self._ws - self._stride

    def stats(self) -> dict:
        return {
            "window_id":           self._window_id,
            "window_size":         self._ws,
            "stride":              self._stride,
            "overlap_ticks":       self.overlap_ticks(),
            "buffer_fill_pct":     round(self.buffer_fill_pct, 1),
            "total_ticks_ingested": self._total_ticks,
            "ticks_since_last_window": self._ticks_since_last_window,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> WindowSnapshot:
        ticks = list(self._buf)[-self._ws:]
        snap = WindowSnapshot(
            ticks=ticks,
            window_id=self._window_id,
            start_ts=ticks[0].timestamp,
            end_ts=ticks[-1].timestamp,
            window_size=self._ws,
        )
        self._window_id += 1
        return snap


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.data_generator import StockDataGenerator
    gen = StockDataGenerator(seed=1)
    gen.start()
    wm  = WindowManager(window_size=20, overlap_fraction=0.5)
    windows_seen = 0
    for tick in gen.iter_ticks(max_ticks=200):
        snap = wm.ingest(tick)
        if snap:
            windows_seen += 1
            print(f"[WIN {snap.window_id:04d}] size={len(snap)} "
                  f"duration={snap.duration_ms:.0f}ms "
                  f"fill={wm.buffer_fill_pct:.0f}%")
    gen.stop()
    print(f"\nTotal windows produced: {windows_seen}")
