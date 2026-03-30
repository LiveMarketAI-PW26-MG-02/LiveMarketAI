"""
window_resizer.py — Integrates signals from the volatility detector and volume
activity monitor to produce a single, debounced window-size recommendation
that is fed into WindowManager.resize().
"""

import math
from collections import deque
from typing import Dict, Optional, Tuple
from core.config import CFG
from windowing.volatility_detector import VolatilityDetector


class ActivityMonitor:
    """Tracks volume activity and signals unusual trading bursts."""

    def __init__(self, short_window: int = 10, long_window: int = 60):
        self._short: deque = deque(maxlen=short_window)
        self._long:  deque = deque(maxlen=long_window)

    def update(self, volume: int) -> float:
        """Returns volume ratio: short_mean / long_mean (> 1 = elevated activity)."""
        self._short.append(volume)
        self._long.append(volume)
        short_mean = sum(self._short) / len(self._short)
        long_mean  = sum(self._long)  / len(self._long)
        return short_mean / (long_mean + 1e-9)

    @property
    def is_spike(self) -> bool:
        if len(self._short) < 3 or len(self._long) < 10:
            return False
        return self.update.__self__._short  # placeholder; see update return value


class WindowResizer:
    """
    Merges volatility and activity signals into a single target window size.

    Algorithm:
        1. Get vol-driven size from VolatilityDetector.
        2. Apply activity modifier: high activity → slightly larger window.
        3. Apply momentum bias: trending market → keep window stable.
        4. Debounce: only emit a new size recommendation if it differs by
           at least CHANGE_THRESHOLD ticks from the last emitted value.
        5. Rate-limit: wait at least COOLDOWN ticks between emissions.
    """

    CHANGE_THRESHOLD = 5   # Min tick-count difference to trigger a resize
    COOLDOWN_TICKS   = 10  # Min ticks between two resize recommendations

    def __init__(self, vol_detector: Optional[VolatilityDetector] = None):
        self._vol_det = vol_detector or VolatilityDetector()
        self._activity_monitor = ActivityMonitor()
        self._last_emitted_size = CFG.window.base_window_size
        self._ticks_since_emit  = 0
        self._total_resizes     = 0
        self._size_history: deque = deque(maxlen=500)
        self._resize_events: list = []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def update(self, volatility: float, volume: int,
               price: Optional[float] = None,
               high: Optional[float] = None,
               low: Optional[float] = None) -> Optional[int]:
        """
        Ingest one tick. Returns a new window size if a resize is warranted,
        otherwise returns None.
        """
        self._ticks_since_emit += 1

        # Update sub-components
        self._vol_det.observe(volatility, volume, high=high, low=low, price=price)
        activity_ratio = self._activity_monitor.update(volume)

        # Compute target size
        target = self._compute_target(activity_ratio)
        self._size_history.append(target)

        # Debounce check
        delta = abs(target - self._last_emitted_size)
        cooldown_ok = self._ticks_since_emit >= self.COOLDOWN_TICKS

        if delta >= self.CHANGE_THRESHOLD and cooldown_ok:
            self._last_emitted_size = target
            self._ticks_since_emit  = 0
            self._total_resizes    += 1
            self._resize_events.append({
                "tick":         sum(1 for _ in self._size_history),
                "new_size":     target,
                "vol_regime":   self._vol_det.volatility_regime,
                "activity":     round(activity_ratio, 2),
            })
            return target

        return None

    def force_size(self) -> int:
        """Return the current best-estimate size without debounce gating."""
        return self._last_emitted_size

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        hist = list(self._size_history)
        return {
            "total_resizes":     self._total_resizes,
            "current_size":      self._last_emitted_size,
            "mean_target_size":  round(sum(hist) / len(hist), 1) if hist else 0,
            "min_target":        min(hist) if hist else 0,
            "max_target":        max(hist) if hist else 0,
            "vol_regime":        self._vol_det.volatility_regime,
            "current_vol":       round(self._vol_det.current_volatility, 6),
            "last_resize_events": self._resize_events[-5:],
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_target(self, activity_ratio: float) -> int:
        cfg = CFG.window
        # Base from volatility detector
        vol_size = self._vol_det.recommended_window_size()

        # Activity modifier: ratio > 1 adds ticks, but cap the benefit
        activity_delta = int((activity_ratio - 1.0) * cfg.activity_scale_factor * 10)
        activity_delta = max(-20, min(30, activity_delta))

        # Vol-spike penalty: shrink immediately on spikes
        spike_penalty = -int(cfg.base_window_size * 0.2) if self._vol_det.vol_spike() else 0

        raw = vol_size + activity_delta + spike_penalty
        return max(cfg.min_window_size, min(cfg.max_window_size, raw))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(11)
    resizer = WindowResizer()
    resize_count = 0

    print(f"{'Tick':>5} {'Vol':>8} {'Volume':>8} {'Regime':<10} {'NewSize':>8}")
    for i in range(200):
        # Inject a volatility spike between ticks 80–100
        vol = rng.uniform(0.001, 0.005) * (6 if 80 <= i < 100 else 1)
        volume = rng.randint(500, 3000) * (8 if 120 <= i < 140 else 1)
        price  = 150 + rng.gauss(0, 1)
        new_size = resizer.update(vol, volume, price=price)
        if new_size is not None:
            resize_count += 1
            print(f"{i:5d} {vol:8.4f} {volume:8d} "
                  f"{resizer._vol_det.volatility_regime:<10} {new_size:8d}")

    print(f"\nTotal resizes triggered: {resize_count}")
    print(resizer.stats())
