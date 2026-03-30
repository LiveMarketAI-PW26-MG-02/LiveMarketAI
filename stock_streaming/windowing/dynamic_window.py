"""
dynamic_window.py — Implements the dynamic window resizing mechanism.
Window size adapts based on a composite score from volatility, volume
activity, and trend clarity to improve classification in changing regimes.
"""

import math
from collections import deque
from typing import Optional, Tuple
from core.config import CFG


class RegimeDector:
    """
    Identifies the current market regime (trending / ranging / volatile)
    from recent price statistics.
    """

    TRENDING  = "TRENDING"
    RANGING   = "RANGING"
    VOLATILE  = "VOLATILE"
    STABLE    = "STABLE"

    def __init__(self, lookback: int = 30):
        self._returns: deque = deque(maxlen=lookback)
        self._vols:    deque = deque(maxlen=lookback)

    def observe(self, ret: float, volatility: float) -> None:
        self._returns.append(ret)
        self._vols.append(volatility)

    def classify(self) -> str:
        if len(self._returns) < 5:
            return self.STABLE
        mean_v = sum(self._vols) / len(self._vols)
        rets   = list(self._returns)
        pos    = sum(1 for r in rets if r > 0)
        neg    = len(rets) - pos
        bias   = abs(pos - neg) / (len(rets) + 1e-9)  # 0 = balanced, 1 = all one direction

        if mean_v > 0.015:           return self.VOLATILE
        if bias > 0.5:               return self.TRENDING
        if mean_v < 0.003 and bias < 0.25: return self.STABLE
        return self.RANGING


class DynamicWindowResizer:
    """
    Computes the recommended window size each tick based on:
        - Rolling volatility (high vol → smaller window, react faster)
        - Volume activity ratio (high activity → larger window, more data)
        - Trend strength (strong trend → medium window)
        - Market regime label
    The recommended size is smoothed with an EMA to avoid jitter.
    """

    def __init__(self):
        cfg = CFG.window
        self._base = cfg.base_window_size
        self._min  = cfg.min_window_size
        self._max  = cfg.max_window_size
        self._vs   = cfg.volatility_scale_factor
        self._as   = cfg.activity_scale_factor
        self._regime = RegimeDector()
        self._smoothed_size: float = float(self._base)
        self._ema_alpha = 0.1          # Slow EMA to prevent thrashing
        self._history: deque = deque(maxlen=500)
        self._current_regime = RegimeDector.STABLE
        self._ticks = 0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def observe(self, volatility: float, volume: int,
                mean_volume: float = 1.0) -> int:
        """
        Feed one tick worth of stats and get back the current recommended size.
        """
        self._ticks += 1

        # Compute the ideal window size this tick
        raw_size = self._compute_raw_size(volatility, volume, mean_volume)

        # EMA smooth
        self._smoothed_size = (self._ema_alpha * raw_size +
                               (1 - self._ema_alpha) * self._smoothed_size)

        clamped = int(max(self._min, min(self._max, round(self._smoothed_size))))
        self._history.append(clamped)
        return clamped

    def recommended_window_size(self) -> int:
        """Return the current smoothed window size recommendation."""
        return int(max(self._min, min(self._max, round(self._smoothed_size))))

    # ------------------------------------------------------------------
    # Regime exposure
    # ------------------------------------------------------------------

    def update_regime(self, ret: float, volatility: float) -> str:
        self._regime.observe(ret, volatility)
        self._current_regime = self._regime.classify()
        return self._current_regime

    @property
    def current_regime(self) -> str:
        return self._current_regime

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        hist = list(self._history)
        return {
            "smoothed_size":    self.recommended_window_size(),
            "regime":           self._current_regime,
            "ticks_processed":  self._ticks,
            "mean_size":        round(sum(hist) / len(hist), 1) if hist else self._base,
            "min_seen":         min(hist) if hist else self._base,
            "max_seen":         max(hist) if hist else self._base,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_raw_size(self, volatility: float, volume: int,
                          mean_volume: float) -> float:
        """
        Core sizing formula:
            size = base / (1 + vol_scale * norm_vol) * (1 + act_scale * norm_vol_ratio)

        High volatility → denominator grows → size shrinks.
        High volume activity → numerator factor grows → size grows.
        """
        # Normalise volatility against a reference level
        ref_vol = 0.005
        norm_vol = volatility / (ref_vol + 1e-9)

        # Volume activity ratio
        vol_ratio = volume / (mean_volume + 1e-9)
        norm_act  = math.log1p(max(0, vol_ratio - 1))  # log-scale, 0 when ratio=1

        # Regime modifier
        regime_mod = {
            RegimeDector.VOLATILE:  0.7,
            RegimeDector.TRENDING:  1.0,
            RegimeDector.RANGING:   1.2,
            RegimeDector.STABLE:    1.3,
        }.get(self._current_regime, 1.0)

        raw = (self._base / (1.0 + self._vs * norm_vol)) \
              * (1.0 + self._as * norm_act * 0.1) \
              * regime_mod

        return float(max(self._min, min(self._max, raw)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(5)
    resizer = DynamicWindowResizer()

    print(f"{'Tick':>5} {'Vol':>8} {'Volume':>8} {'Regime':<12} {'Size':>6}")
    for i in range(60):
        vol = rng.uniform(0.001, 0.03) * (3 if 20 <= i < 35 else 1)
        volume = rng.randint(500, 5000) * (5 if 40 <= i < 50 else 1)
        ret = rng.gauss(0, vol)
        resizer.update_regime(ret, vol)
        size = resizer.observe(vol, volume, mean_volume=2000)
        if i % 5 == 0:
            print(f"{i:5d} {vol:8.4f} {volume:8d} {resizer.current_regime:<12} {size:6d}")

    print(resizer.stats())
