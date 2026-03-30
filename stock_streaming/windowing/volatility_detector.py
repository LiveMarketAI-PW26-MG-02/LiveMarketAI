"""
volatility_detector.py — Real-time volatility estimation and regime detection
used to drive the dynamic window resizer. Combines Parkinson, Garman-Klass,
and rolling-return volatility estimators into a composite signal.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple
from core.config import CFG


class ParkinsonsVolatility:
    """
    Parkinson (1980) high-low volatility estimator.
    Uses only high and low prices — no close-to-close dependency.
    """

    SCALE = 1.0 / (4.0 * math.log(2))

    def __init__(self, window: int = 20):
        self._buf: deque = deque(maxlen=window)

    def update(self, high: float, low: float) -> Optional[float]:
        if high <= 0 or low <= 0 or high < low:
            return None
        self._buf.append(math.log(high / low) ** 2)
        if len(self._buf) < 2:
            return None
        return math.sqrt(self.SCALE * sum(self._buf) / len(self._buf))


class RollingReturnVolatility:
    """Classic close-to-close return standard deviation."""

    def __init__(self, window: int = 20):
        self._returns: deque = deque(maxlen=window)
        self._last_price: Optional[float] = None

    def update(self, price: float) -> Optional[float]:
        if self._last_price is not None and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self._returns.append(ret)
        self._last_price = price
        if len(self._returns) < 2:
            return None
        mean = sum(self._returns) / len(self._returns)
        var  = sum((r - mean) ** 2 for r in self._returns) / (len(self._returns) - 1)
        return math.sqrt(var)


class VolatilityDetector:
    """
    Composite real-time volatility detector.

    Combines:
        1. Rolling return std (fast, noisy)
        2. Parkinson high-low estimator (smoother, needs H/L)
        3. EMA of the composite (dampens transient spikes)

    Outputs:
        - current_volatility: float
        - volatility_regime:  str ("LOW", "NORMAL", "HIGH", "EXTREME")
        - recommended_window_size: int
    """

    THRESHOLDS = {
        "LOW":     0.003,
        "NORMAL":  0.008,
        "HIGH":    0.018,
        "EXTREME": float("inf"),
    }

    def __init__(self, window: int = 20):
        self._return_vol  = RollingReturnVolatility(window=window)
        self._parkinson   = ParkinsonsVolatility(window=window)
        self._vol_ema: Optional[float] = None
        self._ema_alpha   = 0.15
        self._history: deque = deque(maxlen=1000)
        self._volume_buf: deque = deque(maxlen=window)
        self._cfg = CFG.window
        self._tick = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def observe(self, volatility_estimate: float, volume: int,
                high: Optional[float] = None, low: Optional[float] = None,
                price: Optional[float] = None) -> None:
        """Ingest one tick's worth of stats."""
        self._tick += 1
        self._volume_buf.append(volume)

        # Parkinson if H/L available
        pk_vol = None
        if high is not None and low is not None and high > low:
            pk_vol = self._parkinson.update(high, low)

        # Return-based if price available
        ret_vol = None
        if price is not None:
            ret_vol = self._return_vol.update(price)

        # Composite: prefer Parkinson when available, else return-based,
        # else fall back to externally provided estimate
        if pk_vol is not None and ret_vol is not None:
            composite = 0.5 * pk_vol + 0.5 * ret_vol
        elif pk_vol is not None:
            composite = pk_vol
        elif ret_vol is not None:
            composite = ret_vol
        else:
            composite = volatility_estimate

        # EMA smooth
        if self._vol_ema is None:
            self._vol_ema = composite
        else:
            self._vol_ema = self._ema_alpha * composite + (1 - self._ema_alpha) * self._vol_ema

        self._history.append(self._vol_ema)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def current_volatility(self) -> float:
        return self._vol_ema or 0.0

    @property
    def volatility_regime(self) -> str:
        v = self.current_volatility
        for regime, threshold in self.THRESHOLDS.items():
            if v <= threshold:
                return regime
        return "EXTREME"

    def recommended_window_size(self) -> int:
        """
        Volatility-driven window size:
            - EXTREME → min_window (react immediately)
            - HIGH    → 60% of base
            - NORMAL  → base
            - LOW     → 140% of base (smooth out the noise)
        """
        multipliers = {
            "LOW":     1.4,
            "NORMAL":  1.0,
            "HIGH":    0.6,
            "EXTREME": 0.33,
        }
        m = multipliers.get(self.volatility_regime, 1.0)
        raw = int(self._cfg.base_window_size * m)
        return max(self._cfg.min_window_size, min(self._cfg.max_window_size, raw))

    def mean_volume(self) -> float:
        if not self._volume_buf:
            return 1.0
        return sum(self._volume_buf) / len(self._volume_buf)

    def vol_spike(self) -> bool:
        """True if current volatility is more than 3× the recent average."""
        if len(self._history) < 10:
            return False
        recent_mean = sum(list(self._history)[-10:]) / 10
        return self.current_volatility > 3 * recent_mean + 1e-9

    def stats(self) -> Dict:
        hist = list(self._history)
        return {
            "current_vol":       round(self.current_volatility, 6),
            "regime":            self.volatility_regime,
            "recommended_ws":    self.recommended_window_size(),
            "vol_spike":         self.vol_spike(),
            "mean_vol":          round(sum(hist) / len(hist), 6) if hist else 0.0,
            "max_vol":           round(max(hist), 6) if hist else 0.0,
            "ticks_processed":   self._tick,
            "mean_volume":       round(self.mean_volume(), 1),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.data_generator import StockDataGenerator
    gen = StockDataGenerator(seed=9)
    gen.start()
    det = VolatilityDetector(window=20)
    prev_price = None

    for i, tick in enumerate(gen.iter_ticks(max_ticks=150)):
        det.observe(0.0, tick.volume, price=tick.price)
        if i % 15 == 0:
            print(f"tick={i:3d} vol={det.current_volatility:.5f} "
                  f"regime={det.volatility_regime:<8} ws={det.recommended_window_size()}")

    gen.stop()
    print(det.stats())
