"""
feature_extractor.py — Incremental feature computation over a rolling window of ticks.
Only newly arriving ticks trigger partial updates; full recomputation is avoided.
"""

import math
from collections import deque
from typing import Dict, List, Optional
from core.data_generator import StockTick
from core.config import CFG


class RollingStats:
    """Welford online algorithm for mean and variance — O(1) per update."""

    def __init__(self, maxlen: int):
        self._buf: deque = deque(maxlen=maxlen)
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def add(self, x: float) -> None:
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            # Remove old value using reverse Welford
            if self._n > 1:
                old_mean = self._mean
                self._mean = (self._mean * self._n - old) / (self._n - 1)
                self._M2 -= (old - old_mean) * (old - self._mean)
                self._n -= 1
        self._buf.append(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._M2 / self._n if self._n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def count(self) -> int:
        return self._n


class IncrementalEMA:
    """Exponential moving average updated tick-by-tick."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class FeatureExtractor:
    """
    Maintains incremental technical indicators over a streaming tick window.
    Features are recomputed partially on each new tick rather than from scratch.
    """

    def __init__(self, window_size: int):
        self._ws = window_size
        self._prices: deque = deque(maxlen=window_size)
        self._volumes: deque = deque(maxlen=window_size)
        self._returns: deque = deque(maxlen=window_size)
        self._price_stats = RollingStats(maxlen=window_size)
        self._volume_stats = RollingStats(maxlen=window_size)
        self._ema_fast = IncrementalEMA(alpha=0.2)
        self._ema_slow = IncrementalEMA(alpha=0.05)
        self._ema_vol = IncrementalEMA(alpha=0.15)
        self._last_price: Optional[float] = None
        self._ticks_seen = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, tick: StockTick) -> None:
        """Ingest one new tick and refresh all incremental indicators."""
        p = tick.price
        v = tick.volume

        if self._last_price is not None:
            ret = (p - self._last_price) / self._last_price
            self._returns.append(ret)
        self._last_price = p

        self._prices.append(p)
        self._volumes.append(v)
        self._price_stats.add(p)
        self._volume_stats.add(float(v))
        self._ema_fast.update(p)
        self._ema_slow.update(p)
        self._ema_vol.update(float(v))
        self._ticks_seen += 1

    def resize(self, new_window_size: int) -> None:
        """Adapt the buffer sizes when the dynamic window changes."""
        if new_window_size == self._ws:
            return
        self._ws = new_window_size
        self._prices = deque(self._prices, maxlen=new_window_size)
        self._volumes = deque(self._volumes, maxlen=new_window_size)
        self._returns = deque(self._returns, maxlen=new_window_size)
        # RollingStats are re-created (cheap since deques are small)
        new_ps = RollingStats(maxlen=new_window_size)
        for p in self._prices:
            new_ps.add(p)
        self._price_stats = new_ps
        new_vs = RollingStats(maxlen=new_window_size)
        for v in self._volumes:
            new_vs.add(v)
        self._volume_stats = new_vs

    # ------------------------------------------------------------------
    # Feature vector
    # ------------------------------------------------------------------

    def extract(self) -> Optional[Dict[str, float]]:
        """Return the current feature dict, or None if not enough data yet."""
        if len(self._prices) < 5 or len(self._returns) < 5:
            return None

        prices = list(self._prices)
        returns = list(self._returns)
        volumes = list(self._volumes)

        volatility = self._price_stats.std / (self._price_stats.mean + 1e-9)
        momentum_5  = self._momentum(prices, 5)
        momentum_10 = self._momentum(prices, 10) if len(prices) >= 10 else 0.0
        momentum_20 = self._momentum(prices, 20) if len(prices) >= 20 else 0.0
        rsi = self._rsi(returns, period=min(14, len(returns)))
        macd_signal = (
            (self._ema_fast.value or 0.0) - (self._ema_slow.value or 0.0)
        ) / (self._ema_slow.value or 1.0)
        vol_ratio = (
            volumes[-1] / (self._volume_stats.mean + 1e-9)
            if self._volume_stats.mean > 0 else 1.0
        )
        z_score = (
            (prices[-1] - self._price_stats.mean) / (self._price_stats.std + 1e-9)
        )
        mean_return = sum(returns) / len(returns)
        return_std   = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        skewness = self._skewness(returns, mean_return, return_std)
        rolling_max = max(prices)
        rolling_min = min(prices)
        drawdown = (prices[-1] - rolling_max) / (rolling_max + 1e-9)
        range_pct  = (rolling_max - rolling_min) / (rolling_min + 1e-9)

        return {
            "volatility":      round(volatility, 6),
            "momentum_5":      round(momentum_5, 6),
            "momentum_10":     round(momentum_10, 6),
            "momentum_20":     round(momentum_20, 6),
            "rsi":             round(rsi, 4),
            "macd_signal":     round(macd_signal, 6),
            "vol_ratio":       round(vol_ratio, 4),
            "z_score":         round(z_score, 4),
            "return_std":      round(return_std, 6),
            "skewness":        round(skewness, 4),
            "drawdown":        round(drawdown, 6),
            "range_pct":       round(range_pct, 6),
            "price":           round(prices[-1], 4),
            "ema_fast":        round(self._ema_fast.value or 0.0, 4),
            "ema_slow":        round(self._ema_slow.value or 0.0, 4),
            "ticks_seen":      float(self._ticks_seen),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _momentum(prices: List[float], period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] - prices[-period - 1]) / (prices[-period - 1] + 1e-9)

    @staticmethod
    def _rsi(returns: List[float], period: int) -> float:
        if not returns:
            return 50.0
        recent = returns[-period:]
        gains  = [r for r in recent if r > 0]
        losses = [-r for r in recent if r < 0]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _skewness(returns: List[float], mean: float, std: float) -> float:
        n = len(returns)
        if n < 3 or std < 1e-12:
            return 0.0
        return sum(((r - mean) / std) ** 3 for r in returns) / n


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.data_generator import StockDataGenerator
    gen = StockDataGenerator(seed=0)
    gen.start()
    fx = FeatureExtractor(window_size=60)
    for i, tick in enumerate(gen.iter_ticks(max_ticks=80)):
        fx.update(tick)
        feats = fx.extract()
        if feats:
            print(f"tick={i:03d} rsi={feats['rsi']:.2f} vol={feats['volatility']:.5f} "
                  f"z={feats['z_score']:.3f}")
    gen.stop()
