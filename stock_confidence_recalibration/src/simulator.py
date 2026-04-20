"""
Stock Data Simulator
====================
Generates realistic synthetic stock data with regime changes, volatility
clustering, and correlated assets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TickData:
    symbol: str
    price: float
    volume: float
    timestamp: float
    returns: float
    regime_label: str


class StockSimulator:
    """
    Simulates realistic multi-asset stock tick data with:
    - Regime changes (bull/bear/sideways)
    - GARCH-like volatility clustering
    - Correlated asset returns
    - Mean reversion in sideways markets
    """

    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"]

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.prices: Dict[str, float] = {s: np.random.uniform(100, 500)
                                          for s in self.SYMBOLS}
        self.regime_probs = {
            "bullish":  {"bullish": 0.85, "bearish": 0.05, "sideways": 0.10},
            "bearish":  {"bullish": 0.05, "bearish": 0.85, "sideways": 0.10},
            "sideways": {"bullish": 0.15, "bearish": 0.15, "sideways": 0.70},
        }
        self.current_regime: str = np.random.choice(
            ["bullish", "bearish", "sideways"])
        self.regime_duration: int = 0
        self.regime_change_every: int = np.random.randint(30, 80)
        self.vol_state: float = 0.02  # GARCH vol state

    def _transition_regime(self):
        self.regime_duration += 1
        if self.regime_duration >= self.regime_change_every:
            probs = self.regime_probs[self.current_regime]
            self.current_regime = np.random.choice(
                list(probs.keys()), p=list(probs.values()))
            self.regime_duration = 0
            self.regime_change_every = np.random.randint(30, 80)

    def _update_volatility(self, last_return: float):
        """GARCH(1,1)-style volatility update."""
        omega = 0.00001
        alpha = 0.10
        beta  = 0.85
        self.vol_state = np.sqrt(
            omega + alpha * last_return**2 + beta * self.vol_state**2)
        self.vol_state = float(np.clip(self.vol_state, 0.005, 0.08))

    def _regime_drift(self) -> float:
        drifts = {"bullish": 0.0008, "bearish": -0.0008, "sideways": 0.0}
        return drifts[self.current_regime]

    def generate_tick(self, timestamp: Optional[float] = None) -> List[TickData]:
        if timestamp is None:
            timestamp = time.time()

        self._transition_regime()
        drift = self._regime_drift()

        # Common market factor
        market_shock = np.random.normal(0, self.vol_state)
        self._update_volatility(market_shock)

        ticks = []
        for symbol in self.SYMBOLS:
            # Idiosyncratic component
            idio   = np.random.normal(0, self.vol_state * 0.6)
            # Beta-weighted market exposure (different betas per stock)
            betas  = {"AAPL": 1.1, "MSFT": 1.0, "GOOGL": 1.2, "AMZN": 1.15,
                      "NVDA": 1.5, "TSLA": 1.8, "META": 1.3, "JPM": 0.9}
            beta   = betas.get(symbol, 1.0)
            ret    = drift + beta * market_shock + idio

            # Volume spike during high volatility
            vol_base = np.random.lognormal(10, 0.5)
            vol_mult = 1.0 + 3.0 * (abs(market_shock) / (self.vol_state + 1e-8))
            volume   = vol_base * vol_mult

            self.prices[symbol] *= (1 + ret)
            self.prices[symbol]  = max(self.prices[symbol], 1.0)

            ticks.append(TickData(
                symbol=symbol,
                price=round(self.prices[symbol], 2),
                volume=round(volume),
                timestamp=timestamp,
                returns=float(ret),
                regime_label=self.current_regime
            ))

        return ticks

    def generate_base_prediction(self, symbol: str, tick: TickData) -> Tuple[float, int]:
        """
        Simulates a base ML model's raw confidence output.
        The model is intentionally miscalibrated to demonstrate recalibration.
        """
        # True signal with noise
        true_prob = 0.5 + 0.2 * np.sign(tick.returns) * min(abs(tick.returns) / 0.02, 1.0)

        # Add regime-specific bias (model has regime blind spots)
        regime_bias = {"bullish": 0.08, "bearish": -0.05, "sideways": 0.02}
        true_prob  += regime_bias.get(tick.regime_label, 0.0)

        # Model overconfidence (systematic miscalibration)
        raw_conf = 0.5 + (true_prob - 0.5) * 1.6  # inflate confidence
        raw_conf = float(np.clip(raw_conf + np.random.normal(0, 0.04), 0.1, 0.95))

        # True direction (ground truth)
        true_up = true_prob > 0.5 + np.random.uniform(-0.1, 0.1)
        direction = 1 if true_up else -1

        return raw_conf, direction
