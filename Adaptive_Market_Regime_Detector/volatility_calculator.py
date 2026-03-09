"""volatility_calculator.py — realized & EWMA volatility"""
import numpy as np


class VolatilityCalculator:
    def realized_vol(self, returns: np.ndarray, window: int = 21) -> float:
        if len(returns) < 2:
            return 0.0
        r = returns[-window:] if len(returns) >= window else returns
        return float(np.std(r))

    def annualized_vol(self, returns: np.ndarray, window: int = 21, periods: int = 252) -> float:
        return self.realized_vol(returns, window) * np.sqrt(periods)

    def ewma_vol(self, returns: np.ndarray, lambda_: float = 0.94) -> float:
        if len(returns) < 2:
            return 0.0
        var = float(np.var(returns[:5])) if len(returns) >= 5 else float(returns[0] ** 2)
        for r in returns:
            var = lambda_ * var + (1 - lambda_) * r ** 2
        return np.sqrt(var)

    def vol_of_vol(self, returns: np.ndarray, window: int = 21) -> float:
        if len(returns) < window * 2:
            return 0.0
        vols = [self.realized_vol(returns[i:i+window]) for i in range(0, len(returns)-window, 5)]
        return float(np.std(vols)) if vols else 0.0
