"""
Inflation regime detection using Hidden Markov Models.
Classifies regimes: low, moderate, high, hyperinflation.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class RegimeDetector:
    """
    Detects inflationary regimes via HMM or threshold-based classification.
    """

    THRESHOLDS = {"deflation": 0.0, "low": 2.0, "moderate": 4.0, "high": 7.0}
    LABELS     = ["deflation", "low", "moderate", "high", "hyperinflation"]

    def __init__(self, n_regimes: int = 4, method: str = "threshold"):
        self.n_regimes = n_regimes
        self.method = method
        self._model = None
        self.is_fitted = False

    def fit(self, series: pd.Series) -> "RegimeDetector":
        if self.method == "hmm":
            try:
                from hmmlearn.hmm import GaussianHMM
                X = series.dropna().values.reshape(-1, 1)
                self._model = GaussianHMM(n_components=self.n_regimes, covariance_type="full",
                                          n_iter=200, random_state=42)
                self._model.fit(X)
            except Exception:
                pass
        self.is_fitted = True
        return self

    def classify(self, value: float) -> str:
        if value < 0:           return "deflation"
        if value < 2.0:         return "low"
        if value < 4.0:         return "moderate"
        if value < 7.0:         return "high"
        return "hyperinflation"

    def regime_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.classify)

    def regime_durations(self, series: pd.Series) -> Dict[str, float]:
        regimes = self.regime_series(series)
        durations: Dict[str, list] = {r: [] for r in self.LABELS}
        current, count = regimes.iloc[0], 1
        for r in regimes.iloc[1:]:
            if r == current:
                count += 1
            else:
                durations[current].append(count)
                current, count = r, 1
        durations[current].append(count)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in durations.items()}

    def transition_matrix(self, series: pd.Series) -> np.ndarray:
        regimes = self.regime_series(series).values
        n = len(self.LABELS)
        idx = {r: i for i, r in enumerate(self.LABELS)}
        mat = np.zeros((n, n))
        for i in range(len(regimes) - 1):
            mat[idx[regimes[i]], idx[regimes[i+1]]] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        return np.divide(mat, row_sums, where=row_sums > 0)
