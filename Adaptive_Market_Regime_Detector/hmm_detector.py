"""hmm_detector.py — Gaussian HMM regime detector"""
import numpy as np
from typing import Tuple
from logger import get_logger

logger = get_logger("hmm")


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False

    def fit(self, returns_df):
        try:
            from hmmlearn import hmm
            # Use all symbols' average returns + vol features
            avg_returns = returns_df.mean(axis=1).values.reshape(-1, 1)
            # Add rolling vol feature
            vol = np.array([np.std(avg_returns[max(0,i-10):i+1])
                            for i in range(len(avg_returns))]).reshape(-1, 1)
            X = np.hstack([avg_returns, vol])
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=1000,
                random_state=42,
            )
            self.model.fit(X)
            self.fitted = True
            logger.info(f"HMM fitted with {self.n_regimes} regimes")
        except ImportError:
            logger.warning("hmmlearn not installed — using threshold classifier")
        except Exception as e:
            logger.error(f"HMM fit error: {e}")

    def predict_latest(self, returns: np.ndarray) -> Tuple[int, float]:
        if not self.fitted or self.model is None:
            return self._threshold_predict(returns)
        try:
            recent = returns[-30:] if len(returns) >= 30 else returns
            avg = np.mean(recent)
            vol = np.std(recent)
            X = np.array([[avg, vol]])
            probs = self.model.predict_proba(X)[0]
            regime = int(np.argmax(probs))
            # Sort regimes by mean vol to get consistent labeling
            means = [self.model.means_[i][1] for i in range(self.n_regimes)]
            sorted_idx = np.argsort(means)
            mapped = int(np.where(sorted_idx == regime)[0][0])
            return mapped, float(probs[regime])
        except Exception as e:
            logger.debug(f"HMM predict fallback: {e}")
            return self._threshold_predict(returns)

    def _threshold_predict(self, returns: np.ndarray) -> Tuple[int, float]:
        if len(returns) < 5:
            return 0, 0.5
        vol = float(np.std(returns[-21:] if len(returns) >= 21 else returns))
        if vol < 0.005:
            return 0, 0.85
        elif vol < 0.012:
            return 1, 0.75
        elif vol < 0.025:
            return 2, 0.70
        else:
            return 3, 0.80
