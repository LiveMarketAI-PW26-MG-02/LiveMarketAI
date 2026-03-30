"""
batch_classifier.py — Static batch classifier trained once on historical data
and used as a fixed-model baseline for comparison against streaming classification.
"""

import time
from typing import Dict, List, Optional, Tuple
from classification.base_classifier import BaseClassifier, LABELS
from core.config import CFG

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def generate_synthetic_training_data(n_samples: int = 3000, seed: int = 42):
    """
    Produce labelled (features, label) pairs from a synthetic GBM simulation.
    Label is derived from the next-period return sign.
    Returns (X: list[dict], y: list[str]).
    """
    import math, random
    rng = random.Random(seed)
    price = 150.0
    mu, sigma = 0.0001, 0.005
    prices, volumes = [], []
    for _ in range(n_samples + 60):
        z = rng.gauss(0, 1)
        price *= math.exp((mu - 0.5 * sigma ** 2) * 1e-4 + sigma * math.sqrt(1e-4) * z)
        prices.append(price)
        volumes.append(rng.randint(100, 10_000))

    X, y = [], []
    for i in range(60, n_samples + 60):
        window_p = prices[i - 60:i]
        window_v = volumes[i - 60:i]
        returns  = [(window_p[j] - window_p[j-1]) / window_p[j-1] for j in range(1, 60)]
        ret_next = (prices[i] - prices[i-1]) / prices[i-1]
        # Coarse label from future return
        if ret_next >  0.005: label = "STRONG_BUY"
        elif ret_next > 0.001: label = "BUY"
        elif ret_next < -0.005: label = "STRONG_SELL"
        elif ret_next < -0.001: label = "SELL"
        else: label = "HOLD"

        mean_r = sum(returns) / len(returns)
        std_r  = math.sqrt(sum((r - mean_r)**2 for r in returns) / len(returns)) + 1e-9
        feats = {
            "volatility":  std_r / (abs(mean_r) + 1e-9),
            "momentum_5":  (window_p[-1] - window_p[-6]) / (window_p[-6] + 1e-9),
            "momentum_10": (window_p[-1] - window_p[-11]) / (window_p[-11] + 1e-9),
            "rsi":         50.0 + mean_r * 5000,
            "z_score":     (window_p[-1] - sum(window_p) / 60) / (std_r * window_p[-1] + 1e-9),
            "vol_ratio":   window_v[-1] / (sum(window_v) / 60),
            "price":       window_p[-1],
            "drawdown":    (window_p[-1] - max(window_p)) / (max(window_p) + 1e-9),
            "range_pct":   (max(window_p) - min(window_p)) / (min(window_p) + 1e-9),
        }
        X.append(feats)
        y.append(label)
    return X, y


class BatchClassifier(BaseClassifier):
    """
    Random-Forest based classifier trained in one offline batch.
    Once fitted it is frozen — no online updates.
    This is the baseline against which the streaming classifier is compared.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 8):
        super().__init__()
        self._n_estimators = n_estimators
        self._max_depth     = max_depth
        self._pipeline: Optional[object] = None
        self._feature_keys: Optional[List[str]] = None
        self._train_accuracy: float = 0.0

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: List[Dict[str, float]], y: List[str]) -> None:
        if not _HAS_SKLEARN or not X:
            return
        self._feature_keys = sorted(X[0].keys())
        X_arr = np.array([[f[k] for k in self._feature_keys] for f in X])
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                n_jobs=-1,
                random_state=CFG.evaluation.random_seed,
                class_weight="balanced",
            )),
        ])
        t0 = time.perf_counter()
        self._pipeline.fit(X_arr, y)
        fit_time = (time.perf_counter() - t0) * 1000
        preds = self._pipeline.predict(X_arr)
        self._train_accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
        self._fitted = True
        print(f"[BatchClassifier] Trained on {len(X)} samples in {fit_time:.0f}ms | "
              f"train_acc={self._train_accuracy:.3f}")

    def predict_raw(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        if not _HAS_SKLEARN or not self._fitted or not self._pipeline:
            label, conf = self._heuristic_predict(features)
            return label, conf, {}

        keys = self._feature_keys or sorted(features.keys())
        fv   = np.array([[features.get(k, 0.0) for k in keys]])
        probas  = self._pipeline.predict_proba(fv)[0]
        classes = list(self._pipeline.named_steps["clf"].classes_)
        proba_dict = {l: 0.0 for l in LABELS}
        for cls, p in zip(classes, probas):
            proba_dict[cls] = float(p)
        label = max(proba_dict, key=proba_dict.get)
        return label, proba_dict[label], proba_dict

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict:
        base = super().diagnostics()
        base.update({
            "train_accuracy":    self._train_accuracy,
            "n_estimators":      self._n_estimators,
            "sklearn_available": _HAS_SKLEARN,
        })
        return base


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic training data …")
    X_train, y_train = generate_synthetic_training_data(n_samples=1000)
    label_counts = {l: y_train.count(l) for l in LABELS}
    print(f"Label distribution: {label_counts}")

    clf = BatchClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    # Evaluate on the last 200 samples
    correct = sum(clf.predict(X_train[i])[0] == y_train[i] for i in range(-200, 0))
    print(f"Last-200 accuracy: {correct / 200:.3f}")
    print(clf.diagnostics())
