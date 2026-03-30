"""
streaming_classifier.py — Online-learning classifier for the streaming pipeline.
Uses scikit-learn's SGDClassifier for incremental partial_fit updates, combined
with an in-memory replay buffer to prevent catastrophic forgetting.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from classification.base_classifier import BaseClassifier, LABELS
from core.config import CFG

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class ReplayBuffer:
    """Fixed-size FIFO buffer storing (feature_vector, label) pairs."""

    def __init__(self, maxlen: int = 2000):
        self._buf: deque = deque(maxlen=maxlen)

    def add(self, fv: List[float], label: str) -> None:
        self._buf.append((fv, label))

    def sample(self, n: int) -> Tuple[List[List[float]], List[str]]:
        items = random.sample(list(self._buf), min(n, len(self._buf)))
        X = [i[0] for i in items]
        y = [i[1] for i in items]
        return X, y

    def __len__(self) -> int:
        return len(self._buf)


class StreamingClassifier(BaseClassifier):
    """
    Incremental SGD-based classifier that updates with each new window.

    - partial_fit() is called on every labelled window (self-supervised label
      derived from the next-tick return; in production this would be the
      true market outcome).
    - A small replay sample prevents distribution drift.
    - Falls back to BaseClassifier._heuristic_predict() before first fit.
    """

    def __init__(self, replay_buffer_size: int = 2000, min_fit_samples: int = 50):
        super().__init__()
        self._replay = ReplayBuffer(maxlen=replay_buffer_size)
        self._min_fit_samples = min_fit_samples
        self._update_count = 0

        if _HAS_SKLEARN:
            self._sgd = SGDClassifier(
                loss="modified_huber",  # gives probability estimates
                max_iter=1,
                tol=None,
                random_state=CFG.evaluation.random_seed,
                class_weight="balanced",
            )
            self._scaler = StandardScaler()
            self._scaler_fitted = False
        else:
            self._sgd = None
            self._scaler = None
            self._scaler_fitted = False

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: List[Dict[str, float]], y: List[str]) -> None:
        """Bulk fit from a list of feature dicts (used for warm-start)."""
        if not _HAS_SKLEARN or not X:
            return
        vecs = [self.features_to_vector(f) for f in X]
        self._fit_vectors(vecs, y)

    def predict_raw(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Return (label, confidence, proba_dict)."""
        if not _HAS_SKLEARN or not self._fitted:
            label, conf = self._heuristic_predict(features)
            return label, conf, {l: 0.0 for l in LABELS}

        fv = self.features_to_vector(features)
        import numpy as np
        X = np.array([fv])
        if self._scaler_fitted:
            X = self._scaler.transform(X)
        probas = self._sgd.predict_proba(X)[0]
        classes = list(self._sgd.classes_)
        proba_dict = {LABELS[i]: 0.0 for i in range(len(LABELS))}
        for cls, p in zip(classes, probas):
            if cls in LABELS:
                proba_dict[cls] = float(p)
        label = max(proba_dict, key=proba_dict.get)
        confidence = proba_dict[label]
        return label, confidence, proba_dict

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, features: Dict[str, float], label: str,
               replay_batch: int = 32) -> None:
        """
        Incremental update: add new sample + replay some old ones.
        Called by the streaming pipeline after each prediction window.
        """
        if not _HAS_SKLEARN:
            return
        fv = self.features_to_vector(features)
        self._replay.add(fv, label)

        if len(self._replay) < self._min_fit_samples:
            return

        # Current sample + replay
        import numpy as np
        X_new  = [fv]
        y_new  = [label]
        if len(self._replay) >= replay_batch:
            X_r, y_r = self._replay.sample(replay_batch)
            X_new.extend(X_r)
            y_new.extend(y_r)

        X_arr = np.array(X_new)
        if not self._scaler_fitted:
            self._scaler.partial_fit(X_arr)
            self._scaler_fitted = True
        X_arr = self._scaler.transform(X_arr)

        self._sgd.partial_fit(X_arr, y_new, classes=LABELS)
        self._fitted = True
        self._update_count += 1

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit_vectors(self, vecs: List[List[float]], y: List[str]) -> None:
        import numpy as np
        X = np.array(vecs)
        if not self._scaler_fitted:
            self._scaler.fit(X)
            self._scaler_fitted = True
        X = self._scaler.transform(X)
        self._sgd.fit(X, y)
        self._fitted = True

    def diagnostics(self) -> Dict:
        base = super().diagnostics()
        base.update({
            "online_updates":    self._update_count,
            "replay_buffer_len": len(self._replay),
            "sklearn_available": _HAS_SKLEARN,
        })
        return base


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.data_generator import StockDataGenerator
    from core.feature_extractor import FeatureExtractor
    import random

    gen = StockDataGenerator(seed=42)
    gen.start()
    fx  = FeatureExtractor(window_size=30)
    clf = StreamingClassifier()

    for i, tick in enumerate(gen.iter_ticks(max_ticks=300)):
        fx.update(tick)
        feats = fx.extract()
        if feats is None:
            continue
        # Self-supervised label from momentum sign
        label = "BUY" if feats["momentum_5"] > 0 else "SELL"
        clf.update(feats, label)
        if i % 50 == 0 and clf._fitted:
            l, c = clf.predict(feats)
            print(f"tick={i:3d} → {l} ({c:.3f})")

    gen.stop()
    print(clf.diagnostics())
