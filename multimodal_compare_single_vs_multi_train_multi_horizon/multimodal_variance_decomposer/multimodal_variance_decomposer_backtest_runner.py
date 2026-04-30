"""
  Multimodal Stock-Market Module
  ZIP Topic  : CompareSingleVsMulti
  Folder     : multimodal_variance_decomposer
  File       : multimodal_variance_decomposer_backtest_runner.py
  Purpose    : Compare Single Vs Multi — Variance Decomposer
"""

import numpy as np
import pandas as pd
import time, logging, copy, json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
MULTIMODAL_MODULE = 'multimodal_variance_decomposer_backtest_runner'
TOPIC             = 'compare_single_vs_multi'
VERSION           = '1.0.0'
HORIZONS          = [1, 5, 10, 21, 63, 126, 252]   # trading-day horizons

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class MultimodalVarianceDecomposerBacktestRunnerConfig:
    """Multimodal multi-horizon config for variance_decomposer."""
    horizons: List[int]       = field(default_factory=lambda: HORIZONS)
    universe_size: int        = 500
    train_window: int         = 756          # 3 yrs
    test_window: int          = 252          # 1 yr
    learning_rate: float      = 1e-3
    l2_reg: float             = 1e-4
    n_estimators: int         = 100
    confidence: float         = 0.95
    n_bootstrap: int          = 1000
    doc_format: str           = 'markdown'   # markdown | json | html

# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class HorizonResult:
    """Stores per-horizon train/test metrics for multimodal comparison."""
    horizon: int
    sharpe_train: float
    sharpe_test: float
    ic_mean: float
    ic_std: float
    max_drawdown: float
    hit_rate: float
    n_trades: int
    pvalue: float = 1.0
    significant: bool = False

# ── Abstract base ─────────────────────────────────────────────────────────────
class MultimodalBaseHorizon(ABC):
    """Abstract multimodal multi-horizon pipeline stage."""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, horizon: int) -> None: ...
    @abstractmethod
    def predict(self, X: np.ndarray, horizon: int) -> np.ndarray: ...
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> HorizonResult: ...

# ── Core engine ───────────────────────────────────────────────────────────────
class MultimodalVarianceDecomposerEngine(MultimodalBaseHorizon):
    """
    Multimodal engine: compare single vs multi — variance decomposer.
    Trains, evaluates, tunes and documents multi-horizon stock signals.
    """
    def __init__(self, cfg: Optional[MultimodalVarianceDecomposerBacktestRunnerConfig] = None):
        self.cfg     = cfg or MultimodalVarianceDecomposerBacktestRunnerConfig()
        self.models  : Dict[int, np.ndarray] = {}   # horizon → weights
        self.results : Dict[int, HorizonResult] = {}
        self.history : List[Dict] = []
        self.docs    : List[str]  = []
        logger.info("Multimodal %s ready  topic=%s", MULTIMODAL_MODULE, TOPIC)

    # ── ABC ──────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, horizon: int) -> None:
        """Ridge-regression fit for one multimodal horizon."""
        XtX = X.T @ X + self.cfg.l2_reg * np.eye(X.shape[1])
        Xty = X.T @ y
        self.models[horizon] = np.linalg.solve(XtX, Xty)

    def predict(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Score assets with multimodal horizon-specific weights."""
        w = self.models.get(horizon)
        if w is None:
            return np.zeros(X.shape[0])
        return X @ w

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> HorizonResult:
        """Evaluate most-recently trained horizon."""
        h = max(self.models) if self.models else 1
        return self._score_horizon(h, X, y)

    # ── Multi-horizon training ───────────────────────────────────────────────
    def train_all_horizons(self, prices: np.ndarray) -> Dict[int, HorizonResult]:
        """Train multimodal model on every configured horizon."""
        for h in self.cfg.horizons:
            X, y = self._build_features(prices, h)