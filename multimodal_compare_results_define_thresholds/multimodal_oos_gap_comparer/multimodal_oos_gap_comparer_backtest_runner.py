"""
  Multimodal Stock-Market Module
  ZIP Topic  : CompareResults
  Folder     : multimodal_oos_gap_comparer
  File       : multimodal_oos_gap_comparer_backtest_runner.py
  Purpose    : Compare Results — Oos Gap Comparer
"""

import numpy as np
import pandas as pd
import time, logging, copy, json, itertools
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
MULTIMODAL_MODULE = 'multimodal_oos_gap_comparer_backtest_runner'
TOPIC             = 'compare_results'
VERSION           = '1.0.0'
DEFAULT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class MultimodalOosGapComparerBacktestRunnerConfig:
    """Multimodal threshold-sweep config for oos_gap_comparer."""
    thresholds: List[float]   = field(default_factory=lambda: DEFAULT_THRESHOLDS)
    universe_size: int        = 500
    lookback: int             = 252
    n_bootstrap: int          = 500
    n_permutations: int       = 200
    confidence: float         = 0.95
    cost_per_trade_bps: float = 5.0
    min_holding_days: int     = 1
    stability_window: int     = 63
    doc_format: str           = 'markdown'

# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ThresholdResult:
    """Per-threshold evaluation record for multimodal sweep."""
    threshold: float
    sharpe: float
    calmar: float
    ic_mean: float
    hit_rate: float
    turnover: float
    cost_adj_sharpe: float
    max_drawdown: float
    n_trades: int
    pvalue: float = 1.0
    stable: bool  = False
    optimal: bool = False

# ── Abstract base ─────────────────────────────────────────────────────────────
class MultimodalThresholdBase(ABC):
    """Abstract interface for every multimodal threshold-analysis module."""
    @abstractmethod
    def sweep(self, signals: np.ndarray,
              returns: np.ndarray) -> Dict[float, ThresholdResult]: ...
    @abstractmethod
    def optimal(self) -> Optional[ThresholdResult]: ...
    @abstractmethod
    def summarize(self) -> str: ...

# ── Core Engine ───────────────────────────────────────────────────────────────
class MultimodalOosGapComparerEngine(MultimodalThresholdBase):
    """
    Multimodal threshold engine: compare results — oos gap comparer.
    Sweeps, evaluates, validates and documents signal thresholds.
    """
    def __init__(self, cfg: Optional[MultimodalOosGapComparerBacktestRunnerConfig] = None):
        self.cfg      = cfg or MultimodalOosGapComparerBacktestRunnerConfig()
        self.results  : Dict[float, ThresholdResult] = {}
        self.history  : List[Dict[str, Any]] = []
        self.docs     : List[str] = []
        self._run_id  : int = 0
        logger.info("Multimodal %s ready  topic=%s", MULTIMODAL_MODULE, TOPIC)

    # ── Threshold sweep ─────────────────────────────────────────────────────
    def sweep(self, signals: np.ndarray,
              returns: np.ndarray) -> Dict[float, ThresholdResult]:
        """Run multimodal grid sweep across all configured thresholds."""
        self.results.clear()
        for t in self.cfg.thresholds:
            self.results[t] = self._eval_threshold(t, signals, returns)
        self._mark_optimal()
        self._run_id += 1
        self.history.append({'run_id': self._run_id,
                             'n_thresh': len(self.cfg.thresholds),
                             'timestamp': datetime.utcnow().isoformat()})
        return self.results

    def _eval_threshold(self, t: float, signals: np.ndarray,
                        returns: np.ndarray) -> ThresholdResult:
        """Compute all metrics for a single multimodal threshold value."""
        mask       = np.abs(signals) >= t
        pos        = np.sign(signals) * mask.astype(float)
        strat_rets = pos[:-1] * returns[1:]
        n_trades   = int(mask.sum())