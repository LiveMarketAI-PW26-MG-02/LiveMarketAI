"""
Core Uncertainty Estimation Engine
Central orchestration for all uncertainty estimation methods.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

from .base_estimator import BaseUncertaintyEstimator
from .monte_carlo import MonteCarloEstimator
from .conformal_prediction import ConformalPredictor

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    TOTAL = "total"


@dataclass
class UncertaintyResult:
    predictions: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_epistemic(self) -> float:
        return float(np.mean(self.epistemic_uncertainty))

    @property
    def mean_aleatoric(self) -> float:
        return float(np.mean(self.aleatoric_uncertainty))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions.tolist(),
            "epistemic_uncertainty": self.epistemic_uncertainty.tolist(),
            "aleatoric_uncertainty": self.aleatoric_uncertainty.tolist(),
            "total_uncertainty": self.total_uncertainty.tolist(),
            "mean_epistemic": self.mean_epistemic,
            "mean_aleatoric": self.mean_aleatoric,
            "metadata": self.metadata,
        }


class UncertaintyEngine:
    """
    Main engine that orchestrates all uncertainty estimation approaches.
    Supports Bayesian, ensemble, MC-dropout, and conformal methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._models: Dict[str, BaseUncertaintyEstimator] = {}
        self._mc_estimator = MonteCarloEstimator(
            n_samples=self.config.get("mc_samples", 100)
        )
        self._conformal = ConformalPredictor(
            alpha=self.config.get("confidence_level", 0.1)
        )
        self._results_cache: Dict[str, UncertaintyResult] = {}
        logger.info("UncertaintyEngine initialized with config: %s", self.config)

    def register_model(self, name: str, model: BaseUncertaintyEstimator) -> None:
        """Register a model with the engine."""
        if name in self._models:
            logger.warning("Overwriting existing model: %s", name)
        self._models[name] = model
        logger.info("Model '%s' registered successfully.", name)

    def unregister_model(self, name: str) -> None:
        """Remove a model from the engine."""
        self._models.pop(name, None)

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def predict(
        self,
        model_name: str,
        X: np.ndarray,
        use_cache: bool = False,
    ) -> UncertaintyResult:
        """
        Run uncertainty-aware prediction using the specified model.
        Returns predictions plus decomposed uncertainty estimates.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not registered. Available: {self.list_models()}")

        cache_key = f"{model_name}_{hash(X.tobytes())}"
        if use_cache and cache_key in self._results_cache:
            logger.debug("Cache hit for model '%s'.", model_name)
            return self._results_cache[cache_key]

        model = self._models[model_name]
        logger.info("Running prediction with model '%s' on %d samples.", model_name, len(X))

        raw_preds, epistemic, aleatoric = model.predict_with_uncertainty(X)
        total = np.sqrt(epistemic**2 + aleatoric**2)

        ci_lower, ci_upper = self._conformal.get_intervals(raw_preds, total)

        result = UncertaintyResult(
            predictions=raw_preds,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_intervals=(ci_lower, ci_upper),
            metadata={"model": model_name, "n_samples": len(X)},
        )

        if use_cache:
            self._results_cache[cache_key] = result

        return result

    def compare_models(
        self, X: np.ndarray, model_names: Optional[List[str]] = None
    ) -> Dict[str, UncertaintyResult]:
        """Run all registered models and return comparative results."""
        names = model_names or self.list_models()
        return {name: self.predict(name, X) for name in names}

    def calibrate(self, model_name: str, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate a registered model using calibration data."""
        model = self._models[model_name]
        model.calibrate(X_cal, y_cal)
        logger.info("Model '%s' calibrated on %d samples.", model_name, len(X_cal))

    def clear_cache(self) -> None:
        self._results_cache.clear()

    def __repr__(self) -> str:
        return f"UncertaintyEngine(models={self.list_models()}, config={self.config})"
