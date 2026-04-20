"""
Abstract base class for all uncertainty estimators.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict, Any


class BaseUncertaintyEstimator(ABC):
    """
    Abstract base for any model that supports uncertainty estimation.
    Subclasses must implement predict_with_uncertainty().
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._is_fitted = False
        self._calibration_data: Optional[Dict] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseUncertaintyEstimator":
        """Train the model on data."""
        ...

    @abstractmethod
    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            predictions: mean predictions shape (n,)
            epistemic: model/knowledge uncertainty shape (n,)
            aleatoric: data/noise uncertainty shape (n,)
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions only."""
        preds, _, _ = self.predict_with_uncertainty(X)
        return preds

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Optional post-hoc calibration. Override in subclasses."""
        self._calibration_data = {"X": X_cal, "y": y_cal}

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_fitted": self._is_fitted,
            "has_calibration": self._calibration_data is not None,
        }

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before prediction.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
