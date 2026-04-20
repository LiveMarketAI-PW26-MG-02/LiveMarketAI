"""
API Schemas
============
Pydantic-compatible data models for request/response validation.
Falls back to plain dataclasses if pydantic is not installed.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class PredictRequest:
    """Single prediction request."""
    features: List[float]
    method: str = "bayesian"
    confidence_level: float = 0.95
    return_samples: bool = False

    def validate(self):
        if not self.features:
            raise ValueError("features must be non-empty")
        if not (0 < self.confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1)")
        valid_methods = {"bayesian", "monte_carlo", "ensemble", "conformal", "gp"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictResponse:
    """Single prediction response."""
    mean: float
    std: float
    epistemic: float
    aleatoric: float
    lower_bound: float
    upper_bound: float
    method: str
    latency_ms: float = 0.0
    samples: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def interval_width(self) -> float:
        return self.upper_bound - self.lower_bound

    def epistemic_fraction(self) -> float:
        total = self.epistemic + self.aleatoric
        return self.epistemic / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRequest:
    """Batch prediction request."""
    instances: List[List[float]]
    method: str = "bayesian"
    confidence_level: float = 0.95

    def validate(self):
        if not self.instances:
            raise ValueError("instances must be non-empty")
        lengths = {len(row) for row in self.instances}
        if len(lengths) > 1:
            raise ValueError("All instances must have the same number of features")
        return self


@dataclass
class CalibrationRequest:
    """Calibration endpoint request."""
    calibration_scores: List[float]
    alpha: float = 0.1
    method: str = "split_conformal"

    def validate(self):
        if not self.calibration_scores:
            raise ValueError("calibration_scores must be non-empty")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        return self


@dataclass
class CalibrationResponse:
    """Calibration endpoint response."""
    quantile_hat: float
    alpha: float
    n_calibration: int
    expected_coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
