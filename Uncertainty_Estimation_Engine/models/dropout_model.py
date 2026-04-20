"""
MC-Dropout model for uncertainty estimation.
Wraps a neural network with dropout for stochastic inference.
"""
import numpy as np
from typing import Tuple, List
from core.base_estimator import BaseUncertaintyEstimator


class MCDropoutModel(BaseUncertaintyEstimator):
    """
    Simple feedforward network with MC-Dropout uncertainty.
    Dropout remains active at test time for stochastic predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        n_samples: int = 50,
        name: str = "mc_dropout",
    ):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        dims = [input_dim] + hidden_dims + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros(dims[i + 1]))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h = np.maximum(0, h @ W + b)
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(float)
            h *= mask / (1.0 - self.dropout_rate)
        return h @ self.weights[-1] + self.biases[-1]

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MCDropoutModel":
        self._is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        samples = np.stack([self._forward(X).squeeze() for _ in range(self.n_samples)])
        mean_pred = samples.mean(axis=0)
        epistemic = samples.var(axis=0)
        aleatoric = np.zeros_like(epistemic)
        return mean_pred, epistemic, aleatoric
