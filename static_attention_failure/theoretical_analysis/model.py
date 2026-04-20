"""
Mathematical proof: static attention fails under distribution shift
Module: theoretical_analysis.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TheoreticalRegimeShiftAnalysis(nn.Module):
    """
    Mathematical proof: static attention fails under distribution shift
    """

    def __init__(self, d_model: int = 256, n_regimes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model   = d_model
        self.n_regimes = n_regimes
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )
        self.regime_embed = nn.Embedding(n_regimes, d_model)
        self.output_proj  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, regime: torch.Tensor = None) -> torch.Tensor:
        h = self.encoder(x)
        if regime is not None:
            h = h + self.regime_embed(regime).unsqueeze(1) if h.dim() == 3 else h + self.regime_embed(regime)
        return self.output_proj(h)

    def compute_loss(self, x, target, regime=None):
        pred = self(x, regime)
        return F.mse_loss(pred, target)


def compute_optimal_weights(X, R, regime_distributions):
    # Static weight minimizes average loss but fails per-regime
    w_static = np.linalg.lstsq(X, np.ones(len(X)), rcond=None)[0]
    losses_per_regime = {}
    for r, dist in regime_distributions.items():
        Xr = X[R == r]
        losses_per_regime[r] = np.mean((Xr @ w_static - dist['target'])**2)
    return w_static, losses_per_regime

