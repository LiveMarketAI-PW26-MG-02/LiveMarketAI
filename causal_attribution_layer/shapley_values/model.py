"""
SHAP-based modality contribution scoring for regime-conditioned attention
Module: shapley_values.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ShapleyAttributionComputer(nn.Module):
    """
    SHAP-based modality contribution scoring for regime-conditioned attention
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


def kernel_shap_attention(model, X, R, n_permutations=256):
    n_sources = X.shape[1]
    shap_vals = np.zeros(n_sources)
    for _ in range(n_permutations):
        perm = np.random.permutation(n_sources)
        for k in range(n_sources):
            mask = np.zeros(n_sources); mask[perm[:k+1]] = 1
            Xm = X * torch.tensor(mask, dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad(): Mt, _ = model(Xm, R)
            shap_vals[perm[k]] += Mt.norm().item()
    return shap_vals / n_permutations

