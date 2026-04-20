"""
Aleatoric (data) uncertainty estimation via learned noise variance
Module: aleatoric_head.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AleatoricUncertaintyHead(nn.Module):
    """
    Aleatoric (data) uncertainty estimation via learned noise variance
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


def heteroscedastic_loss(mu, log_var, target):
    var = log_var.exp().clamp(min=1e-6)
    return (0.5 * ((target - mu).pow(2) / var + log_var)).mean()

