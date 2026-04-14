"""
Epistemic (model) uncertainty via MC Dropout and deep ensembles
Module: epistemic_head.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EpistemicUncertaintyHead(nn.Module):
    """
    Epistemic (model) uncertainty via MC Dropout and deep ensembles
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


def mc_dropout_variance(model, x, n_samples=50):
    model.train()
    preds = torch.stack([model(x) for _ in range(n_samples)], 0)
    return preds.mean(0), preds.var(0)

