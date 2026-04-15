"""
Learnable smoothing kernel adapting to local regime stability
Module: smoothing_kernel.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveSmoothingKernel(nn.Module):
    """
    Learnable smoothing kernel adapting to local regime stability
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


class GaussianSmoothingKernel(nn.Module):
    def __init__(self, max_window=20):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.max_window = max_window
    def forward(self, alpha_seq):
        sigma = self.log_sigma.exp().clamp(0.5, self.max_window/2)
        w = torch.arange(self.max_window, dtype=torch.float32) - self.max_window//2
        kernel = torch.exp(-0.5*(w/sigma)**2); kernel /= kernel.sum()
        return F.conv1d(alpha_seq.unsqueeze(1), kernel.view(1,1,-1),
                        padding=self.max_window//2).squeeze(1)

