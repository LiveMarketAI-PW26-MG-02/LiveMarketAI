"""Dynamic α weight learning via regime-aware MLP."""
import torch, torch.nn as nn, torch.nn.functional as F

class DynamicWeightLearner(nn.Module):
    def __init__(self, regime_dim, n_sources, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(regime_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, n_sources))
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, regime_embed):
        logits = self.mlp(regime_embed) / self.temperature.clamp(min=0.1)
        return F.softmax(logits, dim=-1)

class RegimeEmbedder(nn.Module):
    def __init__(self, n_regimes, regime_dim=64):
        super().__init__()
        self.hard = nn.Embedding(n_regimes, regime_dim)
        self.soft_proj = nn.Linear(n_regimes, regime_dim)
    def forward(self, R_hard=None, R_soft=None):
        return self.soft_proj(R_soft) if R_soft is not None else self.hard(R_hard)
