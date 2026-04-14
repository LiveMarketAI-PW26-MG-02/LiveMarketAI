"""Mt = Σ αi(Rt) Xi – regime-conditioned attention."""
import torch, torch.nn as nn, torch.nn.functional as F

class RegimeConditionedAttention(nn.Module):
    def __init__(self, d_model=256, n_regimes=4, n_sources=8):
        super().__init__()
        self.regime_queries = nn.Embedding(n_regimes, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj   = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
    def forward(self, X, R):
        q = self.regime_queries(R).unsqueeze(1)
        K, V = self.key_proj(X), self.value_proj(X)
        alpha = F.softmax((q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        Mt = (alpha @ V).squeeze(1)
        return self.out_proj(Mt), alpha.squeeze(1)
