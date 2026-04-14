"""Training loop for regime-conditioned attention."""
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from attention_equation import RegimeConditionedAttention

class AttentionTrainer:
    def __init__(self, d_model=256, n_regimes=4, n_sources=8, lr=3e-4):
        self.model = RegimeConditionedAttention(d_model, n_regimes, n_sources)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        self.criterion = nn.MSELoss()
    def train_step(self, X, R, targets):
        self.model.train(); self.optimizer.zero_grad()
        Mt, alpha = self.model(X, R)
        loss = self.criterion(Mt, targets)
        entropy = -(alpha * (alpha+1e-8).log()).sum(-1).mean()
        (loss - 0.01*entropy).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        return loss.item(), alpha.detach()
    def evaluate(self, X, R, targets):
        self.model.eval()
        with torch.no_grad():
            Mt, alpha = self.model(X, R)
            loss = self.criterion(Mt, targets)
        return loss.item(), Mt, alpha
