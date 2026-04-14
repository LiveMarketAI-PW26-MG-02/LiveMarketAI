"""Ablation: static vs dynamic attention."""
import torch, torch.nn as nn, numpy as np
from attention_equation import RegimeConditionedAttention

class StaticAttention(nn.Module):
    def __init__(self, d_model, n_sources):
        super().__init__()
        self.v=nn.Linear(d_model,d_model); self.o=nn.Linear(d_model,d_model)
    def forward(self, X, R=None):
        return self.o(self.v(X).mean(1)), torch.ones(X.size(0),X.size(1))/X.size(1)

def run_ablation(n=5):
    results={"static":[],"dynamic":[]}
    for _ in range(n):
        X=torch.randn(128,8,256); R=torch.randint(0,4,(128,)); T=torch.randn(128,256)
        for name,model in [("static",StaticAttention(256,8)),("dynamic",RegimeConditionedAttention(256,4,8))]:
            model.eval()
            with torch.no_grad(): Mt,_=model(X,R)
            results[name].append(nn.functional.mse_loss(Mt,T).item())
    for k,v in results.items(): print(f"{k}: mean={np.mean(v):.4f}±{np.std(v):.4f}")
