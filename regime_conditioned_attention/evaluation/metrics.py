"""Evaluation metrics for regime-conditioned attention."""
import torch, numpy as np
from scipy.stats import entropy as scipy_entropy

def attention_consistency(alpha_seq, regime_seq):
    within=[]
    for r in np.unique(regime_seq):
        mask=regime_seq==r
        if mask.sum()>1: within.append(alpha_seq[mask].std(0).mean())
    global_std=alpha_seq.std(0).mean()
    return {"within_regime_std":float(np.mean(within)),"global_std":float(global_std),
            "consistency_ratio":float(np.mean(within)/(global_std+1e-8))}

def regime_separation_score(alpha_by_regime):
    vs=list(alpha_by_regime.values())
    dists=[np.abs(np.array(vs[i])-np.array(vs[j])).sum() for i in range(len(vs)) for j in range(i+1,len(vs))]
    return float(np.mean(dists)) if dists else 0.0

def attention_entropy(alpha): return float(scipy_entropy(alpha+1e-8))
def reconstruction_mse(pred,true): return float(torch.nn.functional.mse_loss(pred,true).item())
