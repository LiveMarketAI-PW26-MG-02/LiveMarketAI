"""Tests for evaluation metrics."""
import numpy as np, torch
from metrics import attention_consistency, regime_separation_score, reconstruction_mse

def test_consistency_ratio():
    alpha=np.tile([0.5,0.3,0.1,0.05,0.03,0.01,0.005,0.005],(100,1))+np.random.randn(100,8)*0.01
    regimes=np.zeros(100,dtype=int)
    assert attention_consistency(alpha,regimes)["consistency_ratio"]<0.5

def test_mse_zero_for_identical():
    t=torch.randn(8,64); assert reconstruction_mse(t,t)<1e-6
