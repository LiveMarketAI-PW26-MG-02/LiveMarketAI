"""Tests for AttentionTrainer."""
import torch
from attention_trainer import AttentionTrainer

def test_loss_decreases():
    trainer=AttentionTrainer(d_model=64,n_regimes=4,n_sources=6)
    X=torch.randn(32,6,64); R=torch.randint(0,4,(32,)); T=torch.randn(32,64)
    losses=[trainer.train_step(X,R,T)[0] for _ in range(30)]
    assert losses[-1]<losses[0]
