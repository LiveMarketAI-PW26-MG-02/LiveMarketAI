"""Unit tests for regime-conditioned attention."""
import pytest, torch
from attention_equation import RegimeConditionedAttention

@pytest.fixture
def model(): return RegimeConditionedAttention(d_model=64,n_regimes=4,n_sources=6)

def test_output_shape(model):
    X=torch.randn(8,6,64); R=torch.randint(0,4,(8,))
    Mt,alpha=model(X,R)
    assert Mt.shape==(8,64); assert alpha.shape==(8,6)

def test_alpha_sums_to_one(model):
    X=torch.randn(16,6,64); R=torch.randint(0,4,(16,))
    _,alpha=model(X,R)
    assert torch.allclose(alpha.sum(-1),torch.ones(16),atol=1e-5)

def test_gradient_flow(model):
    X=torch.randn(4,6,64); R=torch.randint(0,4,(4,))
    Mt,_=model(X,R); Mt.sum().backward()
    for p in model.parameters(): assert p.grad is not None
