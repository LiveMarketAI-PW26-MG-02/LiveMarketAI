"""Tests for RegimeAttentionDataset."""
from dataset_builder import RegimeAttentionDataset

def test_dataset_lengths():
    ds=RegimeAttentionDataset(n=100,d=64,S=6); assert len(ds)==100

def test_item_shapes():
    ds=RegimeAttentionDataset(n=10,d=64,S=6); X,R,T=ds[0]
    assert X.shape==(6,64); assert T.shape==(64,)
