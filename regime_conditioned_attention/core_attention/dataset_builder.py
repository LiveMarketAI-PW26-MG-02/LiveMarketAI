"""Synthetic regime-conditioned multi-modal dataset."""
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader

class RegimeAttentionDataset(Dataset):
    REGIME_WEIGHTS = {
        0:[0.5,0.3,0.1,0.05,0.02,0.01,0.01,0.01],
        1:[0.1,0.1,0.4,0.2,0.1,0.05,0.03,0.02],
        2:[0.2,0.2,0.2,0.2,0.1,0.05,0.03,0.02],
        3:[0.05,0.05,0.05,0.05,0.6,0.1,0.05,0.05]}
    def __init__(self, n=4000, d=256, S=8, K=4):
        torch.manual_seed(42)
        self.X = torch.randn(n,S,d); self.R = torch.randint(0,K,(n,))
        self.T = torch.stack([(torch.tensor(self.REGIME_WEIGHTS[self.R[i].item()]).unsqueeze(-1)*self.X[i]).sum(0) for i in range(n)])
    def __len__(self): return len(self.R)
    def __getitem__(self,i): return self.X[i], self.R[i], self.T[i]

def get_loaders(bs=64):
    ds=RegimeAttentionDataset(); n=len(ds)
    tr,va=int(.8*n),int(.1*n)
    sets=torch.utils.data.random_split(ds,[tr,va,n-tr-va])
    return tuple(DataLoader(s,bs,shuffle=(i==0)) for i,s in enumerate(sets))
