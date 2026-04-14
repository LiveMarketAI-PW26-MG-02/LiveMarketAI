"""K-fold CV for regime attention."""
import torch, numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from dataset_builder import RegimeAttentionDataset
from attention_trainer import AttentionTrainer

def kfold_cv(k=5, epochs=10):
    dataset=RegimeAttentionDataset(n_samples=2000 if False else 2000); kf=KFold(n_splits=k,shuffle=True,random_state=42)
    fold_losses=[]
    for fold,(tr_idx,va_idx) in enumerate(kf.split(range(len(dataset)))):
        trainer=AttentionTrainer()
        for _ in range(epochs):
            for X,R,T in DataLoader(Subset(dataset,tr_idx),64,shuffle=True):
                trainer.train_step(X,R,T)
        losses=[trainer.evaluate(X,R,T)[0] for X,R,T in DataLoader(Subset(dataset,va_idx),64)]
        fold_losses.append(np.mean(losses)); print(f"Fold {fold+1}: {fold_losses[-1]:.4f}")
    print(f"CV={np.mean(fold_losses):.4f}±{np.std(fold_losses):.4f}")
    return fold_losses
