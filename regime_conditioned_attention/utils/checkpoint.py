"""Model checkpointing."""
import torch
from pathlib import Path

class Checkpointer:
    def __init__(self, save_dir="checkpoints", top_k=3):
        self.save_dir=Path(save_dir); self.save_dir.mkdir(parents=True,exist_ok=True)
        self.top_k=top_k; self.best_score=float("inf"); self.saved=[]
    def maybe_save(self, model, optimizer, epoch, val_loss):
        path=self.save_dir/f"epoch_{epoch:04d}_loss_{val_loss:.4f}.pt"
        if val_loss<self.best_score:
            self.best_score=val_loss
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "opt_state":optimizer.state_dict(),"val_loss":val_loss},path)
            self.saved.append((val_loss,path)); self.saved.sort(key=lambda x:x[0])
            if len(self.saved)>self.top_k: _,old=self.saved.pop(); old.unlink(missing_ok=True)
            return True
        return False
