"""
gradient_replay.py — Requirement 5
Stock Gradient Replay Mechanism
---------------------------------
Stores gradient snapshots from previous training cycles and blends them with
current gradients during incremental updates to prevent catastrophic forgetting
and stabilise convergence.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import config

torch.manual_seed(config.SEED)


# ─── Gradient Snapshot ────────────────────────────────────────────────────────

class GradientSnapshot:
    """
    A frozen copy of a model's gradient tensors at a particular training step.
    Snapshots are stored on CPU to save GPU memory.
    """

    def __init__(self, model: nn.Module, step: int, loss_val: float):
        self.step     = step
        self.loss_val = loss_val
        self.grads: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.grads[name] = param.grad.detach().cpu().clone()

    def __repr__(self) -> str:
        n = len(self.grads)
        return f"GradientSnapshot(step={self.step}, loss={self.loss_val:.5f}, params={n})"


# ─── Gradient Replay Buffer ───────────────────────────────────────────────────

class StockGradientReplayBuffer:
    """
    Req 5 — Gradient Replay Buffer.

    Maintains a FIFO buffer of GradientSnapshot objects.
    During an incremental step, `replay()` blends stored historical gradients
    into the current model gradients before the optimiser step:

        effective_grad = (1 - replay_weight) * current_grad
                       +      replay_weight  * mean(historic_grads)

    This stabilises incremental updates by "remembering" gradient directions
    from previous training cycles.
    """

    def __init__(self,
                 buffer_size:   int   = config.REPLAY_BUFFER_SZ,
                 replay_weight: float = config.REPLAY_WEIGHT):
        self.buffer_size   = buffer_size
        self.replay_weight = replay_weight
        self._buffer: Deque[GradientSnapshot] = deque(maxlen=buffer_size)
        self._step = 0
        print(f"[GradientReplay] buffer_size={buffer_size}, "
              f"replay_weight={replay_weight:.2f}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def capture(self, model: nn.Module, loss_val: float) -> GradientSnapshot:
        """
        Save a snapshot of current gradients (call after loss.backward()).
        Automatically evicts the oldest snapshot when the buffer is full.
        """
        snap = GradientSnapshot(model, step=self._step, loss_val=loss_val)
        self._buffer.append(snap)
        self._step += 1
        return snap

    def replay(self, model: nn.Module) -> Dict[str, float]:
        """
        Blend stored historical gradients into the model's current gradients.
        Must be called *after* loss.backward() and *before* optimizer.step().

        Returns per-layer blend statistics.
        """
        if len(self._buffer) == 0:
            return {}

        # Compute mean gradient across all snapshots for each parameter
        mean_grads: Dict[str, torch.Tensor] = {}
        for snap in self._buffer:
            for name, g in snap.grads.items():
                if name not in mean_grads:
                    mean_grads[name] = g.clone()
                else:
                    mean_grads[name] += g

        n = len(self._buffer)
        for name in mean_grads:
            mean_grads[name] /= n

        # Blend into current gradients
        stats: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None and name in mean_grads:
                hist_g = mean_grads[name].to(param.grad.device)
                blended = ((1 - self.replay_weight) * param.grad
                           + self.replay_weight   * hist_g)
                cosine  = self._cosine_sim(param.grad, hist_g)
                param.grad.copy_(blended)
                stats[name] = float(cosine)

        return stats

    def selective_replay(self,
                         model: nn.Module,
                         layer_keywords: List[str]) -> Dict[str, float]:
        """
        Only replay gradients for layers whose names contain any keyword.
        Useful for applying replay only to the decision layers.
        """
        if len(self._buffer) == 0:
            return {}

        mean_grads = self._compute_mean_grads()
        stats: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not any(kw in name for kw in layer_keywords):
                continue
            if name not in mean_grads:
                continue

            hist_g = mean_grads[name].to(param.grad.device)
            blended = ((1 - self.replay_weight) * param.grad
                       + self.replay_weight   * hist_g)
            stats[name] = float(self._cosine_sim(param.grad, hist_g))
            param.grad.copy_(blended)

        return stats

    @property
    def size(self) -> int:
        return len(self._buffer)

    def summary(self) -> str:
        losses = [s.loss_val for s in self._buffer]
        return (f"GradientReplayBuffer | stored={self.size}/{self.buffer_size} | "
                f"replay_weight={self.replay_weight:.2f} | "
                f"mean_hist_loss={np.mean(losses):.5f}" if losses
                else "GradientReplayBuffer | empty")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _compute_mean_grads(self) -> Dict[str, torch.Tensor]:
        mean_grads: Dict[str, torch.Tensor] = {}
        for snap in self._buffer:
            for name, g in snap.grads.items():
                if name not in mean_grads:
                    mean_grads[name] = g.clone()
                else:
                    mean_grads[name] += g
        n = len(self._buffer)
        for name in mean_grads:
            mean_grads[name] /= n
        return mean_grads

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        af = a.flatten().float()
        bf = b.flatten().float()
        dot = (af * bf).sum()
        denom = (af.norm() * bf.norm()) + 1e-9
        return float((dot / denom).clamp(-1, 1))


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.optim as optim
    from incremental_model import DualStockModel

    model  = DualStockModel()
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    replay = StockGradientReplayBuffer(buffer_size=5, replay_weight=0.3)

    for step in range(8):
        x    = torch.randn(16, config.SEQUENCE_LEN, config.INPUT_SIZE)
        pred, _ = model(x)
        y    = torch.randn(16, 1)
        loss = nn.MSELoss()(pred, y)
        opt.zero_grad()
        loss.backward()

        snap = replay.capture(model, loss.item())
        print(f"Step {step}: {snap}")
        if step > 0:
            stats = replay.replay(model)
            print(f"  Replay stats (cosine sim, first 2): "
                  f"{dict(list(stats.items())[:2])}")
        opt.step()

    print(replay.summary())
