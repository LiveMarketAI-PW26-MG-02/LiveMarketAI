"""
incremental_model.py — Requirements 1, 4, 7, 8
------------------------------------------------
• Req 1  — Selective Weight Update: gradient-magnitude-based param selection
• Req 4  — Dual-Model Architecture: StableBaseModel + FastAdaptModel → Ensemble
• Req 7  — Layer-Freezing Strategy: lower LSTM layers frozen; head updated only
• Req 8  — Confidence-Guided Updates: MC-Dropout uncertainty → update trigger
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import config

torch.manual_seed(config.SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Base LSTM Stock Model
# ═══════════════════════════════════════════════════════════════════════════════

class StockLSTM(nn.Module):
    """
    Multi-layer LSTM for stock return prediction.
    Layers 0 … FROZEN_LAYERS-1 can be frozen (req 7).
    """

    def __init__(self,
                 input_size:  int = config.INPUT_SIZE,
                 hidden_size: int = config.HIDDEN_SIZE,
                 num_layers:  int = config.NUM_LAYERS,
                 dropout:     float = config.DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Build individual LSTM layers so we can freeze selectively
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size  if i == 0 else hidden_size,
                hidden_size,
                num_layers  = 1,
                batch_first = True,
                dropout     = 0.0
            )
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, config.OUTPUT_SIZE)
        )

    def forward(self, x: torch.Tensor,
                use_dropout: bool = False) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        Returns: (batch, 1) predictions
        """
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
            if use_dropout:
                out = self.dropout(out)
        # Take last time-step
        out = out[:, -1, :]
        return self.head(out)

    def freeze_base_layers(self, n_frozen: int = config.FROZEN_LAYERS) -> None:
        """Req 7: Freeze the bottom n_frozen LSTM layers."""
        for i, layer in enumerate(self.lstm_layers):
            requires = (i >= n_frozen)
            for p in layer.parameters():
                p.requires_grad = requires
        print(f"[StockLSTM] Froze layers 0–{n_frozen-1}; "
              f"layers {n_frozen}+ + head remain trainable.")

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_params())
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Selective Weight Update Mechanism  (Req 1)
# ═══════════════════════════════════════════════════════════════════════════════

class SelectiveWeightUpdater:
    """
    After a backward pass, identifies the top-K % parameters by gradient
    magnitude and zeros out gradients for all others → only most relevant
    weights are updated this step.
    """

    def __init__(self, model: nn.Module, top_k: float = config.TOP_K_PARAMS):
        self.model = model
        self.top_k = top_k

    def apply(self) -> Dict[str, int]:
        """
        Must be called *after* loss.backward() and *before* optimizer.step().
        Returns stats on how many params were masked.
        """
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.abs().flatten())

        if not grads:
            return {"kept": 0, "zeroed": 0}

        all_grads = torch.cat(grads)
        k         = max(1, int(len(all_grads) * self.top_k))
        threshold = torch.kthvalue(all_grads, len(all_grads) - k + 1).values.item()

        zeroed = 0
        kept   = 0
        for p in self.model.parameters():
            if p.grad is not None:
                mask        = p.grad.abs() >= threshold
                p.grad     *= mask.float()
                zeroed     += (~mask).sum().item()
                kept        += mask.sum().item()

        return {"kept": kept, "zeroed": zeroed}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fast-Adapting Incremental Head  (Req 4)
# ═══════════════════════════════════════════════════════════════════════════════

class FastAdaptHead(nn.Module):
    """
    A lightweight MLP that sits on top of frozen base-model features and
    adapts rapidly to recent market conditions.
    """

    def __init__(self, input_dim: int = config.HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, config.OUTPUT_SIZE)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dual-Model Ensemble  (Req 4)
# ═══════════════════════════════════════════════════════════════════════════════

class DualStockModel(nn.Module):
    """
    Combines a stable base model (StockLSTM) with a rapidly adapting head
    (FastAdaptHead).  Final prediction = blend of both outputs.

        pred = stable_blend * base_pred + adapt_blend * adapt_pred
    """

    def __init__(self,
                 stable_blend: float = config.STABLE_BLEND,
                 adapt_blend:  float = config.ADAPT_BLEND):
        super().__init__()
        self.stable_blend = stable_blend
        self.adapt_blend  = adapt_blend

        self.base_model   = StockLSTM()
        self.adapt_head   = FastAdaptHead()

        # Internal hook to expose base-model features for adapt head
        self._features: Optional[torch.Tensor] = None
        self.base_model.head.register_forward_hook(self._feature_hook)

    def _feature_hook(self, module, inp, out):
        """Capture penultimate features from base model head."""
        self._features = inp[0].detach()

    def forward(self, x: torch.Tensor,
                use_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (ensemble_pred, base_pred) — both (batch, 1).
        """
        base_pred  = self.base_model(x, use_dropout=use_dropout)
        adapt_pred = self.adapt_head(self._features) if self._features is not None \
                     else torch.zeros_like(base_pred)
        ensemble   = self.stable_blend * base_pred + self.adapt_blend * adapt_pred
        return ensemble, base_pred

    def freeze_base(self) -> None:
        """Freeze the entire base model; only adapt head trains."""
        for p in self.base_model.parameters():
            p.requires_grad = False

    def unfreeze_head_only(self) -> None:
        """Req 7: freeze base LSTM layers, unfreeze head + adapt head."""
        self.base_model.freeze_base_layers()
        for p in self.adapt_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Confidence-Guided Update  (Req 8)
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceGuardedPredictor:
    """
    Wraps a DualStockModel and uses Monte-Carlo Dropout to estimate prediction
    uncertainty.  An incremental update is recommended only when uncertainty
    exceeds a threshold (low-confidence region).
    """

    def __init__(self,
                 model: DualStockModel,
                 threshold: float = config.CONFIDENCE_THRESHOLD,
                 n_samples: int   = config.MC_DROPOUT_SAMPLES):
        self.model     = model
        self.threshold = threshold
        self.n_samples = n_samples

    @torch.no_grad()
    def predict_with_uncertainty(
            self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Runs `n_samples` stochastic forward passes with dropout active.
        Returns:
            mean_pred   (B,)  — point estimate
            uncertainty (B,)  — std across MC samples
            should_update     — True if mean uncertainty > threshold
        """
        self.model.train()           # enable dropout
        preds = []
        for _ in range(self.n_samples):
            ens, _ = self.model(x, use_dropout=True)
            preds.append(ens.squeeze(-1).cpu().numpy())
        self.model.eval()

        preds_arr   = np.stack(preds, axis=0)   # (n_samples, B)
        mean_pred   = preds_arr.mean(axis=0)
        uncertainty = preds_arr.std(axis=0)

        should_update = bool(uncertainty.mean() > self.threshold)
        return mean_pred, uncertainty, should_update


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DualStockModel()
    model.unfreeze_head_only()
    pc = model.base_model.param_count()
    print(f"Param counts: {pc}")

    x      = torch.randn(8, config.SEQUENCE_LEN, config.INPUT_SIZE)
    ens, base = model(x)
    print(f"Ensemble pred: {ens.shape}, Base pred: {base.shape}")

    cgp     = ConfidenceGuardedPredictor(model)
    mean, unc, upd = cgp.predict_with_uncertainty(x)
    print(f"MC mean: {mean[:3]}, uncertainty: {unc[:3]}, update_triggered: {upd}")

    swu = SelectiveWeightUpdater(model)
    loss = ens.sum()
    loss.backward()
    stats = swu.apply()
    print(f"SelectiveWeightUpdater: {stats}")
