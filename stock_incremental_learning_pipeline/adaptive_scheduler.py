"""
adaptive_scheduler.py — Requirement 6
Stock Adaptive Learning Rate Scheduler
---------------------------------------
Dynamically adjusts per-parameter-group learning rates based on:
  • recent return volatility (VIX-proxy from streaming data)
  • loss trajectory (plateau / spike detection)
  • regime change signals from the drift detector

The LR multiplier follows:
    scale = clip( K * sigma_recent / sigma_baseline,  LR_SCALE_MIN, LR_SCALE_MAX )
    lr_t  = base_lr * scale
"""

import numpy as np
import torch
import torch.optim as optim
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import config


class MarketVolatilityTracker:
    """
    Maintains a rolling window of return magnitudes and exposes the current
    annualised (proxy) volatility, relative to the long-run baseline.
    """

    def __init__(self, window: int = 30, baseline_window: int = 200):
        self._window   = window
        self._baseline = baseline_window
        self._returns: Deque[float] = deque(maxlen=baseline_window)
        self._vol_history: List[float] = []

    def update(self, close_prices: np.ndarray) -> float:
        """Feed a batch of close prices; returns current volatility ratio."""
        rets = np.diff(close_prices) / (np.abs(close_prices[:-1]) + 1e-9)
        self._returns.extend(rets.tolist())

        if len(self._returns) < self._window:
            return 1.0

        recent   = list(self._returns)[-self._window:]
        baseline = list(self._returns)

        sigma_r  = float(np.std(recent))
        sigma_b  = float(np.std(baseline)) + 1e-9
        ratio    = sigma_r / sigma_b
        self._vol_history.append(ratio)
        return ratio

    @property
    def volatility_history(self) -> List[float]:
        return self._vol_history


class StockAdaptiveLRScheduler:
    """
    Req 6 — Adaptive Learning Rate Scheduler.

    Wraps a PyTorch optimiser and scales its learning rate(s) each update step
    according to market volatility:

        high volatility → smaller LR (avoid overshooting)
        low  volatility → larger  LR (exploit stable regime)

    Also detects loss spikes and temporarily halves the LR for recovery.

    Usage
    -----
        scheduler = StockAdaptiveLRScheduler(optimizer, base_lr=1e-3)
        for batch in stream:
            vol_ratio = vol_tracker.update(batch["close"].values)
            scheduler.step(vol_ratio, current_loss)
    """

    def __init__(self,
                 optimizer:     optim.Optimizer,
                 base_lr:       float = config.INCR_LR,
                 lr_scale_min:  float = config.LR_SCALE_MIN,
                 lr_scale_max:  float = config.LR_SCALE_MAX,
                 volatility_k:  float = config.LR_VOLATILITY_K,
                 spike_factor:  float = 2.5,
                 warmup_steps:  int   = 5):
        self.optimizer    = optimizer
        self.base_lr      = base_lr
        self.lr_scale_min = lr_scale_min
        self.lr_scale_max = lr_scale_max
        self.vol_k        = volatility_k
        self.spike_factor = spike_factor
        self.warmup_steps = warmup_steps

        self._step         = 0
        self._loss_window: Deque[float] = deque(maxlen=10)
        self._lr_history:  List[Dict]   = []
        self._recovery_steps = 0

        # Backup original base LR per param-group
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"[AdaptiveLR] base_lr={base_lr}, scale=[{lr_scale_min}, {lr_scale_max}]")

    # ─── Public API ──────────────────────────────────────────────────────────

    def step(self,
             vol_ratio:    float,
             current_loss: Optional[float] = None) -> List[float]:
        """
        Compute new LR(s) and apply to all param-groups.

        Parameters
        ----------
        vol_ratio    : current σ / baseline σ from MarketVolatilityTracker
        current_loss : optional scalar loss value for spike detection

        Returns
        -------
        List of new LRs (one per param-group).
        """
        self._step += 1

        # 1) Warmup: linearly scale from 0 to base_lr
        if self._step <= self.warmup_steps:
            warmup_frac = self._step / self.warmup_steps
        else:
            warmup_frac = 1.0

        # 2) Volatility-based scale (inverse: high vol → low LR)
        #    scale = 1 / (1 + K * (vol_ratio - 1))  clipped to [min, max]
        vol_shift = vol_ratio - 1.0          # 0 at baseline
        raw_scale = 1.0 / (1.0 + self.vol_k * max(vol_shift, 0))
        scale     = float(np.clip(raw_scale, self.lr_scale_min, self.lr_scale_max))

        # 3) Loss-spike recovery
        spike_halve = 1.0
        if current_loss is not None:
            self._loss_window.append(current_loss)
            if len(self._loss_window) >= 5:
                mean_l = float(np.mean(list(self._loss_window)[:-1]))
                if current_loss > self.spike_factor * mean_l:
                    spike_halve = 0.5
                    self._recovery_steps = 3
                    print(f"  [AdaptiveLR] Loss spike detected (loss={current_loss:.5f} "
                          f"vs mean={mean_l:.5f}) → LR halved for 3 steps")

        if self._recovery_steps > 0:
            spike_halve = 0.5
            self._recovery_steps -= 1

        # 4) Apply to each param-group
        new_lrs = []
        for i, pg in enumerate(self.optimizer.param_groups):
            blr     = self._base_lrs[i]
            new_lr  = blr * warmup_frac * scale * spike_halve
            pg["lr"] = float(np.clip(new_lr, blr * self.lr_scale_min,
                                             blr * self.lr_scale_max))
            new_lrs.append(pg["lr"])

        self._lr_history.append({
            "step":        self._step,
            "vol_ratio":   vol_ratio,
            "scale":       scale,
            "spike_halve": spike_halve,
            "lrs":         new_lrs.copy(),
        })
        return new_lrs

    def get_current_lrs(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def reset_to_base(self) -> None:
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self._base_lrs[i]
        self._recovery_steps = 0

    @property
    def lr_history(self) -> List[Dict]:
        return self._lr_history

    def summary(self) -> Dict:
        if not self._lr_history:
            return {}
        lrs = [h["lrs"][0] for h in self._lr_history]
        vols = [h["vol_ratio"] for h in self._lr_history]
        return {
            "steps":       self._step,
            "min_lr":      float(min(lrs)),
            "max_lr":      float(max(lrs)),
            "mean_lr":     float(np.mean(lrs)),
            "mean_vol":    float(np.mean(vols)),
            "lr_history":  lrs,
            "vol_history": vols,
        }


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from incremental_model import DualStockModel

    model     = DualStockModel()
    opt       = optim.Adam(model.parameters(), lr=config.INCR_LR)
    sched     = StockAdaptiveLRScheduler(opt)
    vol_track = MarketVolatilityTracker()

    rng = np.random.default_rng(42)
    for i in range(20):
        # Simulate close prices; inject volatility spike at step 10
        sigma = 3.0 if i == 10 else 0.5
        prices = 100 + np.cumsum(rng.normal(0, sigma, 50))
        ratio  = vol_track.update(prices)
        loss   = abs(rng.normal(0.01, 0.005)) * (5 if i == 10 else 1)
        new_lrs = sched.step(vol_ratio=ratio, current_loss=loss)
        print(f"Step {i:02d} | vol_ratio={ratio:.3f} | loss={loss:.5f} | lr={new_lrs[0]:.6f}")

    print(sched.summary())
