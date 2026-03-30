"""
pipeline.py — Main Stock Incremental Learning Pipeline
=======================================================
Orchestrates all components:
  • Full base-model training on historical data
  • Streaming incremental updates (drift-triggered, confidence-guided)
  • Comparative evaluation: incremental vs full-retraining
  • Result logging and CSV export

Run: python pipeline.py
"""

import os
import time
import logging
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional

import config
from data_generator      import (generate_all_historical, streaming_batches,
                                  build_sequences)
from memory_buffer        import StockMemoryBuffer
from drift_detector       import StockDriftDetector
from incremental_model    import (DualStockModel, SelectiveWeightUpdater,
                                   ConfidenceGuardedPredictor)
from gradient_replay      import StockGradientReplayBuffer
from adaptive_scheduler   import StockAdaptiveLRScheduler, MarketVolatilityTracker

# ─── Setup ───────────────────────────────────────────────────────────────────

os.makedirs(config.MODEL_DIR,   exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
DEVICE = torch.device(config.DEVICE)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def arrays_to_loader(X: np.ndarray, y: np.ndarray,
                     batch_size: int = config.BATCH_SIZE,
                     shuffle: bool = True) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    return DataLoader(TensorDataset(Xt, yt),
                      batch_size=batch_size, shuffle=shuffle)


def evaluate(model: DualStockModel,
             loader: DataLoader,
             criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    losses, maes, mapes = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            pred, _ = model(Xb)
            loss     = criterion(pred, yb)
            mae      = (pred - yb).abs().mean()
            mape     = ((pred - yb).abs() / (yb.abs() + 1e-8)).mean() * 100
            losses.append(loss.item())
            maes.append(mae.item())
            mapes.append(mape.item())
    return {
        "loss": float(np.mean(losses)),
        "mae":  float(np.mean(maes)),
        "mape": float(np.mean(mapes)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Full Base-Model Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_base_model(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val:   np.ndarray,
                     y_val:   np.ndarray) -> DualStockModel:
    log.info("=" * 60)
    log.info("PHASE 1 — Full Base-Model Training")
    log.info("=" * 60)

    model     = DualStockModel().to(DEVICE)
    model.unfreeze_all()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.BASE_LR,
                           weight_decay=config.WEIGHT_DECAY)
    sched     = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.FULL_EPOCHS)

    train_loader = arrays_to_loader(X_train, y_train)
    val_loader   = arrays_to_loader(X_val,   y_val, shuffle=False)

    best_val  = float("inf")
    best_ckpt = None

    for epoch in range(1, config.FULL_EPOCHS + 1):
        model.train()
        ep_losses = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss     = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
        sched.step()

        val_m = evaluate(model, val_loader, criterion)
        if val_m["loss"] < best_val:
            best_val  = val_m["loss"]
            best_ckpt = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{config.FULL_EPOCHS} | "
                     f"train_loss={np.mean(ep_losses):.5f} | "
                     f"val_loss={val_m['loss']:.5f} | "
                     f"val_mae={val_m['mae']:.5f}")

    model.load_state_dict(best_ckpt)
    ckpt_path = os.path.join(config.MODEL_DIR, "base_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"  Base model saved → {ckpt_path} (best val_loss={best_val:.5f})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Incremental Update Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_incremental_pipeline(
        base_model: DualStockModel,
        buffer:     StockMemoryBuffer,
        X_val:      np.ndarray,
        y_val:      np.ndarray
) -> Tuple[List[Dict], List[Dict]]:
    """
    Streams STREAM_BATCHES batches.  Uses:
      • DriftDetector to decide WHEN to update
      • ConfidenceGuardedPredictor to decide WHETHER to update per batch
      • SelectiveWeightUpdater for parameter masking
      • GradientReplayBuffer for gradient stabilisation
      • AdaptiveLRScheduler for dynamic LR
      • MemoryBuffer for rehearsal
    Returns incremental_metrics and retraining_metrics (for comparison).
    """
    log.info("=" * 60)
    log.info("PHASE 2 — Incremental Update Pipeline")
    log.info("=" * 60)

    # ── Clone model for full-retraining baseline (Req 9) ────────────────────
    retrain_model = copy.deepcopy(base_model).to(DEVICE)

    # ── Prepare incremental model (Req 7: freeze base, train head) ──────────
    incr_model = copy.deepcopy(base_model).to(DEVICE)
    incr_model.unfreeze_head_only()
    log.info(f"  Incremental model params: {incr_model.base_model.param_count()}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, incr_model.parameters()),
        lr=config.INCR_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # ── Component initialisation ─────────────────────────────────────────────
    drift_det   = StockDriftDetector(method="ks_test")
    cgp         = ConfidenceGuardedPredictor(incr_model)
    swu         = SelectiveWeightUpdater(incr_model, top_k=config.TOP_K_PARAMS)
    replay_buf  = StockGradientReplayBuffer()
    vol_tracker = MarketVolatilityTracker()
    lr_sched    = StockAdaptiveLRScheduler(optimizer)
    val_loader  = arrays_to_loader(X_val, y_val, shuffle=False)

    incr_metrics:    List[Dict] = []
    retrain_metrics: List[Dict] = []

    X_cumul_hist: List[np.ndarray] = []
    y_cumul_hist: List[np.ndarray] = []

    for batch_idx, batch_df in streaming_batches():
        log.info(f"\n--- Batch {batch_idx + 1}/{config.STREAM_BATCHES} "
                 f"(size={len(batch_df)}) ---")

        # ── Build sequences for this batch ───────────────────────────────────
        try:
            X_b, y_b = build_sequences(batch_df)
        except ValueError:
            log.warning("  Skipping batch — insufficient rows")
            continue
        if len(X_b) < 2:
            log.warning("  Skipping batch — < 2 sequences")
            continue

        # ── Drift detection ──────────────────────────────────────────────────
        close_vals = batch_df["close"].values
        drift_trig, drift_score = drift_det.update(close_vals)
        log.info(f"  Drift score={drift_score:.4f}  triggered={drift_trig}")

        # ── Adaptive LR ──────────────────────────────────────────────────────
        vol_ratio = vol_tracker.update(close_vals)

        # ── Confidence check ─────────────────────────────────────────────────
        X_t  = torch.tensor(X_b[:16], dtype=torch.float32).to(DEVICE)
        _, unc, conf_trig = cgp.predict_with_uncertainty(X_t)
        log.info(f"  Mean uncertainty={unc.mean():.5f}  conf_trigger={conf_trig}")

        # ── Decide whether to run incremental update ─────────────────────────
        should_update = drift_trig or conf_trig
        update_ran    = False
        incr_loss_val = None

        if should_update:
            # Add new data to buffer
            buffer.add(X_b, y_b)

            # Mix buffer rehearsal samples with current batch
            if buffer.size >= config.BATCH_SIZE:
                X_mem, y_mem = buffer.sample(min(128, buffer.size))
                X_mix = np.concatenate([X_b, X_mem], axis=0)
                y_mix = np.concatenate([y_b, y_mem], axis=0)
            else:
                X_mix, y_mix = X_b, y_b

            loader_mix = arrays_to_loader(X_mix, y_mix,
                                          batch_size=config.BATCH_SIZE)
            incr_model.train()

            for ep in range(config.INCR_EPOCHS):
                ep_losses = []
                for Xb_, yb_ in loader_mix:
                    optimizer.zero_grad()
                    pred, _ = incr_model(Xb_)
                    loss     = criterion(pred, yb_)
                    loss.backward()

                    # Req 5: gradient replay
                    replay_buf.replay(incr_model)

                    # Req 1: selective weight update
                    swu.apply()

                    nn.utils.clip_grad_norm_(
                        [p for p in incr_model.parameters() if p.requires_grad], 1.0)

                    # Capture gradient snapshot
                    replay_buf.capture(incr_model, loss.item())

                    # Req 6: adaptive LR step
                    lr_sched.step(vol_ratio=vol_ratio, current_loss=loss.item())

                    optimizer.step()
                    ep_losses.append(loss.item())

            incr_loss_val = float(np.mean(ep_losses))
            update_ran    = True
            log.info(f"  Incremental update done | train_loss={incr_loss_val:.5f}")
        else:
            log.info("  No update triggered this batch")

        # ── Evaluate incremental model ────────────────────────────────────────
        val_m = evaluate(incr_model, val_loader, criterion)
        log.info(f"  Incremental val: loss={val_m['loss']:.5f}  "
                 f"mae={val_m['mae']:.5f}  mape={val_m['mape']:.2f}%")

        incr_metrics.append({
            "batch":        batch_idx + 1,
            "drift_score":  round(drift_score, 4),
            "drift_trig":   drift_trig,
            "conf_trig":    conf_trig,
            "update_ran":   update_ran,
            "vol_ratio":    round(vol_ratio, 4),
            "current_lr":   round(lr_sched.get_current_lrs()[0], 8),
            "val_loss":     round(val_m["loss"], 6),
            "val_mae":      round(val_m["mae"],  6),
            "val_mape":     round(val_m["mape"], 4),
            "replay_size":  replay_buf.size,
            "buffer_size":  buffer.size,
        })

        # ── Req 9: Retrain baseline (accumulate ALL data, full retrain) ──────
        X_cumul_hist.append(X_b)
        y_cumul_hist.append(y_b)

        if (batch_idx + 1) % 5 == 0:   # retrain every 5 batches for speed
            X_all = np.concatenate(X_cumul_hist, axis=0)
            y_all = np.concatenate(y_cumul_hist, axis=0)
            t0    = time.time()
            retrain_model = _full_retrain(retrain_model, X_all, y_all,
                                          epochs=config.FULL_EPOCHS // 3)
            retrain_time  = time.time() - t0
            ret_val = evaluate(retrain_model, val_loader, criterion)
            log.info(f"  [RETRAIN] val_loss={ret_val['loss']:.5f}  "
                     f"time={retrain_time:.1f}s")
            retrain_metrics.append({
                "batch":       batch_idx + 1,
                "val_loss":    round(ret_val["loss"], 6),
                "val_mae":     round(ret_val["mae"],  6),
                "val_mape":    round(ret_val["mape"], 4),
                "retrain_sec": round(retrain_time, 2),
                "n_samples":   len(X_all),
            })

    return incr_metrics, retrain_metrics


def _full_retrain(model: DualStockModel,
                  X: np.ndarray, y: np.ndarray,
                  epochs: int = 10) -> DualStockModel:
    model.unfreeze_all()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.BASE_LR)
    loader    = arrays_to_loader(X, y)

    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss     = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Comparative Evaluation & Report  (Req 9)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comparison_report(incr_metrics:    List[Dict],
                                retrain_metrics: List[Dict]) -> None:
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — Comparative Evaluation (Req 9)")
    log.info("=" * 60)

    import csv

    # Save incremental metrics
    if incr_metrics:
        path = os.path.join(config.RESULTS_DIR, "incremental_metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=incr_metrics[0].keys())
            w.writeheader(); w.writerows(incr_metrics)
        log.info(f"  Incremental metrics → {path}")

    # Save retrain metrics
    if retrain_metrics:
        path = os.path.join(config.RESULTS_DIR, "retrain_metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=retrain_metrics[0].keys())
            w.writeheader(); w.writerows(retrain_metrics)
        log.info(f"  Retraining metrics  → {path}")

    # Summary comparison
    if incr_metrics and retrain_metrics:
        incr_vals   = [m["val_loss"] for m in incr_metrics  if m["update_ran"]]
        retrain_vals= [m["val_loss"] for m in retrain_metrics]
        retrain_secs= [m["retrain_sec"] for m in retrain_metrics]
        updates     = sum(1 for m in incr_metrics if m["update_ran"])
        total       = len(incr_metrics)

        summary = {
            "incremental": {
                "mean_val_loss":      round(float(np.mean(incr_vals)), 6) if incr_vals else None,
                "updates_triggered":  updates,
                "total_batches":      total,
                "update_rate_%":      round(100 * updates / max(total, 1), 1),
                "mean_lr":            round(float(np.mean([m["current_lr"] for m in incr_metrics])), 8),
            },
            "full_retraining": {
                "mean_val_loss":      round(float(np.mean(retrain_vals)), 6) if retrain_vals else None,
                "mean_retrain_sec":   round(float(np.mean(retrain_secs)), 2) if retrain_secs else None,
            },
            "efficiency_gain_%": None,
        }
        if retrain_secs:
            incr_sec_est = updates * config.INCR_EPOCHS * 0.5  # rough estimate
            total_retrain = sum(retrain_secs)
            gain = 100 * (1 - incr_sec_est / (total_retrain + 1e-9))
            summary["efficiency_gain_%"] = round(gain, 1)

        rpt_path = os.path.join(config.RESULTS_DIR, "comparison_summary.json")
        with open(rpt_path, "w") as f:
            json.dump(summary, f, indent=2)

        log.info("\n  ─── Summary ─────────────────────────────────────")
        log.info(f"  Incremental  mean_val_loss : {summary['incremental']['mean_val_loss']}")
        log.info(f"  Full-Retrain mean_val_loss : {summary['full_retraining']['mean_val_loss']}")
        log.info(f"  Retrain avg time/cycle     : {summary['full_retraining']['mean_retrain_sec']}s")
        log.info(f"  Efficiency gain (est.)     : {summary['efficiency_gain_%']}%")
        log.info(f"  Update rate                : {summary['incremental']['update_rate_%']}%")
        log.info(f"  Comparison report          → {rpt_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("Stock Incremental Learning Pipeline — START")
    log.info(f"Tickers: {config.TICKERS}")
    log.info(f"Device : {DEVICE}")

    # ── Generate historical data ─────────────────────────────────────────────
    log.info("\nGenerating historical stock data …")
    hist_df = generate_all_historical()
    log.info(f"  Historical rows: {len(hist_df)}")

    X, y = build_sequences(hist_df)
    log.info(f"  Sequences: X={X.shape}  y={y.shape}")

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    # ── Phase 1: Full training ───────────────────────────────────────────────
    base_model = train_base_model(X_train, y_train, X_val, y_val)

    # ── Seed memory buffer from training data ────────────────────────────────
    buffer = StockMemoryBuffer()
    seed_idx = np.random.choice(len(X_train), size=min(200, len(X_train)),
                                replace=False)
    buffer.add(X_train[seed_idx], y_train[seed_idx])
    log.info(f"\n  Memory buffer seeded: {buffer.summary()}")

    # ── Phase 2: Incremental pipeline ───────────────────────────────────────
    incr_metrics, retrain_metrics = run_incremental_pipeline(
        base_model, buffer, X_val, y_val)

    # ── Phase 3: Comparison report ───────────────────────────────────────────
    generate_comparison_report(incr_metrics, retrain_metrics)

    elapsed = time.time() - t_start
    log.info(f"\nPipeline complete in {elapsed:.1f}s")
    log.info(f"Results saved in '{config.RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
