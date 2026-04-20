# Stock Incremental Learning Pipeline

A production-grade, research-quality incremental learning system for stock
market prediction.  All nine requirements are implemented as independent,
composable Python modules.

---

## Quick Start

```
# 1. Install dependencies (Windows)
setup.bat

# 2. Run the full pipeline
run.bat
```

On Linux / macOS:
```bash
pip install numpy scipy pandas torch matplotlib tqdm
python pipeline.py
python visualise.py
```

---

## File Manifest

| File | Implements | Description |
|------|-----------|-------------|
| `config.py` | All | Central hyper-parameters & paths |
| `data_generator.py` | — | Synthetic OHLCV + indicators, streaming batches |
| `memory_buffer.py` | Req 2 | Reservoir / FIFO / priority buffer |
| `drift_detector.py` | Req 3 | KS-test / Page-Hinkley / Wasserstein drift detection |
| `incremental_model.py` | Req 1, 4, 7, 8 | DualStockModel, SelectiveWeightUpdater, ConfidenceGuardedPredictor |
| `gradient_replay.py` | Req 5 | Gradient snapshot buffer + blending |
| `adaptive_scheduler.py` | Req 6 | Volatility-aware LR scheduler |
| `pipeline.py` | Req 9 | Full orchestration + comparison evaluation |
| `visualise.py` | Req 9 | Dashboard charts |
| `setup.bat` | — | Windows dependency installer |
| `run.bat` | — | Windows pipeline launcher |

---

## Requirements Implementation Detail

### Req 1 — Selective Weight Update (`incremental_model.py → SelectiveWeightUpdater`)
After each backward pass the updater computes the absolute gradient magnitude
across all parameters, finds the top-K % threshold, and zeroes gradients below
it.  Only the most "information-rich" weights are updated per step.

### Req 2 — Memory Buffer (`memory_buffer.py → StockMemoryBuffer`)
Three strategies:
- **reservoir** — uniform random sampling over the entire stream (default)
- **fifo** — sliding window of the most recent N samples
- **priority** — hard-example mining: keeps samples with highest prediction error

The buffer is seeded from historical training data before streaming begins,
then mixed with each incremental batch to provide rehearsal.

### Req 3 — Drift-Aware Trigger (`drift_detector.py → StockDriftDetector`)
Maintains reference and current rolling windows of returns and computes a
drift score using the two-sample Kolmogorov-Smirnov statistic.  An incremental
update is only triggered when `KS > DRIFT_THRESHOLD`.  The reference window
slides forward after each confirmed drift event.

### Req 4 — Dual-Model Architecture (`incremental_model.py → DualStockModel`)
- **StableBaseModel** (StockLSTM): trained on all historical data, changes slowly
- **FastAdaptHead** (FastAdaptHead MLP): updates rapidly on recent batches
- Final prediction: `0.6 × base_pred + 0.4 × adapt_pred`

### Req 5 — Gradient Replay (`gradient_replay.py → StockGradientReplayBuffer`)
Stores up to `REPLAY_BUFFER_SZ` gradient snapshots.  Before each optimiser
step the replay buffer blends the mean of all historic gradients with the
current gradient:  `eff_grad = (1-w)·curr + w·hist_mean`.

### Req 6 — Adaptive LR Scheduler (`adaptive_scheduler.py → StockAdaptiveLRScheduler`)
Computes a volatility ratio `σ_recent / σ_baseline` from streaming close
prices.  The LR is scaled inversely: high volatility → smaller LR to avoid
overshooting.  A loss-spike detector halves the LR for 3 steps after any
sudden loss increase.

### Req 7 — Layer Freezing (`incremental_model.py → StockLSTM.freeze_base_layers`)
The LSTM stack has `NUM_LAYERS` layers.  The bottom `FROZEN_LAYERS` layers
are frozen during incremental updates; only the top layer(s) and the
prediction head are trained.  `unfreeze_all()` is called for full retraining.

### Req 8 — Confidence-Guided Updates (`incremental_model.py → ConfidenceGuardedPredictor`)
Monte-Carlo Dropout runs `MC_DROPOUT_SAMPLES` stochastic forward passes and
measures the standard deviation of predictions as uncertainty.  An update is
triggered only when mean uncertainty exceeds `CONFIDENCE_THRESHOLD`.

### Req 9 — Comparative Evaluation (`pipeline.py → generate_comparison_report`)
Every 5 batches a shadow model is fully retrained on all accumulated data
(simulating the naive baseline).  Metrics (val loss, MAE, MAPE, training time)
are saved to CSV and a JSON summary compares efficiency gains.

---

## Output Files

After running the pipeline, `results/` contains:

| File | Contents |
|------|----------|
| `incremental_metrics.csv` | Per-batch: drift score, val loss/MAE/MAPE, LR, buffer size |
| `retrain_metrics.csv` | Per retrain-cycle: val loss/MAE/MAPE, training time |
| `comparison_summary.json` | Aggregate efficiency comparison |
| `pipeline.log` | Full execution log with timestamps |
| `pipeline_dashboard.png` | 8-panel visualisation dashboard |

`models/` contains:
| File | Contents |
|------|----------|
| `base_model.pt` | Trained base LSTM weights |

---

## Configuration

Edit `config.py` to change:
- `TICKERS` / `FEATURE_COLS` — which stocks and features to use
- `BUFFER_STRATEGY` — `"reservoir"` | `"fifo"` | `"priority"`
- `DRIFT_THRESHOLD` — sensitivity of the drift detector
- `TOP_K_PARAMS` — fraction of parameters updated per incremental step
- `FROZEN_LAYERS` — how many base LSTM layers to freeze
- `STABLE_BLEND` / `ADAPT_BLEND` — dual-model ensemble weights
- `FULL_EPOCHS` / `INCR_EPOCHS` — training durations
