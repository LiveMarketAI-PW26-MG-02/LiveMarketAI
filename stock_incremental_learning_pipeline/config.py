"""
config.py — Central configuration for the Stock Incremental Learning Pipeline
"""

# ─── Data & Market Settings ────────────────────────────────────────────────────
TICKERS          = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                    "META", "NVDA", "JPM", "BAC", "GS"]
FEATURE_COLS     = ["open", "high", "low", "close", "volume",
                    "rsi", "macd", "bb_upper", "bb_lower", "atr"]
SEQUENCE_LEN     = 20          # look-back window (time-steps)
PRED_HORIZON     = 1           # steps ahead to predict
INITIAL_SAMPLES  = 2_000       # synthetic historical rows per ticker
STREAM_BATCHES   = 30          # number of incremental data batches
STREAM_BATCH_SZ  = 50          # rows per streaming batch

# ─── Model Architecture ────────────────────────────────────────────────────────
INPUT_SIZE       = len(FEATURE_COLS)
HIDDEN_SIZE      = 64
NUM_LAYERS       = 3           # LSTM layers (lower layers = frozen base)
FROZEN_LAYERS    = 2           # req 7: freeze bottom N layers
OUTPUT_SIZE      = 1           # next-close prediction (regression)
DROPOUT          = 0.2

# ─── Training Hyper-parameters ────────────────────────────────────────────────
BASE_LR          = 1e-3        # initial learning rate
INCR_LR          = 5e-4        # incremental head LR
WEIGHT_DECAY     = 1e-5
FULL_EPOCHS      = 30          # full-retraining epochs
INCR_EPOCHS      = 5           # incremental-update epochs
BATCH_SIZE       = 64

# ─── Memory Buffer (req 2) ────────────────────────────────────────────────────
BUFFER_SIZE      = 500         # max historical samples kept
BUFFER_STRATEGY  = "reservoir" # "reservoir" | "fifo" | "priority"

# ─── Drift Detector (req 3) ──────────────────────────────────────────────────
DRIFT_WINDOW     = 50          # rolling window for drift detection
DRIFT_THRESHOLD  = 0.05        # KS-statistic threshold to trigger update
DRIFT_CHECK_FREQ = 5           # check every N batches

# ─── Adaptive LR Scheduler (req 6) ───────────────────────────────────────────
LR_SCALE_MIN     = 0.1         # minimum LR multiplier
LR_SCALE_MAX     = 5.0         # maximum LR multiplier
LR_VOLATILITY_K  = 10.0        # sensitivity to market volatility

# ─── Gradient Replay (req 5) ─────────────────────────────────────────────────
REPLAY_BUFFER_SZ = 20          # gradient snapshots to store
REPLAY_WEIGHT    = 0.3         # blend weight for replayed gradients

# ─── Confidence-Guided Updates (req 8) ───────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.02    # trigger update if pred error > threshold (normalised)
MC_DROPOUT_SAMPLES   = 10      # Monte-Carlo dropout passes for uncertainty

# ─── Selective Weight Update (req 1) ─────────────────────────────────────────
TOP_K_PARAMS     = 0.20        # update top-20 % most-sensitive parameters

# ─── Dual-Model Architecture (req 4) ─────────────────────────────────────────
STABLE_BLEND     = 0.6         # weight of stable base model in ensemble
ADAPT_BLEND      = 0.4         # weight of fast-adapting incremental model

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR        = "models"
RESULTS_DIR      = "results"
LOG_FILE         = "results/pipeline.log"

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED             = 42
DEVICE           = "cpu"       # "cpu" | "cuda"
VERBOSE          = True
