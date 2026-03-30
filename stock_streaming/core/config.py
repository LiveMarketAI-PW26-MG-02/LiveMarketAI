"""
config.py — Central configuration for the stock streaming classification system.
All tunable parameters live here so every module imports from one source of truth.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------------
@dataclass
class WindowConfig:
    """Parameters that govern the sliding / dynamic window behaviour."""
    base_window_size: int = 60          # Default window length (ticks)
    min_window_size: int = 20           # Smallest allowed window
    max_window_size: int = 180          # Largest allowed window
    overlap_fraction: float = 0.5       # Fraction of window that overlaps with next
    resize_cooldown_ticks: int = 10     # Minimum ticks between consecutive resizes
    volatility_scale_factor: float = 2.0  # How aggressively volatility shrinks window
    activity_scale_factor: float = 1.5   # How aggressively activity enlarges window


# ---------------------------------------------------------------------------
# Streaming configuration
# ---------------------------------------------------------------------------
@dataclass
class StreamConfig:
    """Parameters controlling the real-time ingestion pipeline."""
    tick_interval_ms: float = 100.0     # Simulated milliseconds between ticks
    max_queue_size: int = 1_000         # Back-pressure queue cap
    inference_timeout_ms: float = 50.0  # Hard deadline for a single inference call
    warm_up_ticks: int = 30             # Ticks before the first prediction is emitted


# ---------------------------------------------------------------------------
# Classification configuration
# ---------------------------------------------------------------------------
@dataclass
class ClassifierConfig:
    """Hyper-parameters for the streaming classifier."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    state_memory_length: int = 5        # How many past predictions inform the next
    confidence_ema_alpha: float = 0.3   # EMA smoothing for confidence scores
    labels: List[str] = field(default_factory=lambda: ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])


# ---------------------------------------------------------------------------
# Smoothing configuration
# ---------------------------------------------------------------------------
@dataclass
class SmoothingConfig:
    """Parameters for the temporal smoothing layer."""
    window_size: int = 10               # Number of recent predictions to smooth over
    method: str = "weighted_majority"   # Options: "majority", "weighted_majority", "ema"
    ema_alpha: float = 0.4              # Used when method == "ema"
    noise_threshold: float = 0.15      # Confidence below this is treated as noise


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Parameters for batch-vs-streaming benchmarking."""
    n_simulation_ticks: int = 2_000
    batch_sizes: List[int] = field(default_factory=lambda: [30, 60, 120, 240])
    latency_percentiles: List[int] = field(default_factory=lambda: [50, 90, 95, 99])
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Master config assembler
# ---------------------------------------------------------------------------
@dataclass
class MasterConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


# Singleton — import this everywhere
CFG = MasterConfig()


def print_config(cfg: MasterConfig = CFG) -> None:
    """Pretty-print the active configuration."""
    import json, dataclasses
    print(json.dumps(dataclasses.asdict(cfg), indent=2))


if __name__ == "__main__":
    print_config()
