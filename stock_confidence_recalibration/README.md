# Online Stock Confidence Recalibration Engine

## Overview
A complete system for real-time recalibration of stock prediction confidence scores
without retraining the underlying model. Processes incoming tick data and continuously
updates calibration state across 9 specialized modules.

## Quick Start
1. Run `setup.bat` to install dependencies (numpy, rich)
2. Run `run.bat` to launch with interactive menu
3. Open `output/recalibration_report.html` for the benchmarking report

## Architecture — 9 Recalibration Modules

| # | Module | File | Description |
|---|--------|------|-------------|
| 1 | VolatilityAwareAdjustment | confidence_engine.py | Scales confidence based on GARCH-like rolling volatility |
| 2 | TimeDecayedWeighting | confidence_engine.py | Exponential decay on historical prediction accuracy |
| 3 | RegimeDependentCalibration | confidence_engine.py | Online Platt scaling per regime (bull/bear/sideways) |
| 4 | AdaptiveConfidenceSmoothing | confidence_engine.py | EMA with adaptive alpha based on vol ratio |
| 5 | MiscalibrationDetector | confidence_engine.py | ECE-based alert system with bin-level analysis |
| 6 | FeedbackCorrectionLoop | confidence_engine.py | Momentum-based bias correction from past errors |
| 7 | ConfidenceDriftTracker | confidence_engine.py | Long-term shift detection (old vs new window gap) |
| 8 | ConfidenceNormalizationLayer | confidence_engine.py | Z-score normalization across assets |
| 9 | CalibrationBenchmark | confidence_engine.py | ECE/Brier/Sharpness: raw vs recalibrated comparison |

## Pipeline Flow
```
Raw Confidence
     │
     ▼
[1] Volatility Adjustment    ← current GARCH vol vs baseline
     │
     ▼
[2] Time-Decay Weighting     ← exponentially weighted accuracy
     │
     ▼
[3] Regime Calibration       ← Platt scaling per market state
     │
     ▼
[4] Adaptive Smoothing       ← EMA with vol-adaptive alpha
     │
     ▼
[5] Miscalibration Check     ← ECE monitoring, alerts
     │
     ▼
[6] Feedback Correction      ← momentum bias correction
     │
     ▼
[7] Drift Tracking           ← long-term reliability monitor
     │
     ▼
[8] Cross-Asset Normalization ← consistent scale across stocks
     │
     ▼
Final Recalibrated Confidence
     │
     ▼
[9] Benchmark Recording      ← raw vs calibrated metrics
```

## Assets Simulated
AAPL · MSFT · GOOGL · AMZN · NVDA · TSLA · META · JPM

## Requirements
- Python 3.8+
- numpy
- rich (for live dashboard)

## Output
- **Terminal**: Live dashboard with per-asset status and stage-by-stage pipeline values
- **HTML**: Interactive report with Chart.js visualizations comparing raw vs recalibrated performance
