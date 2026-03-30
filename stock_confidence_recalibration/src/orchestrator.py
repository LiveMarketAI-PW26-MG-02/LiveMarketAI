"""
Online Recalibration Orchestrator
===================================
Ties all modules together into a real-time pipeline that processes
incoming stock ticks and continuously recalibrates confidence.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from confidence_engine import (
    VolatilityAwareAdjustment,
    TimeDecayedWeighting,
    RegimeDependentCalibration,
    AdaptiveConfidenceSmoothing,
    MiscalibrationDetector,
    FeedbackCorrectionLoop,
    ConfidenceDriftTracker,
    ConfidenceNormalizationLayer,
    CalibrationBenchmark,
    MarketRegime,
    StockPrediction,
    CalibrationState,
)
from simulator import StockSimulator, TickData


@dataclass
class PipelineResult:
    symbol: str
    timestamp: float
    raw_confidence: float
    final_confidence: float
    regime: str
    volatility: float
    vol_label: str
    ece: float
    miscalibrated: bool
    drift_direction: str
    actual_correct: Optional[int] = None
    stage_confidences: Dict[str, float] = field(default_factory=dict)


class OnlineRecalibrationPipeline:
    """
    Full pipeline that processes a stream of stock ticks through all
    9 recalibration modules in sequence.
    """

    def __init__(self):
        self.vol_adjust    = VolatilityAwareAdjustment(window=20)
        self.time_decay    = TimeDecayedWeighting(half_life_seconds=500.0)
        self.regime_cal    = RegimeDependentCalibration(regime_window=30)
        self.smoother      = AdaptiveConfidenceSmoothing(base_window=10)
        self.miscal_detect = MiscalibrationDetector(n_bins=10, detection_threshold=0.08)
        self.feedback      = FeedbackCorrectionLoop(lr=0.05, momentum=0.85)
        self.drift_tracker = ConfidenceDriftTracker(drift_window=80, alert_threshold=0.12)
        self.normalizer    = ConfidenceNormalizationLayer(target_mean=0.62, target_std=0.14)
        self.benchmark     = CalibrationBenchmark()

        # Pending predictions awaiting outcome confirmation
        self.pending: Dict[str, List[StockPrediction]] = {}
        self.calibration_states: Dict[str, CalibrationState] = {}
        self.results_log: List[PipelineResult] = []
        self.tick_count = 0

    def _ensure_state(self, symbol: str):
        if symbol not in self.calibration_states:
            self.calibration_states[symbol] = CalibrationState(symbol=symbol)
        if symbol not in self.pending:
            self.pending[symbol] = []

    def process_tick(self, tick: TickData, raw_confidence: float,
                     predicted_direction: int) -> PipelineResult:
        """Process a single stock tick through the full recalibration pipeline."""
        symbol = tick.symbol
        ts     = tick.timestamp
        self._ensure_state(symbol)
        self.tick_count += 1

        # ----- Update price/return histories -----
        self.vol_adjust.update(symbol, tick.returns)
        self.regime_cal.update_price(symbol, tick.price)

        vol      = self.vol_adjust.get_volatility(symbol)
        vol_hist = list(self.vol_adjust.vol_history.get(symbol, []))
        base_vol = float(np.mean(vol_hist)) if len(vol_hist) > 3 else 0.02
        vol_lbl  = self.vol_adjust.get_volatility_regime_label(symbol)
        regime   = self.regime_cal.detect_regime(symbol)

        # ----- Stage 1: Volatility-aware adjustment -----
        c1 = self.vol_adjust.adjust_confidence(symbol, raw_confidence)

        # ----- Stage 2: Time-decayed weighting -----
        c2 = self.time_decay.time_decay_adjustment(symbol, c1, ts)

        # ----- Stage 3: Regime-dependent calibration -----
        c3 = self.regime_cal.calibrate(symbol, c2, regime)

        # ----- Stage 4: Adaptive smoothing -----
        c4 = self.smoother.update_and_smooth(symbol, c3, vol, base_vol)

        # ----- Stage 5: Feedback correction -----
        c5 = self.feedback.apply_correction(symbol, c4)

        # ----- Stage 6: Normalization -----
        c6 = self.normalizer.normalize(symbol, c5)

        # ----- Resolve pending predictions (simulate next-tick outcome) -----
        if self.pending[symbol]:
            last_pred = self.pending[symbol][-1]
            # Outcome: did the predicted direction match actual price movement?
            actual_dir = 1 if tick.returns > 0 else -1
            correct    = int(last_pred.predicted_direction == actual_dir)
            last_pred.actual_direction = actual_dir

            conf_used = last_pred.recalibrated_confidence or last_pred.raw_confidence

            # Feed outcome back into all learning modules
            self.time_decay.add_prediction(symbol, last_pred.timestamp, conf_used, correct)
            self.regime_cal.record_outcome(symbol, regime, conf_used, correct)
            self.miscal_detect.record(symbol, conf_used, correct)
            self.feedback.record_error(symbol, conf_used, correct)
            self.drift_tracker.record(symbol, conf_used, correct, ts)
            self.benchmark.record(symbol, last_pred.raw_confidence, conf_used, correct, ts)

            # Update calibration state
            state = self.calibration_states[symbol]
            state.total_predictions += 1
            state.correct_predictions += correct

        # ----- Miscalibration detection -----
        miscal_info = self.miscal_detect.detect_miscalibration(symbol, ts)
        ece         = miscal_info["ece"]
        is_miscal   = miscal_info["miscalibrated"]

        # ----- Drift tracking -----
        drift_info = self.drift_tracker.compute_drift(symbol)

        # ----- Register this prediction as pending -----
        prediction = StockPrediction(
            symbol=symbol,
            timestamp=ts,
            raw_confidence=raw_confidence,
            predicted_direction=predicted_direction,
            recalibrated_confidence=c6,
            regime=regime,
            volatility=vol,
        )
        self.pending[symbol].append(prediction)
        if len(self.pending[symbol]) > 20:
            self.pending[symbol] = self.pending[symbol][-20:]

        result = PipelineResult(
            symbol=symbol,
            timestamp=ts,
            raw_confidence=raw_confidence,
            final_confidence=c6,
            regime=regime.value,
            volatility=vol,
            vol_label=vol_lbl,
            ece=ece,
            miscalibrated=is_miscal,
            drift_direction=drift_info["direction"],
            stage_confidences={
                "raw":        raw_confidence,
                "vol_adj":    c1,
                "time_decay": c2,
                "regime_cal": c3,
                "smoothed":   c4,
                "feedback":   c5,
                "normalized": c6,
            }
        )
        self.results_log.append(result)
        return result

    def get_benchmark_report(self) -> List[Dict]:
        return self.benchmark.all_symbols_report()

    def get_drift_report(self) -> Dict[str, Dict]:
        return {sym: self.drift_tracker.compute_drift(sym)
                for sym in self.calibration_states}

    def get_cross_asset_summary(self) -> Dict:
        return self.normalizer.get_cross_asset_summary()

    def get_accuracy_table(self) -> Dict[str, Dict]:
        table = {}
        for sym, state in self.calibration_states.items():
            total = state.total_predictions
            if total > 0:
                table[sym] = {
                    "total":    total,
                    "accuracy": round(state.correct_predictions / total, 4),
                    "ece":      round(self.miscal_detect.compute_ece(sym), 4),
                    "ece_trend": self.miscal_detect.get_ece_trend(sym),
                    "reliability": round(self.drift_tracker.get_reliability_score(sym), 4),
                }
        return table
