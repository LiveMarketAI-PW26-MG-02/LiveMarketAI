"""
Online Stock Confidence Recalibration Engine
============================================
Continuously updates prediction confidence using newly arriving stock data
without retraining the full model.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class StockPrediction:
    symbol: str
    timestamp: float
    raw_confidence: float
    predicted_direction: int   # +1 up, -1 down
    actual_direction: Optional[int] = None
    recalibrated_confidence: Optional[float] = None
    regime: Optional[MarketRegime] = None
    volatility: Optional[float] = None


@dataclass
class CalibrationState:
    """Per-asset calibration state maintained online."""
    symbol: str
    ece_history: deque = field(default_factory=lambda: deque(maxlen=200))
    correction_bias: float = 0.0
    drift_score: float = 0.0
    regime_corrections: Dict[str, float] = field(default_factory=lambda: {
        "bullish": 0.0, "bearish": 0.0, "sideways": 0.0
    })
    total_predictions: int = 0
    correct_predictions: int = 0
    smoothed_confidence: float = 0.5
    last_update: float = field(default_factory=time.time)


class VolatilityAwareAdjustment:
    """
    Module 1: Stock Volatility-Aware Confidence Adjustment Layer
    Dynamically scales confidence based on current market volatility.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.returns_buffer: Dict[str, deque] = {}
        self.vol_history: Dict[str, deque] = {}

    def update(self, symbol: str, price_return: float):
        if symbol not in self.returns_buffer:
            self.returns_buffer[symbol] = deque(maxlen=self.window)
            self.vol_history[symbol] = deque(maxlen=100)
        self.returns_buffer[symbol].append(price_return)

    def get_volatility(self, symbol: str) -> float:
        buf = self.returns_buffer.get(symbol, deque())
        if len(buf) < 3:
            return 0.02  # default 2%
        vol = float(np.std(list(buf)))
        if symbol in self.vol_history:
            self.vol_history[symbol].append(vol)
        return vol

    def adjust_confidence(self, symbol: str, raw_confidence: float) -> float:
        vol = self.get_volatility(symbol)
        hist = self.vol_history.get(symbol, deque())
        baseline_vol = float(np.mean(list(hist))) if len(hist) > 5 else 0.02

        # High volatility → compress confidence toward 0.5 (more uncertain)
        # Low volatility  → expand confidence (more reliable)
        vol_ratio = vol / (baseline_vol + 1e-8)
        vol_ratio = np.clip(vol_ratio, 0.1, 5.0)

        # Scaling: above baseline shrinks confidence, below expands it
        adjustment = 1.0 / (1.0 + np.log1p(vol_ratio - 1.0))
        adjusted = 0.5 + (raw_confidence - 0.5) * adjustment
        return float(np.clip(adjusted, 0.01, 0.99))

    def get_volatility_regime_label(self, symbol: str) -> str:
        vol = self.get_volatility(symbol)
        hist = self.vol_history.get(symbol, deque())
        baseline = float(np.mean(list(hist))) if len(hist) > 5 else 0.02
        ratio = vol / (baseline + 1e-8)
        if ratio > 1.5:
            return "HIGH_VOL"
        elif ratio < 0.7:
            return "LOW_VOL"
        return "NORMAL_VOL"


class TimeDecayedWeighting:
    """
    Module 2: Time-Decayed Stock Confidence Weighting Scheme
    Older predictions gradually lose influence over recent signals.
    """

    def __init__(self, half_life_seconds: float = 3600.0):
        self.half_life = half_life_seconds
        self.decay_lambda = np.log(2) / half_life_seconds
        self.prediction_history: Dict[str, List[Tuple[float, float, int]]] = {}

    def add_prediction(self, symbol: str, timestamp: float,
                       confidence: float, correct: int):
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        self.prediction_history[symbol].append((timestamp, confidence, correct))
        # Keep last 500 predictions
        if len(self.prediction_history[symbol]) > 500:
            self.prediction_history[symbol] = self.prediction_history[symbol][-500:]

    def get_weighted_accuracy(self, symbol: str, current_time: float) -> float:
        history = self.prediction_history.get(symbol, [])
        if not history:
            return 0.5

        weights, outcomes = [], []
        for ts, conf, correct in history:
            age = current_time - ts
            w = np.exp(-self.decay_lambda * age)
            weights.append(w)
            outcomes.append(correct)

        weights = np.array(weights)
        outcomes = np.array(outcomes)
        total_w = weights.sum()
        if total_w < 1e-10:
            return 0.5
        return float((weights * outcomes).sum() / total_w)

    def time_decay_adjustment(self, symbol: str, raw_confidence: float,
                              current_time: float) -> float:
        weighted_acc = self.get_weighted_accuracy(symbol, current_time)
        # Blend raw confidence with historical weighted accuracy
        # If model has been accurate recently → trust raw confidence more
        trust_weight = np.clip(weighted_acc, 0.3, 0.9)
        adjusted = trust_weight * raw_confidence + (1 - trust_weight) * 0.5
        return float(np.clip(adjusted, 0.01, 0.99))


class RegimeDependentCalibration:
    """
    Module 3: Stock Regime-Dependent Calibration System
    Calibration behaves differently across bullish, bearish, sideways states.
    """

    def __init__(self, regime_window: int = 30):
        self.regime_window = regime_window
        self.price_history: Dict[str, deque] = {}
        self.regime_accuracy: Dict[str, Dict[str, deque]] = {}

        # Platt scaling parameters per regime per symbol
        self.platt_params: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def update_price(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.regime_window + 1)
            self.regime_accuracy[symbol] = {
                r.value: deque(maxlen=100) for r in MarketRegime
            }
            self.platt_params[symbol] = {
                r.value: (1.0, 0.0) for r in MarketRegime  # (scale, bias)
            }
        self.price_history[symbol].append(price)

    def detect_regime(self, symbol: str) -> MarketRegime:
        prices = list(self.price_history.get(symbol, []))
        if len(prices) < 10:
            return MarketRegime.SIDEWAYS

        p_short = np.array(prices[-6:])
        p_long  = np.array(prices)
        short_ret = float(np.mean(np.diff(p_short) / (p_short[:-1] + 1e-8)))
        long_ret  = float(np.mean(np.diff(p_long)  / (p_long[:-1]  + 1e-8)))
        vol       = float(np.std( np.diff(p_long)  / (p_long[:-1]  + 1e-8)))

        if short_ret > 0.002 and long_ret > 0.001:
            return MarketRegime.BULLISH
        elif short_ret < -0.002 and long_ret < -0.001:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.SIDEWAYS

    def record_outcome(self, symbol: str, regime: MarketRegime,
                       confidence: float, correct: int):
        if symbol in self.regime_accuracy:
            self.regime_accuracy[symbol][regime.value].append((confidence, correct))
            self._update_platt(symbol, regime)

    def _update_platt(self, symbol: str, regime: MarketRegime):
        """Online Platt scaling update using gradient descent."""
        history = list(self.regime_accuracy[symbol][regime.value])
        if len(history) < 10:
            return

        scale, bias = self.platt_params[symbol][regime.value]
        lr = 0.01
        recent = history[-20:]
        for conf, outcome in recent:
            logit = scale * conf + bias
            prob  = 1 / (1 + np.exp(-logit))
            err   = prob - outcome
            scale -= lr * err * conf
            bias  -= lr * err

        self.platt_params[symbol][regime.value] = (
            float(np.clip(scale, 0.1, 5.0)),
            float(np.clip(bias, -3.0, 3.0))
        )

    def calibrate(self, symbol: str, raw_confidence: float,
                  regime: MarketRegime) -> float:
        if symbol not in self.platt_params:
            return raw_confidence

        scale, bias = self.platt_params[symbol][regime.value]
        logit    = scale * raw_confidence + bias
        calibrated = 1 / (1 + np.exp(-logit))
        return float(np.clip(calibrated, 0.01, 0.99))


class AdaptiveConfidenceSmoothing:
    """
    Module 4: Adaptive Stock Confidence Smoothing
    Aggregates confidence across recent windows to reduce instability.
    """

    def __init__(self, base_window: int = 10, vol_scale: float = 2.0):
        self.base_window = base_window
        self.vol_scale   = vol_scale
        self.conf_history: Dict[str, deque] = {}
        self.ema_alpha: Dict[str, float] = {}

    def update_and_smooth(self, symbol: str, confidence: float,
                          volatility: float, baseline_vol: float = 0.02) -> float:
        if symbol not in self.conf_history:
            self.conf_history[symbol] = deque(maxlen=50)
            self.ema_alpha[symbol] = 0.3

        # Adaptive window: wider window during high volatility
        vol_ratio = volatility / (baseline_vol + 1e-8)
        adaptive_alpha = max(0.05, 0.3 / (1.0 + self.vol_scale * np.log1p(vol_ratio - 1)))
        self.ema_alpha[symbol] = float(np.clip(adaptive_alpha, 0.05, 0.5))

        self.conf_history[symbol].append(confidence)
        history = list(self.conf_history[symbol])

        # Exponential weighted moving average
        alpha = self.ema_alpha[symbol]
        ema = history[0]
        for c in history[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return float(np.clip(ema, 0.01, 0.99))


class MiscalibrationDetector:
    """
    Module 5: Miscalibration Detection Module
    Identifies when predicted confidence deviates from actual outcome probabilities.
    """

    def __init__(self, n_bins: int = 10, detection_threshold: float = 0.1):
        self.n_bins = n_bins
        self.threshold = detection_threshold
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.bin_data: Dict[str, List[Tuple[float, int]]] = {}
        self.ece_history: Dict[str, deque] = {}
        self.alert_log: List[Dict] = []

    def record(self, symbol: str, confidence: float, correct: int):
        if symbol not in self.bin_data:
            self.bin_data[symbol] = []
            self.ece_history[symbol] = deque(maxlen=50)
        self.bin_data[symbol].append((confidence, correct))
        if len(self.bin_data[symbol]) > 500:
            self.bin_data[symbol] = self.bin_data[symbol][-500:]

    def compute_ece(self, symbol: str) -> float:
        """Expected Calibration Error computation."""
        data = self.bin_data.get(symbol, [])
        if len(data) < 20:
            return 0.0

        confs = np.array([d[0] for d in data])
        outs  = np.array([d[1] for d in data])
        ece   = 0.0
        n     = len(data)

        for i in range(self.n_bins):
            lo, hi = self.bin_edges[i], self.bin_edges[i+1]
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            avg_conf = confs[mask].mean()
            avg_acc  = outs[mask].mean()
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)

        if symbol in self.ece_history:
            self.ece_history[symbol].append(ece)
        return float(ece)

    def detect_miscalibration(self, symbol: str, timestamp: float) -> Dict:
        ece = self.compute_ece(symbol)
        is_miscalibrated = ece > self.threshold
        result = {
            "symbol": symbol,
            "ece": ece,
            "miscalibrated": is_miscalibrated,
            "timestamp": timestamp,
            "severity": "HIGH" if ece > 0.2 else ("MEDIUM" if ece > 0.1 else "LOW")
        }
        if is_miscalibrated:
            self.alert_log.append(result)
        return result

    def get_ece_trend(self, symbol: str) -> str:
        hist = list(self.ece_history.get(symbol, []))
        if len(hist) < 5:
            return "STABLE"
        trend = np.polyfit(range(len(hist)), hist, 1)[0]
        if trend > 0.005:
            return "WORSENING"
        elif trend < -0.005:
            return "IMPROVING"
        return "STABLE"


class FeedbackCorrectionLoop:
    """
    Module 6: Feedback-Driven Stock Confidence Correction Loop
    Uses past prediction errors to dynamically adjust future confidence outputs.
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.error_history: Dict[str, deque] = {}
        self.correction_velocity: Dict[str, float] = {}
        self.cumulative_correction: Dict[str, float] = {}

    def record_error(self, symbol: str, confidence: float, correct: int):
        if symbol not in self.error_history:
            self.error_history[symbol] = deque(maxlen=100)
            self.correction_velocity[symbol] = 0.0
            self.cumulative_correction[symbol] = 0.0

        # Error = predicted probability - actual outcome
        error = confidence - float(correct)
        self.error_history[symbol].append(error)

    def get_correction(self, symbol: str) -> float:
        if symbol not in self.error_history:
            return 0.0

        errors = list(self.error_history[symbol])
        if not errors:
            return 0.0

        # Recent bias estimate (weighted toward recent errors)
        weights = np.exp(np.linspace(-2, 0, len(errors)))
        weights /= weights.sum()
        recent_bias = float(np.dot(weights, errors))

        # Momentum update
        vel = self.correction_velocity.get(symbol, 0.0)
        vel = self.momentum * vel - self.lr * recent_bias
        self.correction_velocity[symbol] = vel
        self.cumulative_correction[symbol] = (
            self.cumulative_correction.get(symbol, 0.0) + vel
        )
        return float(np.clip(vel, -0.3, 0.3))

    def apply_correction(self, symbol: str, confidence: float) -> float:
        correction = self.get_correction(symbol)
        corrected = confidence - correction  # subtract bias
        return float(np.clip(corrected, 0.01, 0.99))


class ConfidenceDriftTracker:
    """
    Module 7: Confidence Drift Tracking for Stock Signals
    Monitors long-term shifts in model confidence reliability.
    """

    def __init__(self, drift_window: int = 100, alert_threshold: float = 0.15):
        self.drift_window = drift_window
        self.alert_threshold = alert_threshold
        self.confidence_series: Dict[str, deque] = {}
        self.accuracy_series:   Dict[str, deque] = {}
        self.drift_alerts: List[Dict] = []

    def record(self, symbol: str, confidence: float, correct: int, timestamp: float):
        if symbol not in self.confidence_series:
            self.confidence_series[symbol] = deque(maxlen=self.drift_window * 2)
            self.accuracy_series[symbol]   = deque(maxlen=self.drift_window * 2)

        self.confidence_series[symbol].append(confidence)
        self.accuracy_series[symbol].append(float(correct))

    def compute_drift(self, symbol: str) -> Dict:
        confs = list(self.confidence_series.get(symbol, []))
        accs  = list(self.accuracy_series.get(symbol, []))
        if len(confs) < self.drift_window * 2:
            return {"drift": 0.0, "alert": False, "direction": "NONE"}

        mid = len(confs) // 2
        old_gap = np.mean(confs[:mid]) - np.mean(accs[:mid])
        new_gap = np.mean(confs[mid:]) - np.mean(accs[mid:])
        drift   = float(new_gap - old_gap)

        alert = abs(drift) > self.alert_threshold
        if alert:
            self.drift_alerts.append({
                "symbol": symbol,
                "drift": drift,
                "timestamp": time.time()
            })

        return {
            "drift": drift,
            "alert": alert,
            "direction": "OVERCONFIDENT" if drift > 0 else ("UNDERCONFIDENT" if drift < 0 else "STABLE"),
            "old_gap": float(old_gap),
            "new_gap": float(new_gap)
        }

    def get_reliability_score(self, symbol: str) -> float:
        drift_info = self.compute_drift(symbol)
        return float(1.0 - min(abs(drift_info["drift"]), 1.0))


class ConfidenceNormalizationLayer:
    """
    Module 8: Confidence Normalization Across Multiple Stock Assets
    Ensures consistent confidence interpretation across different stocks.
    """

    def __init__(self, target_mean: float = 0.65, target_std: float = 0.15):
        self.target_mean = target_mean
        self.target_std  = target_std
        self.asset_stats: Dict[str, Dict[str, deque]] = {}

    def update_stats(self, symbol: str, confidence: float):
        if symbol not in self.asset_stats:
            self.asset_stats[symbol] = {
                "confidences": deque(maxlen=200),
                "rolling_mean": deque(maxlen=50),
                "rolling_std":  deque(maxlen=50)
            }
        self.asset_stats[symbol]["confidences"].append(confidence)

        confs = list(self.asset_stats[symbol]["confidences"])
        if len(confs) >= 10:
            self.asset_stats[symbol]["rolling_mean"].append(np.mean(confs[-20:]))
            self.asset_stats[symbol]["rolling_std"].append(np.std(confs[-20:]) + 1e-6)

    def normalize(self, symbol: str, confidence: float) -> float:
        self.update_stats(symbol, confidence)
        stats = self.asset_stats.get(symbol, {})
        means = list(stats.get("rolling_mean", []))
        stds  = list(stats.get("rolling_std", []))

        if not means or not stds:
            return confidence

        asset_mean = float(np.mean(means[-5:]))
        asset_std  = float(np.mean(stds[-5:]))

        # Z-score normalize then rescale to target distribution
        z = (confidence - asset_mean) / (asset_std + 1e-8)
        normalized = self.target_mean + z * self.target_std
        return float(np.clip(normalized, 0.01, 0.99))

    def get_cross_asset_summary(self) -> Dict:
        summary = {}
        for sym, stats in self.asset_stats.items():
            confs = list(stats["confidences"])
            if confs:
                summary[sym] = {
                    "mean": float(np.mean(confs)),
                    "std":  float(np.std(confs)),
                    "count": len(confs)
                }
        return summary


class CalibrationBenchmark:
    """
    Module 9: Real-Time Stock Calibration Benchmarking Study
    Compares raw vs recalibrated confidence for improved decision reliability.
    """

    def __init__(self):
        self.raw_records:  Dict[str, List[Tuple[float, int]]] = {}
        self.cal_records:  Dict[str, List[Tuple[float, int]]] = {}
        self.timestamps:   Dict[str, List[float]] = {}

    def record(self, symbol: str, raw_conf: float, cal_conf: float,
               correct: int, timestamp: float):
        for store, conf in [(self.raw_records, raw_conf), (self.cal_records, cal_conf)]:
            if symbol not in store:
                store[symbol] = []
            store[symbol].append((conf, correct))

        if symbol not in self.timestamps:
            self.timestamps[symbol] = []
        self.timestamps[symbol].append(timestamp)

    def compute_metrics(self, records: Dict[str, List[Tuple[float, int]]],
                        symbol: str) -> Dict:
        data = records.get(symbol, [])
        if len(data) < 5:
            return {"ece": 0.0, "accuracy": 0.5, "brier": 0.25, "sharpness": 0.0}

        confs = np.array([d[0] for d in data])
        outs  = np.array([d[1] for d in data])

        # ECE
        bins  = np.linspace(0, 1, 11)
        ece   = 0.0
        n     = len(data)
        for i in range(10):
            mask = (confs >= bins[i]) & (confs < bins[i+1])
            if mask.sum() > 0:
                ece += (mask.sum() / n) * abs(confs[mask].mean() - outs[mask].mean())

        return {
            "ece":       float(ece),
            "accuracy":  float(outs.mean()),
            "brier":     float(np.mean((confs - outs) ** 2)),
            "sharpness": float(np.std(confs)),
            "mean_conf": float(confs.mean()),
            "n":         n
        }

    def full_report(self, symbol: str) -> Dict:
        raw = self.compute_metrics(self.raw_records, symbol)
        cal = self.compute_metrics(self.cal_records, symbol)
        return {
            "symbol": symbol,
            "raw":    raw,
            "calibrated": cal,
            "ece_improvement":    float(raw["ece"] - cal["ece"]),
            "brier_improvement":  float(raw["brier"] - cal["brier"]),
            "sharpness_change":   float(cal["sharpness"] - raw["sharpness"])
        }

    def all_symbols_report(self) -> List[Dict]:
        symbols = set(self.raw_records.keys()) | set(self.cal_records.keys())
        return [self.full_report(s) for s in sorted(symbols)]
