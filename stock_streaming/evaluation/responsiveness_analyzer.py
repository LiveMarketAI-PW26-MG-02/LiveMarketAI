"""
responsiveness_analyzer.py — Quantifies the trade-off between prediction
responsiveness and signal stability when using different window sizes.

Key questions answered:
    1. How quickly does each window size detect a regime change?
    2. How stable is the label output during a calm period?
    3. What is the sweet-spot window for a given volatility regime?
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from core.config import CFG
from core.data_generator import StockDataGenerator
from core.feature_extractor import FeatureExtractor
from classification.base_classifier import BaseClassifier, LABELS
from windowing.temporal_smoother import TemporalSmoother


class RegimeChangeDetector:
    """
    Injects a synthetic regime shift at a known tick and measures how many
    ticks later each window configuration detects it.
    """

    def __init__(self, shift_tick: int = 100, shift_magnitude: float = 0.015):
        self.shift_tick      = shift_tick
        self.shift_magnitude = shift_magnitude
        self._detected: Dict[int, Optional[int]] = {}   # ws → detection_tick

    def check_detection(self, window_size: int, tick_index: int,
                        label: str, prev_label: str) -> bool:
        """Returns True if this tick represents a detected label change."""
        if window_size not in self._detected:
            self._detected[window_size] = None
        if (tick_index >= self.shift_tick and
                prev_label in ("HOLD", "BUY", "STRONG_BUY") and
                label in ("SELL", "STRONG_SELL") and
                self._detected[window_size] is None):
            self._detected[window_size] = tick_index
            return True
        return False

    def detection_lag(self, window_size: int) -> Optional[int]:
        detected = self._detected.get(window_size)
        if detected is None:
            return None
        return detected - self.shift_tick


class WindowResponsivenessAnalyzer:
    """
    Runs a single synthetic stream through multiple window-size configurations
    and reports the responsiveness / stability trade-off for each.

    Metrics per window size:
        - detection_lag_ticks : how many ticks after a regime shift the label changes
        - flip_rate           : label changes per prediction (lower = more stable)
        - mean_confidence     : average confidence (lower = more uncertainty)
        - stability_score     : composite stability metric (higher = more stable)
        - responsiveness_score: inverse detection lag (higher = faster)
    """

    DEFAULT_WINDOW_SIZES = [20, 40, 60, 90, 120, 180]
    N_TICKS              = 400
    REGIME_SHIFT_TICK    = 200

    def __init__(self, window_sizes: Optional[List[int]] = None, seed: int = 42):
        self._sizes  = window_sizes or self.DEFAULT_WINDOW_SIZES
        self._seed   = seed
        self._results: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> Dict[int, Dict]:
        print("=" * 60)
        print("  Window Responsiveness vs Stability Analysis")
        print("=" * 60)
        print(f"  Window sizes tested : {self._sizes}")
        print(f"  Stream length       : {self.N_TICKS} ticks")
        print(f"  Regime shift at     : tick {self.REGIME_SHIFT_TICK}\n")

        # Build a single shared tick stream (same for all window sizes)
        gen   = StockDataGenerator(seed=self._seed)
        ticks = gen.generate_batch(self.N_TICKS)
        # Inject a downward regime shift after REGIME_SHIFT_TICK
        self._inject_shift(ticks, self.REGIME_SHIFT_TICK)

        for ws in self._sizes:
            self._results[ws] = self._evaluate_window_size(ws, ticks)

        self._print_results()
        return self._results

    # ------------------------------------------------------------------
    # Per-window evaluation
    # ------------------------------------------------------------------

    def _evaluate_window_size(self, ws: int, ticks: list) -> Dict:
        fx       = FeatureExtractor(window_size=ws)
        smoother = TemporalSmoother(window_size=max(3, ws // 6), method="weighted_majority")
        rcd      = RegimeChangeDetector(shift_tick=self.REGIME_SHIFT_TICK)

        labels_out   = []
        confs_out    = []
        flip_count   = 0
        prev_label   = None
        n_predictions = 0

        for i, tick in enumerate(ticks):
            fx.update(tick)
            feats = fx.extract()
            if feats is None:
                continue

            raw_label, raw_conf = BaseClassifier._heuristic_predict(feats)
            sm_label, sm_conf   = smoother.smooth(raw_label, raw_conf)

            if prev_label and sm_label != prev_label:
                flip_count += 1
                rcd.check_detection(ws, i, sm_label, prev_label)

            labels_out.append(sm_label)
            confs_out.append(sm_conf)
            prev_label = sm_label
            n_predictions += 1

        n = max(1, n_predictions)
        flip_rate     = flip_count / n
        mean_conf     = sum(confs_out) / len(confs_out) if confs_out else 0.0
        detection_lag = rcd.detection_lag(ws)

        # Composite scores (normalised 0–1)
        stability_score     = max(0.0, 1.0 - flip_rate * 10)
        responsiveness_score = (
            1.0 / (1.0 + detection_lag) if detection_lag is not None else 0.0
        )

        return {
            "window_size":         ws,
            "n_predictions":       n_predictions,
            "flip_count":          flip_count,
            "flip_rate":           round(flip_rate, 5),
            "mean_confidence":     round(mean_conf, 4),
            "detection_lag_ticks": detection_lag,
            "stability_score":     round(stability_score, 4),
            "responsiveness_score": round(responsiveness_score, 4),
            "composite_score":     round(
                0.5 * stability_score + 0.5 * responsiveness_score, 4),
        }

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def _print_results(self) -> None:
        header = (f"{'WS':>5} {'Preds':>6} {'Flips':>6} "
                  f"{'FlipRate':>9} {'MeanConf':>9} "
                  f"{'DetLag':>7} {'Stability':>10} {'Response':>10} {'Score':>7}")
        print(header)
        print("-" * len(header))
        for ws, r in sorted(self._results.items()):
            dl = str(r["detection_lag_ticks"]) if r["detection_lag_ticks"] is not None else "N/A"
            print(f"{ws:5d} {r['n_predictions']:6d} {r['flip_count']:6d} "
                  f"{r['flip_rate']:9.5f} {r['mean_confidence']:9.4f} "
                  f"{dl:>7} {r['stability_score']:10.4f} "
                  f"{r['responsiveness_score']:10.4f} {r['composite_score']:7.4f}")

    def best_window(self) -> int:
        """Return the window size with the highest composite score."""
        return max(self._results, key=lambda ws: self._results[ws]["composite_score"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_shift(ticks: list, from_tick: int) -> None:
        """Artificially drive prices down after from_tick to simulate a bear shift."""
        price = ticks[from_tick].price if from_tick < len(ticks) else 150.0
        for i in range(from_tick, len(ticks)):
            price *= 0.997          # −0.3% per tick compounding
            ticks[i].price = round(price, 4)
            ticks[i].bid   = round(price - 0.02, 4)
            ticks[i].ask   = round(price + 0.02, 4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    analyzer = WindowResponsivenessAnalyzer(seed=42)
    results  = analyzer.run()
    best_ws  = analyzer.best_window()
    print(f"\n★  Best window size (composite score): {best_ws}")
    print(f"   {results[best_ws]}")
