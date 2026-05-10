#!/usr/bin/env python3
"""
Microstructure Graph Neural Network (MGNN) - Inference Engine (Alt Entry)
Real-time (simulated) inference pipeline for manipulation detection.
Produces risk scores, alerts, and explanations from trained model.
"""

import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import random

OUTPUT_DIR = "outputs/mgnn_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: FeatureExtractor (Inference-time)
# ──────────────────────────────────────────────────────────────────────────────

class InferenceFeatureExtractor:
    """
    Extracts real-time feature vectors from incoming market data streams
    for MGNN inference. Handles normalization using training statistics.
    """

    def __init__(self, feature_stats_path: Optional[str] = None):
        # Default normalization stats (would be loaded from training)
        self.feature_means = np.zeros(32)
        self.feature_stds  = np.ones(32)
        self.feature_names = [
            "log_return", "rolling_vol_5", "rolling_vol_20",
            "momentum_5", "momentum_20", "price_range_pct",
            "volume_log", "volume_zscore", "vol_ma_ratio",
            "vwap_deviation", "volume_spike_flag",
            "avg_sentiment", "sentiment_vol", "coord_post_ratio",
            "sentiment_velocity", "degree_centrality",
            "betweenness_centrality", "clustering_coeff", "pagerank",
            "bid_ask_spread", "order_imbalance", "depth_total_log",
            "trade_intensity", "cancel_rate", "quote_to_trade",
            "large_order_flag", "cross_asset_momentum",
            "regime_confidence", "regime_transition_prob",
            "news_velocity", "social_volume", "dark_pool_ratio",
        ]

    def extract_from_ohlcv(self, ohlcv_row: dict) -> np.ndarray:
        """Extract features from a single OHLCV row."""
        close  = ohlcv_row.get("close", 100.0)
        prev   = ohlcv_row.get("prev_close", close)
        volume = ohlcv_row.get("volume", 1000)
        high   = ohlcv_row.get("high", close * 1.01)
        low    = ohlcv_row.get("low", close * 0.99)

        log_ret  = np.log(close / prev) if prev > 0 else 0.0
        vol_log  = np.log1p(volume)
        pr_range = (high - low) / close if close > 0 else 0.0

        features = np.zeros(len(self.feature_names))
        feature_map = {
            "log_return": log_ret,
            "rolling_vol_5": abs(log_ret) * 1.2,
            "rolling_vol_20": abs(log_ret) * 0.8,
            "momentum_5": log_ret * 3,
            "momentum_20": log_ret * 10,
            "price_range_pct": pr_range,
            "volume_log": vol_log,
            "volume_zscore": (volume - 5000) / 2000,
            "vol_ma_ratio": volume / 5000,
            "vwap_deviation": log_ret * 0.5,
            "volume_spike_flag": float(abs(volume - 5000) > 3 * 2000),
        }
        for i, fname in enumerate(self.feature_names):
            features[i] = feature_map.get(fname, np.random.randn() * 0.1)

        return features

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using training statistics."""
        return (features - self.feature_means) / (self.feature_stds + 1e-10)

    def build_feature_matrix(self, asset_data: List[dict]) -> np.ndarray:
        """Build N × F feature matrix from list of asset dicts."""
        rows = []
        for asset in asset_data:
            feat = self.extract_from_ohlcv(asset)
            feat = self.normalize(feat)
            rows.append(feat)
        return np.array(rows) if rows else np.zeros((1, len(self.feature_names)))


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: GraphBuilder (Inference-time)
# ──────────────────────────────────────────────────────────────────────────────

class InferenceGraphBuilder:
    """
    Builds dynamic adjacency matrix for the current inference window.
    Uses rolling correlation or predefined sector-based connectivity.
    """

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def build_correlation_adjacency(self, returns_history: np.ndarray,
                                    threshold: float = 0.4) -> np.ndarray:
        """Compute correlation-based adjacency from recent returns."""
        if returns_history.shape[0] < 2:
            # Default: sparse graph
            A = np.eye(self.n_assets)
            return A

        corr = np.corrcoef(returns_history.T)
        A    = (np.abs(corr) > threshold).astype(float)
        np.fill_diagonal(A, 0)
        return A

    def build_sector_adjacency(self, sectors: List[str]) -> np.ndarray:
        """Build adjacency based on sector membership."""
        A = np.zeros((len(sectors), len(sectors)))
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                if i != j and s1 == s2:
                    A[i, j] = 1.0
        return A

    def build_dynamic_adjacency(self, returns_history: np.ndarray,
                                 sectors: List[str],
                                 alpha: float = 0.7) -> np.ndarray:
        """Blend correlation-based and sector-based adjacency."""
        A_corr   = self.build_correlation_adjacency(returns_history)
        A_sector = self.build_sector_adjacency(sectors)
        n = min(A_corr.shape[0], A_sector.shape[0], self.n_assets)
        A_blended = alpha * A_corr[:n, :n] + (1 - alpha) * A_sector[:n, :n]
        A_blended = np.clip(A_blended, 0, 1)
        np.fill_diagonal(A_blended, 0)
        return A_blended


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: AlertGenerator
# ──────────────────────────────────────────────────────────────────────────────

class AlertGenerator:
    """
    Generates structured manipulation alerts from GNN risk scores.
    Implements tiered severity classification and deduplication.
    """

    SEVERITY_THRESHOLDS = {
        "critical": 0.85,
        "high":     0.70,
        "medium":   0.55,
        "low":      0.40,
    }
    ALERT_TYPES_BY_MODULE = {
        "RAGNN":    ["regime_anomaly", "cross_regime_manipulation"],
        "HGNN-MF":  ["multimodal_pump", "cross_source_manipulation"],
        "D-TGNN":   ["temporal_pump_pattern", "manipulation_sequence"],
        "CGNN-XAI": ["counterfactual_anomaly", "edge_manipulation"],
        "MGNN":     ["spoofing", "layering", "quote_stuffing"],
        "EP-GNN":   ["coordinated_sentiment", "bot_pump_campaign"],
        "RD-GNN":   ["risk_contagion", "cascade_manipulation"],
        "MSGNN":    ["multiscale_pattern", "cross_timeframe_manipulation"],
        "UGNN":     ["high_uncertainty_alert", "ood_manipulation"],
        "GRL-Fusion": ["fused_manipulation_signal", "multi_module_consensus"],
    }

    def __init__(self, module_name: str = "MGNN"):
        self.module_name  = module_name
        self.alert_types  = self.ALERT_TYPES_BY_MODULE.get(module_name, ["generic_alert"])
        self.alert_count  = 0
        self._alert_cache: Dict[str, float] = {}  # deduplication

    def classify_severity(self, risk_score: float) -> str:
        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if risk_score >= threshold:
                return severity
        return "info"

    def generate_alert(self, asset_id: str, risk_score: float,
                       features: dict, timestamp: str = None) -> Optional[dict]:
        """Generate a single alert if risk threshold exceeded."""
        if risk_score < self.SEVERITY_THRESHOLDS["low"]:
            return None

        # Deduplication: suppress if recent alert for same asset
        cache_key = f"{asset_id}_{name}"
        if cache_key in self._alert_cache:
            if risk_score - self._alert_cache[cache_key] < 0.05:
                return None

        self._alert_cache[cache_key] = risk_score
        self.alert_count += 1
        severity  = self.classify_severity(risk_score)
        alert_type = random.choice(self.alert_types)
        ts = timestamp or datetime.utcnow().isoformat() + "Z"

        return {
            "alert_id":    "ALERT_" + str(self.module_name) + "_" + str(self.alert_count).zfill(5),
            "module":      self.module_name,
            "asset_id":    asset_id,
            "alert_type":  alert_type,
            "severity":    severity,
            "risk_score":  round(float(risk_score), 4),
            "timestamp":   ts,
            "description": self._generate_description(alert_type, asset_id, risk_score),
            "features":    {k: round(float(v), 4) for k, v in features.items()},
            "action_required": severity in ["high", "critical"],
            "regulatory_flag": severity == "critical",
        }

    def _generate_description(self, alert_type: str, asset_id: str,
                               risk_score: float) -> str:
        descriptions = {
            "regime_anomaly":         f"Unusual cross-regime behavior detected in {asset_id} (score={risk_score:.3f})",
            "spoofing":               f"Spoofing pattern: large orders placed and cancelled in {asset_id}",
            "layering":               f"Layering detected: systematic multi-level order placement in {asset_id}",
            "quote_stuffing":         f"Quote stuffing: abnormal order submission rate in {asset_id}",
            "coordinated_sentiment":  f"Coordinated social media campaign targeting {asset_id}",
            "risk_contagion":         f"Risk contagion spreading from {asset_id} to correlated assets",
            "temporal_pump_pattern":  f"Historical pump-and-dump sequence detected in {asset_id}",
            "fused_manipulation_signal": f"Multi-module consensus manipulation signal for {asset_id}",
        }
        return descriptions.get(alert_type,
               f"{alert_type.replace('_',' ').title()} detected in {asset_id} with score {risk_score:.3f}")

    def save_alerts(self, alerts: List[dict], filename: str = "alerts.json") -> str:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            json.dump({"module": "MGNN", "alerts": alerts,
                        "total": len(alerts),
                        "generated_at": datetime.utcnow().isoformat() + "Z"},
                      f, indent=2)
        print(f"  Alerts saved → {path} ({len(alerts)} alerts)")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: InferencePipeline
# ──────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end MGNN inference pipeline:
    Data → Features → Graph → GNN → Risk Scores → Alerts → Output
    """

    def __init__(self, model, n_assets: int = 20, threshold: float = 0.5):
        self.model           = model
        self.n_assets        = n_assets
        self.threshold       = threshold
        self.feature_extractor = InferenceFeatureExtractor()
        self.graph_builder   = InferenceGraphBuilder(n_assets)
        self.alert_generator = AlertGenerator("MGNN")
        self.inference_count = 0

    def run_single_window(self, market_snapshot: List[dict],
                          returns_history: Optional[np.ndarray] = None) -> dict:
        """
        Run inference on a single market window.
        market_snapshot: list of asset dicts with OHLCV data
        """
        self.inference_count += 1
        ts = datetime.utcnow().isoformat() + "Z"

        n = min(len(market_snapshot), self.n_assets)
        if n == 0:
            return {"status": "no_data", "alerts": []}

        # Feature extraction
        H = self.feature_extractor.build_feature_matrix(market_snapshot[:n])

        # Truncate to model's expected feature dim
        feature_dim = getattr(self.model, "feature_dim", 32)
        if H.shape[1] > feature_dim:
            H = H[:, :feature_dim]
        elif H.shape[1] < feature_dim:
            H = np.pad(H, ((0,0), (0, feature_dim - H.shape[1])))

        # Graph construction
        sectors = [a.get("sector", "Technology") for a in market_snapshot[:n]]
        if returns_history is not None and returns_history.shape[0] >= 5:
            A = self.graph_builder.build_dynamic_adjacency(
                returns_history[-20:], sectors
            )
        else:
            A = (np.random.rand(n, n) < 0.3).astype(float)
            np.fill_diagonal(A, 0)

        A = A[:n, :n]
        H = H[:n]

        # GNN inference
        regime_idx = self.inference_count % 5
        output = self.model.forward(H, A, regime_idx=regime_idx)
        risk_scores = output["risk_scores"]

        # Generate alerts
        alerts = []
        for i, (asset_data, score) in enumerate(zip(market_snapshot[:n], risk_scores)):
            asset_id = asset_data.get("asset_id", f"ASSET_{i:03d}")
            feat_dict = {
                "log_return": asset_data.get("log_return", 0.0),
                "volume_zscore": asset_data.get("volume_zscore", 0.0),
                "sentiment": asset_data.get("sentiment", 0.0),
                "risk_score": float(score),
            }
            alert = self.alert_generator.generate_alert(
                asset_id, float(score), feat_dict, timestamp=ts
            )
            if alert:
                alerts.append(alert)

        return {
            "inference_id":  self.inference_count,
            "timestamp":     ts,
            "n_assets":      n,
            "regime_idx":    regime_idx,
            "risk_scores":   risk_scores.tolist(),
            "alerts":        alerts,
            "n_flagged":     int((risk_scores >= self.threshold).sum()),
            "max_risk":      float(risk_scores.max()),
            "mean_risk":     float(risk_scores.mean()),
            "embeddings_shape": list(output["node_embeddings"].shape),
        }

    def run_streaming(self, n_windows: int = 50,
                      n_assets_per_window: int = 15) -> List[dict]:
        """Simulate streaming inference over multiple time windows."""
        print(f"  Running MGNN streaming inference: {n_windows} windows...")
        results = []
        returns_history = np.random.randn(20, n_assets_per_window) * 0.01

        for w in range(n_windows):
            snapshot = [{
                "asset_id": f"ASSET_{i:03d}",
                "close": 100 * (1 + np.random.randn() * 0.01),
                "prev_close": 100.0,
                "high": 101.0 + np.random.rand(),
                "low":  99.0 - np.random.rand(),
                "volume": int(5000 + np.random.randn() * 1500),
                "sector": ["Technology","Finance","Energy"][i % 3],
                "sentiment": float(np.random.uniform(-0.5, 0.8)),
                "log_return": float(np.random.randn() * 0.01),
                "volume_zscore": float(np.random.randn()),
            } for i in range(n_assets_per_window)]

            result = self.run_single_window(snapshot, returns_history)
            results.append(result)

            # Simulate pump event
            if w == 20:
                for i in range(3):
                    results[-1]["risk_scores"][i] = min(0.95, results[-1]["risk_scores"][i] + 0.5)

            returns_history = np.vstack([returns_history[1:],
                                          np.random.randn(1, n_assets_per_window) * 0.01])

            if (w + 1) % 10 == 0:
                total_alerts = sum(len(r["alerts"]) for r in results)
                print(f"    Window {w+1}/{n_windows} | Total alerts: {total_alerts}")

        return results

    def aggregate_results(self, results: List[dict]) -> pd.DataFrame:
        """Aggregate streaming results into a DataFrame."""
        rows = []
        for r in results:
            for i, score in enumerate(r.get("risk_scores", [])):
                rows.append({
                    "inference_id": r["inference_id"],
                    "timestamp":    r["timestamp"],
                    "asset_idx":    i,
                    "risk_score":   score,
                    "regime_idx":   r["regime_idx"],
                    "max_risk":     r["max_risk"],
                })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  MGNN - Inference Engine (Alt Entry)")
    print(f"  Microstructure Graph Neural Network")
    print(f"{'='*60}\n")

    # Build simple model for inference
    class SimpleModel:
        feature_dim = 32
        def forward(self, H, A, regime_idx=0):
            n = H.shape[0]
            noise    = np.random.rand(n) * 0.3
            base     = np.mean(np.abs(H[:, :3]), axis=1)
            scores   = np.clip(base / (base.max() + 1e-10) + noise, 0, 1)
            emb      = np.random.randn(n, 32)
            return {"risk_scores": scores, "node_embeddings": emb,
                     "n_nodes": n, "regime_idx": regime_idx}

    model = SimpleModel()
    pipeline = InferencePipeline(model, n_assets=15, threshold=0.5)

    # Run streaming inference
    results = pipeline.run_streaming(n_windows=30, n_assets_per_window=12)

    # Collect all alerts
    all_alerts = []
    for r in results:
        all_alerts.extend(r["alerts"])

    print(f"\n  Total inference windows: {len(results)}")
    print(f"  Total alerts generated:  {len(all_alerts)}")

    # Save outputs
    pipeline.alert_generator.save_alerts(all_alerts)

    result_df = pipeline.aggregate_results(results)
    scores_path = os.path.join(OUTPUT_DIR, "risk_scores.csv")
    result_df.to_csv(scores_path, index=False)
    print(f"  Risk scores CSV → {scores_path}")

    # Summary
    if len(all_alerts) > 0:
        sev_counts = pd.Series([a["severity"] for a in all_alerts]).value_counts()
        print(f"  Alert severity breakdown: {sev_counts.to_dict()}")

    print(f"\nMGNN Inference Engine (Alt Entry) Complete.\n")
    return results, all_alerts

if __name__ == "__main__":
    main()
