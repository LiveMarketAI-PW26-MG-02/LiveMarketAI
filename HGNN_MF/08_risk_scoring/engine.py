#!/usr/bin/env python3
"""
Heterogeneous Graph Neural Fusion (HGNN-MF) - Risk Scoring Engine (Alt Entry)
Computes heterogeneous_fusion_score via ensemble of price, volume, sentiment, and graph scorers.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

OUTPUT_DIR = "outputs/hgnn_mf_risk"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

class ComponentScorer:
    """Base component scorer for HGNN-MF manipulation risk."""
    def __init__(self, weight: float = 0.25, name: str = "base"):
        self.weight = weight
        self.name   = name

    def score(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self):
        return f"ComponentScorer(name={self.name}, weight={self.weight})"

class PriceScorer(ComponentScorer):
    """Price anomaly risk scorer."""
    def __init__(self):
        super().__init__(0.30, "price")

    def score(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(0.0, index=df.index)
        if "log_return" in df.columns:
            lr = df["log_return"]
            z  = (lr - lr.mean()) / (lr.std() + 1e-10)
            s += np.clip(np.abs(z) / 3.0, 0, 1) * 0.4
        if "rolling_vol_5" in df.columns and "rolling_vol_20" in df.columns:
            vr = df["rolling_vol_5"] / (df["rolling_vol_20"] + 1e-10)
            s += np.clip((vr - 1.0) / 2.0, 0, 1) * 0.3
        if "momentum_5" in df.columns:
            s += np.clip(np.abs(df["momentum_5"]) / 0.05, 0, 1) * 0.3
        return np.clip(s, 0, 1)

class VolumeScorer(ComponentScorer):
    """Volume anomaly risk scorer."""
    def __init__(self):
        super().__init__(0.25, "volume")

    def score(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(0.0, index=df.index)
        if "volume_zscore" in df.columns:
            s += np.clip(np.abs(df["volume_zscore"]) / 2.5, 0, 1) * 0.6
        if "volume_spike_flag" in df.columns:
            s += df["volume_spike_flag"].astype(float) * 0.4
        return np.clip(s, 0, 1)

class SentimentScorer(ComponentScorer):
    """Sentiment-based manipulation risk scorer."""
    def __init__(self):
        super().__init__(0.25, "sentiment")

    def score(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(0.0, index=df.index)
        if "coordinated_post_ratio" in df.columns:
            s += np.clip(df["coordinated_post_ratio"] / 0.3, 0, 1) * 0.5
        if "sentiment_velocity" in df.columns:
            s += np.clip(np.abs(df["sentiment_velocity"]) / 0.2, 0, 1) * 0.5
        return np.clip(s, 0, 1)

class GraphScorer(ComponentScorer):
    """Graph centrality-based risk scorer."""
    def __init__(self):
        super().__init__(0.20, "graph")

    def score(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(0.0, index=df.index)
        if "betweenness_centrality" in df.columns:
            bc = df["betweenness_centrality"]
            s += np.clip((bc - bc.mean()) / (bc.std() + 1e-10) / 2.0, 0, 1) * 0.5
        if "pagerank" in df.columns:
            pr = df["pagerank"]
            s += np.clip((pr - pr.mean()) / (pr.std() + 1e-10) / 2.0, 0, 1) * 0.5
        return np.clip(s, 0, 1)

class EnsembleRiskScorer:
    """
    Ensemble scorer combining price, volume, sentiment, and graph signals.
    Module: HGNN-MF | Primary risk metric: heterogeneous_fusion_score
    """
    SEVERITY_BINS = [-0.001, 0.40, 0.55, 0.70, 0.85, 1.001]
    SEVERITY_LABELS = ["info", "low", "medium", "high", "critical"]

    def __init__(self):
        self.scorers = [PriceScorer(), VolumeScorer(), SentimentScorer(), GraphScorer()]

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        ensemble = pd.Series(0.0, index=df.index)
        for scorer in self.scorers:
            try:
                comp = scorer.score(df)
                result[f"risk_{scorer.name}"] = comp
                ensemble += scorer.weight * comp
            except Exception:
                pass
        result["heterogeneous_fusion_score"] = np.clip(ensemble, 0, 1)
        result["hgnn_mf_severity"] = pd.cut(
            result["heterogeneous_fusion_score"], bins=self.SEVERITY_BINS, labels=self.SEVERITY_LABELS
        ).astype(str)
        result["hgnn_mf_label"] = (result["heterogeneous_fusion_score"] >= 0.5).astype(int)
        return result

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Find F1-optimal threshold."""
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 33):
            p = (scores >= t).astype(int)
            tp = int(((p==1)&(labels==1)).sum())
            fp = int(((p==1)&(labels==0)).sum())
            fn = int(((p==0)&(labels==1)).sum())
            prec = tp/(tp+fp+1e-10); rec = tp/(tp+fn+1e-10)
            f1 = 2*prec*rec/(prec+rec+1e-10)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t

class RegimeAggregator:
    """Aggregates risk scores by market regime for HGNN-MF."""
    REGIMES = ["bull","bear","sideways","crisis","recovery"]

    def aggregate(self, df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        if "regime" not in df.columns:
            df = df.copy()
            df["regime"] = np.random.choice(self.REGIMES, len(df))
        return df.groupby("regime")[score_col].agg(
            mean_risk="mean", max_risk="max", n="count"
        ).reset_index()

class RegulatoryScorer:
    """Maps risk scores to regulatory violation categories."""
    VIOLATIONS = {
        "pump_and_dump": ["MAR Art 12(1)(a)", "SEC 10b-5"],
        "spoofing":      ["MAD Art 1(2)(c)", "Dodd-Frank 747"],
        "wash_trading":  ["MAR Art 12(1)(a)", "CEA 4c(a)(1)"],
    }

    def assess(self, alert_type: str, score: float) -> dict:
        viols = self.VIOLATIONS.get(alert_type, ["MAR Art 12"])
        return {
            "regulatory_score": min(1.0, score * 1.1),
            "sar_required":     score >= 0.75,
            "violations":       viols,
            "deadline_hours":   24 if score >= 0.75 else 72,
        }

def main():
    print(f"\n{'='*60}\n  HGNN-MF Risk Scoring\n{'='*60}")
    n = 50
    df = pd.DataFrame({
        "asset_id":               [f"ASSET_{i:03d}" for i in range(n)],
        "log_return":             np.random.randn(n)*0.02,
        "rolling_vol_5":          np.abs(np.random.randn(n))*0.015,
        "rolling_vol_20":         np.abs(np.random.randn(n))*0.010,
        "momentum_5":             np.random.randn(n)*0.04,
        "volume_zscore":          np.random.randn(n)*1.5,
        "volume_spike_flag":      (np.random.rand(n) < 0.1).astype(int),
        "avg_sentiment_1h":       np.random.uniform(-0.5, 0.8, n),
        "sentiment_velocity":     np.random.randn(n)*0.1,
        "coordinated_post_ratio": np.random.beta(2,8,n),
        "betweenness_centrality": np.random.beta(1,5,n),
        "pagerank":               np.random.dirichlet(np.ones(n)),
        "is_manipulation":        (np.random.rand(n) < 0.15).astype(int),
    })

    scorer    = EnsembleRiskScorer()
    scored_df = scorer.score_batch(df)
    print(f"  Mean {'heterogeneous_fusion_score'}: {scored_df['heterogeneous_fusion_score'].mean():.4f}")
    print(f"  High/Critical: {(scored_df['heterogeneous_fusion_score'] >= 0.7).sum()}")

    agg = RegimeAggregator()
    regime_stats = agg.aggregate(scored_df, 'heterogeneous_fusion_score')
    print(f"  Regime stats:\n{regime_stats.to_string(index=False)}")

    scored_df.to_csv(os.path.join(OUTPUT_DIR, "risk_scores.csv"), index=False)
    alerts = [
        {"alert_id": f"ALERT_{i:03d}", "asset_id": r["asset_id"],
          "risk_score": r["heterogeneous_fusion_score"], "severity": r["hgnn_mf_severity"], "module": "HGNN-MF"}
        for i, r in scored_df[scored_df["hgnn_mf_label"]==1].iterrows()
    ]
    with open(os.path.join(OUTPUT_DIR, "alerts.json"), "w") as f:
        json.dump({"module":"HGNN-MF","alerts":alerts}, f, indent=2)
    print(f"  Saved: risk_scores.csv + alerts.json ({len(alerts)} alerts)")
    print(f"\nHGNN-MF Risk Scoring Complete.\n")

if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED RISK SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveThresholdManager:
    """Dynamically adjusts detection thresholds based on market conditions."""

    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold
        self.history: List[float] = []

    def update(self, new_score: float) -> None:
        self.history.append(new_score)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def adaptive_threshold(self, regime: str = "normal") -> float:
        regime_adjustments = {
            "crisis":   -0.10,  # lower threshold = more sensitive in crisis
            "bear":     -0.05,
            "bull":     +0.05,  # higher threshold = less sensitive in bull
            "sideways":  0.00,
            "recovery": -0.03,
        }
        adj = regime_adjustments.get(regime, 0.0)
        if self.history:
            recent_mean = np.mean(self.history[-50:])
            if recent_mean > 0.7:
                adj -= 0.05  # reduce FP rate when many high scores
        return float(np.clip(self.base_threshold + adj, 0.2, 0.9))


class RiskScoreCalibrator:
    """Calibrates raw GNN scores to well-calibrated probabilities."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.a = 1.0
        self.b = 0.0

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling parameters."""
        if len(raw_scores) < 10:
            return
        best_loss = float("inf")
        for a in np.linspace(0.5, 2.0, 20):
            for b in np.linspace(-1.0, 1.0, 20):
                cal = 1.0 / (1.0 + np.exp(-(a * raw_scores + b)))
                cal = np.clip(cal, 1e-7, 1 - 1e-7)
                loss = -np.mean(labels * np.log(cal) + (1 - labels) * np.log(1 - cal))
                if loss < best_loss:
                    best_loss = loss
                    self.a, self.b = a, b

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.clip(1.0 / (1.0 + np.exp(-(self.a * raw_scores + self.b))), 0, 1)

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.fit(scores, labels)
        return self.transform(scores)


class AlertDeduplicator:
    """Deduplicates alerts to prevent alert fatigue."""

    def __init__(self, cooldown_steps: int = 10, score_delta_threshold: float = 0.05):
        self.cooldown      = cooldown_steps
        self.delta         = score_delta_threshold
        self._last_alert: Dict[str, dict] = {}
        self._step         = 0

    def should_alert(self, asset_id: str, score: float) -> bool:
        self._step += 1
        last = self._last_alert.get(asset_id)
        if last is None:
            self._last_alert[asset_id] = {"step": self._step, "score": score}
            return score >= 0.4

        steps_since = self._step - last["step"]
        score_delta = score - last["score"]

        if steps_since >= self.cooldown or score_delta >= self.delta:
            self._last_alert[asset_id] = {"step": self._step, "score": score}
            return score >= 0.4
        return False

    def stats(self) -> dict:
        return {"total_assets": len(self._last_alert),
                "current_step": self._step,
                "cooldown":     self.cooldown}


class PortfolioRiskAggregator:
    """Aggregates individual asset risks to portfolio-level manipulation risk."""

    def portfolio_var(self, scores: np.ndarray, confidence: float = 0.95) -> float:
        """Value-at-Risk analogue for manipulation risk."""
        return float(np.percentile(scores, 100 * confidence))

    def concentration_risk(self, scores: np.ndarray, asset_ids: list) -> dict:
        """Herfindahl-Hirschman Index for manipulation risk concentration."""
        if scores.sum() == 0:
            return {"hhi": 0.0, "top_5_pct": 0.0}
        shares = scores / (scores.sum() + 1e-10)
        hhi    = float((shares ** 2).sum())
        top5   = float(shares[np.argsort(shares)[::-1][:5]].sum())
        top_assets = [asset_ids[i] for i in np.argsort(scores)[::-1][:3]]
        return {"hhi": round(hhi, 4), "top_5_pct": round(top5, 4),
                "top_3_assets": top_assets[:3]}

    def compute_all(self, scored_df: pd.DataFrame, score_col: str) -> dict:
        scores = scored_df[score_col].values
        assets = scored_df.get("asset_id", pd.Series(range(len(scored_df)))).tolist()
        return {
            "var_95":           self.portfolio_var(scores),
            "var_99":           self.portfolio_var(scores, 0.99),
            "mean_risk":        float(scores.mean()),
            "max_risk":         float(scores.max()),
            "systemic_index":   float(np.percentile(scores, 90)),
            "concentration":    self.concentration_risk(scores, assets),
            "n_critical":       int((scores >= 0.85).sum()),
            "n_high":           int((scores >= 0.70).sum()),
        }


class TimeSeriesRiskTracker:
    """Tracks risk score evolution over time for trend analysis."""

    def __init__(self, window: int = 20):
        self.window  = window
        self.history: Dict[str, List[float]] = {}

    def update(self, asset_id: str, score: float) -> dict:
        if asset_id not in self.history:
            self.history[asset_id] = []
        self.history[asset_id].append(score)
        if len(self.history[asset_id]) > self.window * 3:
            self.history[asset_id] = self.history[asset_id][-self.window * 3:]
        h = np.array(self.history[asset_id])
        trend = float(np.polyfit(range(len(h)), h, 1)[0]) if len(h) >= 2 else 0.0
        return {
            "asset_id":    asset_id,
            "current":     score,
            "rolling_mean": float(np.mean(h[-self.window:])) if len(h) >= self.window else float(h.mean()),
            "rolling_max":  float(np.max(h[-self.window:]))  if len(h) >= self.window else float(h.max()),
            "trend":       round(trend, 6),
            "acceleration": round(float(h[-1] - h[-2]) if len(h) >= 2 else 0.0, 6),
        }

    def escalating_assets(self, threshold: float = 0.003) -> list:
        return [aid for aid, scores in self.history.items()
                if len(scores) >= 3 and
                np.polyfit(range(len(scores[-10:])), scores[-10:], 1)[0] > threshold]

