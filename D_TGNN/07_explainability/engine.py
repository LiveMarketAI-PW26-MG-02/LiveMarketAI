#!/usr/bin/env python3
"""
Dynamic Temporal Graph Neural Network (D-TGNN) - Explainability Engine (Alt Entry)
Implements XAI techniques for D-TGNN manipulation detection:
feature importance, attention visualization, counterfactual explanations,
graph substructure attribution, and natural language explanation generation.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import random

OUTPUT_DIR = "outputs/d_tgnn_xai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: FeatureImportanceExplainer
# ──────────────────────────────────────────────────────────────────────────────

class FeatureImportanceExplainer:
    """
    Computes feature importance scores using multiple XAI approaches:
    - Gradient-based saliency
    - Permutation importance
    - SHAP-inspired additive feature attribution
    """

    FEATURE_NAMES = [
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

    def __init__(self, model):
        self.model = model
        self.n_features = len(self.FEATURE_NAMES)

    def gradient_saliency(self, H: np.ndarray, A: np.ndarray,
                          node_idx: int, regime_idx: int = 0) -> np.ndarray:
        """
        Approximate gradient saliency by perturbing each feature
        and measuring change in risk score.
        gradient_i ≈ (score(H + eps_i) - score(H)) / eps
        """
        eps = 1e-3
        output_base = self.model.forward(H, A, regime_idx=regime_idx)
        base_score  = output_base["risk_scores"][node_idx]

        saliency = np.zeros(H.shape[1])
        for feat_idx in range(H.shape[1]):
            H_perturbed = H.copy()
            H_perturbed[node_idx, feat_idx] += eps
            output_p = self.model.forward(H_perturbed, A, regime_idx=regime_idx)
            pert_score = output_p["risk_scores"][node_idx]
            saliency[feat_idx] = (pert_score - base_score) / eps

        return np.abs(saliency)

    def permutation_importance(self, H: np.ndarray, A: np.ndarray,
                                y: np.ndarray, n_repeats: int = 5) -> np.ndarray:
        """
        Permutation feature importance: for each feature, shuffle its values
        across nodes and measure increase in loss (accuracy drop).
        """
        output_base = self.model.forward(H, A, regime_idx=0)
        base_scores = output_base["risk_scores"]
        base_loss   = float(np.mean((base_scores - y) ** 2))

        importances = np.zeros(H.shape[1])
        for feat_idx in range(H.shape[1]):
            feat_losses = []
            for _ in range(n_repeats):
                H_perm = H.copy()
                H_perm[:, feat_idx] = np.random.permutation(H_perm[:, feat_idx])
                output_p    = self.model.forward(H_perm, A, regime_idx=0)
                perm_scores = output_p["risk_scores"]
                feat_losses.append(float(np.mean((perm_scores - y) ** 2)))
            importances[feat_idx] = np.mean(feat_losses) - base_loss

        return np.maximum(importances, 0)

    def shap_additive(self, H: np.ndarray, A: np.ndarray,
                      node_idx: int) -> Dict[str, float]:
        """
        SHAP-inspired additive feature attribution using coalition sampling.
        phi_i = E[f(x) | S union {i}] - E[f(x) | S]
        Approximated via marginal contribution over random coalitions.
        """
        n_features  = H.shape[1]
        n_samples   = 20
        phi         = np.zeros(n_features)

        for feat_idx in range(n_features):
            marginals = []
            for _ in range(n_samples):
                # Random subset of features
                coalition = np.random.choice([True, False], size=n_features)

                # With feature
                H_with = H.copy()
                H_with[node_idx, ~coalition] = 0.0   # mask out non-coalition
                H_with[node_idx, feat_idx]   = H[node_idx, feat_idx]
                out_w  = self.model.forward(H_with, A, regime_idx=0)
                score_w = out_w["risk_scores"][node_idx]

                # Without feature
                H_without = H_with.copy()
                H_without[node_idx, feat_idx] = 0.0
                out_wo = self.model.forward(H_without, A, regime_idx=0)
                score_wo = out_wo["risk_scores"][node_idx]

                marginals.append(score_w - score_wo)

            phi[feat_idx] = np.mean(marginals)

        feat_names = self.FEATURE_NAMES[:n_features]
        return dict(zip(feat_names, phi.tolist()))

    def explain_node(self, H: np.ndarray, A: np.ndarray,
                     node_idx: int, y: np.ndarray,
                     regime_idx: int = 0) -> dict:
        """Generate comprehensive feature explanation for a single node."""
        saliency     = self.gradient_saliency(H, A, node_idx, regime_idx)
        perm_imp     = self.permutation_importance(H, A, y, n_repeats=3)
        shap_values  = self.shap_additive(H, A, node_idx)

        # Normalize
        n = H.shape[1]
        saliency_norm = saliency / (saliency.sum() + 1e-10)
        perm_norm     = perm_imp / (perm_imp.sum() + 1e-10)

        feat_names = self.FEATURE_NAMES[:n]
        combined   = {
            fname: {
                "saliency":     float(saliency_norm[i]),
                "permutation":  float(perm_norm[i]) if i < len(perm_norm) else 0.0,
                "shap":         float(shap_values.get(fname, 0.0)),
                "combined_imp": float((saliency_norm[i] + perm_norm[i]) / 2
                                       if i < len(perm_norm) else saliency_norm[i]),
            }
            for i, fname in enumerate(feat_names)
        }
        # Top features
        top_k = sorted(combined.items(), key=lambda x: x[1]["combined_imp"], reverse=True)[:5]
        return {
            "node_idx":      node_idx,
            "regime_idx":    regime_idx,
            "feature_importances": combined,
            "top_5_features": [{k: v} for k, v in top_k],
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: GraphSubstructureExplainer
# ──────────────────────────────────────────────────────────────────────────────

class GraphSubstructureExplainer:
    """
    Explains GNN decisions by identifying critical graph substructures.
    Implements GNNExplainer-inspired edge and node masking.
    """

    def __init__(self, model, n_iterations: int = 30):
        self.model       = model
        self.n_iters     = n_iterations

    def explain_edges(self, H: np.ndarray, A: np.ndarray,
                      target_node: int, regime_idx: int = 0) -> np.ndarray:
        """
        Find edge importance mask by optimizing:
        max MI(Y_v, GNN(H, A*M)) where M is learnable mask.
        Approximated via iterative soft masking.
        """
        N = H.shape[0]
        edge_mask = np.ones((N, N)) * 0.5  # Initialize soft mask

        output_ref = self.model.forward(H, A, regime_idx=regime_idx)
        ref_score  = output_ref["risk_scores"][target_node]

        for iteration in range(self.n_iters):
            # Gradient of score w.r.t. edge mask
            for i in range(N):
                for j in range(N):
                    if A[i, j] > 0:
                        A_masked = A * edge_mask
                        A_masked[i, j] *= 0.0  # Remove edge
                        out_m = self.model.forward(H, A_masked, regime_idx=regime_idx)
                        delta = ref_score - out_m["risk_scores"][target_node]
                        # Update mask toward preserving important edges
                        edge_mask[i, j] = min(1.0, max(0.0,
                            edge_mask[i, j] + 0.01 * delta))

        return edge_mask

    def get_important_subgraph(self, edge_mask: np.ndarray, A: np.ndarray,
                                threshold: float = 0.7) -> dict:
        """Extract subgraph with high importance edges."""
        important_edges = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0 and edge_mask[i, j] >= threshold:
                    important_edges.append({
                        "source": int(i),
                        "target": int(j),
                        "importance": float(edge_mask[i, j]),
                    })
        important_nodes = list(set(
            [e["source"] for e in important_edges] +
            [e["target"] for e in important_edges]
        ))
        return {
            "important_edges": important_edges,
            "important_nodes": important_nodes,
            "subgraph_size":   len(important_nodes),
            "n_important_edges": len(important_edges),
        }

    def motif_detection(self, A: np.ndarray) -> dict:
        """Detect graph motifs relevant to manipulation patterns."""
        N = A.shape[0]
        motifs = {
            "triangles": 0, "stars": 0,
            "chains": 0, "hubs": [],
        }

        degrees = A.sum(axis=1)
        motifs["hubs"] = [int(i) for i in np.where(degrees > degrees.mean() + degrees.std())[0]]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i < j < k and A[i,j] > 0 and A[j,k] > 0 and A[i,k] > 0:
                        motifs["triangles"] += 1

        # Stars: hub with many connections
        motifs["stars"] = len(motifs["hubs"])

        return motifs


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: CounterfactualExplainer
# ──────────────────────────────────────────────────────────────────────────────

class CounterfactualExplainer:
    """
    Generates counterfactual explanations:
    'What minimal change to features/graph would change the manipulation prediction?'
    """

    def __init__(self, model, threshold: float = 0.5):
        self.model     = model
        self.threshold = threshold

    def find_counterfactual(self, H: np.ndarray, A: np.ndarray,
                             node_idx: int, target_label: int = 0,
                             max_iters: int = 50,
                             step_size: float = 0.05) -> dict:
        """
        Find minimal feature perturbation that flips the prediction.
        Starts from the original features and perturbs toward lower/higher risk.
        """
        H_cf   = H.copy()
        output = self.model.forward(H_cf, A, regime_idx=0)
        orig_score  = float(output["risk_scores"][node_idx])
        orig_pred   = int(orig_score >= self.threshold)

        history = [orig_score]
        for step in range(max_iters):
            output_cf = self.model.forward(H_cf, A, regime_idx=0)
            score_cf  = float(output_cf["risk_scores"][node_idx])
            pred_cf   = int(score_cf >= self.threshold)
            history.append(score_cf)

            if pred_cf == target_label:
                break  # Found counterfactual

            # Perturbation direction: toward target label
            direction = -1 if target_label == 0 else +1
            noise     = np.random.randn(H_cf.shape[1]) * step_size
            H_cf[node_idx] += direction * noise

        output_final = self.model.forward(H_cf, A, regime_idx=0)
        final_score  = float(output_final["risk_scores"][node_idx])
        final_pred   = int(final_score >= self.threshold)

        # Compute delta
        delta = H_cf[node_idx] - H[node_idx]

        return {
            "original_score":     orig_score,
            "original_pred":      orig_pred,
            "counterfactual_score": final_score,
            "counterfactual_pred":  final_pred,
            "prediction_flipped": final_pred == target_label,
            "n_iterations":       len(history),
            "score_history":      [round(s, 4) for s in history],
            "feature_delta":      delta.tolist(),
            "l2_distance":        float(np.linalg.norm(delta)),
            "sparsity":           float((np.abs(delta) < 0.01).mean()),
        }

    def generate_diverse_counterfactuals(self, H: np.ndarray, A: np.ndarray,
                                          node_idx: int, n_cf: int = 5) -> List[dict]:
        """Generate multiple diverse counterfactuals for robustness."""
        counterfactuals = []
        for k in range(n_cf):
            # Vary step size for diversity
            step_size = 0.02 * (k + 1)
            cf = self.find_counterfactual(H, A, node_idx,
                                           target_label=0,
                                           step_size=step_size)
            cf["cf_index"] = k
            counterfactuals.append(cf)
        return counterfactuals


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: NaturalLanguageExplainer
# ──────────────────────────────────────────────────────────────────────────────

class NaturalLanguageExplainer:
    """
    Converts quantitative explanations into human-readable narratives
    for compliance officers and regulators.
    """

    MODULE_DESCRIPTION = "Dynamic Temporal Graph Neural Network (D-TGNN)"

    TEMPLATE = """
MANIPULATION DETECTION EXPLANATION
===================================
Module: {module}
Asset: {asset_id}
Detection Timestamp: {timestamp}

RISK ASSESSMENT
--------------
Risk Score: {risk_score:.4f} (Threshold: {threshold:.2f})
Severity: {severity}
Prediction: {prediction}

TOP CONTRIBUTING FACTORS
-----------------------
{top_factors_text}

GRAPH CONTEXT
-------------
{graph_context}

COUNTERFACTUAL INSIGHT
----------------------
{counterfactual_text}

RECOMMENDED ACTION
------------------
{action_text}

Explanation generated by D-TGNN XAI module.
Model version: 1.0.0 | Confidence: {confidence:.2%}
"""

    def generate_explanation(self, asset_id: str, risk_score: float,
                              top_features: Dict[str, float],
                              graph_info: dict,
                              cf_info: dict,
                              threshold: float = 0.5) -> str:
        """Generate natural language explanation for a manipulation alert."""
        severity = "CRITICAL" if risk_score > 0.85 else                    "HIGH" if risk_score > 0.70 else                    "MEDIUM" if risk_score > 0.55 else "LOW"
        prediction = "MANIPULATION DETECTED" if risk_score >= threshold else "NORMAL"

        # Format top features
        top_factors_text = ""
        for i, (fname, importance) in enumerate(
            sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5], 1
        ):
            direction = "↑ elevated" if importance > 0 else "↓ suppressed"
            readable  = fname.replace("_", " ").title()
            top_factors_text += f"  {i}. {readable}: {direction} (importance={abs(importance):.4f})\n"

        # Graph context
        n_imp_edges = graph_info.get("n_important_edges", 0)
        n_imp_nodes = graph_info.get("subgraph_size", 0)
        graph_context = (f"  Critical subgraph: {n_imp_nodes} nodes, "
                         f"{n_imp_edges} high-importance edges identified.\n"
                         f"  Manipulation motifs detected: {graph_info.get('motifs', {})}\n"
                         f"  Hub nodes involved: {graph_info.get('hubs', [])[:3]}")

        # Counterfactual
        if cf_info.get("prediction_flipped"):
            cf_text = (f"  If feature changes with L2 distance={cf_info['l2_distance']:.4f} "
                       f"were applied, the prediction would change from "
                       f"{cf_info['original_pred']} to {cf_info['counterfactual_pred']}.\n"
                       f"  Score would change: {cf_info['original_score']:.4f} → "
                       f"{cf_info['counterfactual_score']:.4f}")
        else:
            cf_text = "  No minimal counterfactual found within search budget."

        # Action
        action_map = {
            "CRITICAL": "IMMEDIATE REVIEW REQUIRED. Escalate to compliance team. "
                        "Consider trading halt pending investigation.",
            "HIGH":     "Urgent review recommended. Flag for surveillance team within 1 hour.",
            "MEDIUM":   "Schedule review within 24 hours. Monitor for pattern escalation.",
            "LOW":      "Log for trend analysis. No immediate action required.",
        }
        action_text = action_map.get(severity, "Monitor and review.")

        return self.TEMPLATE.format(
            module=self.MODULE_DESCRIPTION,
            asset_id=asset_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            risk_score=risk_score,
            threshold=threshold,
            severity=severity,
            prediction=prediction,
            top_factors_text=top_factors_text,
            graph_context=graph_context,
            counterfactual_text=cf_text,
            action_text=action_text,
            confidence=max(0, min(1, risk_score if risk_score >= threshold else 1 - risk_score)),
        )

    def save_explanation(self, text: str, asset_id: str,
                         filename: str = None) -> str:
        fname = filename or f"explanation_{asset_id}.txt"
        path  = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            f.write(text)
        return path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  D-TGNN - Explainability Engine (Alt Entry)")
    print(f"  Dynamic Temporal Graph Neural Network")
    print(f"{'='*60}\n")

    class SimpleModel:
        feature_dim = 16
        def forward(self, H, A, regime_idx=0):
            n = H.shape[0]
            scores = np.clip(np.mean(np.abs(H[:, :3]), axis=1) + np.random.rand(n)*0.2, 0, 1)
            emb    = np.random.randn(n, 16)
            return {"risk_scores": scores, "node_embeddings": emb,
                     "n_nodes": n, "regime_idx": regime_idx}

    n_nodes = 10
    H = np.random.randn(n_nodes, 16) * 0.5
    A = (np.random.rand(n_nodes, n_nodes) < 0.3).astype(float)
    np.fill_diagonal(A, 0)
    y = (np.random.rand(n_nodes) < 0.2).astype(float)
    model = SimpleModel()

    print("[1/5] Feature importance analysis...")
    feat_exp = FeatureImportanceExplainer(model)
    node_exp = feat_exp.explain_node(H, A, node_idx=0, y=y, regime_idx=0)
    print(f"      Top 3 features: {[list(x.keys())[0] for x in node_exp['top_5_features'][:3]]}")

    print("[2/5] Graph substructure explanation...")
    graph_exp   = GraphSubstructureExplainer(model, n_iterations=5)
    edge_mask   = graph_exp.explain_edges(H, A, target_node=0)
    subgraph    = graph_exp.get_important_subgraph(edge_mask, A)
    motifs      = graph_exp.motif_detection(A)
    print(f"      Important subgraph: {subgraph['subgraph_size']} nodes, "
          f"{subgraph['n_important_edges']} edges")

    print("[3/5] Counterfactual generation...")
    cf_exp = CounterfactualExplainer(model, threshold=0.5)
    cf     = cf_exp.find_counterfactual(H, A, node_idx=0, target_label=0)
    print(f"      CF flipped: {cf['prediction_flipped']} | L2: {cf['l2_distance']:.4f}")

    print("[4/5] Natural language explanation...")
    nl_exp = NaturalLanguageExplainer()
    output = model.forward(H, A)
    risk   = float(output["risk_scores"][0])
    shap_v = feat_exp.shap_additive(H, A, node_idx=0)
    graph_info = {**subgraph, "motifs": motifs, "hubs": motifs["hubs"][:3]}
    explanation_text = nl_exp.generate_explanation(
        "ASSET_000", risk, shap_v, graph_info, cf
    )
    exp_path = nl_exp.save_explanation(explanation_text, "ASSET_000", "explanation.txt")
    print(f"      Explanation saved → {exp_path}")

    print("[5/5] Saving JSON explanations...")
    xai_output = {
        "module":        "D-TGNN",
        "asset_id":      "ASSET_000",
        "risk_score":    risk,
        "node_explanation": node_exp,
        "graph_substructure": subgraph,
        "motifs":        motifs,
        "counterfactual": cf,
        "top_shap_features": sorted(shap_v.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
    }
    xai_path = os.path.join(OUTPUT_DIR, "xai_results.json")
    with open(xai_path, "w") as f:
        json.dump(xai_output, f, indent=2)
    print(f"      XAI results → {xai_path}")

    print(f"\nD-TGNN Explainability Engine (Alt Entry) Complete.\n")
    return node_exp, subgraph, cf, explanation_text

if __name__ == "__main__":
    main()
