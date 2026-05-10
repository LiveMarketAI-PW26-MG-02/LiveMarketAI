#!/usr/bin/env python3
"""
Counterfactual Graph Neural Network (CGNN-XAI) - GNN Model — model.py Entry
Implements the CounterfactualGNN graph neural network architecture
for financial manipulation detection.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import random

OUTPUT_DIR = "outputs/cgnn_xai_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# GNN LAYER IMPLEMENTATIONS (NumPy-based, no deep learning framework required)
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer:
    """
    Graph Convolutional Layer (GCN-style) implemented with NumPy.
    Propagates: H' = sigma(D^-1/2 A D^-1/2 H W)
    Used as a base building block in CGNN-XAI.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation
        self.use_bias     = bias
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features) if bias else None

    def normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Symmetric normalization: D^-1/2 A D^-1/2"""
        A_hat = A + np.eye(A.shape[0])
        d     = A_hat.sum(axis=1)
        d_inv = np.where(d > 0, d ** -0.5, 0)
        D_inv = np.diag(d_inv)
        return D_inv @ A_hat @ D_inv

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass through GCN layer."""
        A_norm = self.normalize_adjacency(A)
        Z = A_norm @ H @ self.W
        if self.use_bias:
            Z = Z + self.b
        return self._activate(Z)

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "tanh":
            return np.tanh(Z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -50, 50)))
        elif self.activation == "leaky_relu":
            return np.where(Z > 0, Z, 0.01 * Z)
        return Z  # linear


class GraphAttentionLayer:
    """
    Graph Attention Layer (GAT-style) with multi-head attention.
    Critical for CGNN-XAI to learn which graph neighbors matter most.
    H'_i = sigma( sum_j alpha_ij * W * h_j )
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, dropout_rate: float = 0.1):
        self.in_features  = in_features
        self.out_features = out_features
        self.n_heads      = n_heads
        self.dropout_rate = dropout_rate
        head_dim          = out_features // n_heads

        self.W_heads  = [
            np.random.randn(in_features, head_dim) * np.sqrt(2.0 / (in_features + head_dim))
            for _ in range(n_heads)
        ]
        self.a_heads  = [
            np.random.randn(2 * head_dim) * 0.01
            for _ in range(n_heads)
        ]

    def attention_scores(self, H: np.ndarray, A: np.ndarray,
                         head_idx: int) -> np.ndarray:
        """Compute attention coefficient matrix for one head."""
        W = self.W_heads[head_idx]
        a = self.a_heads[head_idx]
        Wh = H @ W                             # (N, head_dim)
        N  = H.shape[0]

        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    concat   = np.concatenate([Wh[i], Wh[j]])
                    raw_e    = np.dot(a, concat)
                    e[i, j]  = max(0.01 * raw_e, raw_e)  # LeakyReLU

        # Mask non-edges with -inf before softmax
        mask       = (A == 0) & ~np.eye(N, dtype=bool)
        e[mask]    = -1e9
        alpha      = self._softmax_rows(e)
        return alpha

    def _softmax_rows(self, X: np.ndarray) -> np.ndarray:
        X_max  = X.max(axis=1, keepdims=True)
        exp_X  = np.exp(X - X_max)
        return exp_X / (exp_X.sum(axis=1, keepdims=True) + 1e-10)

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Multi-head attention forward pass."""
        head_outputs = []
        for h in range(self.n_heads):
            alpha  = self.attention_scores(H, A, h)
            Wh     = H @ self.W_heads[h]
            output = alpha @ Wh
            head_outputs.append(output)
        # Concatenate (or average) heads
        concatenated = np.concatenate(head_outputs, axis=1)
        # Truncate or pad to out_features if needed
        if concatenated.shape[1] > self.out_features:
            concatenated = concatenated[:, :self.out_features]
        return np.maximum(0, concatenated)  # ReLU activation


class RegimeAwareMessagePassing:
    """
    Regime-conditioned message passing layer.
    Edge weights are modulated by the current market regime embedding.
    Core innovation of CGNN-XAI: different regimes require different attention patterns.
    """

    def __init__(self, feature_dim: int, regime_dim: int, n_regimes: int = 5):
        self.feature_dim = feature_dim
        self.regime_dim  = regime_dim
        self.n_regimes   = n_regimes

        # Per-regime weight matrices
        self.regime_W = [
            np.random.randn(feature_dim, feature_dim) * 0.1
            for _ in range(n_regimes)
        ]
        # Regime embedding table
        self.regime_embeddings = np.random.randn(n_regimes, regime_dim) * 0.1

    def get_regime_embedding(self, regime_idx: int) -> np.ndarray:
        return self.regime_embeddings[regime_idx % self.n_regimes]

    def regime_modulated_propagation(self, H: np.ndarray, A: np.ndarray,
                                     regime_idx: int) -> np.ndarray:
        """
        Message passing with regime-specific weight matrix.
        H' = sigma(A_norm * H * W_regime)
        """
        W = self.regime_W[regime_idx % self.n_regimes]
        # Normalize adjacency
        d    = A.sum(axis=1) + 1e-10
        D_inv = np.diag(1.0 / d)
        A_norm = D_inv @ A

        H_new = A_norm @ H @ W
        return np.tanh(H_new)

    def compute_regime_aware_edge_weights(self, A: np.ndarray,
                                          regime_embedding: np.ndarray) -> np.ndarray:
        """
        Dynamically update edge weights based on regime embedding.
        Regime-specific correlation strengths vary by market state.
        """
        regime_scalar = np.dot(regime_embedding, regime_embedding) / (
            np.linalg.norm(regime_embedding) ** 2 + 1e-10
        )
        # Scale existing edges
        A_weighted = A * (0.5 + 0.5 * regime_scalar)
        return A_weighted


class TemporalGraphLayer:
    """
    Temporal aggregation layer combining GRU-style memory with graph propagation.
    Captures evolving manipulation patterns over time in CGNN-XAI.
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim

        # GRU weight matrices
        self.W_z = np.random.randn(feature_dim + hidden_dim, hidden_dim) * 0.1  # update gate
        self.W_r = np.random.randn(feature_dim + hidden_dim, hidden_dim) * 0.1  # reset gate
        self.W_h = np.random.randn(feature_dim + hidden_dim, hidden_dim) * 0.1  # new hidden

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def gru_step(self, h_prev: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Single GRU step for temporal state update."""
        combined = np.concatenate([h_prev, x], axis=-1)
        z = self.sigmoid(combined @ self.W_z)        # update gate
        r = self.sigmoid(combined @ self.W_r)        # reset gate
        combined_r = np.concatenate([r * h_prev, x], axis=-1)
        h_tilde = np.tanh(combined_r @ self.W_h)    # candidate hidden
        h_new   = (1 - z) * h_prev + z * h_tilde    # final hidden
        return h_new

    def forward_sequence(self, H_sequence: List[np.ndarray],
                         A: np.ndarray) -> np.ndarray:
        """Process a sequence of graph snapshots."""
        N     = H_sequence[0].shape[0]
        h     = np.zeros((N, self.hidden_dim))

        graph_conv = GraphConvLayer(self.hidden_dim, self.hidden_dim)

        for H_t in H_sequence:
            # Project input to hidden_dim if needed
            if H_t.shape[1] != self.hidden_dim:
                # Simple projection
                proj_W = np.random.randn(H_t.shape[1], self.hidden_dim) * 0.1
                H_t_proj = H_t @ proj_W
            else:
                H_t_proj = H_t

            # GRU update per node
            for i in range(N):
                h[i] = self.gru_step(h[i], H_t_proj[i])

            # Graph propagation on updated hidden state
            h = graph_conv.forward(h, A)

        return h


# ──────────────────────────────────────────────────────────────────────────────
# CLASS: Main GNN Model
# ──────────────────────────────────────────────────────────────────────────────

class CounterfactualGNN:
    """
    Counterfactual Graph Neural Network (CGNN-XAI) - Main GNN Model
    Multi-layer graph neural network with module-specific innovations
    for financial market manipulation detection.

    Architecture:
      1. Input feature projection
      2. Module-specific message passing layers
      3. Temporal aggregation (optional)
      4. Risk scoring head
      5. Explanation extraction
    """

    def __init__(self, config: dict):
        self.config        = config
        self.feature_dim   = config.get("feature_dim", 32)
        self.hidden_dim    = config.get("hidden_dim", 64)
        self.n_layers      = config.get("n_layers", 3)
        self.n_regimes     = config.get("n_regimes", 5)
        self.dropout_rate  = config.get("dropout_rate", 0.1)
        self.n_heads       = config.get("n_heads", 4)
        self.use_temporal  = config.get("use_temporal", True)
        self._build_layers()

    def _build_layers(self):
        """Construct GNN layer stack."""
        self.input_proj = np.random.randn(self.feature_dim, self.hidden_dim) * 0.1

        self.gcn_layers = [
            GraphConvLayer(self.hidden_dim, self.hidden_dim,
                           activation="relu" if i < self.n_layers - 1 else "tanh")
            for i in range(self.n_layers)
        ]
        self.gat_layer  = GraphAttentionLayer(
            self.hidden_dim, self.hidden_dim, n_heads=self.n_heads
        )
        self.regime_mp  = RegimeAwareMessagePassing(
            self.hidden_dim, 16, n_regimes=self.n_regimes
        )
        if self.use_temporal:
            self.temporal   = TemporalGraphLayer(self.feature_dim, self.hidden_dim)

        # Risk scoring head: hidden_dim → 1
        self.score_W    = np.random.randn(self.hidden_dim, 1) * 0.1
        self.score_b    = np.zeros(1)

    def forward(self, H: np.ndarray, A: np.ndarray,
                regime_idx: int = 0,
                H_sequence: Optional[List[np.ndarray]] = None) -> dict:
        """
        Full forward pass through the CGNN-XAI GNN.

        Args:
            H: Node feature matrix (N × feature_dim)
            A: Adjacency matrix (N × N)
            regime_idx: Current market regime index
            H_sequence: Optional temporal sequence of feature matrices

        Returns:
            Dictionary with node embeddings, risk scores, attention weights
        """
        N = H.shape[0]

        # 1. Input projection
        if H.shape[1] != self.hidden_dim:
            if H.shape[1] < self.feature_dim:
                H_padded = np.pad(H, ((0,0), (0, self.feature_dim - H.shape[1])))
            else:
                H_padded = H[:, :self.feature_dim]
            H_proj = H_padded @ self.input_proj
        else:
            H_proj = H

        # 2. Regime-aware message passing
        regime_embed = self.regime_mp.get_regime_embedding(regime_idx)
        A_weighted   = self.regime_mp.compute_regime_aware_edge_weights(A, regime_embed)
        H_regime     = self.regime_mp.regime_modulated_propagation(H_proj, A_weighted, regime_idx)

        # 3. Stack GCN layers
        H_curr = H_regime
        layer_outputs = []
        for layer in self.gcn_layers:
            H_curr = layer.forward(H_curr, A)
            layer_outputs.append(H_curr.copy())

        # 4. Attention layer for weighted aggregation
        H_att  = self.gat_layer.forward(H_curr, A)
        # Residual connection
        if H_att.shape == H_regime.shape:
            H_final = H_att + H_regime
        else:
            H_final = H_att

        # 5. Temporal aggregation
        if self.use_temporal and H_sequence is not None and len(H_sequence) > 1:
            H_temporal = self.temporal.forward_sequence(H_sequence, A)
            # Fuse with graph output
            min_dim  = min(H_final.shape[1], H_temporal.shape[1])
            H_final  = (H_final[:, :min_dim] + H_temporal[:, :min_dim]) / 2.0

        # 6. Risk scoring
        if H_final.shape[1] != self.hidden_dim:
            H_for_score = H_final[:, :self.hidden_dim] if H_final.shape[1] > self.hidden_dim                           else np.pad(H_final, ((0,0), (0, self.hidden_dim - H_final.shape[1])))
        else:
            H_for_score = H_final

        risk_logits = H_for_score @ self.score_W + self.score_b    # (N, 1)
        risk_scores = 1.0 / (1.0 + np.exp(-risk_logits.flatten())) # sigmoid

        return {
            "node_embeddings":  H_final,
            "risk_scores":      risk_scores,
            "layer_outputs":    layer_outputs,
            "regime_embedding": regime_embed,
            "weighted_adjacency": A_weighted,
            "n_nodes":          N,
            "regime_idx":       regime_idx,
        }

    def predict_manipulation(self, H: np.ndarray, A: np.ndarray,
                             threshold: float = 0.5,
                             regime_idx: int = 0) -> dict:
        """Run forward pass and generate binary manipulation predictions."""
        output    = self.forward(H, A, regime_idx)
        scores    = output["risk_scores"]
        preds     = (scores >= threshold).astype(int)

        n_flagged = int(preds.sum())
        return {
            "predictions":   preds.tolist(),
            "risk_scores":   scores.tolist(),
            "n_flagged":     n_flagged,
            "flagged_ratio": float(n_flagged / len(preds)) if len(preds) > 0 else 0.0,
            "threshold":     threshold,
        }

    def get_node_importance(self, output: dict) -> np.ndarray:
        """Extract node importance scores from embeddings via L2 norm."""
        embeddings = output["node_embeddings"]
        return np.linalg.norm(embeddings, axis=1)

    def save_embeddings(self, output: dict, asset_ids: List[str],
                        filename: str = "embeddings.json") -> str:
        """Save node embeddings to JSON file."""
        path = os.path.join(OUTPUT_DIR, filename)
        emb_data = {
            "module": "CGNN-XAI",
            "regime_idx": int(output["regime_idx"]),
            "embeddings": [
                {
                    "asset_id": aid,
                    "embedding": output["node_embeddings"][i].tolist(),
                    "risk_score": float(output["risk_scores"][i]),
                    "importance": float(np.linalg.norm(output["node_embeddings"][i])),
                }
                for i, aid in enumerate(asset_ids[:output["n_nodes"]])
            ],
        }
        with open(path, "w") as f:
            json.dump(emb_data, f, indent=2)
        print(f"  Embeddings saved → {path}")
        return path

    def get_config(self) -> dict:
        return self.config.copy()


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY & UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config: Optional[dict] = None) -> CounterfactualGNN:
    """Factory function to build CGNN-XAI model with default or custom config."""
    default_config = {
        "feature_dim":  32,
        "hidden_dim":   64,
        "n_layers":     3,
        "n_regimes":    5,
        "dropout_rate": 0.1,
        "n_heads":      4,
        "use_temporal": True,
        "module":       "CGNN-XAI",
    }
    if config:
        default_config.update(config)
    model = CounterfactualGNN(default_config)
    print(f"  Built CGNN-XAI model with config: {default_config}")
    return model


def build_random_graph(n_nodes: int = 20, edge_density: float = 0.3,
                       feature_dim: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Build a random graph for testing the GNN."""
    A = (np.random.rand(n_nodes, n_nodes) < edge_density).astype(float)
    np.fill_diagonal(A, 0)  # No self loops
    A = np.maximum(A, A.T)  # Symmetric
    H = np.random.randn(n_nodes, feature_dim) * 0.5
    return H, A


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  CGNN-XAI - GNN Model Module")
    print(f"  Counterfactual Graph Neural Network")
    print(f"{'='*60}\n")

    # Build model
    model = build_model()

    # Generate test graph
    n_nodes = 15
    asset_ids = [f"ASSET_{i:03d}" for i in range(n_nodes)]
    H, A = build_random_graph(n_nodes=n_nodes, feature_dim=32)

    print(f"[1/4] Input: {n_nodes} nodes | Feature dim: {H.shape[1]}")
    print(f"      Adjacency matrix: {A.shape} | Density: {A.mean():.3f}")

    # Forward pass
    print("[2/4] Running forward pass...")
    for regime_idx in range(3):
        output = model.forward(H, A, regime_idx=regime_idx)
        print(f"      Regime {regime_idx}: avg risk = {output['risk_scores'].mean():.4f}")

    # Prediction
    print("[3/4] Generating manipulation predictions...")
    result = model.predict_manipulation(H, A, threshold=0.5, regime_idx=0)
    print(f"      Flagged: {result['n_flagged']} / {n_nodes} nodes "
          f"({result['flagged_ratio']:.1%})")

    # Save embeddings
    print("[4/4] Saving embeddings...")
    output = model.forward(H, A, regime_idx=0)
    emb_path = model.save_embeddings(output, asset_ids)

    # Risk scores CSV
    scores_df = pd.DataFrame({
        "asset_id":   asset_ids,
        "risk_score": output["risk_scores"].tolist(),
        "importance": model.get_node_importance(output).tolist(),
        "prediction": (output["risk_scores"] >= 0.5).astype(int).tolist(),
        "regime_idx": 0,
        "module":     "CGNN-XAI",
    })
    csv_path = os.path.join(OUTPUT_DIR, "risk_scores.csv")
    scores_df.to_csv(csv_path, index=False)
    print(f"  Risk scores → {csv_path}")

    print(f"\nCGNN-XAI GNN Model Module Complete.\n")
    return model, output

if __name__ == "__main__":
    main()
