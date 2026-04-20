#!/usr/bin/env python3
"""
FCMDR Cross-Model Divergence Ring
Monitors gradient divergence across independently trained models.
Quarantines nodes whose gradient directions deviate beyond stochastic tolerance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelState:
    node_id:  str
    weights:  np.ndarray
    gradients: np.ndarray
    loss:     float
    quarantined: bool = False


class DivergenceRingMonitor:
    """
    Maintains a ring of N independently-trained models.
    Computes pairwise cosine similarity of gradient vectors.
    Quarantines outliers beyond threshold.
    """

    def __init__(self, n_models: int = 4, weight_dim: int = 16,
                 quarantine_threshold: float = 0.3):
        self.n_models   = n_models
        self.weight_dim = weight_dim
        self.threshold  = quarantine_threshold
        self.models: Dict[str, ModelState] = {}
        rng = np.random.default_rng(42)
        for i in range(n_models):
            nid = f"ring-node-{i:02d}"
            self.models[nid] = ModelState(
                node_id  = nid,
                weights  = rng.standard_normal(weight_dim).astype(np.float32) * 0.1,
                gradients = np.zeros(weight_dim, dtype=np.float32),
                loss     = float("inf"),
            )

    def submit_gradients(self, node_id: str, gradients: np.ndarray, loss: float) -> bool:
        if node_id not in self.models:
            return False
        self.models[node_id].gradients = gradients.copy()
        self.models[node_id].loss      = loss
        return True

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))

    def check_divergence(self) -> List[str]:
        """Return list of node_ids that should be quarantined."""
        active = [m for m in self.models.values() if not m.quarantined]
        if len(active) < 2:
            return []

        # Compute median gradient direction
        grads  = np.array([m.gradients for m in active])
        median = np.median(grads, axis=0)
        quarantine = []
        for m in active:
            sim = self._cosine_sim(m.gradients, median)
            if sim < self.threshold:
                m.quarantined = True
                quarantine.append(m.node_id)
                print(f"[FCMDR] QUARANTINE: {m.node_id} | cosine_sim={sim:.4f} < {self.threshold}")
            else:
                print(f"[FCMDR] OK: {m.node_id} | cosine_sim={sim:.4f}")
        return quarantine

    def fedavg_healthy(self) -> np.ndarray:
        """Average weights of non-quarantined nodes only."""
        healthy = [m for m in self.models.values() if not m.quarantined]
        if not healthy:
            return np.zeros(self.weight_dim, dtype=np.float32)
        return np.mean([m.weights for m in healthy], axis=0).astype(np.float32)

    def inject_poisoned_gradients(self, node_id: str, poison_scale: float = 10.0):
        """Simulate a poisoning attack on a specific node."""
        if node_id in self.models:
            rng = np.random.default_rng()
            self.models[node_id].gradients = (
                rng.standard_normal(self.weight_dim).astype(np.float32) * poison_scale
            )
            print(f"[FCMDR] POISON injected into {node_id}")


if __name__ == "__main__":
    ring = DivergenceRingMonitor(n_models=4, quarantine_threshold=0.3)

    # Normal round
    print("=== Normal Training Round ===")
    rng = np.random.default_rng()
    for nid in ring.models:
        g = rng.standard_normal(16).astype(np.float32) * 0.01  # similar gradients
        ring.submit_gradients(nid, g, loss=0.05)
    quarantined = ring.check_divergence()
    print(f"Quarantined: {quarantined}")

    # Poisoning attack
    print("\n=== Poisoning Attack Round ===")
    for nid in list(ring.models.keys())[:3]:
        g = rng.standard_normal(16).astype(np.float32) * 0.01
        ring.submit_gradients(nid, g, loss=0.05)
    ring.inject_poisoned_gradients(list(ring.models.keys())[3])
    quarantined = ring.check_divergence()
    print(f"Quarantined: {quarantined}")
    global_w = ring.fedavg_healthy()
    print(f"Global weight norm (healthy only): {np.linalg.norm(global_w):.4f}")
