#!/usr/bin/env python3
"""
FSPM Session Physics Model
Models interaction cadence entropy, propagation delay signatures,
subscription graph topology, and latency variance drift.
"""

import numpy as np


class SessionPhysicsModel:
    """
    Builds a legitimacy manifold for session dynamics.
    Features: [cadence_entropy, prop_delay_mean, prop_delay_std,
               sub_graph_density, latency_drift, ...11 more]
    """

    INPUT_DIM  = 16
    HIDDEN_DIM = 32

    def __init__(self, node_id: str):
        self.node_id = node_id
        rng = np.random.default_rng(abs(hash(node_id)) % (2**31))
        self.W1 = rng.standard_normal((self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2 = rng.standard_normal((self.HIDDEN_DIM, 1)).astype(np.float32) * 0.1
        self.b2 = np.zeros(1, dtype=np.float32)
        self.loss = float("inf")

    @staticmethod
    def extract_session_features(session: dict) -> np.ndarray:
        delays   = np.array(session.get("prop_delays", [0.01] * 20))
        cadences = np.array(session.get("cadence", [0.5] * 20))
        subs     = np.array(session.get("subscriptions", [1, 2, 3]))

        h_bins, _ = np.histogram(cadences, bins=8, density=True)
        cadence_entropy = float(-np.sum(h_bins * np.log2(h_bins + 1e-9)))

        graph_density = len(subs) / (len(subs)**2 + 1e-9)

        return np.array([
            cadence_entropy,
            float(np.mean(delays)),
            float(np.std(delays) + 1e-9),
            float(np.percentile(delays, 95)),
            graph_density,
            float(np.var(delays)),
            float(np.max(delays) - np.min(delays)),
            float(np.diff(delays).std() if len(delays) > 1 else 0),  # latency drift
            float(np.mean(cadences)),
            float(np.std(cadences) + 1e-9),
            float(np.corrcoef(delays, cadences)[0,1] if len(delays)==len(cadences) and len(delays)>2 else 0),
            float(len(subs)),
            float(np.max(delays)),
            float(np.min(delays)),
            float(np.median(cadences)),
            float(np.sum(delays > np.mean(delays) + 2*np.std(delays))),
        ], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        return 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)))

    def train(self, sessions: list, lr: float = 1e-3) -> float:
        X = np.array([self.extract_session_features(s[0]) for s in sessions], dtype=np.float32)
        y = np.array([float(s[1]) for s in sessions]).reshape(-1, 1)
        preds = self.forward(X)
        err   = preds - y
        n     = len(sessions)
        hh    = np.tanh(X @ self.W1 + self.b1)
        dW2   = (hh.T @ err) / n
        db2   = err.mean(axis=0)
        dh    = (err @ self.W2.T) * (1 - hh**2)
        dW1   = (X.T @ dh) / n
        db1   = dh.mean(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.loss = float(np.mean(err**2))
        return self.loss

    def congruence_score(self, session: dict) -> float:
        feats = self.extract_session_features(session)
        return float(self.forward(feats.reshape(1, -1))[0, 0])


def simulate_legitimate_session() -> dict:
    rng = np.random.default_rng()
    return {
        "prop_delays":   list(rng.exponential(0.01, 20)),
        "cadence":       list(rng.exponential(0.5, 20)),
        "subscriptions": list(range(rng.integers(3, 8))),
    }


def simulate_hijacked_session() -> dict:
    rng = np.random.default_rng()
    return {
        "prop_delays":   list(rng.exponential(0.1, 20)),   # 10x normal delay
        "cadence":       list(rng.exponential(0.05, 20)),  # 10x faster cadence
        "subscriptions": list(range(rng.integers(20, 50))), # anomalous fan-out
    }


if __name__ == "__main__":
    model = SessionPhysicsModel("fspm-demo")
    sessions = [(simulate_legitimate_session(), True) for _ in range(40)]
    sessions += [(simulate_hijacked_session(), False) for _ in range(10)]

    for rnd in range(8):
        loss = model.train(sessions)
        print(f"Round {rnd+1:2d} | loss={loss:.6f}")

    print(f"\nLegit session score : {model.congruence_score(simulate_legitimate_session()):.4f}")
    print(f"Hijacked session    : {model.congruence_score(simulate_hijacked_session()):.4f}")
