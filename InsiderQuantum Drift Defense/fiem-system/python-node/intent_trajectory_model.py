#!/usr/bin/env python3
"""
FIEM Intent Embedding Monitor
Encodes privileged command sequences into trajectory embeddings.
Detects slow-moving insider threats via vector field deviation.
"""

import numpy as np
from collections import deque


class IntentTrajectoryModel:
    """
    Models privilege evolution as a dynamic vector field trajectory.
    Each time step: command embedding → trajectory update → drift detection.
    """

    EMBED_DIM   = 16
    HIDDEN_DIM  = 32
    WINDOW_SIZE = 20

    # Known privilege command vocabulary (simulated)
    CMD_VOCAB = [
        "read_position", "write_order", "cancel_order", "read_risk",
        "modify_limit", "export_data", "read_audit", "admin_access",
        "bulk_export", "system_config", "delete_log", "elevate_priv",
        "read_pnl", "read_portfolio", "exec_trade", "batch_trade",
    ]

    def __init__(self, node_id: str):
        self.node_id = node_id
        rng = np.random.default_rng(abs(hash(node_id)) % (2**31))

        # Command embedding matrix
        self.cmd_embed = rng.standard_normal(
            (len(self.CMD_VOCAB), self.EMBED_DIM)
        ).astype(np.float32) * 0.1

        # Trajectory model weights
        self.W1   = rng.standard_normal((self.EMBED_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b1   = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2   = rng.standard_normal((self.HIDDEN_DIM, 1)).astype(np.float32) * 0.1
        self.b2   = np.zeros(1, dtype=np.float32)
        self.loss = float("inf")

        # Sliding window for trajectory state
        self._trajectory: deque = deque(maxlen=self.WINDOW_SIZE)

    def embed_command(self, cmd: str) -> np.ndarray:
        if cmd in self.CMD_VOCAB:
            idx = self.CMD_VOCAB.index(cmd)
        else:
            idx = abs(hash(cmd)) % len(self.CMD_VOCAB)
        return self.cmd_embed[idx]

    def embed_sequence(self, commands: list) -> np.ndarray:
        """Aggregate sequence into single trajectory vector."""
        embeds = np.array([self.embed_command(c) for c in commands], dtype=np.float32)
        # Weighted recency: more recent commands have higher weight
        weights = np.exp(np.linspace(-1, 0, len(embeds)))
        weights /= weights.sum()
        return (embeds * weights.reshape(-1, 1)).sum(axis=0)

    def forward(self, x: np.ndarray) -> float:
        h = np.tanh(x @ self.W1 + self.b1)
        return float(1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)[0])))

    def train(self, sequences: list, lr: float = 1e-3) -> float:
        X = np.array([self.embed_sequence(s[0]) for s in sequences], dtype=np.float32)
        y = np.array([float(s[1]) for s in sequences]).reshape(-1, 1)
        preds = np.array([[self.forward(x)] for x in X], dtype=np.float32)
        err   = preds - y
        n     = len(sequences)
        h_all = np.tanh(X @ self.W1 + self.b1)
        dW2   = (h_all.T @ err) / n
        db2   = err.mean(axis=0)
        dh    = (err @ self.W2.T) * (1 - h_all**2)
        dW1   = (X.T @ dh) / n
        db1   = dh.mean(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.loss = float(np.mean(err**2))
        return self.loss

    def intent_score(self, commands: list) -> float:
        """Returns congruence score: 1.0 = normal insider behaviour."""
        embed = self.embed_sequence(commands)
        return self.forward(embed)

    def update_trajectory(self, commands: list) -> float:
        score = self.intent_score(commands)
        self._trajectory.append(score)
        return score

    @property
    def drift_velocity(self) -> float:
        """Rate of change of intent score over sliding window."""
        t = list(self._trajectory)
        if len(t) < 2:
            return 0.0
        return float(np.mean(np.diff(t)))


def simulate_normal_session(n: int = 10) -> list:
    rng   = np.random.default_rng()
    safe  = ["read_position", "read_pnl", "read_portfolio", "read_risk",
             "exec_trade", "read_audit"]
    return list(rng.choice(safe, n))


def simulate_insider_escalation(n: int = 10) -> list:
    rng       = np.random.default_rng()
    escalate  = ["admin_access", "bulk_export", "delete_log", "elevate_priv",
                 "export_data", "system_config"]
    return list(rng.choice(escalate, n))


if __name__ == "__main__":
    model = IntentTrajectoryModel("fiem-demo")
    normal_seqs   = [(simulate_normal_session(), True) for _ in range(40)]
    escalate_seqs = [(simulate_insider_escalation(), False) for _ in range(10)]

    for rnd in range(8):
        loss = model.train(normal_seqs + escalate_seqs)
        print(f"Round {rnd+1:2d} | loss={loss:.6f}")

    print(f"\nNormal score     : {model.intent_score(simulate_normal_session()):.4f}")
    print(f"Escalation score : {model.intent_score(simulate_insider_escalation()):.4f}")

    print("\n--- Simulating Slow Drift Detection ---")
    for step in range(15):
        cmds  = simulate_normal_session() if step < 8 else simulate_insider_escalation()
        score = model.update_trajectory(cmds)
        drift = model.drift_velocity
        print(f"Step {step+1:2d} | score={score:.4f} | drift={drift:+.4f}"
              f"{' *** ALERT ***' if drift < -0.05 else ''}")
