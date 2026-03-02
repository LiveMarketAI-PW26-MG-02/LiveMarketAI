#!/usr/bin/env python3
"""
FBIB Behavioral Identity Model
Trains on command invocation timing, session cadence, trade volume patterns,
and execution sequence embeddings to build behavioral identity signatures.
"""

import numpy as np
import time


class BehavioralIdentityModel:
    """
    Embeds credential usage behaviour into a learned identity vector.
    Features: [cmd_timing_entropy, trade_vol_mean, trade_vol_std,
               session_cadence_iat, exec_seq_hash_mod, ...12 more micro-features]
    """

    INPUT_DIM  = 16
    HIDDEN_DIM = 32

    def __init__(self, node_id: str):
        self.node_id = node_id
        rng = np.random.default_rng(abs(hash(node_id)) % (2**31))
        self.W  = rng.standard_normal((self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b  = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.Wo = rng.standard_normal((self.HIDDEN_DIM, 1)).astype(np.float32) * 0.1
        self.bo = np.zeros(1, dtype=np.float32)
        self.loss = float("inf")

    def _encode_session(self, session_events: np.ndarray) -> np.ndarray:
        """Encode session events into 16-dimensional behavioral feature vector."""
        n = len(session_events)
        if n < 2:
            return np.zeros(self.INPUT_DIM, dtype=np.float32)
        iats  = np.diff(session_events[:, 0])                  # inter-arrival times
        vols  = session_events[:, 1]                            # trade volumes
        seqs  = session_events[:, 2] if session_events.shape[1] > 2 else np.zeros(n)
        feats = np.array([
            float(np.mean(iats)),
            float(np.std(iats) + 1e-9),
            float(-np.sum(np.histogram(iats, bins=8, density=True)[0] *
                          np.log(np.histogram(iats, bins=8, density=True)[0] + 1e-9))),
            float(np.mean(vols)),
            float(np.std(vols) + 1e-9),
            float(np.percentile(vols, 95)),
            float(np.mean(seqs)),
            float(np.std(seqs) + 1e-9),
            float(n / (session_events[-1, 0] - session_events[0, 0] + 1e-9)),
            float(np.max(iats) - np.min(iats)),
            float(np.corrcoef(iats, vols[:len(iats)])[0, 1] if len(iats) > 2 else 0),
            float(np.sum(vols > np.percentile(vols, 90))),
            float(np.sum(iats < np.percentile(iats, 10))),
            float(np.var(seqs)),
            float(np.median(iats)),
            float(np.max(vols) / (np.mean(vols) + 1e-9)),
        ], dtype=np.float32)
        return feats

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W + self.b)
        return 1.0 / (1.0 + np.exp(-(h @ self.Wo + self.bo)))

    def train(self, sessions: list, lr: float = 1e-3) -> float:
        """Train on a list of (events_array, is_legitimate) tuples."""
        X = np.array([self._encode_session(s[0]) for s in sessions], dtype=np.float32)
        y = np.array([float(s[1]) for s in sessions], dtype=np.float32).reshape(-1, 1)
        preds = self.forward(X)
        err   = preds - y
        n     = len(sessions)
        h     = np.tanh(X @ self.W + self.b)
        dWo   = (h.T @ err) / n
        dbo   = err.mean(axis=0)
        dh    = (err @ self.Wo.T) * (1 - h**2)
        dW    = (X.T @ dh) / n
        db    = dh.mean(axis=0)
        self.Wo -= lr * dWo
        self.bo -= lr * dbo
        self.W  -= lr * dW
        self.b  -= lr * db
        self.loss = float(np.mean(err**2))
        return self.loss

    def identity_score(self, events: np.ndarray) -> float:
        """Score 0.0 = behaviorally alien, 1.0 = fully congruent."""
        feat = self._encode_session(events)
        return float(self.forward(feat.reshape(1, -1))[0, 0])


def simulate_session(legitimate: bool = True, n_events: int = 50) -> np.ndarray:
    rng = np.random.default_rng()
    if legitimate:
        times  = np.cumsum(rng.exponential(0.5, n_events))
        vols   = rng.lognormal(5, 0.3, n_events)
        seqs   = np.arange(n_events, dtype=float)
    else:
        # Attacker: faster cadence, unusual volumes, non-sequential cmds
        times  = np.cumsum(rng.exponential(0.05, n_events))
        vols   = rng.lognormal(8, 1.5, n_events)
        seqs   = rng.choice(1000, n_events).astype(float)
    return np.column_stack([times, vols, seqs])


if __name__ == "__main__":
    model = BehavioralIdentityModel("fbib-demo")
    sessions = [(simulate_session(True), True) for _ in range(40)]
    sessions += [(simulate_session(False), False) for _ in range(10)]

    for rnd in range(5):
        loss = model.train(sessions)
        print(f"Round {rnd+1:2d} | loss={loss:.6f}")

    legit  = model.identity_score(simulate_session(True))
    attack = model.identity_score(simulate_session(False))
    print(f"\nLegit score  : {legit:.4f}")
    print(f"Attacker score: {attack:.4f}")
    print(f"Delta: {legit - attack:.4f} (positive = correctly differentiated)")
