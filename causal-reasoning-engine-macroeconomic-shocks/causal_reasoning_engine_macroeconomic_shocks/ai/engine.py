from __future__ import annotations

import math

import numpy as np



class Engine:
    """Linear structural causal model. Propagates interventions through a small
    DAG via the reduced-form (I - W)^-1 and reports per-path effects."""

    NODES = ["rates", "credit", "fx", "equity", "growth"]

    def __init__(self) -> None:
        # weight[i][j] = direct causal effect of node j on node i
        self.W = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0, 0.0, 0.0],
            [-0.5, -0.4, 0.2, 0.0, 0.0],
            [-0.3, -0.3, 0.0, 0.4, 0.0],
        ])
        self.reduced = np.linalg.inv(np.eye(len(self.NODES)) - self.W)

    def explain(self, features: dict) -> dict:
        shock = np.array([float(features.get(n, 0.0)) for n in self.NODES])
        response = self.reduced @ shock
        effects = [{"node": self.NODES[i], "total_effect": round(float(response[i]), 4)}
                   for i in range(len(self.NODES))]
        driver = max(effects, key=lambda e: abs(e["total_effect"]))
        return {"primitive": "causal", "summary":
                f"Largest propagated effect on {driver['node']} "
                f"({driver['total_effect']:+.3f}).",
                "shock": {self.NODES[i]: float(shock[i]) for i in range(len(self.NODES))},
                "effects": effects}
