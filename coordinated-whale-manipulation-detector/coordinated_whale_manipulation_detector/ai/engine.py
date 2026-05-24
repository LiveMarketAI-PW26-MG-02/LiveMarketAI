from __future__ import annotations

import math

import numpy as np



class Engine:
    """PageRank-based influence over a transaction/holder graph."""

    def __init__(self, damping: float = 0.85) -> None:
        self.damping = damping

    def explain(self, features: dict) -> dict:
        edges = features.get("edges") or [["A", "B"], ["B", "C"], ["C", "A"], ["A", "C"]]
        nodes = sorted({n for e in edges for n in e})
        index = {n: i for i, n in enumerate(nodes)}
        size = len(nodes)
        M = np.zeros((size, size))
        for src, dst in edges:
            M[index[dst], index[src]] += 1.0
        col_sums = M.sum(axis=0)
        for j in range(size):
            if col_sums[j] > 0:
                M[:, j] /= col_sums[j]
            else:
                M[:, j] = 1.0 / size
        rank = np.ones(size) / size
        for _ in range(100):
            rank = (1 - self.damping) / size + self.damping * (M @ rank)
        ranking = sorted(({"node": nodes[i], "influence": round(float(rank[i]), 4)}
                          for i in range(size)), key=lambda r: r["influence"], reverse=True)
        return {"primitive": "graph", "summary":
                f"Most influential node: {ranking[0]['node']}.",
                "ranking": ranking}
