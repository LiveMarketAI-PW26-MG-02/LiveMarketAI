"""Reliability diagram data generation."""
import numpy as np
from typing import Dict, List


class ReliabilityDiagram:
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute(self, probs: np.ndarray, y: np.ndarray) -> Dict[str, List]:
        edges = np.linspace(0, 1, self.n_bins + 1)
        accs, confs, counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            n = mask.sum()
            counts.append(int(n))
            if n:
                accs.append(float(y[mask].mean()))
                confs.append(float(probs[mask].mean()))
            else:
                accs.append(float((lo+hi)/2))
                confs.append(float((lo+hi)/2))
        return {"accuracies": accs, "confidences": confs, "counts": counts,
                "bin_centers": [(lo+hi)/2 for lo,hi in zip(edges[:-1],edges[1:])]}

    def ece_from_diagram(self, data: Dict, n: int) -> float:
        return sum(c*abs(a-cf)/n for a,cf,c in
                   zip(data["accuracies"], data["confidences"], data["counts"]))
