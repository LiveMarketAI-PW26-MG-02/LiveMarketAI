"""Calibration curve visualization."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


class CalibrationPlotter:
    def __init__(self, figsize=(7, 7)):
        self.figsize = figsize

    def reliability_diagram(self, data: Dict, title="Reliability Diagram",
                             save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        confs = data["confidences"]
        accs = data["accuracies"]
        counts = data["counts"]
        bar_width = 1.0 / len(confs)
        bars = ax.bar(confs, accs, width=bar_width*0.9, alpha=0.7, color="steelblue",
                      label="Accuracy per bin")
        ax.plot([0,1],[0,1], "r--", linewidth=2, label="Perfect calibration")
        ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color="red")
        ax2 = ax.twinx()
        ax2.bar(confs, counts, width=bar_width*0.9, alpha=0.2, color="gray", label="Counts")
        ax2.set_ylabel("Sample counts")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.legend(loc="upper left"); plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
