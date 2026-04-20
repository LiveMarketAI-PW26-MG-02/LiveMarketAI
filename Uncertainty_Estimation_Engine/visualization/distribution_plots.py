"""Predictive distribution visualization."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class DistributionPlotter:
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize

    def plot_predictive_samples(self, X_test, samples, y_true=None,
                                 title="Predictive Samples",
                                 save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        x = X_test[:, 0] if X_test.ndim > 1 else X_test
        n_samples = min(50, samples.shape[0])
        for i in range(n_samples):
            ax.plot(x, samples[i], alpha=0.05, color="royalblue", linewidth=0.8)
        ax.plot(x, samples.mean(axis=0), "r-", linewidth=2, label="Mean")
        if y_true is not None:
            ax.scatter(x, y_true, s=20, zorder=5, color="black", label="True values")
        ax.set_title(title); ax.legend(); plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_epistemic_vs_aleatoric(self, epistemic, aleatoric,
                                    save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(epistemic, aleatoric, alpha=0.4, s=10, c="purple")
        ax.set_xlabel("Epistemic Uncertainty"); ax.set_ylabel("Aleatoric Uncertainty")
        ax.set_title("Epistemic vs Aleatoric Uncertainty"); plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
