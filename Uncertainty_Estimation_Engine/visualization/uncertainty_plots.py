"""Uncertainty visualization tools."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class UncertaintyPlotter:
    def __init__(self, figsize: Tuple[int,int] = (10, 6), style: str = "seaborn-v0_8"):
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            pass

    def plot_predictions(self, X, y, mean, lower, upper,
                         title="Prediction with Uncertainty",
                         save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        order = np.argsort(X[:, 0]) if X.ndim > 1 else np.argsort(X)
        x_plot = X[order, 0] if X.ndim > 1 else X[order]
        ax.scatter(x_plot, y[order], s=15, alpha=0.6, label="Observations", color="steelblue")
        ax.plot(x_plot, mean[order], "r-", label="Predicted Mean", linewidth=2)
        ax.fill_between(x_plot, lower[order], upper[order], alpha=0.25, color="red", label="90% CI")
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("y")
        ax.legend(); plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_uncertainty_decomposition(self, epistemic, aleatoric, title="Uncertainty Decomposition",
                                       save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        total = np.sqrt(epistemic**2 + aleatoric**2)
        for ax, data, name, color in zip(
            axes, [epistemic, aleatoric, total],
            ["Epistemic", "Aleatoric", "Total"], ["royalblue","darkorange","forestgreen"]
        ):
            ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white")
            ax.set_title(name); ax.set_xlabel("Uncertainty"); ax.set_ylabel("Count")
            ax.axvline(data.mean(), color="black", linestyle="--", alpha=0.8, label=f"Mean={data.mean():.3f}")
            ax.legend()
        fig.suptitle(title); plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
