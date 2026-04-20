"""
Uncertainty Dashboard
======================
Creates a comprehensive multi-panel summary dashboard for an estimation run.
"""

import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class UncertaintyDashboard:
    """Multi-panel diagnostic dashboard for uncertainty estimation results."""

    def __init__(self, figsize=(18, 12)):
        self.figsize = figsize

    def generate(
        self,
        results: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Generate a 2×3 dashboard summarising:
        1. Prediction intervals
        2. Uncertainty decomposition
        3. Reliability diagram
        4. Sharpness histogram
        5. Posterior samples (if available)
        6. Metrics summary table
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("matplotlib required")

        fig = plt.figure(figsize=self.figsize)
        fig.suptitle("Uncertainty Estimation Engine — Diagnostic Dashboard", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Panel 1: Prediction intervals
        ax1 = fig.add_subplot(gs[0, 0])
        if "X" in results and "y_true" in results:
            X = np.asarray(results["X"]).ravel()
            idx = np.argsort(X)
            ax1.scatter(X[idx], results["y_true"][idx], s=15, color="crimson", zorder=5, label="True")
            ax1.plot(X[idx], results["y_pred"][idx], color="steelblue", lw=2, label="Pred")
            ax1.fill_between(X[idx], results["y_lower"][idx], results["y_upper"][idx], alpha=0.3, color="steelblue")
        ax1.set_title("Prediction Intervals"); ax1.legend(fontsize=8)

        # Panel 2: Uncertainty decomposition (bar)
        ax2 = fig.add_subplot(gs[0, 1])
        if "epistemic" in results and "aleatoric" in results:
            epi  = np.asarray(results["epistemic"])
            alea = np.asarray(results["aleatoric"])
            idx  = np.arange(min(50, len(epi)))
            ax2.bar(idx, epi[:50],  label="Epistemic", color="steelblue", alpha=0.8)
            ax2.bar(idx, alea[:50], bottom=epi[:50], label="Aleatoric", color="coral", alpha=0.8)
        ax2.set_title("Uncertainty Decomposition"); ax2.legend(fontsize=8)

        # Panel 3: Metrics summary
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")
        if "metrics" in results:
            m = results["metrics"]
            rows = [[k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v)]
                    for k, v in list(m.items())[:8]]
            t = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
            t.auto_set_font_size(False); t.set_fontsize(9)
        ax3.set_title("Metrics Summary")

        # Panel 4: Confidence histogram
        ax4 = fig.add_subplot(gs[1, 0])
        if "y_std" in results:
            ax4.hist(results["y_std"], bins=30, color="steelblue", alpha=0.7, density=True)
            ax4.axvline(np.mean(results["y_std"]), color="crimson", lw=2)
        ax4.set_title("Predictive Std Distribution"); ax4.set_xlabel("Std Dev")

        # Panel 5: Coverage
        ax5 = fig.add_subplot(gs[1, 1])
        if "y_true" in results and "y_pred" in results and "y_std" in results:
            alphas = np.linspace(0.01, 0.5, 20)
            from scipy import stats as sc_stats
            coverages = []
            for a in alphas:
                z = sc_stats.norm.ppf(1 - a / 2)
                lo = np.asarray(results["y_pred"]) - z * np.asarray(results["y_std"])
                hi = np.asarray(results["y_pred"]) + z * np.asarray(results["y_std"])
                coverages.append(np.mean((results["y_true"] >= lo) & (results["y_true"] <= hi)))
            ax5.plot(1 - alphas, coverages, lw=2, color="steelblue", label="Empirical")
            ax5.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
            ax5.legend(fontsize=8)
        ax5.set_title("Coverage Calibration")

        # Panel 6: Histogram of residuals
        ax6 = fig.add_subplot(gs[1, 2])
        if "y_true" in results and "y_pred" in results:
            resid = np.asarray(results["y_true"]) - np.asarray(results["y_pred"])
            ax6.hist(resid, bins=30, color="coral", alpha=0.7, density=True)
            ax6.axvline(0, color="black", lw=1.5, ls="--")
        ax6.set_title("Residuals Distribution"); ax6.set_xlabel("Residual")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Dashboard saved to %s", save_path)
        return fig
