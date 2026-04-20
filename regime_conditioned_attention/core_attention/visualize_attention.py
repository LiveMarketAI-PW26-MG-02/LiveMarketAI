"""Attention weight visualization."""
import matplotlib.pyplot as plt, numpy as np

def plot_attention_heatmap(alphas: dict):
    regimes=list(alphas.keys()); weights=np.stack([alphas[r] for r in regimes])
    fig,ax=plt.subplots(figsize=(10,4))
    im=ax.imshow(weights,aspect="auto",cmap="YlOrRd")
    ax.set_xticks(range(weights.shape[1])); ax.set_xticklabels([f"X{i}" for i in range(weights.shape[1])])
    ax.set_yticks(range(len(regimes))); ax.set_yticklabels(regimes)
    plt.colorbar(im,ax=ax,label="α"); ax.set_title("Regime-conditioned Attention Weights")
    plt.tight_layout(); return fig

def plot_weight_trajectory(alpha_series, regime_series):
    fig,axes=plt.subplots(alpha_series.shape[1],1,figsize=(12,2*alpha_series.shape[1]),sharex=True)
    cmap=plt.cm.Set1
    for i,ax in enumerate(axes):
        for r in np.unique(regime_series):
            mask=regime_series==r
            ax.scatter(np.where(mask)[0],alpha_series[mask,i],s=4,c=[cmap(r/4)],label=f"R{r}")
        ax.set_ylabel(f"α{i}"); ax.set_ylim(0,1)
    axes[-1].set_xlabel("Time step"); fig.suptitle("Attention Weight Trajectory"); plt.tight_layout(); return fig
