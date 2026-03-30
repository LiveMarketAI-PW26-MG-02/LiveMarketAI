"""
visualise.py — Results Visualisation for Stock Incremental Learning Pipeline
Produces charts saved to results/ and displayed interactively (if possible).
"""

import os
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[visualise] matplotlib not available — skipping plots.")

RESULTS_DIR = "results"


def load_csv(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _flt(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]


def _int_col(rows, key):
    return [int(r[key]) for r in rows if r.get(key) not in (None, "", "None")]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not HAS_MPL:
        return

    incr    = load_csv("incremental_metrics.csv")
    retrain = load_csv("retrain_metrics.csv")
    summary = load_json("comparison_summary.json")

    if not incr:
        print("[visualise] No incremental_metrics.csv found. "
              "Run the pipeline first.")
        return

    batches      = _int_col(incr, "batch")
    incr_loss    = _flt(incr,  "val_loss")
    incr_mae     = _flt(incr,  "val_mae")
    drift_scores = _flt(incr,  "drift_score")
    vol_ratios   = _flt(incr,  "vol_ratio")
    current_lrs  = _flt(incr,  "current_lr")
    buf_sizes    = _flt(incr,  "buffer_size")
    update_flags = [r["update_ran"] == "True" for r in incr]

    ret_batches  = _int_col(retrain, "batch")
    ret_loss     = _flt(retrain, "val_loss")
    ret_sec      = _flt(retrain, "retrain_sec")

    fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
    fig.suptitle("Stock Incremental Learning Pipeline — Results",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.93,  bottom=0.06)

    PANEL   = "#1a1d27"
    GREEN   = "#00e676"
    ORANGE  = "#ff9100"
    BLUE    = "#40c4ff"
    RED     = "#ff5252"
    YELLOW  = "#ffea00"
    GREY    = "#546e7a"

    def _ax(row, col, colspan=1):
        a = fig.add_subplot(gs[row, col:col+colspan])
        a.set_facecolor(PANEL)
        for spine in a.spines.values():
            spine.set_edgecolor("#2a2d3a")
        a.tick_params(colors="#aaaaaa", labelsize=8)
        a.xaxis.label.set_color("#aaaaaa")
        a.yaxis.label.set_color("#aaaaaa")
        return a

    # ── 1. Val Loss: Incremental vs Retrain ──────────────────────────────────
    ax1 = _ax(0, 0, 2)
    ax1.set_title("Val Loss: Incremental vs Full Retraining (Req 9)",
                  color="white", fontsize=9)
    ax1.plot(batches, incr_loss, color=GREEN,  lw=1.5, label="Incremental")
    if ret_loss:
        ax1.plot(ret_batches, ret_loss, color=ORANGE, lw=1.5,
                 linestyle="--", label="Full Retrain")
    # Mark update triggers
    for i, (b, upd) in enumerate(zip(batches, update_flags)):
        if upd and i < len(incr_loss):
            ax1.axvline(b, color=BLUE, alpha=0.25, lw=0.8)
    ax1.set_xlabel("Batch"); ax1.set_ylabel("Val MSE Loss")
    ax1.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # ── 2. MAE over batches ───────────────────────────────────────────────────
    ax2 = _ax(0, 2)
    ax2.set_title("Incremental Val MAE", color="white", fontsize=9)
    ax2.plot(batches, incr_mae, color=BLUE, lw=1.5)
    ax2.set_xlabel("Batch"); ax2.set_ylabel("MAE")

    # ── 3. Drift Scores & Triggers ───────────────────────────────────────────
    ax3 = _ax(1, 0)
    ax3.set_title("Drift Scores (Req 3)", color="white", fontsize=9)
    ax3.plot(batches[:len(drift_scores)], drift_scores, color=RED, lw=1.5)
    thr = float(summary.get("incremental", {}).get("threshold",
                            next((r for r in [0.05]), 0.05)))
    ax3.axhline(0.05, color=YELLOW, lw=1, linestyle="--", label="threshold")
    for b, d, s in zip(batches, drift_scores, [r["drift_trig"]=="True" for r in incr]):
        if s:
            ax3.scatter(b, d, color=YELLOW, s=40, zorder=5)
    ax3.set_xlabel("Batch"); ax3.set_ylabel("KS Stat")
    ax3.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # ── 4. Adaptive LR Schedule ──────────────────────────────────────────────
    ax4 = _ax(1, 1)
    ax4.set_title("Adaptive LR Schedule (Req 6)", color="white", fontsize=9)
    ax4.plot(batches[:len(current_lrs)], current_lrs, color=GREEN, lw=1.5)
    ax4.set_xlabel("Batch"); ax4.set_ylabel("Learning Rate")
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ── 5. Volatility Ratio ───────────────────────────────────────────────────
    ax5 = _ax(1, 2)
    ax5.set_title("Market Volatility Ratio (Req 6)", color="white", fontsize=9)
    ax5.plot(batches[:len(vol_ratios)], vol_ratios, color=ORANGE, lw=1.5)
    ax5.axhline(1.0, color=GREY, lw=0.8, linestyle="--")
    ax5.set_xlabel("Batch"); ax5.set_ylabel("σ_recent / σ_baseline")

    # ── 6. Memory Buffer Growth ───────────────────────────────────────────────
    ax6 = _ax(2, 0)
    ax6.set_title("Memory Buffer Size (Req 2)", color="white", fontsize=9)
    ax6.fill_between(batches[:len(buf_sizes)], buf_sizes, alpha=0.4,
                     color=BLUE)
    ax6.plot(batches[:len(buf_sizes)], buf_sizes, color=BLUE, lw=1.5)
    ax6.axhline(500, color=GREY, lw=0.8, linestyle="--", label="capacity")
    ax6.set_xlabel("Batch"); ax6.set_ylabel("Samples")
    ax6.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # ── 7. Full Retrain Time ──────────────────────────────────────────────────
    ax7 = _ax(2, 1)
    ax7.set_title("Full Retrain Time per Cycle (Req 9)", color="white", fontsize=9)
    if ret_sec and ret_batches:
        ax7.bar(ret_batches, ret_sec, color=ORANGE, alpha=0.8, width=1.5)
    ax7.set_xlabel("Batch"); ax7.set_ylabel("Seconds")

    # ── 8. Update Trigger Pie ─────────────────────────────────────────────────
    ax8 = _ax(2, 2)
    ax8.set_title("Update Trigger Distribution", color="white", fontsize=9)
    n_upd    = sum(update_flags)
    n_no_upd = len(update_flags) - n_upd
    drift_only  = sum(1 for r in incr
                      if r["drift_trig"]=="True" and r["conf_trig"]!="True")
    conf_only   = sum(1 for r in incr
                      if r["conf_trig"]=="True" and r["drift_trig"]!="True")
    both        = sum(1 for r in incr
                      if r["drift_trig"]=="True" and r["conf_trig"]=="True")
    no_upd      = n_no_upd
    labels  = ["Drift only", "Confidence only", "Both", "No update"]
    sizes   = [drift_only, conf_only, both, no_upd]
    colors  = [RED, BLUE, YELLOW, GREY]
    valid   = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if valid:
        ls, ss, cs = zip(*valid)
        ax8.pie(ss, labels=ls, colors=cs, autopct="%1.0f%%",
                textprops={"color": "white", "fontsize": 7},
                wedgeprops={"edgecolor": "#0f1117", "linewidth": 1})

    out_path = os.path.join(RESULTS_DIR, "pipeline_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[visualise] Dashboard saved → {out_path}")

    # ── Print text summary ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  COMPARATIVE EVALUATION SUMMARY (Req 9)")
    print("=" * 55)
    if summary:
        incr_s = summary.get("incremental", {})
        ret_s  = summary.get("full_retraining", {})
        print(f"  Incremental  mean val loss  : {incr_s.get('mean_val_loss')}")
        print(f"  Full-Retrain mean val loss  : {ret_s.get('mean_val_loss')}")
        print(f"  Retrain avg time / cycle    : {ret_s.get('mean_retrain_sec')}s")
        print(f"  Update rate                 : {incr_s.get('update_rate_%')}%")
        print(f"  Efficiency gain (est.)      : {summary.get('efficiency_gain_%')}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
