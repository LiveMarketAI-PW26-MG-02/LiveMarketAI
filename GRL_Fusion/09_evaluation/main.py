#!/usr/bin/env python3
"""
Graph Reinforcement Learning Fusion (GRL-Fusion) - Evaluation Module
Comprehensive evaluation: classification metrics, regime analysis, calibration, graph metrics.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

OUTPUT_DIR = "outputs/grl_fusion_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

class BinaryClassificationEvaluator:
    """Full binary classification metric suite for GRL-Fusion."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        p    = (y_pred >= self.threshold).astype(int)
        tp   = int(((p==1)&(y_true==1)).sum())
        fp   = int(((p==1)&(y_true==0)).sum())
        fn   = int(((p==0)&(y_true==1)).sum())
        tn   = int(((p==0)&(y_true==0)).sum())
        prec = tp/(tp+fp+1e-10); rec=tp/(tp+fn+1e-10)
        spec = tn/(tn+fp+1e-10)
        f1   = 2*prec*rec/(prec+rec+1e-10)
        acc  = (tp+tn)/(tp+fp+fn+tn+1e-10)
        mcc  = (tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-10)
        auc  = self._auc(y_true, y_pred)
        ap   = self._ap(y_true, y_pred)
        brier= float(np.mean((y_pred-y_true)**2))
        return dict(
            precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
            specificity=round(spec,4), accuracy=round(acc,4), mcc=round(mcc,4),
            auc_roc=round(auc,4), average_precision=round(ap,4),
            brier_score=round(brier,4), threshold=self.threshold,
            tp=tp,fp=fp,fn=fn,tn=tn,
            n_pos=int(y_true.sum()), n_neg=int((1-y_true).sum()),
            prevalence=round(float(y_true.mean()),4),
        )

    def _auc(self, y_true, y_pred):
        idx = np.argsort(y_pred)[::-1]
        yt  = y_true[idx]
        n_pos = y_true.sum(); n_neg = len(y_true)-n_pos
        if n_pos==0 or n_neg==0: return 0.5
        cum_p=cum_n=auc=0
        for lab in yt:
            if lab==1: cum_p+=1
            else: auc+=cum_p; cum_n+=1
        return float(auc/(n_pos*n_neg+1e-10))

    def _ap(self, y_true, y_pred):
        idx=np.argsort(y_pred)[::-1]; yt=y_true[idx]
        n_pos=y_true.sum()
        if n_pos==0: return 0.0
        cum_p=ap=0
        for i,lab in enumerate(yt):
            cum_p+=lab; ap+=cum_p/(i+1)*lab
        return float(ap/(n_pos+1e-10))

    def threshold_sweep(self, y_true, y_pred):
        rows = []
        for t in np.linspace(0.05, 0.95, 19):
            old = self.threshold; self.threshold=t
            m=self.compute(y_true,y_pred); m["threshold"]=t; rows.append(m)
            self.threshold=old
        return pd.DataFrame(rows)

class RegimeEvaluator:
    """Regime-stratified evaluation for GRL-Fusion."""
    REGIMES=["bull","bear","sideways","crisis","recovery"]

    def evaluate(self, df, score_col, label_col="is_manipulation", regime_col="regime"):
        if regime_col not in df.columns:
            df=df.copy(); df[regime_col]=np.random.choice(self.REGIMES,len(df))
        ev=BinaryClassificationEvaluator()
        rows=[]
        for reg in df[regime_col].unique():
            sub=df[df[regime_col]==reg]
            if len(sub)<5: continue
            m=ev.compute(sub[label_col].values.astype(float),sub[score_col].values.astype(float))
            m["regime"]=reg; m["n_samples"]=len(sub)
            rows.append(m)
        return pd.DataFrame(rows)

class CalibrationEvaluator:
    """ECE and reliability diagram for GRL-Fusion probability calibration."""

    def ece(self, y_true, y_pred, n_bins=10):
        bins=np.linspace(0,1,n_bins+1); ece=0.0; n=len(y_true)
        bin_data=[]
        for i in range(n_bins):
            mask=(y_pred>=bins[i])&(y_pred<bins[i+1])
            if mask.sum()==0: continue
            conf=float(y_pred[mask].mean()); acc=float(y_true[mask].mean())
            cnt=int(mask.sum())
            ece+=cnt/n*abs(conf-acc)
            bin_data.append({"bin":f"[{bins[i]:.2f},{bins[i+1]:.2f})", "conf":round(conf,4), "acc":round(acc,4), "count":cnt})
        return {"ece":round(ece,4), "n_bins":n_bins, "bins":bin_data}

class GraphEvaluator:
    """Graph-aware metrics for GRL-Fusion GNN evaluation."""

    def neighborhood_consistency(self, scores, labels, A):
        threshold=0.5; high=(scores>=threshold)
        total=0.0; cnt=0
        for i in range(len(scores)):
            nb=np.where(A[i]>0)[0]
            if len(nb)==0: continue
            total+=(scores[nb].mean() if high[i] else 1-scores[nb].mean()); cnt+=1
        return float(total/(cnt+1e-10))

    def link_auc(self, emb, A):
        n=emb.shape[0]
        norms=np.linalg.norm(emb,axis=1,keepdims=True)+1e-10
        en=emb/norms; scores=[]; labs=[]
        for i in range(n):
            for j in range(i+1,n):
                scores.append(float(en[i]@en[j])); labs.append(int(A[i,j]>0))
        if not labs: return 0.5
        ys=np.array(scores); yl=np.array(labs)
        ys=(ys-ys.min())/(ys.max()-ys.min()+1e-10)
        ev=BinaryClassificationEvaluator(0.5)
        return ev._auc(yl,ys)

def main():
    print(f"\n{'='*60}\n  GRL-Fusion Evaluation Module\n{'='*60}")
    n=200
    y_true=( np.random.rand(n)<0.15).astype(float)
    y_pred=np.clip(y_true*0.7+np.random.rand(n)*0.4, 0, 1)
    A=(np.random.rand(n,n)<0.05).astype(float); np.fill_diagonal(A,0)
    emb=np.random.randn(n,32)
    regimes=np.random.choice(["bull","bear","sideways","crisis","recovery"],n)

    clf=BinaryClassificationEvaluator(0.5)
    m=clf.compute(y_true,y_pred)
    print(f"  F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f} | AP={m['average_precision']:.4f}")

    df_eval=pd.DataFrame({"risk_score":y_pred,"is_manipulation":y_true,"regime":regimes})
    reg_ev=RegimeEvaluator()
    reg_df=reg_ev.evaluate(df_eval,"risk_score")
    print(f"  Regime rows: {len(reg_df)}")

    cal=CalibrationEvaluator()
    ece=cal.ece(y_true,y_pred)
    print(f"  ECE={ece['ece']:.4f}")

    gev=GraphEvaluator()
    nc=gev.neighborhood_consistency(y_pred,y_true,A)
    lauc=gev.link_auc(emb,A)
    print(f"  Neighborhood consistency={nc:.4f} | Link AUC={lauc:.4f}")

    thresh_df=clf.threshold_sweep(y_true,y_pred)
    best=thresh_df.loc[thresh_df["f1"].idxmax()]
    print(f"  Optimal threshold={best['threshold']:.2f} (F1={best['f1']:.4f})")

    report={"module":"GRL-Fusion","full_name":"Graph Reinforcement Learning Fusion","evaluated_at":datetime.utcnow().isoformat()+"Z",
              "n_samples":n,"classification":m,
              "calibration":ece,"graph":{"neighborhood_consistency":nc,"link_auc":lauc},
              "optimal_threshold":float(best["threshold"])}
    rp=os.path.join(OUTPUT_DIR,"evaluation_report.json")
    with open(rp,"w") as f: json.dump(report,f,indent=2)
    thresh_df.to_csv(os.path.join(OUTPUT_DIR,"threshold_analysis.csv"),index=False)
    reg_df.to_csv(os.path.join(OUTPUT_DIR,"regime_evaluation.csv"),index=False)
    print(f"  Saved: evaluation_report.json, threshold_analysis.csv\n")

if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED EVALUATION COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionRecallAnalyzer:
    """Detailed precision-recall curve analysis."""

    def pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray, n_points: int = 50) -> pd.DataFrame:
        thresholds = np.linspace(0, 1, n_points)
        rows = []
        for t in thresholds:
            p = (y_scores >= t).astype(int)
            tp = int(((p==1)&(y_true==1)).sum())
            fp = int(((p==1)&(y_true==0)).sum())
            fn = int(((p==0)&(y_true==1)).sum())
            prec = tp/(tp+fp+1e-10)
            rec  = tp/(tp+fn+1e-10)
            rows.append({"threshold": t, "precision": prec, "recall": rec,
                         "f1": 2*prec*rec/(prec+rec+1e-10), "tp": tp, "fp": fp, "fn": fn})
        return pd.DataFrame(rows)

    def best_f1_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        df = self.pr_curve(y_true, y_scores)
        return float(df.loc[df["f1"].idxmax(), "threshold"])

    def area_under_pr(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        df = self.pr_curve(y_true, y_scores).sort_values("recall")
        auprc = 0.0
        for i in range(1, len(df)):
            dr = df["recall"].iloc[i] - df["recall"].iloc[i-1]
            avg_p = (df["precision"].iloc[i] + df["precision"].iloc[i-1]) / 2
            auprc += dr * avg_p
        return float(abs(auprc))


class CrossValidationEvaluator:
    """K-fold cross-validation evaluation."""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        n = len(scores)
        fold_size = n // self.n_folds
        fold_metrics = []
        ev = BinaryClassificationEvaluator()
        for k in range(self.n_folds):
            val_start = k * fold_size
            val_end   = val_start + fold_size if k < self.n_folds - 1 else n
            val_mask  = np.zeros(n, dtype=bool)
            val_mask[val_start:val_end] = True
            y_val  = labels[val_mask]
            s_val  = scores[val_mask]
            if y_val.sum() == 0 or len(y_val) == 0:
                continue
            m = ev.compute(y_val, s_val)
            fold_metrics.append(m)
        if not fold_metrics:
            return {"cv_f1_mean": 0.0, "cv_f1_std": 0.0, "cv_auc_mean": 0.0}
        f1s  = [m["f1"]      for m in fold_metrics]
        aucs = [m["auc_roc"] for m in fold_metrics]
        return {
            "cv_f1_mean":  round(float(np.mean(f1s)), 4),
            "cv_f1_std":   round(float(np.std(f1s)), 4),
            "cv_auc_mean": round(float(np.mean(aucs)), 4),
            "cv_auc_std":  round(float(np.std(aucs)), 4),
            "n_folds":     len(fold_metrics),
        }


class BootstrapConfidenceInterval:
    """Bootstrap confidence intervals for evaluation metrics."""

    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap

    def ci_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
              alpha: float = 0.05) -> dict:
        ev = BinaryClassificationEvaluator()
        f1_samples = []
        n = len(y_true)
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]
            if yt.sum() == 0:
                continue
            f1_samples.append(ev.compute(yt, yp)["f1"])
        f1_samples = np.array(f1_samples)
        lo = np.percentile(f1_samples, 100 * alpha / 2)
        hi = np.percentile(f1_samples, 100 * (1 - alpha / 2))
        return {
            "f1_mean": round(float(f1_samples.mean()), 4),
            "f1_ci_lower": round(float(lo), 4),
            "f1_ci_upper": round(float(hi), 4),
            "n_bootstrap":  self.n_bootstrap,
        }


class ModelComparison:
    """Compare multiple model variants on same dataset."""

    def compare(self, models: dict, y_true: np.ndarray) -> pd.DataFrame:
        rows = []
        ev = BinaryClassificationEvaluator()
        for model_name, y_pred in models.items():
            m = ev.compute(y_true, y_pred)
            m["model"] = model_name
            rows.append(m)
        df = pd.DataFrame(rows)
        if "f1" in df.columns:
            df = df.sort_values("f1", ascending=False)
        return df

    def statistical_test(self, scores_a: np.ndarray, scores_b: np.ndarray,
                          y_true: np.ndarray) -> dict:
        """McNemar's test for comparing two models."""
        preds_a = (scores_a >= 0.5).astype(int)
        preds_b = (scores_b >= 0.5).astype(int)
        b = int(((preds_a == 1) & (preds_b == 0) & (y_true == 1)).sum())
        c = int(((preds_a == 0) & (preds_b == 1) & (y_true == 1)).sum())
        if b + c == 0:
            return {"mcnemar_statistic": 0.0, "significant": False}
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        significant = statistic > 3.841  # chi2 critical value at 0.05
        return {"mcnemar_statistic": round(float(statistic), 4),
                "significant": bool(significant),
                "b": b, "c": c}


class EvaluationReporter:
    """Generates comprehensive evaluation report."""

    def generate_report(self, module_name: str, metrics: dict,
                        regime_df: pd.DataFrame,
                        cv_results: dict, bootstrap_ci: dict,
                        output_dir: str) -> str:
        report = {
            "module":          module_name,
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "main_metrics":    metrics,
            "cross_validation": cv_results,
            "bootstrap_ci":    bootstrap_ci,
            "regime_count":    len(regime_df),
        }
        path = os.path.join(output_dir, "full_evaluation_report.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path

    def print_summary(self, metrics: dict) -> None:
        print(f"  ┌─ Classification Metrics ─────────────────────┐")
        print(f"  │ F1:        {metrics.get('f1', 0):.4f}                         │")
        print(f"  │ AUC-ROC:   {metrics.get('auc_roc', 0):.4f}                         │")
        print(f"  │ Precision: {metrics.get('precision', 0):.4f}                         │")
        print(f"  │ Recall:    {metrics.get('recall', 0):.4f}                         │")
        print(f"  │ MCC:       {metrics.get('mcc', 0):.4f}                         │")
        print(f"  └──────────────────────────────────────────────┘")

