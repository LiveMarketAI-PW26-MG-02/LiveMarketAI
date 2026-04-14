"""Generate Markdown evaluation report."""
import json
from datetime import datetime

def generate_report(results, output_path="report.md"):
    lines=["# Regime-Conditioned Attention Report",
           f"Generated: {datetime.utcnow().isoformat()} UTC","",
           "## Summary","",
           f"- Best val loss : {results.get('best_val_loss','N/A')}",
           f"- Regime sep    : {results.get('regime_sep','N/A')}","",
           "| Regime | "+" | ".join(f"X{i}" for i in range(8))+" |",
           "|--------|"+"---------|"*8]
    for r,w in results.get("regime_weights",{}).items():
        lines.append(f"| {r} | "+" | ".join(f"{x:.3f}" for x in w)+" |")
    lines+=["","```json",json.dumps(results.get("ablation",{}),indent=2),"```"]
    with open(output_path,"w") as f: f.write("
".join(lines))
    print(f"Report → {output_path}")
