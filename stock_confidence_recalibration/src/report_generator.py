"""
HTML Report Generator
=====================
Generates a standalone HTML report with charts showing
raw vs recalibrated confidence performance.
"""

import json
import os
from typing import Dict, List
from datetime import datetime


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stock Confidence Recalibration Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin-bottom: 28px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .card h2 {{ font-size: 1rem; color: var(--accent); margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
  .chart-wrap {{ position: relative; height: 240px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 10px; border-bottom: 1px solid rgba(48,54,61,0.5); }}
  tr:last-child td {{ border: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }} .purple {{ color: var(--purple); }}
  .badge.bull {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge.bear {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge.side {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
  .stat-row {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .stat {{ background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.2); border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 120px; }}
  .stat .val {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
  .stat .lbl {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .tag {{ font-size: 0.72rem; padding: 1px 6px; border-radius: 3px; }}
  .improving {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .worsening {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .stable {{ background: rgba(139,148,158,0.15); color: var(--muted); }}
</style>
</head>
<body>
<h1>⚡ Online Stock Confidence Recalibration</h1>
<p class="subtitle">Generated: {timestamp} &nbsp;|&nbsp; {total_ticks} ticks processed &nbsp;|&nbsp; {n_assets} assets &nbsp;|&nbsp; Runtime: {runtime:.1f}s</p>

<div class="stat-row">
  <div class="stat"><div class="val">{avg_ece_improvement:+.3f}</div><div class="lbl">Avg ECE Improvement</div></div>
  <div class="stat"><div class="val">{avg_brier_improvement:+.3f}</div><div class="lbl">Avg Brier Improvement</div></div>
  <div class="stat"><div class="val">{n_miscal_alerts}</div><div class="lbl">Miscal Alerts Fired</div></div>
  <div class="stat"><div class="val">{n_drift_alerts}</div><div class="lbl">Drift Alerts Fired</div></div>
  <div class="stat"><div class="val">{avg_accuracy:.1%}</div><div class="lbl">Avg Prediction Accuracy</div></div>
</div>

<div class="grid">
  <div class="card full-width">
    <h2>📊 Benchmark: Raw vs Recalibrated ECE</h2>
    <div class="chart-wrap"><canvas id="eceChart"></canvas></div>
  </div>
  <div class="card">
    <h2>🎯 Brier Score Comparison</h2>
    <div class="chart-wrap"><canvas id="brierChart"></canvas></div>
  </div>
  <div class="card">
    <h2>📈 Confidence Sharpness Change</h2>
    <div class="chart-wrap"><canvas id="sharpChart"></canvas></div>
  </div>
  <div class="card full-width">
    <h2>🔬 Per-Asset Detailed Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Predictions</th><th>Accuracy</th>
          <th>Raw ECE</th><th>Cal ECE</th><th>ECE Δ</th>
          <th>Raw Brier</th><th>Cal Brier</th><th>ECE Trend</th><th>Reliability</th>
        </tr>
      </thead>
      <tbody id="detailRows"></tbody>
    </table>
  </div>
  <div class="card">
    <h2>⚠️ Miscalibration Alerts</h2>
    <table>
      <thead><tr><th>Symbol</th><th>ECE</th><th>Severity</th><th>Time</th></tr></thead>
      <tbody id="miscalRows"></tbody>
    </table>
  </div>
  <div class="card">
    <h2>📉 Confidence Drift Report</h2>
    <table>
      <thead><tr><th>Symbol</th><th>Drift</th><th>Direction</th><th>Alert</th></tr></thead>
      <tbody id="driftRows"></tbody>
    </table>
  </div>
</div>

<script>
const DATA = {data_json};

const symbols = DATA.benchmark.map(d => d.symbol);
const rawEce  = DATA.benchmark.map(d => d.raw?.ece ?? 0);
const calEce  = DATA.benchmark.map(d => d.calibrated?.ece ?? 0);
const rawBrier = DATA.benchmark.map(d => d.raw?.brier ?? 0);
const calBrier = DATA.benchmark.map(d => d.calibrated?.brier ?? 0);
const sharp   = DATA.benchmark.map(d => d.sharpness_change ?? 0);

const baseOpts = (title) => ({{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }}, title: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
  }}
}});

new Chart('eceChart', {{
  type: 'bar',
  data: {{
    labels: symbols,
    datasets: [
      {{ label: 'Raw ECE',         data: rawEce,  backgroundColor: 'rgba(248,81,73,0.7)' }},
      {{ label: 'Calibrated ECE',  data: calEce,  backgroundColor: 'rgba(63,185,80,0.7)' }}
    ]
  }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ ...baseOpts().plugins.legend, position: 'top' }} }} }}
}});

new Chart('brierChart', {{
  type: 'bar',
  data: {{
    labels: symbols,
    datasets: [
      {{ label: 'Raw Brier',        data: rawBrier, backgroundColor: 'rgba(248,81,73,0.7)' }},
      {{ label: 'Calibrated Brier', data: calBrier, backgroundColor: 'rgba(88,166,255,0.7)' }}
    ]
  }},
  options: baseOpts()
}});

new Chart('sharpChart', {{
  type: 'bar',
  data: {{
    labels: symbols,
    datasets: [{{
      label: 'Sharpness Δ',
      data: sharp,
      backgroundColor: sharp.map(v => v > 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)')
    }}]
  }},
  options: baseOpts()
}});

// Detail table
const tbody = document.getElementById('detailRows');
DATA.benchmark.forEach(d => {{
  const acc = DATA.accuracy[d.symbol] || {{}};
  const eceD = (d.raw?.ece ?? 0) - (d.calibrated?.ece ?? 0);
  const trendCls = acc.ece_trend === 'IMPROVING' ? 'improving' : acc.ece_trend === 'WORSENING' ? 'worsening' : 'stable';
  const accV = acc.accuracy || 0;
  const accCls = accV > 0.55 ? 'green' : accV < 0.45 ? 'red' : 'yellow';
  tbody.innerHTML += `<tr>
    <td><b>${{d.symbol}}</b></td>
    <td>${{acc.total || 0}}</td>
    <td class="${{accCls}}">${{(accV*100).toFixed(1)}}%</td>
    <td class="red">${{(d.raw?.ece??0).toFixed(4)}}</td>
    <td class="green">${{(d.calibrated?.ece??0).toFixed(4)}}</td>
    <td class="${{eceD>0?'green':'red'}}">${{eceD>0?'+':''}}${{eceD.toFixed(4)}}</td>
    <td class="red">${{(d.raw?.brier??0).toFixed(4)}}</td>
    <td class="green">${{(d.calibrated?.brier??0).toFixed(4)}}</td>
    <td><span class="tag ${{trendCls}}">${{acc.ece_trend||'STABLE'}}</span></td>
    <td>${{(acc.reliability||0).toFixed(3)}}</td>
  </tr>`;
}});

// Miscal alerts
const mRows = document.getElementById('miscalRows');
(DATA.miscal_alerts || []).slice(-10).forEach(a => {{
  const sev = a.severity;
  const cls = sev==='HIGH'?'red':sev==='MEDIUM'?'yellow':'green';
  mRows.innerHTML += `<tr><td><b>${{a.symbol}}</b></td><td>${{a.ece.toFixed(4)}}</td>
    <td class="${{cls}}">${{sev}}</td><td>t=${{(a.timestamp||0).toFixed(0)}}</td></tr>`;
}});
if (!DATA.miscal_alerts?.length) mRows.innerHTML = '<tr><td colspan="4" style="color:#8b949e">No alerts</td></tr>';

// Drift report
const dRows = document.getElementById('driftRows');
Object.entries(DATA.drift || {{}}).forEach(([sym, d]) => {{
  const cls = d.alert ? 'red' : 'green';
  const dir = d.direction || 'STABLE';
  dRows.innerHTML += `<tr><td><b>${{sym}}</b></td><td>${{(d.drift||0).toFixed(4)}}</td>
    <td>${{dir}}</td><td class="${{cls}}">${{d.alert?'⚠ YES':'✓ NO'}}</td></tr>`;
}});
</script>
</body>
</html>"""


def generate_html_report(pipeline, total_ticks: int, runtime: float,
                         output_path: str = "output/report.html"):
    benchmark = pipeline.get_benchmark_report()
    accuracy  = pipeline.get_accuracy_table()
    drift     = pipeline.get_drift_report()
    miscal    = pipeline.miscal_detect.alert_log[-20:]  # last 20 alerts

    avg_ece_imp   = sum(d.get("ece_improvement", 0)   for d in benchmark) / max(len(benchmark), 1)
    avg_brier_imp = sum(d.get("brier_improvement", 0) for d in benchmark) / max(len(benchmark), 1)
    avg_acc       = sum(v.get("accuracy", 0) for v in accuracy.values()) / max(len(accuracy), 1)
    n_drift_alerts = sum(1 for d in drift.values() if d.get("alert"))

    data_json = json.dumps({
        "benchmark":    benchmark,
        "accuracy":     accuracy,
        "drift":        drift,
        "miscal_alerts": miscal,
    }, default=str)

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_ticks=total_ticks,
        n_assets=len(accuracy),
        runtime=runtime,
        avg_ece_improvement=avg_ece_imp,
        avg_brier_improvement=avg_brier_imp,
        n_miscal_alerts=len(miscal),
        n_drift_alerts=n_drift_alerts,
        avg_accuracy=avg_acc,
        data_json=data_json,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
