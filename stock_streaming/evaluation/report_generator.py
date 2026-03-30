"""
report_generator.py — Aggregates results from all evaluation modules and
produces a human-readable text report plus a JSON data export.
"""

import json
import time
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


class Section:
    """One section of a report with a title and body lines."""

    def __init__(self, title: str):
        self.title = title
        self._lines: List[str] = []

    def add(self, line: str) -> "Section":
        self._lines.append(line)
        return self

    def add_dict(self, d: Dict, indent: int = 2) -> "Section":
        pad = " " * indent
        for k, v in d.items():
            self._lines.append(f"{pad}{str(k):<30}: {v}")
        return self

    def add_table(self, headers: List[str], rows: List[List[Any]],
                  col_width: int = 14) -> "Section":
        hdr = "  " + "".join(h.ljust(col_width) for h in headers)
        self._lines.append(hdr)
        self._lines.append("  " + "-" * (col_width * len(headers)))
        for row in rows:
            self._lines.append("  " + "".join(str(v).ljust(col_width) for v in row))
        return self

    def render(self) -> str:
        width = 60
        sep = "=" * width
        body = "\n".join(self._lines)
        return f"\n{sep}\n  {self.title}\n{sep}\n{body}"


class ReportGenerator:
    """
    Collects result dicts from evaluation modules and writes a final report.

    Usage:
        rg = ReportGenerator()
        rg.add_section("Latency", latency_benchmark.summary())
        rg.add_section("Batch vs Streaming", comparison_results)
        rg.add_section("Responsiveness", responsiveness_results)
        rg.save("my_report")
    """

    def __init__(self, title: str = "Stock Streaming Classification System — Evaluation Report"):
        self._title    = title
        self._sections: List[Section] = []
        self._raw_data: Dict[str, Any] = {}
        self._created_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_section(self, title: str, data: Dict) -> "ReportGenerator":
        sec = Section(title)
        sec.add_dict(data)
        self._sections.append(sec)
        self._raw_data[title] = data
        return self

    def add_latency_section(self, summary: Dict) -> "ReportGenerator":
        sec = Section(f"Latency Benchmark — {summary.get('name', '')}")
        sec.add(f"  Samples         : {summary.get('n_predictions', 0)}")
        sec.add(f"  Mean latency    : {summary.get('mean_ms', summary.get('mean_latency_ms', 0)):.3f} ms")
        sec.add(f"  P90 latency     : {summary.get('p90_ms', summary.get('p90_latency_ms', 0)):.3f} ms")
        sec.add(f"  P99 latency     : {summary.get('p99_ms', summary.get('p99_latency_ms', 0)):.3f} ms")
        sla = summary.get("sla_compliance", 0)
        sec.add(f"  SLA compliance  : {sla*100:.1f}%")
        sec.add(f"  Timeout count   : {summary.get('timeout_count', 0)}")
        self._sections.append(sec)
        self._raw_data[sec.title] = summary
        return self

    def add_comparison_section(self, batch: Dict, streaming: Dict) -> "ReportGenerator":
        sec = Section("Batch vs Streaming Comparison")
        metrics = [
            ("Mean latency (ms)",  "mean_latency_ms"),
            ("P99 latency (ms)",   "p99_latency_ms"),
            ("SLA compliance",     "sla_compliance"),
            ("Label flip rate",    "flip_rate"),
            ("Mean confidence",    "mean_confidence"),
            ("Predictions",        "n_predictions"),
        ]
        headers = ["Metric", "Batch", "Streaming", "Delta"]
        rows = []
        for label, key in metrics:
            bv = batch.get(key, "N/A")
            sv = streaming.get(key, "N/A")
            if isinstance(bv, float) and isinstance(sv, float) and bv != 0:
                delta = f"{(sv - bv) / abs(bv) * 100:+.1f}%"
            else:
                delta = "N/A"
            rows.append([label,
                         f"{bv:.4f}" if isinstance(bv, float) else str(bv),
                         f"{sv:.4f}" if isinstance(sv, float) else str(sv),
                         delta])
        sec.add_table(headers, rows, col_width=16)
        self._sections.append(sec)
        self._raw_data["comparison"] = {"batch": batch, "streaming": streaming}
        return self

    def add_responsiveness_section(self, results: Dict[int, Dict]) -> "ReportGenerator":
        sec = Section("Window Size Responsiveness vs Stability")
        headers = ["WinSize", "FlipRate", "DetLag", "Stability", "Response", "Score"]
        rows = []
        for ws, r in sorted(results.items()):
            dl = str(r.get("detection_lag_ticks", "N/A"))
            rows.append([
                ws,
                f"{r.get('flip_rate', 0):.5f}",
                dl,
                f"{r.get('stability_score', 0):.4f}",
                f"{r.get('responsiveness_score', 0):.4f}",
                f"{r.get('composite_score', 0):.4f}",
            ])
        sec.add_table(headers, rows, col_width=12)
        best = max(results, key=lambda w: results[w].get("composite_score", 0))
        sec.add(f"\n  ★ Recommended window size: {best}")
        self._sections.append(sec)
        self._raw_data["responsiveness"] = {str(k): v for k, v in results.items()}
        return self

    # ------------------------------------------------------------------
    # Render and save
    # ------------------------------------------------------------------

    def render(self) -> str:
        width = 60
        lines = [
            "=" * width,
            f"  {self._title}",
            f"  Generated: {self._created_at}",
            "=" * width,
        ]
        for sec in self._sections:
            lines.append(sec.render())
        lines.append("\n" + "=" * width)
        lines.append("  END OF REPORT")
        lines.append("=" * width + "\n")
        return "\n".join(lines)

    def save(self, filename_stem: str = "evaluation_report",
             output_dir: Optional[str] = None):
        """
        Write the report as both .txt and .json.
        Returns (txt_path, json_path).
        """
        out_dir = output_dir or REPORT_DIR
        os.makedirs(out_dir, exist_ok=True)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{filename_stem}_{ts}"

        txt_path  = os.path.join(out_dir, f"{stem}.txt")
        json_path = os.path.join(out_dir, f"{stem}.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.render())

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "title":      self._title,
                "created_at": self._created_at,
                "data":       self._raw_data,
            }, f, indent=2, default=str)

        print(f"\n[ReportGenerator] Saved:\n  TXT  → {txt_path}\n  JSON → {json_path}")
        return txt_path, json_path

    def print_report(self) -> None:
        print(self.render())



# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rg = ReportGenerator()

    rg.add_latency_section({
        "name": "StreamingClassifier",
        "n_predictions": 900,
        "mean_ms": 1.23,
        "p90_ms":  2.10,
        "p99_ms":  5.44,
        "sla_compliance": 0.982,
        "timeout_count": 3,
    })

    rg.add_comparison_section(
        batch={
            "n_predictions": 900, "mean_latency_ms": 8.2,
            "p99_latency_ms": 22.1, "sla_compliance": 0.81,
            "flip_rate": 0.32, "mean_confidence": 0.61,
        },
        streaming={
            "n_predictions": 900, "mean_latency_ms": 1.4,
            "p99_latency_ms": 5.5, "sla_compliance": 0.98,
            "flip_rate": 0.18, "mean_confidence": 0.67,
        }
    )

    rg.add_responsiveness_section({
        20:  {"flip_rate": 0.08, "detection_lag_ticks": 5,  "stability_score": 0.2, "responsiveness_score": 0.17, "composite_score": 0.18},
        60:  {"flip_rate": 0.04, "detection_lag_ticks": 12, "stability_score": 0.6, "responsiveness_score": 0.08, "composite_score": 0.34},
        120: {"flip_rate": 0.02, "detection_lag_ticks": 25, "stability_score": 0.8, "responsiveness_score": 0.04, "composite_score": 0.42},
    })

    rg.print_report()
