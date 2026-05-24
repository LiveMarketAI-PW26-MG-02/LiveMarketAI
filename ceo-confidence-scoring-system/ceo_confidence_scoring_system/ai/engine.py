from __future__ import annotations

import math

import numpy as np



class Engine:
    """Weighted-evidence trust scorer with a transparent breakdown."""

    EVIDENCE = {"fundamentals": 0.30, "analyst_consensus": 0.20,
                "price_action": 0.20, "liquidity": 0.15, "source_reliability": 0.15}

    def explain(self, features: dict) -> dict:
        breakdown = []
        total = 0.0
        for name, weight in self.EVIDENCE.items():
            value = max(0.0, min(1.0, float(features.get(name, 0.5))))
            contribution = weight * value
            total += contribution
            breakdown.append({"evidence": name, "weight": weight,
                              "value": round(value, 3),
                              "contribution": round(contribution, 4)})
        score = 1.0 / (1.0 + math.exp(-6 * (total - 0.5)))
        breakdown.sort(key=lambda b: b["contribution"], reverse=True)
        tier = "high" if score >= 0.66 else "medium" if score >= 0.4 else "low"
        return {"primitive": "trust", "trust_score": round(score, 4), "tier": tier,
                "summary": f"Trust {score:.0%} ({tier}); strongest evidence: "
                           f"{breakdown[0]['evidence']}.",
                "breakdown": breakdown}
