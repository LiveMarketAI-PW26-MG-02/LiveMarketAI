"""regime_classifier.py — rule-based regime + trend classifier"""
import numpy as np
from typing import Dict, List


class RegimeClassifier:
    def classify(self, vol: float, returns: np.ndarray) -> str:
        avg_return = float(np.mean(returns[-10:])) if len(returns) >= 10 else 0
        if vol < 0.005:
            return "LOW_VOL"
        elif vol < 0.012:
            return "MEDIUM_VOL"
        elif vol < 0.025:
            return "HIGH_VOL"
        else:
            return "CRISIS"

    def trend(self, returns: np.ndarray) -> str:
        if len(returns) < 5:
            return "—"
        ma5 = float(np.mean(returns[-5:]))
        ma20 = float(np.mean(returns[-20:])) if len(returns) >= 20 else ma5
        if ma5 > ma20 and ma5 > 0:
            return "↑ Uptrend"
        elif ma5 < ma20 and ma5 < 0:
            return "↓ Downtrend"
        else:
            return "→ Sideways"

    def market_summary(self, results: List[Dict]) -> Dict:
        if not results:
            return {"market_regime": "UNKNOWN", "confidence": 0, "recommendation": ""}
        regimes = [r["regime"] for r in results]
        vols = [r["realized_vol"] for r in results]
        avg_vol = float(np.mean(vols)) if vols else 0
        crisis_count = regimes.count("CRISIS")
        high_count = regimes.count("HIGH_VOL")
        low_count = regimes.count("LOW_VOL")

        if crisis_count >= len(results) * 0.4:
            market_regime = "CRISIS"
            conf = 0.85
            rec = "Reduce exposure, activate defensive strategies"
        elif high_count >= len(results) * 0.5:
            market_regime = "HIGH_VOL"
            conf = 0.75
            rec = "Apply volatility scaling, reduce position sizes"
        elif low_count >= len(results) * 0.6:
            market_regime = "LOW_VOL"
            conf = 0.80
            rec = "Normal allocation, monitor for regime shifts"
        else:
            market_regime = "MEDIUM_VOL"
            conf = 0.65
            rec = "Moderate allocation, watch macro signals"

        return {
            "market_regime": market_regime,
            "confidence": conf,
            "avg_vol": avg_vol,
            "recommendation": rec,
        }
