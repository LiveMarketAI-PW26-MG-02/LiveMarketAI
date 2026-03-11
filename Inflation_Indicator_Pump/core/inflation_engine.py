"""
Central Inflation Engine.
Orchestrates data ingestion, indicator computation, and forecasting.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InflationSnapshot:
    """Point-in-time snapshot of all computed inflation indicators."""
    timestamp: pd.Timestamp
    cpi_yoy: float
    core_cpi_yoy: float
    ppi_yoy: float
    pce_yoy: float
    breakeven_10y: float
    commodity_index: float
    trimmed_mean: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": str(self.timestamp),
            "cpi_yoy": self.cpi_yoy,
            "core_cpi_yoy": self.core_cpi_yoy,
            "ppi_yoy": self.ppi_yoy,
            "pce_yoy": self.pce_yoy,
            "breakeven_10y": self.breakeven_10y,
            "commodity_index": self.commodity_index,
            "trimmed_mean": self.trimmed_mean,
            "metadata": self.metadata,
        }

    @property
    def inflation_regime(self) -> str:
        if self.cpi_yoy < 1.0: return "deflation"
        if self.cpi_yoy < 2.0: return "low"
        if self.cpi_yoy < 4.0: return "moderate"
        if self.cpi_yoy < 7.0: return "high"
        return "hyperinflation"


class InflationEngine:
    """
    Main orchestrator for the Inflation Indicators system.
    Registers indicators, runs the pipeline, and exposes results.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._indicators: Dict[str, Any] = {}
        self._snapshots: List[InflationSnapshot] = []
        self._latest: Optional[InflationSnapshot] = None
        logger.info("InflationEngine initialised. Config: %s", self.config)

    def register(self, name: str, indicator) -> None:
        self._indicators[name] = indicator
        logger.debug("Registered indicator: %s", name)

    def list_indicators(self) -> List[str]:
        return list(self._indicators.keys())

    def run(self, data: Dict[str, pd.DataFrame]) -> InflationSnapshot:
        """Compute all registered indicators and build a snapshot."""
        results: Dict[str, float] = {}
        for name, ind in self._indicators.items():
            try:
                val = ind.compute(data.get(name, pd.DataFrame()))
                results[name] = float(val) if np.isfinite(val) else 0.0
            except Exception as exc:
                logger.warning("Indicator '%s' failed: %s", name, exc)
                results[name] = 0.0

        snap = InflationSnapshot(
            timestamp=pd.Timestamp.now(),
            cpi_yoy=results.get("cpi", 0.0),
            core_cpi_yoy=results.get("core_cpi", 0.0),
            ppi_yoy=results.get("ppi", 0.0),
            pce_yoy=results.get("pce", 0.0),
            breakeven_10y=results.get("breakeven", 0.0),
            commodity_index=results.get("commodity", 100.0),
            trimmed_mean=results.get("trimmed_mean", 0.0),
            metadata={"n_indicators": len(results)},
        )
        self._snapshots.append(snap)
        self._latest = snap
        return snap

    @property
    def latest(self) -> Optional[InflationSnapshot]:
        return self._latest

    def history(self) -> List[Dict]:
        return [s.to_dict() for s in self._snapshots]

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self._snapshots])
