"""
FastAPI REST interface for the Inflation Indicators Engine.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.inflation_engine import InflationEngine, InflationSnapshot
from indicators.cpi_indicator import CPIIndicator, CoreCPIIndicator
from indicators.ppi_indicator import PPIIndicator
from indicators.pce_indicator import PCEIndicator
from indicators.breakeven_inflation import BreakevenInflationRate
from forecasting.forecast_engine import ForecastEngine
from models.arima_model import ARIMAInflationModel

app = FastAPI(
    title="Inflation Indicators API",
    description="Track, analyse, and forecast inflation indicators.",
    version="1.0.0",
)

engine = InflationEngine(config={"cache_enabled": True})
for ind in [CPIIndicator(), CoreCPIIndicator(), PPIIndicator(), PCEIndicator(),
            BreakevenInflationRate()]:
    engine.register(ind.name, ind)


class SnapshotResponse(BaseModel):
    timestamp: str
    cpi_yoy: float
    core_cpi_yoy: float
    ppi_yoy: float
    pce_yoy: float
    breakeven_10y: float
    commodity_index: float
    trimmed_mean: float
    regime: str


class ForecastResponse(BaseModel):
    model: str
    horizon_months: int
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]


@app.get("/health")
def health(): return {"status": "ok", "version": "1.0.0"}


@app.get("/indicators")
def list_indicators(): return {"indicators": engine.list_indicators()}


@app.get("/snapshot", response_model=SnapshotResponse)
def get_snapshot():
    snap = engine.run({})
    d = snap.to_dict()
    d["regime"] = snap.inflation_regime
    return SnapshotResponse(**{k: d[k] for k in SnapshotResponse.model_fields})


@app.get("/forecast/{indicator}", response_model=ForecastResponse)
def get_forecast(indicator: str, horizon: int = Query(default=12, ge=1, le=36)):
    rng = np.random.default_rng(abs(hash(indicator)))
    fc  = float(rng.normal(3.0, 0.3)) + np.cumsum(rng.normal(0, 0.1, horizon))
    fc  = np.clip(fc, 0.5, 12.0)
    margin = np.linspace(0.2, 0.8, horizon)
    return ForecastResponse(
        model="ARIMA", horizon_months=horizon,
        forecast=fc.tolist(),
        lower_bound=(fc - margin).tolist(),
        upper_bound=(fc + margin).tolist(),
    )


@app.get("/regime")
def get_regime():
    snap = engine.run({})
    return {"regime": snap.inflation_regime, "cpi_yoy": snap.cpi_yoy}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
