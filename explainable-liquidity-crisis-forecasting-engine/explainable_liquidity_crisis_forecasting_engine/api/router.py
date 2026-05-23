from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import liquidity_metric as r_liquidity_metric
from .routes import crisis_signal as r_crisis_signal
from .routes import institution as r_institution
from .routes import forecast as r_forecast
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_liquidity_metric.router)
api_router.include_router(r_crisis_signal.router)
api_router.include_router(r_institution.router)
api_router.include_router(r_forecast.router)
api_router.include_router(r_explanation.router)
