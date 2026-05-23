from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import indicator as r_indicator
from .routes import shock as r_shock
from .routes import causal_link as r_causal_link
from .routes import scenario as r_scenario
from .routes import forecast as r_forecast

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_indicator.router)
api_router.include_router(r_shock.router)
api_router.include_router(r_causal_link.router)
api_router.include_router(r_scenario.router)
api_router.include_router(r_forecast.router)
