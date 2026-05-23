from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import commodity_tick as r_commodity_tick
from .routes import manipulation_alert as r_manipulation_alert
from .routes import pattern as r_pattern
from .routes import venue as r_venue
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_commodity_tick.router)
api_router.include_router(r_manipulation_alert.router)
api_router.include_router(r_pattern.router)
api_router.include_router(r_venue.router)
api_router.include_router(r_explanation.router)
