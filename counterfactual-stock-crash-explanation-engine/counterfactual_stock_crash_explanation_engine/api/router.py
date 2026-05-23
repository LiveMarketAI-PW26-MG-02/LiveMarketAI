from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import crash_event as r_crash_event
from .routes import market_snapshot as r_market_snapshot
from .routes import scenario as r_scenario
from .routes import counterfactual as r_counterfactual
from .routes import explanation_report as r_explanation_report

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_crash_event.router)
api_router.include_router(r_market_snapshot.router)
api_router.include_router(r_scenario.router)
api_router.include_router(r_counterfactual.router)
api_router.include_router(r_explanation_report.router)
