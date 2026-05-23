from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import derivative as r_derivative
from .routes import risk_factor as r_risk_factor
from .routes import decomposition as r_decomposition
from .routes import greeks as r_greeks
from .routes import stress_scenario as r_stress_scenario

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_derivative.router)
api_router.include_router(r_risk_factor.router)
api_router.include_router(r_decomposition.router)
api_router.include_router(r_greeks.router)
api_router.include_router(r_stress_scenario.router)
