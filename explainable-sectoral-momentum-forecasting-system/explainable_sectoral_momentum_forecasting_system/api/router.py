from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import sector as r_sector
from .routes import momentum_score as r_momentum_score
from .routes import forecast as r_forecast
from .routes import feature_contribution as r_feature_contribution
from .routes import constituent as r_constituent

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_sector.router)
api_router.include_router(r_momentum_score.router)
api_router.include_router(r_forecast.router)
api_router.include_router(r_feature_contribution.router)
api_router.include_router(r_constituent.router)
