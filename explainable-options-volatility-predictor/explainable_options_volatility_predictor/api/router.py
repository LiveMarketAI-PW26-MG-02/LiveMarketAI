from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import option as r_option
from .routes import vol_surface as r_vol_surface
from .routes import prediction as r_prediction
from .routes import feature_contribution as r_feature_contribution
from .routes import underlying as r_underlying

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_option.router)
api_router.include_router(r_vol_surface.router)
api_router.include_router(r_prediction.router)
api_router.include_router(r_feature_contribution.router)
api_router.include_router(r_underlying.router)
