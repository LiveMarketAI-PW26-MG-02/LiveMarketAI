from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import emotion_signal as r_emotion_signal
from .routes import volatility_forecast as r_volatility_forecast
from .routes import source as r_source
from .routes import feature_contribution as r_feature_contribution
from .routes import snapshot as r_snapshot

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_emotion_signal.router)
api_router.include_router(r_volatility_forecast.router)
api_router.include_router(r_source.router)
api_router.include_router(r_feature_contribution.router)
api_router.include_router(r_snapshot.router)
