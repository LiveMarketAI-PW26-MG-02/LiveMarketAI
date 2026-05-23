from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import sentiment_signal as r_sentiment_signal
from .routes import confidence_index as r_confidence_index
from .routes import cohort as r_cohort
from .routes import source as r_source
from .routes import snapshot as r_snapshot

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_sentiment_signal.router)
api_router.include_router(r_confidence_index.router)
api_router.include_router(r_cohort.router)
api_router.include_router(r_source.router)
api_router.include_router(r_snapshot.router)
