from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import media_clip as r_media_clip
from .routes import authenticity_signal as r_authenticity_signal
from .routes import acoustic_feature as r_acoustic_feature
from .routes import subject as r_subject
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_media_clip.router)
api_router.include_router(r_authenticity_signal.router)
api_router.include_router(r_acoustic_feature.router)
api_router.include_router(r_subject.router)
api_router.include_router(r_explanation.router)
