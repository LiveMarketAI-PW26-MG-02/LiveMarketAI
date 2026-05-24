from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import transcript as r_transcript
from .routes import confidence_score as r_confidence_score
from .routes import executive as r_executive
from .routes import evidence as r_evidence
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_transcript.router)
api_router.include_router(r_confidence_score.router)
api_router.include_router(r_executive.router)
api_router.include_router(r_evidence.router)
api_router.include_router(r_explanation.router)
