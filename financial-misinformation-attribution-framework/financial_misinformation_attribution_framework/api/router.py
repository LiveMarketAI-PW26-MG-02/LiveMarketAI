from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import claim as r_claim
from .routes import source as r_source
from .routes import attribution_result as r_attribution_result
from .routes import narrative as r_narrative
from .routes import evidence as r_evidence

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_claim.router)
api_router.include_router(r_source.router)
api_router.include_router(r_attribution_result.router)
api_router.include_router(r_narrative.router)
api_router.include_router(r_evidence.router)
