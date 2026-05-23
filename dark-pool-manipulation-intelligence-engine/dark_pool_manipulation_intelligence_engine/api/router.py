from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import dark_pool_print as r_dark_pool_print
from .routes import block as r_block
from .routes import manipulation_signal as r_manipulation_signal
from .routes import venue as r_venue
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_dark_pool_print.router)
api_router.include_router(r_block.router)
api_router.include_router(r_manipulation_signal.router)
api_router.include_router(r_venue.router)
api_router.include_router(r_explanation.router)
