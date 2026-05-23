from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import e_t_f as r_e_t_f
from .routes import sector as r_sector
from .routes import rotation_signal as r_rotation_signal
from .routes import allocation as r_allocation
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_e_t_f.router)
api_router.include_router(r_sector.router)
api_router.include_router(r_rotation_signal.router)
api_router.include_router(r_allocation.router)
api_router.include_router(r_explanation.router)
