from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import advisor as r_advisor
from .routes import allocation as r_allocation
from .routes import rationale as r_rationale
from .routes import client_profile as r_client_profile
from .routes import compliance_check as r_compliance_check

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_advisor.router)
api_router.include_router(r_allocation.router)
api_router.include_router(r_rationale.router)
api_router.include_router(r_client_profile.router)
api_router.include_router(r_compliance_check.router)
