from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import company as r_company
from .routes import financials as r_financials
from .routes import news_signal as r_news_signal
from .routes import warning_score as r_warning_score
from .routes import filing as r_filing

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_company.router)
api_router.include_router(r_financials.router)
api_router.include_router(r_news_signal.router)
api_router.include_router(r_warning_score.router)
api_router.include_router(r_filing.router)
