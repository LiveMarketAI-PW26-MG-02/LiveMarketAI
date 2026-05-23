from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router
from .config import get_settings
from .core.errors import DomainError, domain_error_handler
from .core.logging import configure_logging, get_logger
from .db.session import init_db
from .middleware.request_context import RequestContextMiddleware

configure_logging()
logger = get_logger("causal_attention_based_trading_explanation_engine")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting %s", settings.title)
    init_db()
    yield
    logger.info("stopping %s", settings.title)


app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(DomainError, domain_error_handler)
app.include_router(api_router)


@app.get("/")
def root():
    return {"service": settings.title, "slug": settings.slug, "docs": "/docs"}
