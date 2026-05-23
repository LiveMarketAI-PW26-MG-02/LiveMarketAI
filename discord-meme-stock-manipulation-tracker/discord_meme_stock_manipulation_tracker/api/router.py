from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import server as r_server
from .routes import post as r_post
from .routes import meme_stock_signal as r_meme_stock_signal
from .routes import user as r_user
from .routes import coordination as r_coordination

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_server.router)
api_router.include_router(r_post.router)
api_router.include_router(r_meme_stock_signal.router)
api_router.include_router(r_user.router)
api_router.include_router(r_coordination.router)
