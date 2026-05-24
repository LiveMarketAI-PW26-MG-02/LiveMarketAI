from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import influencer as r_influencer
from .routes import post as r_post
from .routes import campaign_signal as r_campaign_signal
from .routes import edge as r_edge
from .routes import detection_alert as r_detection_alert

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_influencer.router)
api_router.include_router(r_post.router)
api_router.include_router(r_campaign_signal.router)
api_router.include_router(r_edge.router)
api_router.include_router(r_detection_alert.router)
