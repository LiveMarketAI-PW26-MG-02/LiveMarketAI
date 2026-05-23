from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ..core.logging import get_logger

logger = get_logger("request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path,
                    response.status_code, elapsed_ms)
        return response
