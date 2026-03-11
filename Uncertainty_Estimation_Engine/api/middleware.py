"""
API Middleware
===============
Request logging, rate limiting, error handling, and CORS support.
"""

import time
import logging
import uuid
from typing import Any, Dict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RequestLogger:
    """Log each HTTP request with timing and request ID."""

    def __init__(self, app: Any):
        self.app = app
        try:
            from flask import request, g
            @app.before_request
            def before():
                g.request_id   = str(uuid.uuid4())[:8]
                g.start_time   = time.perf_counter()
                logger.info("[%s] %s %s", g.request_id, request.method, request.path)

            @app.after_request
            def after(response):
                elapsed = (time.perf_counter() - getattr(g, "start_time", time.perf_counter())) * 1000
                logger.info(
                    "[%s] %d  %.1f ms",
                    getattr(g, "request_id", "-"), response.status_code, elapsed
                )
                response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
                return response
        except Exception:
            pass


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, app: Any, requests_per_minute: int = 60):
        self.rpm    = requests_per_minute
        self.window = 60.0
        self._log: Dict[str, deque] = defaultdict(deque)

        try:
            from flask import request, jsonify
            @app.before_request
            def check_rate():
                ip  = request.remote_addr or "unknown"
                now = time.time()
                dq  = self._log[ip]
                while dq and now - dq[0] > self.window:
                    dq.popleft()
                if len(dq) >= self.rpm:
                    return jsonify({"error": "Rate limit exceeded. Retry after 60 s."}), 429
                dq.append(now)
        except Exception:
            pass


class ErrorHandler:
    """Global Flask error handler returning JSON responses."""

    def __init__(self, app: Any):
        try:
            from flask import jsonify

            @app.errorhandler(400)
            def bad_request(e):
                return jsonify({"error": "Bad Request", "detail": str(e)}), 400

            @app.errorhandler(404)
            def not_found(e):
                return jsonify({"error": "Not Found"}), 404

            @app.errorhandler(405)
            def method_not_allowed(e):
                return jsonify({"error": "Method Not Allowed"}), 405

            @app.errorhandler(500)
            def server_error(e):
                logger.exception("Unhandled server error")
                return jsonify({"error": "Internal Server Error"}), 500

            @app.errorhandler(Exception)
            def handle_exception(e):
                logger.exception("Unhandled exception")
                return jsonify({"error": type(e).__name__, "detail": str(e)}), 500

        except Exception:
            pass


class CORSMiddleware:
    """Add CORS headers to allow cross-origin requests."""

    ALLOWED_ORIGINS = {"*"}

    def __init__(self, app: Any, allowed_origins=None):
        if allowed_origins:
            self.ALLOWED_ORIGINS = set(allowed_origins)
        try:
            @app.after_request
            def add_cors(response):
                response.headers["Access-Control-Allow-Origin"]  = "*"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
                response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
                return response
        except Exception:
            pass
