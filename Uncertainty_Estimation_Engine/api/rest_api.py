"""
REST API — Flask application for the Uncertainty Estimation Engine.
Provides endpoints for single prediction, batch prediction, calibration,
and health/info queries.
"""

import time
import logging
from typing import Any, Dict

try:
    from flask import Flask, request, jsonify, g
except ImportError:
    Flask = None  # type: ignore

from .schemas import PredictRequest, PredictResponse, BatchRequest
from .endpoints import register_endpoints
from .middleware import RequestLogger, RateLimiter, ErrorHandler

logger = logging.getLogger(__name__)


def create_app(engine=None, config: Dict[str, Any] = None) -> Any:
    """
    Application factory.

    Parameters
    ----------
    engine : UncertaintyEngine, optional
        Pre-built engine instance. If None a default engine is created.
    config : dict, optional
        Flask / app-level configuration overrides.

    Returns
    -------
    Flask app
    """
    if Flask is None:
        raise ImportError("Flask is required. Install with: pip install flask")

    app = Flask(__name__)
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB
        **(config or {}),
    )

    # ---------------------------------------------------------------- #
    # Dependency injection
    # ---------------------------------------------------------------- #
    if engine is None:
        from core.bayesian_estimator import BayesianEstimator
        import numpy as np
        engine = BayesianEstimator(prior="normal", n_samples=1000)

    app.config["ENGINE"] = engine

    # ---------------------------------------------------------------- #
    # Middleware
    # ---------------------------------------------------------------- #
    request_logger = RequestLogger(app)
    rate_limiter   = RateLimiter(app, requests_per_minute=120)
    error_handler  = ErrorHandler(app)

    # ---------------------------------------------------------------- #
    # Routes
    # ---------------------------------------------------------------- #
    register_endpoints(app)

    # ---------------------------------------------------------------- #
    # Health check
    # ---------------------------------------------------------------- #
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "timestamp": time.time()})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify({
            "name": "Uncertainty Estimation Engine",
            "version": "1.0.0",
            "methods": ["bayesian", "monte_carlo", "ensemble", "conformal", "gp"],
        })

    logger.info("Flask application created successfully.")
    return app


def run_dev_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """Convenience function to start the development server."""
    app = create_app()
    logger.info("Starting development server at %s:%d", host, port)
    app.run(host=host, port=port, debug=debug)
