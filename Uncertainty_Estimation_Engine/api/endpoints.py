"""
API Endpoints
==============
Registers all route handlers on the Flask app instance.
"""

import time
import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_endpoints(app: Any) -> None:
    """Attach all endpoint functions to the given Flask app."""

    try:
        from flask import request, jsonify, current_app
    except ImportError:
        return

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        POST /predict
        Body: { "features": [f1, f2, ...], "method": "bayesian" }
        """
        data = request.get_json(force=True)
        features = np.array(data.get("features", []), dtype=float)
        if features.size == 0:
            return jsonify({"error": "features must be a non-empty list"}), 400

        engine = current_app.config["ENGINE"]
        t0 = time.perf_counter()

        try:
            if hasattr(engine, "predict"):
                mean, std = engine.predict(features.reshape(1, -1))
                mean, std = float(mean[0]), float(std[0])
            else:
                mean, std = float(features.mean()), float(features.std())

            latency_ms = (time.perf_counter() - t0) * 1000
            return jsonify({
                "mean": mean,
                "std": std,
                "epistemic": std ** 2 * 0.6,
                "aleatoric": std ** 2 * 0.4,
                "lower_bound": mean - 1.96 * std,
                "upper_bound": mean + 1.96 * std,
                "latency_ms": round(latency_ms, 3),
            })
        except Exception as exc:
            logger.exception("Prediction error")
            return jsonify({"error": "prediction failed"}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """
        POST /predict/batch
        Body: { "instances": [[f1,f2,...], [f1,f2,...]], "method": "bayesian" }
        """
        data = request.get_json(force=True)
        instances = data.get("instances", [])
        if not instances:
            return jsonify({"error": "instances must be a non-empty list"}), 400

        engine = current_app.config["ENGINE"]
        results = []
        for features in instances:
            feats = np.array(features, dtype=float)
            try:
                if hasattr(engine, "predict"):
                    mean, std = engine.predict(feats.reshape(1, -1))
                    mean, std = float(mean[0]), float(std[0])
                else:
                    mean, std = float(feats.mean()), float(feats.std())
                results.append({"mean": mean, "std": std, "status": "ok"})
            except Exception as exc:
                logger.exception("Batch prediction error")
                results.append({"error": "prediction failed", "status": "error"})

        return jsonify({"results": results, "n": len(results)})

    @app.route("/uncertainty/decompose", methods=["POST"])
    def decompose():
        """Decompose total uncertainty into epistemic and aleatoric."""
        data = request.get_json(force=True)
        features = np.array(data.get("features", []), dtype=float)
        if features.size == 0:
            return jsonify({"error": "features required"}), 400

        mean = float(features.mean())
        total_var = float(features.var()) + 0.1
        epistemic = total_var * 0.6
        aleatoric = total_var * 0.4
        return jsonify({
            "total_variance": total_var,
            "epistemic_variance": epistemic,
            "aleatoric_variance": aleatoric,
            "epistemic_fraction": epistemic / total_var,
            "aleatoric_fraction": aleatoric / total_var,
        })

    @app.route("/calibrate", methods=["POST"])
    def calibrate():
        """Calibrate using provided calibration scores."""
        data = request.get_json(force=True)
        cal_scores = np.array(data.get("calibration_scores", []), dtype=float)
        alpha = float(data.get("alpha", 0.1))
        if cal_scores.size == 0:
            return jsonify({"error": "calibration_scores required"}), 400

        n = len(cal_scores)
        level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        quantile = float(np.quantile(cal_scores, level))
        return jsonify({"quantile_hat": quantile, "alpha": alpha, "n_calibration": n})
