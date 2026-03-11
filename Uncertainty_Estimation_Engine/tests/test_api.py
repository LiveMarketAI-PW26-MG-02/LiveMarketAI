"""Tests for the REST API endpoints."""

import json
import pytest
import numpy as np

try:
    from api.rest_api import create_app
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@pytest.fixture
def client():
    if not HAS_FLASK:
        pytest.skip("Flask not installed")
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"


def test_info_endpoint(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "name" in data
    assert "methods" in data


def test_predict_valid(client):
    payload = {"features": [1.0, 2.0, 3.0], "method": "bayesian"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    for key in ("mean", "std", "lower_bound", "upper_bound"):
        assert key in data


def test_predict_empty_features(client):
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 400


def test_predict_batch_valid(client):
    payload = {"instances": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["n"] == 3
    assert len(data["results"]) == 3


def test_predict_batch_empty(client):
    resp = client.post("/predict/batch", json={"instances": []})
    assert resp.status_code == 400


def test_decompose_endpoint(client):
    payload = {"features": [1.0, 0.5, -0.3, 2.1]}
    resp = client.post("/uncertainty/decompose", json=payload)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert abs(data["epistemic_fraction"] + data["aleatoric_fraction"] - 1.0) < 1e-6


def test_calibrate_endpoint(client):
    scores = np.abs(np.random.default_rng(0).standard_normal(100)).tolist()
    payload = {"calibration_scores": scores, "alpha": 0.1}
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "quantile_hat" in data
    assert data["quantile_hat"] > 0
