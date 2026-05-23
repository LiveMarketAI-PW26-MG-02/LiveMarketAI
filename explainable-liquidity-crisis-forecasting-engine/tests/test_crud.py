from fastapi.testclient import TestClient

from explainable_liquidity_crisis_forecasting_engine.main import app


def test_create_and_list_liquidity_metric():
    with TestClient(app) as c:
        created = c.post("/liquidity_metrics", json={})
        assert created.status_code == 201
        listed = c.get("/liquidity_metrics")
        assert listed.status_code == 200
        assert isinstance(listed.json(), list)


def test_auth_token_flow():
    with TestClient(app) as c:
        bad = c.post("/auth/token", json={"email": "x", "password": "y"})
        assert bad.status_code == 401
        ok = c.post("/auth/token",
                    json={"email": "analyst@desk.io", "password": "change-me"})
        assert ok.status_code == 200
        token = ok.json()["access_token"]
        me = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
