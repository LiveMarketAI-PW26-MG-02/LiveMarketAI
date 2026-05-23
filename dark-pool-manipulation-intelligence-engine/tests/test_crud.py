from fastapi.testclient import TestClient

from dark_pool_manipulation_intelligence_engine.main import app


def test_create_and_list_dark_pool_print():
    with TestClient(app) as c:
        created = c.post("/dark_pool_prints", json={})
        assert created.status_code == 201
        listed = c.get("/dark_pool_prints")
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
