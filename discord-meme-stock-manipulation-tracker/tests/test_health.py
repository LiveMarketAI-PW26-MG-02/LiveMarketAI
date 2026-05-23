from fastapi.testclient import TestClient

from discord_meme_stock_manipulation_tracker.main import app


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_root_lists_docs():
    with TestClient(app) as c:
        assert c.get("/").json()["slug"] == "discord-meme-stock-manipulation-tracker"
