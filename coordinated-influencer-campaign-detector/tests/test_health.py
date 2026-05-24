from fastapi.testclient import TestClient

from coordinated_influencer_campaign_detector.main import app


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_root_lists_docs():
    with TestClient(app) as c:
        assert c.get("/").json()["slug"] == "coordinated-influencer-campaign-detector"
