import pytest
from src.watchlist_service import WatchlistService
from src.models import CreateWatchlistRequest, WatchItem

@pytest.fixture
def svc():
    return WatchlistService()

def test_create_watchlist(svc):
    req = CreateWatchlistRequest(name="Tech Picks")
    wl = svc.create("u1", req)
    assert wl.name == "Tech Picks"
    assert wl.user_id == "u1"

def test_add_and_remove_item(svc):
    wl = svc.create("u1", CreateWatchlistRequest(name="Growth"))
    svc.add_item(wl.id, WatchItem(symbol="NVDA", alert_above=900.0))
    assert any(i.symbol == "NVDA" for i in svc.get(wl.id).items)
    svc.remove_item(wl.id, "NVDA")
    assert not any(i.symbol == "NVDA" for i in svc.get(wl.id).items)

def test_alert_trigger(svc):
    wl = svc.create("u1", CreateWatchlistRequest(name="Alerts"))
    svc.add_item(wl.id, WatchItem(symbol="TSLA", alert_above=200.0))
    fired = svc.check_alerts("TSLA", 205.0)
    assert len(fired) == 1
    assert fired[0]["type"] == "ABOVE"
