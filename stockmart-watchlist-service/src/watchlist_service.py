from typing import Dict, List, Optional
from datetime import datetime
from .models import Watchlist, WatchItem, CreateWatchlistRequest

class WatchlistService:
    def __init__(self):
        self._lists: Dict[str, Watchlist] = {}

    def create(self, user_id: str, req: CreateWatchlistRequest) -> Watchlist:
        wl = Watchlist(user_id=user_id, **req.model_dump())
        self._lists[wl.id] = wl
        return wl

    def get(self, wl_id: str) -> Optional[Watchlist]:
        return self._lists.get(wl_id)

    def list_for_user(self, user_id: str) -> List[Watchlist]:
        return [w for w in self._lists.values() if w.user_id == user_id]

    def list_public(self) -> List[Watchlist]:
        return [w for w in self._lists.values() if w.is_public]

    def add_item(self, wl_id: str, item: WatchItem) -> Watchlist:
        wl = self._require(wl_id)
        wl.items = [i for i in wl.items if i.symbol != item.symbol]
        wl.items.append(item)
        wl.updated_at = datetime.utcnow()
        return wl

    def remove_item(self, wl_id: str, symbol: str) -> Watchlist:
        wl = self._require(wl_id)
        wl.items = [i for i in wl.items if i.symbol != symbol]
        wl.updated_at = datetime.utcnow()
        return wl

    def delete(self, wl_id: str) -> bool:
        if wl_id in self._lists:
            del self._lists[wl_id]
            return True
        return False

    def check_alerts(self, symbol: str, price: float) -> List[dict]:
        triggered = []
        for wl in self._lists.values():
            for item in wl.items:
                if item.symbol != symbol:
                    continue
                if item.alert_above and price >= item.alert_above:
                    triggered.append({"watchlist_id": wl.id, "symbol": symbol,
                                       "type": "ABOVE", "threshold": item.alert_above, "price": price})
                if item.alert_below and price <= item.alert_below:
                    triggered.append({"watchlist_id": wl.id, "symbol": symbol,
                                       "type": "BELOW", "threshold": item.alert_below, "price": price})
        return triggered

    def _require(self, wl_id: str) -> Watchlist:
        wl = self._lists.get(wl_id)
        if not wl:
            raise KeyError(f"Watchlist {wl_id} not found")
        return wl
