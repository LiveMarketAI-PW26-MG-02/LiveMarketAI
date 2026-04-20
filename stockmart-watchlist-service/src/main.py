from fastapi import FastAPI, HTTPException
from .models import CreateWatchlistRequest, WatchItem
from .watchlist_service import WatchlistService

app = FastAPI(title="StockMart Watchlist Service", version="1.0.0")
svc = WatchlistService()

@app.post("/users/{user_id}/watchlists", status_code=201)
def create(user_id: str, req: CreateWatchlistRequest):
    return svc.create(user_id, req)

@app.get("/users/{user_id}/watchlists")
def list_user(user_id: str):
    return svc.list_for_user(user_id)

@app.get("/watchlists/public")
def list_public():
    return svc.list_public()

@app.get("/watchlists/{wl_id}")
def get(wl_id: str):
    wl = svc.get(wl_id)
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    return wl

@app.post("/watchlists/{wl_id}/items")
def add_item(wl_id: str, item: WatchItem):
    try:
        return svc.add_item(wl_id, item)
    except KeyError as e:
        raise HTTPException(404, str(e))

@app.delete("/watchlists/{wl_id}/items/{symbol}")
def remove_item(wl_id: str, symbol: str):
    try:
        return svc.remove_item(wl_id, symbol)
    except KeyError as e:
        raise HTTPException(404, str(e))

@app.delete("/watchlists/{wl_id}", status_code=204)
def delete(wl_id: str):
    if not svc.delete(wl_id):
        raise HTTPException(404, "Watchlist not found")
