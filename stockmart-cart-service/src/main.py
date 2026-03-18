from fastapi import FastAPI, HTTPException
from .models import CartItem, CheckoutResult
from .cart_service import CartService

app = FastAPI(title="StockMart Cart Service", version="1.0.0")
svc = CartService()

@app.post("/carts/{user_id}", status_code=201)
def create_cart(user_id: str):
    return svc.create_cart(user_id)

@app.get("/carts/{cart_id}")
def get_cart(cart_id: str):
    cart = svc.get_cart(cart_id)
    if not cart:
        raise HTTPException(404, "Cart not found or expired")
    return cart

@app.post("/carts/{cart_id}/items")
def add_item(cart_id: str, item: CartItem):
    try:
        return svc.add_item(cart_id, item)
    except KeyError:
        raise HTTPException(404, "Cart not found")

@app.delete("/carts/{cart_id}/items/{symbol}/{side}")
def remove_item(cart_id: str, symbol: str, side: str):
    try:
        return svc.remove_item(cart_id, symbol, side)
    except KeyError:
        raise HTTPException(404, "Cart not found")

@app.post("/carts/{cart_id}/checkout", response_model=CheckoutResult)
def checkout(cart_id: str):
    try:
        return svc.checkout(cart_id)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
