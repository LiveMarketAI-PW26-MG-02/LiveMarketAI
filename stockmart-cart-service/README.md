# stockmart-cart-service 🛒

Shopping-cart metaphor for stock order staging — add symbols to cart, review, checkout.

## Features
- Add / remove symbols from cart
- Quantity & order-type per line item
- Checkout triggers order placement
- Cart expiry (15 min TTL)

## Stack
Python 3.11 · FastAPI · Pydantic v2

## Quickstart
```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```
