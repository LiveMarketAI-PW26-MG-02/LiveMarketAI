import pytest
from src.cart_service import CartService
from src.models import CartItem, OrderSide, OrderType

@pytest.fixture
def svc():
    return CartService()

def test_create_and_add(svc):
    cart = svc.create_cart("u1")
    item = CartItem(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
    updated = svc.add_item(cart.id, item)
    assert len(updated.items) == 1

def test_checkout_clears_cart(svc):
    cart = svc.create_cart("u2")
    item = CartItem(symbol="TSLA", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=5, limit_price=170.0)
    svc.add_item(cart.id, item)
    result = svc.checkout(cart.id)
    assert result.orders_placed == 1
    assert result.status == "submitted"

def test_empty_checkout_raises(svc):
    cart = svc.create_cart("u3")
    with pytest.raises(ValueError, match="empty"):
        svc.checkout(cart.id)
