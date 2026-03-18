from datetime import datetime, timedelta
from typing import Dict, Optional
from .models import Cart, CartItem, CheckoutResult

CART_TTL_MINUTES = 15

class CartService:
    def __init__(self):
        self._carts: Dict[str, Cart] = {}

    def create_cart(self, user_id: str) -> Cart:
        cart = Cart(
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(minutes=CART_TTL_MINUTES)
        )
        self._carts[cart.id] = cart
        return cart

    def get_cart(self, cart_id: str) -> Optional[Cart]:
        cart = self._carts.get(cart_id)
        if cart and cart.expires_at and datetime.utcnow() > cart.expires_at:
            cart.items = []
            return None
        return cart

    def add_item(self, cart_id: str, item: CartItem) -> Cart:
        cart = self._get_active(cart_id)
        # Replace if same symbol+side
        cart.items = [i for i in cart.items if not (i.symbol == item.symbol and i.side == item.side)]
        cart.items.append(item)
        return cart

    def remove_item(self, cart_id: str, symbol: str, side: str) -> Cart:
        cart = self._get_active(cart_id)
        cart.items = [i for i in cart.items if not (i.symbol == symbol and i.side == side)]
        return cart

    def checkout(self, cart_id: str) -> CheckoutResult:
        cart = self._get_active(cart_id)
        if not cart.items:
            raise ValueError("Cart is empty")
        n = len(cart.items)
        cart.checked_out = True
        cart.items = []
        return CheckoutResult(
            cart_id=cart_id,
            orders_placed=n,
            total_items=n,
            status="submitted"
        )

    def _get_active(self, cart_id: str) -> Cart:
        cart = self._carts.get(cart_id)
        if not cart:
            raise KeyError("Cart not found")
        if cart.checked_out:
            raise ValueError("Cart already checked out")
        return cart
