from __future__ import annotations

from ..models.wallet import Wallet
from .base import CRUDRepository


class WalletRepository(CRUDRepository[Wallet]):
    def __init__(self) -> None:
        super().__init__(Wallet)


wallet_repository = WalletRepository()
