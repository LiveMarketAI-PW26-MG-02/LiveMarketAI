from __future__ import annotations

from ..models.transaction import Transaction
from .base import CRUDRepository


class TransactionRepository(CRUDRepository[Transaction]):
    def __init__(self) -> None:
        super().__init__(Transaction)


transaction_repository = TransactionRepository()
