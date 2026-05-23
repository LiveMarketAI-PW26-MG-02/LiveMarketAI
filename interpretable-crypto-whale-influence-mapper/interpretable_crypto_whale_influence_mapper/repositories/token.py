from __future__ import annotations

from ..models.token import Token
from .base import CRUDRepository


class TokenRepository(CRUDRepository[Token]):
    def __init__(self) -> None:
        super().__init__(Token)


token_repository = TokenRepository()
