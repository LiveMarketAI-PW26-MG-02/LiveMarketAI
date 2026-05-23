from __future__ import annotations

from ..models.user import User
from .base import CRUDRepository


class UserRepository(CRUDRepository[User]):
    def __init__(self) -> None:
        super().__init__(User)


user_repository = UserRepository()
