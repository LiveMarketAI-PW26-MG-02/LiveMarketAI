from __future__ import annotations

from ..models.server import Server
from .base import CRUDRepository


class ServerRepository(CRUDRepository[Server]):
    def __init__(self) -> None:
        super().__init__(Server)


server_repository = ServerRepository()
