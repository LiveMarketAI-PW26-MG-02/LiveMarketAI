from __future__ import annotations

from ..models.client_profile import ClientProfile
from .base import CRUDRepository


class ClientProfileRepository(CRUDRepository[ClientProfile]):
    def __init__(self) -> None:
        super().__init__(ClientProfile)


client_profile_repository = ClientProfileRepository()
