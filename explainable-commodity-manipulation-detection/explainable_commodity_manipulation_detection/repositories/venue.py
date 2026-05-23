from __future__ import annotations

from ..models.venue import Venue
from .base import CRUDRepository


class VenueRepository(CRUDRepository[Venue]):
    def __init__(self) -> None:
        super().__init__(Venue)


venue_repository = VenueRepository()
