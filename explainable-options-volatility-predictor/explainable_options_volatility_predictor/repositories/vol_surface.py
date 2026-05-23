from __future__ import annotations

from ..models.vol_surface import VolSurface
from .base import CRUDRepository


class VolSurfaceRepository(CRUDRepository[VolSurface]):
    def __init__(self) -> None:
        super().__init__(VolSurface)


vol_surface_repository = VolSurfaceRepository()
