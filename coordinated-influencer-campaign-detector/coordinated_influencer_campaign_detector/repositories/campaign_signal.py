from __future__ import annotations

from ..models.campaign_signal import CampaignSignal
from .base import CRUDRepository


class CampaignSignalRepository(CRUDRepository[CampaignSignal]):
    def __init__(self) -> None:
        super().__init__(CampaignSignal)


campaign_signal_repository = CampaignSignalRepository()
