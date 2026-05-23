from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FeatureContributionBase(BaseModel):
    prediction_id: Optional[int] = None
    feature: Optional[str] = None
    contribution: Optional[float] = None


class FeatureContributionCreate(FeatureContributionBase):
    pass


class FeatureContributionRead(FeatureContributionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
