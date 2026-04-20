from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PriceObservationSchema(BaseModel):
    sequence_ordinal: int
    closing_price: float
    observed_at: datetime

    class Config:
        from_attributes = True


class TimeIndexSchema(BaseModel):
    sequence_ordinal: int
    time_marker: float
    reference_at: datetime

    class Config:
        from_attributes = True


class ActivityFrequencySchema(BaseModel):
    sequence_ordinal: int
    frequency_count: int
    interval_seconds: float
    recorded_at: datetime

    class Config:
        from_attributes = True


class InstrumentBase(BaseModel):
    symbol: str = Field(..., max_length=20)
    name: str = Field(..., max_length=255)
    exchange: str = Field(default="NSE", max_length=50)
    sector: Optional[str] = Field(None, max_length=100)


class InstrumentCreate(InstrumentBase):
    pass


class InstrumentSchema(InstrumentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MultimodalProfileSchema(BaseModel):
    instrument_id: int
    symbol: str
    name: str
    exchange: str
    sector: Optional[str]
    closing_price_sequence: List[PriceObservationSchema]
    time_index_sequence: List[TimeIndexSchema]
    activity_frequency_stream: List[ActivityFrequencySchema]
    profile_depth: int
    assembled_at: datetime

    class Config:
        from_attributes = True


class InstrumentListItem(BaseModel):
    id: int
    symbol: str
    name: str
    exchange: str
    sector: Optional[str]
    latest_close: Optional[float]
    observation_count: int
    activity_count: int

    class Config:
        from_attributes = True


class InstrumentListResponse(BaseModel):
    instruments: List[InstrumentListItem]
    total: int
    page: int
    page_size: int
