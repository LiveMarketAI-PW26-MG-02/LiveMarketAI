from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from db.database import get_db
from services.instrument_service import InstrumentDiscoveryService
from models.schemas import InstrumentListResponse, MultimodalProfileSchema
from typing import Any, Dict

router = APIRouter()


def get_service(db: Session = Depends(get_db)) -> InstrumentDiscoveryService:
    return InstrumentDiscoveryService(db)


@router.post("/instruments/seed", response_model=Dict[str, Any])
async def seed_instruments(
    background_tasks: BackgroundTasks,
    service: InstrumentDiscoveryService = Depends(get_service),
):
    """
    Seeds instrument list and persists all three multimodal streams:
    closing price sequence, time index sequence, activity frequency stream.
    """
    result = await service.seed_instruments()
    return {"status": "seeded", "detail": result}


@router.get("/instruments", response_model=InstrumentListResponse)
async def list_instruments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: InstrumentDiscoveryService = Depends(get_service),
):
    """
    Returns paginated instrument list with latest close, observation count,
    and activity count — the platform's instrument browsing foundation.
    """
    return await service.get_instruments_list(page=page, page_size=page_size)


@router.get("/instruments/{symbol}/profile", response_model=MultimodalProfileSchema)
async def get_multimodal_profile(
    symbol: str,
    service: InstrumentDiscoveryService = Depends(get_service),
):
    """
    Returns the authoritative per-instrument multimodal profile:
    - closing_price_sequence (dimension 1)
    - time_index_sequence (dimension 2)
    - activity_frequency_stream (dimension 3)

    All three structured numeric streams are arithmetically assembled
    and aligned by sequence_ordinal.
    """
    profile = await service.get_multimodal_profile(symbol.upper())
    if not profile:
        raise HTTPException(status_code=404, detail=f"Instrument {symbol.upper()} not found.")
    return profile


@router.get("/instruments/{symbol}/enriched", response_model=Dict[str, Any])
async def get_enriched_profile(
    symbol: str,
    service: InstrumentDiscoveryService = Depends(get_service),
):
    """
    Returns multimodal profile enriched with Mistral AI analysis and YouTube research videos.
    """
    result = await service.get_enriched_profile(symbol.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"Instrument {symbol.upper()} not found.")
    return result


@router.get("/instruments/{symbol}/depth", response_model=Dict[str, Any])
async def get_market_depth(
    symbol: str,
    service: InstrumentDiscoveryService = Depends(get_service),
):
    """
    Returns real-time market depth (bid/ask spread) via ICICI Breeze.
    """
    from services.breeze_service import fetch_market_depth
    depth = await fetch_market_depth(symbol.upper())
    return depth
