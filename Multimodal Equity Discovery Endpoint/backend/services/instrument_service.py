import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from repositories.instrument_repo import (
    InstrumentRepository,
    PriceObservationRepository,
    TimeIndexRepository,
    ActivityFrequencyRepository,
)
from models.equity_models import PriceObservation, TimeIndex, ActivityFrequency
from models.schemas import (
    InstrumentCreate,
    MultimodalProfileSchema,
    InstrumentListResponse,
    InstrumentListItem,
)
from services.alphavantage_service import fetch_daily_prices, fetch_instruments_list
from services.breeze_service import fetch_activity_stream
from services.mistral_service import analyze_instrument_profile
from services.youtube_service import fetch_equity_videos

logger = logging.getLogger(__name__)


class InstrumentDiscoveryService:
    def __init__(self, db: Session):
        self.db = db
        self.inst_repo = InstrumentRepository(db)
        self.price_repo = PriceObservationRepository(db)
        self.time_repo = TimeIndexRepository(db)
        self.activity_repo = ActivityFrequencyRepository(db)

    async def seed_instruments(self) -> Dict[str, Any]:
        raw_list = await fetch_instruments_list()
        seeded = 0
        for item in raw_list:
            inst = self.inst_repo.upsert(InstrumentCreate(**item))
            await self._persist_multimodal_streams(inst.id, inst.symbol)
            seeded += 1
        return {"seeded": seeded, "total": len(raw_list)}

    async def _persist_multimodal_streams(self, instrument_id: int, symbol: str) -> None:
        # ----- closing price sequence -----
        price_data = await fetch_daily_prices(symbol, n=90)
        self.price_repo.clear_for_instrument(instrument_id)
        price_records = []
        for i, row in enumerate(price_data):
            dt = datetime.strptime(row["date"], "%Y-%m-%d") if isinstance(row["date"], str) else row["date"]
            price_records.append(PriceObservation(
                instrument_id=instrument_id,
                closing_price=row["close"],
                observed_at=dt,
                sequence_ordinal=i + 1,
            ))
        self.price_repo.bulk_insert(price_records)

        # ----- time index sequence -----
        self.time_repo.clear_for_instrument(instrument_id)
        time_records = []
        for i, row in enumerate(price_data):
            dt = datetime.strptime(row["date"], "%Y-%m-%d") if isinstance(row["date"], str) else row["date"]
            epoch_marker = dt.timestamp()
            time_records.append(TimeIndex(
                instrument_id=instrument_id,
                time_marker=epoch_marker,
                reference_at=dt,
                sequence_ordinal=i + 1,
            ))
        self.time_repo.bulk_insert(time_records)

        # ----- activity frequency stream -----
        activity_data = await fetch_activity_stream(symbol, n=90)
        self.activity_repo.clear_for_instrument(instrument_id)
        act_records = []
        for row in activity_data:
            act_records.append(ActivityFrequency(
                instrument_id=instrument_id,
                frequency_count=row["frequency_count"],
                interval_seconds=row["interval_seconds"],
                recorded_at=row["recorded_at"] if isinstance(row["recorded_at"], datetime)
                else datetime.utcnow() - timedelta(days=90 - row["sequence_ordinal"]),
                sequence_ordinal=row["sequence_ordinal"],
            ))
        self.activity_repo.bulk_insert(act_records)

    async def get_instruments_list(self, page: int = 1, page_size: int = 50) -> InstrumentListResponse:
        instruments, total = self.inst_repo.get_all(page=page, page_size=page_size)
        items = []
        for inst in instruments:
            latest_close = self.inst_repo.get_latest_close(inst.id)
            obs_count = self.inst_repo.count_observations(inst.id)
            act_count = self.inst_repo.count_activity(inst.id)
            items.append(InstrumentListItem(
                id=inst.id,
                symbol=inst.symbol,
                name=inst.name,
                exchange=inst.exchange,
                sector=inst.sector,
                latest_close=latest_close,
                observation_count=obs_count,
                activity_count=act_count,
            ))
        return InstrumentListResponse(
            instruments=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_multimodal_profile(self, symbol: str) -> Optional[MultimodalProfileSchema]:
        inst = self.inst_repo.get_by_symbol(symbol)
        if not inst:
            return None

        prices = self.price_repo.get_for_instrument(inst.id)
        times = self.time_repo.get_for_instrument(inst.id)
        activities = self.activity_repo.get_for_instrument(inst.id)

        if not prices or not times or not activities:
            await self._persist_multimodal_streams(inst.id, inst.symbol)
            prices = self.price_repo.get_for_instrument(inst.id)
            times = self.time_repo.get_for_instrument(inst.id)
            activities = self.activity_repo.get_for_instrument(inst.id)

        profile_depth = min(len(prices), len(times), len(activities))

        return MultimodalProfileSchema(
            instrument_id=inst.id,
            symbol=inst.symbol,
            name=inst.name,
            exchange=inst.exchange,
            sector=inst.sector,
            closing_price_sequence=[
                {
                    "sequence_ordinal": p.sequence_ordinal,
                    "closing_price": p.closing_price,
                    "observed_at": p.observed_at,
                }
                for p in prices[:profile_depth]
            ],
            time_index_sequence=[
                {
                    "sequence_ordinal": t.sequence_ordinal,
                    "time_marker": t.time_marker,
                    "reference_at": t.reference_at,
                }
                for t in times[:profile_depth]
            ],
            activity_frequency_stream=[
                {
                    "sequence_ordinal": a.sequence_ordinal,
                    "frequency_count": a.frequency_count,
                    "interval_seconds": a.interval_seconds,
                    "recorded_at": a.recorded_at,
                }
                for a in activities[:profile_depth]
            ],
            profile_depth=profile_depth,
            assembled_at=datetime.utcnow(),
        )

    async def get_enriched_profile(self, symbol: str) -> Dict[str, Any]:
        profile = await self.get_multimodal_profile(symbol)
        if not profile:
            return {}

        closes = [p["closing_price"] for p in profile.closing_price_sequence]
        summary = (
            f"Symbol: {symbol}, N={len(closes)}, "
            f"latest_close={closes[-1] if closes else 0:.2f}, "
            f"range=[{min(closes):.2f}, {max(closes):.2f}]"
        )
        analysis = await analyze_instrument_profile(symbol, summary)
        videos = await fetch_equity_videos(symbol)

        return {
            "profile": profile.dict(),
            "analysis": analysis,
            "related_videos": videos,
        }
