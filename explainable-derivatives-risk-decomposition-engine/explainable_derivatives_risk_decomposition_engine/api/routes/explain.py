from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ...services.xai_service import explain_payload

router = APIRouter(prefix="/explain", tags=["explainability"])


class ExplainRequest(BaseModel):
    features: dict


@router.post("")
def explain(req: ExplainRequest):
    return explain_payload(req.features)
