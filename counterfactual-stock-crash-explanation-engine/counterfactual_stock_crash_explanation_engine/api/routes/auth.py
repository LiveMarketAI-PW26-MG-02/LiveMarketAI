from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...core.security import create_token, get_current_subject

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo identity store; replace with a real user repository in production.
_USERS = {"analyst@desk.io": "change-me"}


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
def issue_token(req: LoginRequest):
    if _USERS.get(req.email) != req.password:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    return TokenResponse(access_token=create_token(req.email))


@router.get("/me")
def me(subject: str = Depends(get_current_subject)):
    return {"subject": subject}
