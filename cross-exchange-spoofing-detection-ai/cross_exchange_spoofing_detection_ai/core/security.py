from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_settings

_bearer = HTTPBearer(auto_error=False)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _sign(payload: bytes, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), payload, hashlib.sha256).digest())


def create_token(subject: str) -> str:
    """Compact HMAC-signed token (JWT-style, dependency-free)."""
    s = get_settings()
    body = {"sub": subject, "exp": int(time.time()) + s.token_ttl_seconds}
    raw = json.dumps(body, separators=(",", ":")).encode()
    return _b64(raw) + "." + _sign(raw, s.secret_key)


def verify_token(token: str) -> str:
    s = get_settings()
    try:
        body_b64, sig = token.split(".", 1)
        raw = _unb64(body_b64)
        if not hmac.compare_digest(sig, _sign(raw, s.secret_key)):
            raise ValueError("bad signature")
        body = json.loads(raw)
        if body.get("exp", 0) < int(time.time()):
            raise ValueError("expired")
        return str(body["sub"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc


def get_current_subject(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    return verify_token(creds.credentials)
