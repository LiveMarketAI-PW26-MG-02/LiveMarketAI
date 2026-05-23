from __future__ import annotations

from ..ai.engine import Engine

_engine = Engine()


def explain_payload(features: dict) -> dict:
    """Run the project's AI primitive and return a structured explanation."""
    return _engine.explain(features)
