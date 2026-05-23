from ai_driven_retail_investor_confidence_meter.ai.engine import Engine


def test_engine_explains():
    engine = Engine()
    out = engine.explain({"text": "pump it now", "leverage": 2.0, "edges": [["A", "B"]]})
    assert isinstance(out, dict)
    assert "summary" in out and "primitive" in out
