from multimodal_bankruptcy_early_warning_system.ai.engine import Engine


def test_engine_explains():
    engine = Engine()
    out = engine.explain({"text": "pump it now", "leverage": 2.0, "edges": [["A", "B"]]})
    assert isinstance(out, dict)
    assert "summary" in out and "primitive" in out
