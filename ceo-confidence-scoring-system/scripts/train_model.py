from ceo_confidence_scoring_system.ai.engine import Engine

if __name__ == "__main__":
    e = Engine()
    print("engine ready:", e.explain({"text": "demo", "leverage": 1.0}))
