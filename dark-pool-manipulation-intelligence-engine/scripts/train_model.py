from dark_pool_manipulation_intelligence_engine.ai.engine import Engine

if __name__ == "__main__":
    e = Engine()
    print("engine ready:", e.explain({"text": "demo", "leverage": 1.0}))
