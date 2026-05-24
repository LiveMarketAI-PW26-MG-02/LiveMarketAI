from coordinated_whale_manipulation_detector.ai.engine import Engine

if __name__ == "__main__":
    e = Engine()
    print("engine ready:", e.explain({"text": "demo", "leverage": 1.0}))
