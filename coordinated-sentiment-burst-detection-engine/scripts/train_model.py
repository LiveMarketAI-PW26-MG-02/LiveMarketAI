from coordinated_sentiment_burst_detection_engine.ai.engine import Engine

if __name__ == "__main__":
    e = Engine()
    print("engine ready:", e.explain({"text": "demo", "leverage": 1.0}))
