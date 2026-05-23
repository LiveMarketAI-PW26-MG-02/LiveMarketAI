from multimodal_bankruptcy_early_warning_system.ai.engine import Engine

if __name__ == "__main__":
    e = Engine()
    print("engine ready:", e.explain({"text": "demo", "leverage": 1.0}))
