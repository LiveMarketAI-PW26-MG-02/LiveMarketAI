#!/usr/bin/env python3
"""Simulation test for Latency Arbitrage Sentinel Python node."""

import numpy as np
from federated_coordinator import LocalFederatedModel, generate_tick_batch, AnomalyPublisher

def test_model_trains():
    model = LocalFederatedModel("test-node")
    ticks = generate_tick_batch(128)
    loss0 = model.train_one_round(ticks, lr=1e-2)
    loss1 = model.train_one_round(ticks, lr=1e-2)
    assert loss1 <= loss0 * 1.5, "Loss should not explode"
    print(f"[PASS] train: loss0={loss0:.6f}  loss1={loss1:.6f}")

def test_anomaly_score_range():
    model  = LocalFederatedModel("test-node")
    ticks  = generate_tick_batch(64)
    scores = [model.anomaly_score(t) for t in ticks]
    assert all(0.0 <= s <= 1.0 for s in scores), "Scores must be in [0,1]"
    print(f"[PASS] score range: min={min(scores):.4f}  max={max(scores):.4f}")

def test_weight_serialisation():
    model = LocalFederatedModel("ser-test")
    raw   = model.weights_to_msgpack()
    assert len(raw) > 100, "Payload too small"
    model2 = LocalFederatedModel("ser-test-2")
    model2.apply_global_weights(raw)
    np.testing.assert_allclose(model.W1, model2.W1, rtol=1e-5)
    print("[PASS] MessagePack weight round-trip OK")

if __name__ == "__main__":
    test_model_trains()
    test_anomaly_score_range()
    test_weight_serialisation()
    print("\n=== All Python tests passed ===")
