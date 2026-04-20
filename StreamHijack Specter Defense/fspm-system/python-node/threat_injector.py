#!/usr/bin/env python3
"""
StreamHijack Specter Defense — Threat Simulation Injector
Injects synthetic anomalous market ticks to stress-test detection.
"""

import argparse
import time
import struct
import numpy as np
import zmq
import msgpack


def parse_args():
    p = argparse.ArgumentParser(description="StreamHijack Specter Defense Threat Injector")
    p.add_argument("--endpoint",     default="tcp://localhost:5570")
    p.add_argument("--anomaly-rate", type=float, default=0.10)
    p.add_argument("--rounds",       type=int,   default=20)
    p.add_argument("--node-id",      default="threat-injector-01")
    return p.parse_args()


def generate_normal_tick(rng):
    base   = 100.0 + rng.normal(0, 0.25)
    spread = rng.exponential(0.05)
    return np.array([
        base - spread / 2,
        base + spread / 2,
        rng.exponential(1000),
        rng.exponential(0.01),
        *rng.normal(0, 0.02, 12),
    ], dtype=np.float32)


def generate_anomalous_tick(rng, attack_type: str = "spread_spike"):
    tick = generate_normal_tick(rng)
    if attack_type == "spread_spike":
        # Artificially widen spread by 20x (PhantomTick injection)
        mid          = (tick[0] + tick[1]) / 2
        tick[0]      = mid - 0.5
        tick[1]      = mid + 0.5
    elif attack_type == "volume_surge":
        tick[2] *= 50.0
    elif attack_type == "latency_flood":
        tick[3]  = rng.exponential(1.0)   # 100x normal IAT
    elif attack_type == "micro_burst":
        tick[4:] = rng.normal(0, 1.0, 12) # 50x normal micro-vol
    return tick


def main():
    args = parse_args()
    rng  = np.random.default_rng()
    ctx  = zmq.Context()
    pub  = ctx.socket(zmq.PUB)
    pub.connect(args.endpoint)
    time.sleep(0.2)

    TOPIC      = b"FSPM_ANOMALY"
    ATTACKS    = ["spread_spike", "volume_surge", "latency_flood", "micro_burst"]
    n_injected = 0

    print(f"[INJECTOR] Starting — anomaly_rate={args.anomaly_rate:.0%}  rounds={args.rounds}")

    for rnd in range(1, args.rounds + 1):
        is_anomaly  = rng.random() < args.anomaly_rate
        attack_type = rng.choice(ATTACKS) if is_anomaly else "none"
        tick        = (generate_anomalous_tick(rng, attack_type)
                       if is_anomaly else generate_normal_tick(rng))

        score = 0.92 if is_anomaly else rng.uniform(0.1, 0.5)
        payload = msgpack.packb({
            b"node_id":   args.node_id.encode(),
            b"score":     score,
            b"threshold": 0.75,
            b"features":  tick.tolist(),
            b"ts_ns":     time.time_ns(),
            b"attack":    attack_type.encode(),
        }, use_bin_type=True)

        pub.send_multipart([TOPIC, payload])

        if is_anomaly:
            n_injected += 1
            print(f"[INJECTOR] Round {rnd:3d} | ANOMALY injected | attack={attack_type} | score={score:.3f}")
        else:
            print(f"[INJECTOR] Round {rnd:3d} | normal tick")

        time.sleep(0.3)

    print(f"\n[INJECTOR] Done. Injected {n_injected} anomalies in {args.rounds} rounds.")
    pub.close()
    ctx.term()


if __name__ == "__main__":
    main()
