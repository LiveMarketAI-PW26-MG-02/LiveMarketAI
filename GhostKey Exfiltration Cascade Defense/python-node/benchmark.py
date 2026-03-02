#!/usr/bin/env python3
"""
GhostKey Exfiltration Cascade Defense — Performance Benchmark
Measures: tick processing throughput, federated round latency, Arrow IPC bandwidth.
"""

import time
import struct
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import msgpack

from federated_coordinator import LocalFederatedModel, generate_tick_batch


def bench_local_inference(n_ticks: int = 10_000) -> None:
    model  = LocalFederatedModel("bench-node")
    ticks  = generate_tick_batch(n_ticks)
    model.train_one_round(ticks)

    t0 = time.perf_counter_ns()
    for t in ticks:
        model.anomaly_score(t)
    elapsed_ns = time.perf_counter_ns() - t0

    ns_per_tick = elapsed_ns / n_ticks
    print(f"[BENCH] Inference   | n={n_ticks}  "
          f"total={elapsed_ns/1e6:.2f} ms  "
          f"per_tick={ns_per_tick:.1f} ns  "
          f"throughput={1e9/ns_per_tick:,.0f} ticks/s")


def bench_arrow_ipc(n_rounds: int = 100) -> None:
    model  = LocalFederatedModel("bench-arrow")
    ticks  = generate_tick_batch(512)
    model.train_one_round(ticks)

    schema = pa.schema([
        pa.field("weights", pa.list_(pa.float32())),
        pa.field("loss",    pa.float64()),
    ])
    total_bytes = 0

    t0 = time.perf_counter_ns()
    for _ in range(n_rounds):
        flat_w = np.concatenate([
            model.W1.ravel(), model.b1, model.W2.ravel(), model.b2,
        ]).tolist()
        table = pa.table({"weights": [flat_w], "loss": [model.loss]}, schema=schema)
        sink  = pa.BufferOutputStream()
        with ipc.new_stream(sink, schema) as wr:
            wr.write_table(table)
        total_bytes += len(sink.getvalue())
    elapsed_ns = time.perf_counter_ns() - t0

    mb_per_s = (total_bytes / 1e6) / (elapsed_ns / 1e9)
    print(f"[BENCH] Arrow IPC   | rounds={n_rounds}  "
          f"total_bytes={total_bytes:,}  "
          f"bandwidth={mb_per_s:.1f} MB/s")


def bench_msgpack_serialise(n_rounds: int = 1_000) -> None:
    model  = LocalFederatedModel("bench-msgpack")
    ticks  = generate_tick_batch(256)
    model.train_one_round(ticks)

    t0 = time.perf_counter_ns()
    for _ in range(n_rounds):
        raw = model.weights_to_msgpack()
        model.apply_global_weights(raw)
    elapsed_ns = time.perf_counter_ns() - t0

    us_per_round = elapsed_ns / n_rounds / 1_000
    print(f"[BENCH] MsgPack RT  | rounds={n_rounds}  "
          f"per_round={us_per_round:.1f} µs")


def bench_federated_training(n_rounds: int = 20) -> None:
    model = LocalFederatedModel("bench-train")
    t0    = time.perf_counter_ns()
    for _ in range(n_rounds):
        ticks = generate_tick_batch(512)
        model.train_one_round(ticks)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"[BENCH] Fed Training| rounds={n_rounds}  "
          f"total={elapsed_ms:.1f} ms  "
          f"per_round={elapsed_ms/n_rounds:.1f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  GhostKey Exfiltration Cascade Defense Performance Benchmark")
    print("=" * 60)
    bench_local_inference(10_000)
    bench_arrow_ipc(100)
    bench_msgpack_serialise(1_000)
    bench_federated_training(20)
    print("=" * 60)
    print("  Benchmark complete.")
