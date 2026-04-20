#!/usr/bin/env python3
"""
Flash Crash Cascade Shield — Python Federated Coordinator Node
System ID : SYS-004
Role      : Federated learning coordinator + anomaly detection
Transport : gRPC (primary), Apache Arrow Flight (bulk), ZeroMQ pub/sub
"""

from __future__ import annotations

import io
import os
import struct
import threading
import time
import logging
import hashlib
import math
from concurrent import futures
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import zmq
import msgpack
import grpc

# ── OpenTelemetry tracing ─────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("dcbm.python.coordinator")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [PY-COORD] %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Simulated local model ─────────────────────────────────────────────────────
class LocalFederatedModel:
    """Lightweight anomaly detection model trained on local tick data."""

    INPUT_DIM  = 16
    HIDDEN_DIM = 32
    OUTPUT_DIM = 1

    def __init__(self, node_id: str):
        self.node_id = node_id
        rng = np.random.default_rng(abs(hash(node_id)) % (2**31))
        self.W1 = rng.standard_normal((self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2 = rng.standard_normal((self.HIDDEN_DIM, self.OUTPUT_DIM)).astype(np.float32) * 0.1
        self.b2 = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        self.loss = float("inf")

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        return 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)))

    # ── Simulated local training ───────────────────────────────────────────────
    def train_one_round(self, ticks: np.ndarray, lr: float = 1e-3) -> float:
        """SGD step on local tick features."""
        with tracer.start_as_current_span("local_train"):
            n = ticks.shape[0]
            labels = self._label_ticks(ticks)
            preds  = self.forward(ticks)
            err    = preds - labels.reshape(-1, 1)
            # Backprop (manual, no framework dependency)
            h      = np.tanh(ticks @ self.W1 + self.b1)
            dW2    = (h.T @ err) / n
            db2    = err.mean(axis=0)
            dh     = (err @ self.W2.T) * (1 - h**2)
            dW1    = (ticks.T @ dh) / n
            db1    = dh.mean(axis=0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.loss = float(np.mean(err**2))
        return self.loss

    # ── Anomaly scoring ───────────────────────────────────────────────────────
    def anomaly_score(self, tick: np.ndarray) -> float:
        return float(self.forward(tick.reshape(1, -1))[0, 0])

    # ── Serialise weights to bytes (MessagePack, binary only) ─────────────────
    def weights_to_msgpack(self) -> bytes:
        payload = {
            b"node_id": self.node_id.encode(),
            b"W1":      self.W1.tobytes(),
            b"b1":      self.b1.tobytes(),
            b"W2":      self.W2.tobytes(),
            b"b2":      self.b2.tobytes(),
            b"loss":    self.loss,
            b"shape_W1": list(self.W1.shape),
            b"shape_W2": list(self.W2.shape),
        }
        return msgpack.packb(payload, use_bin_type=True)

    # ── Apply global weights from aggregator ──────────────────────────────────
    def apply_global_weights(self, raw: bytes) -> None:
        d   = msgpack.unpackb(raw, raw=True)
        w1  = np.frombuffer(d[b"W1"], dtype=np.float32).reshape(d[b"shape_W1"])
        b1  = np.frombuffer(d[b"b1"], dtype=np.float32)
        w2  = np.frombuffer(d[b"W2"], dtype=np.float32).reshape(d[b"shape_W2"])
        b2  = np.frombuffer(d[b"b2"], dtype=np.float32)
        self.W1, self.b1, self.W2, self.b2 = w1, b1, w2, b2
        log.info("Applied global weights from aggregator")

    # ── Dummy label: ticks > 2-sigma spread are anomalous ─────────────────────
    @staticmethod
    def _label_ticks(ticks: np.ndarray) -> np.ndarray:
        spread = ticks[:, 1] - ticks[:, 0]
        mu, sd = spread.mean(), spread.std() + 1e-9
        return ((spread - mu) / sd > 2.0).astype(np.float32)


# ── Market data simulator ─────────────────────────────────────────────────────
def generate_tick_batch(n: int = 256) -> np.ndarray:
    """Generate synthetic market tick feature matrix (n × 16)."""
    rng  = np.random.default_rng()
    base = rng.uniform(99.5, 100.5, (n, 1))
    spread = rng.exponential(0.05, (n, 1))
    bid  = base - spread / 2
    ask  = base + spread / 2
    vol  = rng.exponential(1000, (n, 1))
    iat  = rng.exponential(0.01, (n, 1))           # inter-arrival time
    vola = rng.standard_normal((n, 12)) * 0.02     # micro-volatility features
    return np.hstack([bid, ask, vol, iat, vola]).astype(np.float32)


# ── Apache Arrow Flight server (bulk weight transfer) ─────────────────────────
class WeightFlightServer(flight.FlightServerBase):
    def __init__(self, location: str, model: LocalFederatedModel):
        super().__init__(location)
        self.model  = model
        self._store = {}

    def do_put(self, context, descriptor, reader, writer):
        key    = descriptor.command.decode()
        table  = reader.read_all()
        self._store[key] = table
        log.info("Flight: received weight table for key=%s  rows=%d", key, table.num_rows)

    def do_get(self, context, ticket):
        key = ticket.ticket.decode()
        if key not in self._store:
            raise flight.FlightServerError(f"No data for key {key}")
        return flight.RecordBatchStream(self._store[key])

    def list_flights(self, context, criteria):
        for key in self._store:
            desc = flight.FlightDescriptor.for_command(key.encode())
            info = flight.FlightInfo(
                self._store[key].schema,
                desc,
                [],
                self._store[key].num_rows,
                -1,
            )
            yield info


# ── ZeroMQ anomaly publisher ──────────────────────────────────────────────────
class AnomalyPublisher:
    TOPIC = b"DCBM_ANOMALY"

    def __init__(self, endpoint: str = "tcp://*:5570"):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(endpoint)
        time.sleep(0.1)

    def publish(self, node_id: str, score: float, threshold: float, features: list) -> None:
        payload = msgpack.packb({
            b"node_id":   node_id.encode(),
            b"score":     score,
            b"threshold": threshold,
            b"features":  features,
            b"ts_ns":     time.time_ns(),
        }, use_bin_type=True)
        self.sock.send_multipart([self.TOPIC, payload])


# ── Main federated coordinator loop ──────────────────────────────────────────
def run_coordinator(rounds: int = 10, node_id: str = "py-coord-01"):
    log.info("=== Flash Crash Cascade Shield — Python Coordinator ===")
    model     = LocalFederatedModel(node_id)
    publisher = AnomalyPublisher()

    for rnd in range(1, rounds + 1):
        with tracer.start_as_current_span(f"federated_round_{rnd}"):
            ticks  = generate_tick_batch(512)
            loss   = model.train_one_round(ticks)
            log.info("Round %2d | loss=%.6f", rnd, loss)

            # Anomaly detection on latest batch
            scores = np.array([model.anomaly_score(t) for t in ticks])
            mu, sd = scores.mean(), scores.std()
            threshold = mu + 2.5 * sd
            anomalies = np.where(scores > threshold)[0]

            if len(anomalies):
                log.warning("Round %2d | %d anomalies detected", rnd, len(anomalies))
                for idx in anomalies[:5]:
                    publisher.publish(
                        node_id, float(scores[idx]), float(threshold),
                        ticks[idx].tolist(),
                    )

            # Serialise weights → Arrow IPC for bulk transfer simulation
            schema = pa.schema([
                pa.field("weights", pa.list_(pa.float32())),
                pa.field("loss",    pa.float64()),
            ])
            flat_w = np.concatenate([
                model.W1.ravel(), model.b1, model.W2.ravel(), model.b2,
            ]).tolist()
            table = pa.table({"weights": [flat_w], "loss": [model.loss]}, schema=schema)
            sink  = pa.BufferOutputStream()
            with ipc.new_stream(sink, schema) as wr:
                wr.write_table(table)
            arrow_bytes = bytes(sink.getvalue())
            log.info("Round %2d | Arrow IPC payload = %d bytes", rnd, len(arrow_bytes))

            time.sleep(0.5)

    log.info("Coordinator finished %d rounds.", rounds)


if __name__ == "__main__":
    run_coordinator()
