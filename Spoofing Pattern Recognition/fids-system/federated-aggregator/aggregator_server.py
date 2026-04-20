#!/usr/bin/env python3
"""
Spoofing Pattern Recognition System — Federated Aggregator gRPC Server
Implements FedAvg over binary weight payloads; no JSON used anywhere.
"""

from __future__ import annotations
import concurrent.futures
import logging
import time
import threading
from collections import defaultdict
from typing import Iterator

import numpy as np
import grpc

# Stubs generated from proto/service.proto
import service_pb2 as pb
import service_pb2_grpc as rpc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [AGGREGATOR] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class FederatedAggregatorServicer(rpc.FederatedAggregatorServicer):
    """FedAvg aggregator — collects weight updates, returns global average."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._rounds: dict[str, list[pb.WeightUpdate]] = defaultdict(list)
        self._global_weights: list[float] = []
        self._global_biases:  list[float] = []
        self._global_loss:    float       = 0.0

    # ── Unary submit ──────────────────────────────────────────────────────────
    def SubmitWeights(self, request: pb.WeightUpdate,
                      context: grpc.ServicerContext) -> pb.AggregationResponse:
        with self._lock:
            self._rounds[request.round_id].append(request)
            updates = self._rounds[request.round_id]
            log.info("Round %s | participants=%d  node=%s  loss=%.6f",
                     request.round_id, len(updates),
                     request.node_id, request.loss)
            if len(updates) >= 2:
                return self._fed_avg(request.round_id, updates)

        return pb.AggregationResponse(
            round_id     = request.round_id,
            global_weights = list(self._global_weights),
            global_biases  = list(self._global_biases),
            global_loss    = self._global_loss,
            participants   = len(self._rounds[request.round_id]),
        )

    # ── Streaming submit ──────────────────────────────────────────────────────
    def StreamWeights(self, request_iterator: Iterator[pb.WeightUpdate],
                      context: grpc.ServicerContext) -> Iterator[pb.AggregationResponse]:
        for update in request_iterator:
            resp = self.SubmitWeights(update, context)
            yield resp

    # ── Consensus vector query ────────────────────────────────────────────────
    def GetConsensusVector(self, request: pb.HealthRequest,
                           context: grpc.ServicerContext) -> pb.ConsensusVector:
        w = self._global_weights or [0.0]
        arr = np.array(w, dtype=np.float64)
        return pb.ConsensusVector(
            round_id    = "latest",
            mean_vector = arr.tolist(),
            std_vector  = (arr * 0.1).tolist(),
            entropy     = float(np.sum(-arr**2 * np.log(np.abs(arr) + 1e-9))),
        )

    # ── Health ping ───────────────────────────────────────────────────────────
    def Ping(self, request: pb.HealthRequest,
             context: grpc.ServicerContext) -> pb.HealthResponse:
        return pb.HealthResponse(healthy=True, message="OK")

    # ── FedAvg implementation ─────────────────────────────────────────────────
    def _fed_avg(self, round_id: str,
                 updates: list[pb.WeightUpdate]) -> pb.AggregationResponse:
        total_samples = sum(u.sample_count or 1 for u in updates)
        if total_samples == 0:
            total_samples = len(updates)

        # Weighted average of weights
        max_len = max(len(u.weights) for u in updates)
        avg_w   = np.zeros(max_len, dtype=np.float64)
        avg_b   = np.zeros(max(len(u.biases) for u in updates) or 1,
                           dtype=np.float64)

        for u in updates:
            frac = (u.sample_count or 1) / total_samples
            if u.weights:
                w = np.array(u.weights, dtype=np.float64)
                avg_w[:len(w)] += frac * w
            if u.biases:
                b = np.array(u.biases, dtype=np.float64)
                avg_b[:len(b)] += frac * b

        avg_loss = sum(u.loss * (u.sample_count or 1)
                       for u in updates) / total_samples

        self._global_weights = avg_w.tolist()
        self._global_biases  = avg_b.tolist()
        self._global_loss    = avg_loss

        log.info("FedAvg round %s | participants=%d  global_loss=%.6f",
                 round_id, len(updates), avg_loss)

        return pb.AggregationResponse(
            round_id       = round_id,
            global_weights = self._global_weights,
            global_biases  = self._global_biases,
            global_loss    = avg_loss,
            participants   = len(updates),
        )


def serve(port: int = 50051):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=16))
    rpc.add_FederatedAggregatorServicer_to_server(
        FederatedAggregatorServicer(), server
    )
    server.add_insecure_port(f"[::]:" + str(port))
    server.start()
    log.info("Spoofing Pattern Recognition System Aggregator listening on port %d", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
