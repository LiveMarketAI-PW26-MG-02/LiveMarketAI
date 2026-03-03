# ARCHITECTURE — InsiderQuantum Drift Defense

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                            InsiderQuantum Drift Defense                         │
│                    Federated Security System                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA LAYER                                      │
│  Exchange A ──┐  Exchange B ──┐  Exchange C ──┐  Exchange D ──┐                  │
│  (Synthetic)  │  (Synthetic)  │  (Synthetic)  │  (Synthetic)  │                  │
└───────────────┼───────────────┼───────────────┼───────────────┘                  │
                │               │               │               │
                ▼               ▼               ▼               ▼
┌───────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  PYTHON NODE  │ │   R NODE    │ │   C++ NODE  │ │  JAVA NODE  │
│  (Coordinator)│ │ (Bayesian)  │ │  (MLpack)   │ │  (Monitor)  │
│               │ │             │ │             │ │             │
│ ┌───────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ Local     │ │ │ │ Robust  │ │ │ │  DET    │ │ │ │ State   │ │
│ │ NN Model  │ │ │ │ MCD/Cov │ │ │ │ Anomaly │ │ │ │ Window  │ │
│ └───────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│               │ │             │ │             │ │             │
│ Arrow Flight  │ │ Arrow IPC   │ │ gRPC client │ │ Arrow IPC   │
│ ZMQ Publisher │ │ ZMQ sub     │ │ ZMQ publish │ │ ZMQ sub     │
└───────┬───────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
        │                │               │               │
        │   ╔════════════╧═══════════════╧═══════════════╧═════╗
        │   ║         gRPC (Protocol Buffers)                   ║
        │   ║  WeightUpdate stream  ──►  FedAvg Aggregator      ║
        │   ╚════════════════════════════════════════════════════╝
        │                          │
        │           ┌──────────────▼──────────────┐
        │           │    FEDERATED AGGREGATOR      │
        │           │    (Python gRPC Server)      │
        │           │                              │
        │           │  FedAvg: Σ wᵢ·nᵢ / Σ nᵢ   │
        │           │  → Global weight vector      │
        │           │  → ConsensusVector proto     │
        │           └──────────────────────────────┘
        │
        │   ╔════════════════════════════════════════════════════╗
        └──►║         ZeroMQ PUB/SUB (binary MessagePack)       ║
            ║  Topic: FIEM_ANOMALY                  ║
            ║  ◄─── All nodes subscribe for real-time alerts    ║
            ╚════════════════════════════════════════════════════╝

## Data Flow

  1. Each node ingests synthetic tick stream
  2. Local model trained on tick features (bid/ask spread, IAT, vol, micro-vol)
  3. Local weights → MessagePack bytes → gRPC SubmitWeights
  4. Aggregator FedAvg → global_weights returned in AggregationResponse
  5. Nodes apply global weights to local models
  6. Anomaly scores computed locally; events → ZMQ PUB (binary)
  7. All nodes subscribe to ZMQ bus for consensus anomaly alerting

## Binary Transport Map

  ┌────────────────┬─────────────────────┬──────────────────────┐
  │ Transport      │ Used For            │ Format               │
  ├────────────────┼─────────────────────┼──────────────────────┤
  │ gRPC / Protobuf│ Weight updates      │ .proto binary wire   │
  │ Arrow IPC      │ Bulk weight batches  │ Arrow stream format  │
  │ Arrow Flight   │ Python bulk transfer │ Arrow Flight RPC     │
  │ ZeroMQ PUB/SUB │ Anomaly events      │ MessagePack binary   │
  │ Shared memory  │ (future: POSIX shm) │ Raw float array      │
  └────────────────┴─────────────────────┴──────────────────────┘

## Federated Learning Architecture

  Federated round n:
    ┌─────────┐    gradient/weights     ┌───────────────┐
    │ Node A  │ ──────────────────────► │               │
    │ Node B  │ ──────────────────────► │  FedAvg Agg   │ ──► global_w
    │ Node C  │ ──────────────────────► │               │
    │ Node D  │ ──────────────────────► └───────────────┘
    └─────────┘     ◄── global_weights ─────────────────

  No raw market data ever leaves a node.
  Only gradient/weight vectors are exchanged.

## Threat Model

  Attack Vector: Federated Intent Embedding Monitor (FIEM): Privileged command sequences are encoded into vector embe...
  Detection:  Multi-node statistical consensus
  Resistance: Attacker must compromise ≥ N/2 + 1 nodes simultaneously
```
