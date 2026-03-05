# Spoofing Pattern Recognition System
**System ID:** SYS-005
**Abbreviation:** FIDS

## Overview
Federated Intent Detection System: Each node observes local order book flickering patterns — large orders placed and immediately cancelled. Federated models learn the statistical signature of legitimate vs spoofed order placement. Global consensus weights enable cross-venue spoofing detection that no single node could achieve independently.

## Architecture
See `ARCHITECTURE.md` for full ASCII diagram.

## Language Nodes
| Node | Language | Role |
|------|----------|------|
| `python-node` | Python 3.11 | Federated coordinator, Arrow Flight server, ZMQ publisher |
| `r-node`      | R 4.3       | Robust Mahalanobis anomaly detection, statistical modelling |
| `cpp-node`    | C++20       | MLpack DET anomaly detection, low-latency ZMQ publisher |
| `java-node`   | Java 21     | Distributed state monitor, Chronicle Queue, Arrow IPC |
| `federated-aggregator` | Python | gRPC FedAvg server |

## Binary Transports Used
- **gRPC + Protocol Buffers** — federated weight submission
- **Apache Arrow IPC / Flight** — bulk weight transfer
- **ZeroMQ PUB/SUB** — real-time anomaly events (binary MessagePack)
- **MessagePack** — binary payload serialisation (no JSON)

## Quick Start
```bash
# 1. Start federated aggregator
cd federated-aggregator
pip install -r requirements.txt
python -m grpc_tools.protoc -I ../proto --python_out=. --grpc_python_out=. ../proto/service.proto
python aggregator_server.py &

# 2. Start Python node
cd ../python-node
pip install -r requirements.txt
python -m grpc_tools.protoc -I ../proto --python_out=. --grpc_python_out=. ../proto/service.proto
python federated_coordinator.py

# 3. Start R node
cd ../r-node
Rscript r_statistical_node.R

# 4. Build and run C++ node
cd ../cpp-node
bash build.sh
./build/fids_node

# 5. Build and run Java node
cd ../java-node
mvn package -DskipTests
java -jar target/fids-java-node-1.0.0-SNAPSHOT.jar

# Or use Docker Compose:
docker-compose up --build
```

## Dependencies
See `BUILD.md` for full dependency list.

## Running Tests
```bash
# Python
cd python-node && python test_simulation.py

# R
cd r-node && Rscript test_r_node.R

# C++
cd cpp-node/build && ctest --output-on-failure

# Java
cd java-node && mvn test
```

## Federated Learning Protocol
1. Each node trains a local anomaly detection model on synthetic market data
2. Model weights are serialised (MessagePack / Arrow IPC) — no raw data leaves the node
3. Weights submitted to gRPC aggregator via `SubmitWeights` RPC
4. Aggregator performs FedAvg (sample-count-weighted average)
5. Global weights broadcast back to all nodes
6. Cycle repeats every configurable round interval

## Compliance
- ✅ Zero JSON usage
- ✅ Binary-only transport
- ✅ No centralised raw data aggregation
- ✅ Federated learning with genuine weight averaging
- ✅ Multi-language polyglot architecture
