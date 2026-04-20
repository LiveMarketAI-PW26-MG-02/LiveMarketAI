# BUILD INSTRUCTIONS — ModelDrift Sabotage Defense

## Prerequisites

### System
- Linux (Ubuntu 22.04+ recommended) or macOS 13+
- Docker & Docker Compose (for containerised deployment)
- 8 GB RAM minimum

### Python Node
- Python 3.11+
- pip packages: see `python-node/requirements.txt`
- Key: `grpcio`, `pyarrow`, `pyzmq`, `msgpack`, `onnxruntime`, `torch`

### R Node
- R 4.3+
- Packages: `arrow`, `MASS`, `robust`, `mvtnorm`, `data.table`
- System: `libzmq3-dev`

### C++ Node
- CMake 3.20+, GCC 12+ or Clang 15+
- Libraries: `libgrpc++-dev`, `libprotobuf-dev`, `libzmq3-dev`
- `libarmadillo-dev`, `libmlpack-dev`, `libarrow-dev`
- OpenBLAS + LAPACK

### Java Node
- JDK 21+
- Maven 3.9+
- Dependencies managed via `pom.xml`

## Build Steps

### Proto Compilation
```bash
# C++
protoc --proto_path=proto --cpp_out=cpp-node/build --grpc_out=cpp-node/build \
       --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/service.proto

# Python / Aggregator
python -m grpc_tools.protoc -I proto --python_out=python-node \
       --grpc_python_out=python-node proto/service.proto

# Java (handled by Maven plugin automatically)
```

### C++ Build
```bash
cd cpp-node
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Java Build
```bash
cd java-node
mvn package -DskipTests
```

### Docker Compose (all nodes)
```bash
docker-compose up --build
```

## Performance Benchmarking
```bash
# C++ node includes timing in output (nanosecond precision)
# Python node logs Arrow IPC payload sizes per round
# Java node logs event rates per round

# To run a 100-round benchmark:
python-node/federated_coordinator.py  # set rounds=100 in main()
```

## Threat Simulation
```bash
# Inject anomalous ticks (spike in spread variance)
python python-node/threat_injector.py --anomaly-rate 0.15 --rounds 50
```
