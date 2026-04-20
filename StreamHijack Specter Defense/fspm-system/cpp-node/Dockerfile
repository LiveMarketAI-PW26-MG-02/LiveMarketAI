FROM ubuntu:22.04

LABEL system="StreamHijack Specter Defense"
LABEL component="cpp-node"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libprotobuf-dev protobuf-compiler \
    libgrpc++-dev protobuf-compiler-grpc libzmq3-dev \
    libarmadillo-dev libmlpack-dev libarrow-dev \
    libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ../proto/service.proto /proto/service.proto
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

CMD ["./build/fspm_node", "aggregator:50051"]
