#!/usr/bin/env bash
set -e

echo "Building Regulatory Singularity Breach Defense C++ node..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$(nproc)
echo "Build complete. Binary: build/fcig_node"

echo "Running tests..."
cd build && ctest --output-on-failure
