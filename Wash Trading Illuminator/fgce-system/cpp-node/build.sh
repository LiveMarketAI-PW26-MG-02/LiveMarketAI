#!/usr/bin/env bash
set -e

echo "Building Wash Trading Illuminator C++ node..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$(nproc)
echo "Build complete. Binary: build/fgce_node"

echo "Running tests..."
cd build && ctest --output-on-failure
