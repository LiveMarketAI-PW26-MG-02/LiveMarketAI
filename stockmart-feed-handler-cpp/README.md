# stockmart-feed-handler-cpp ⚡

Market data feed handler — parse FIX/ITCH-like messages, normalize, distribute.

## Features
- FIX-style message parser
- Normalized tick publisher
- Multi-symbol subscription manager
- Throughput benchmarking

## Stack
C++17 · CMake · GoogleTest

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/feed_handler
```
