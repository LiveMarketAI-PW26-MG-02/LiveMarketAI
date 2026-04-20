cmake_minimum_required(VERSION 3.20)
project(ftcm_cpp_node CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── Find packages ─────────────────────────────────────────────────────────────
find_package(Protobuf  REQUIRED)
find_package(gRPC      CONFIG REQUIRED)
find_package(Arrow     REQUIRED)
find_package(ZeroMQ    REQUIRED)
find_package(mlpack    REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenMP)
find_package(Threads   REQUIRED)

# ── Proto generation ──────────────────────────────────────────────────────────
set(PROTO_FILE "${CMAKE_SOURCE_DIR}/../proto/service.proto")
get_filename_component(PROTO_DIR "${PROTO_FILE}" DIRECTORY)

set(PROTO_SRCS "${CMAKE_BINARY_DIR}/service.pb.cc")
set(PROTO_HDRS "${CMAKE_BINARY_DIR}/service.pb.h")
set(GRPC_SRCS  "${CMAKE_BINARY_DIR}/service.grpc.pb.cc")
set(GRPC_HDRS  "${CMAKE_BINARY_DIR}/service.grpc.pb.h")

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
  COMMAND protoc
    --proto_path=${PROTO_DIR}
    --cpp_out=${CMAKE_BINARY_DIR}
    --grpc_out=${CMAKE_BINARY_DIR}
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(ftcm_node
  src/main.cpp
  src/anomaly_detector.cpp
  src/zmq_publisher.cpp
  src/federated_client.cpp
  ${PROTO_SRCS}
  ${GRPC_SRCS}
)

target_include_directories(ftcm_node PRIVATE
  ${CMAKE_BINARY_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${MLPACK_INCLUDE_DIRS}
)

target_link_libraries(ftcm_node
  gRPC::grpc++
  protobuf::libprotobuf
  Arrow::arrow_shared
  ${ZeroMQ_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  mlpack
  Threads::Threads
  ${OpenMP_CXX_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_compile_options(ftcm_node PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_ftcm test/test_anomaly.cpp src/anomaly_detector.cpp)
target_include_directories(test_ftcm PRIVATE src ${ARMADILLO_INCLUDE_DIRS} ${MLPACK_INCLUDE_DIRS})
target_link_libraries(test_ftcm ${ARMADILLO_LIBRARIES} mlpack Threads::Threads)
add_test(NAME AnomalyDetectorTests COMMAND test_ftcm)
