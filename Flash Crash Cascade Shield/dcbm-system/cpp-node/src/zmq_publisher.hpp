#pragma once
// ZeroMQ binary publisher — Flash Crash Cascade Shield

#include <zmq.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace dcbm {

class ZmqPublisher {
public:
  explicit ZmqPublisher(const std::string& endpoint = "tcp://localhost:5570");
  ~ZmqPublisher();

  void publish_anomaly(const std::string& node_id,
                       double             score,
                       double             threshold,
                       const std::vector<float>& features);

private:
  zmq::context_t context_;
  zmq::socket_t  socket_;
  std::string    topic_;
};

} // namespace dcbm
