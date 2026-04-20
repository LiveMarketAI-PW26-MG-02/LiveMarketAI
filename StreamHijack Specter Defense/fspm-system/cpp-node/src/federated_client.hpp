#pragma once
// gRPC federated client — StreamHijack Specter Defense

#include <grpcpp/grpcpp.h>
#include "service.grpc.pb.h"
#include <vector>
#include <string>
#include <memory>

namespace fspm {

class FederatedClient {
public:
  explicit FederatedClient(const std::string& server_address);

  bool submit_weights(const std::string& node_id,
                      const std::string& round_id,
                      const std::vector<float>& weights,
                      double loss,
                      long   sample_count,
                      const std::vector<uint8_t>& arrow_payload);

  bool ping(const std::string& node_id);

private:
  std::shared_ptr<grpc::Channel>            channel_;
  std::unique_ptr<FederatedAggregator::Stub> stub_;
};

} // namespace fspm
