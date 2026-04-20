#include "federated_client.hpp"
#include <grpcpp/grpcpp.h>
#include <iostream>

namespace fspm {

FederatedClient::FederatedClient(const std::string& server_address)
  : channel_(grpc::CreateChannel(server_address,
                                  grpc::InsecureChannelCredentials())),
    stub_(FederatedAggregator::NewStub(channel_))
{}

bool FederatedClient::submit_weights(const std::string& node_id,
                                      const std::string& round_id,
                                      const std::vector<float>& weights,
                                      double loss,
                                      long   sample_count,
                                      const std::vector<uint8_t>& arrow_payload) {
  WeightUpdate req;
  req.set_node_id(node_id);
  req.set_round_id(round_id);
  for (auto w : weights) req.add_weights(w);
  req.set_loss(loss);
  req.set_sample_count(sample_count);
  req.set_arrow_payload(arrow_payload.data(), arrow_payload.size());

  AggregationResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->SubmitWeights(&ctx, req, &resp);

  if (!status.ok()) {
    std::cerr << "[CPP] gRPC error: " << status.error_message() << "\n";
    return false;
  }
  std::cout << "[CPP] Round " << round_id
            << " | global_loss=" << resp.global_loss()
            << " | participants=" << resp.participants() << "\n";
  return true;
}

bool FederatedClient::ping(const std::string& node_id) {
  HealthRequest  req;
  HealthResponse resp;
  grpc::ClientContext ctx;
  req.set_node_id(node_id);
  auto status = stub_->Ping(&ctx, req, &resp);
  return status.ok() && resp.healthy();
}

} // namespace fspm
