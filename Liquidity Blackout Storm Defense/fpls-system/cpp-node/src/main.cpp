// ── Liquidity Blackout Storm Defense — C++ Node Entry Point ──────────────────────────────────────
#include "anomaly_detector.hpp"
#include "zmq_publisher.hpp"
#include "federated_client.hpp"

#include <armadillo>
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <cstring>

using namespace fpls;
using namespace std::chrono_literals;

// Simulate market tick matrix (dim × n)
arma::mat generate_ticks(int n = 512, int dim = 16) {
  std::mt19937 rng(std::random_device{}());
  std::normal_distribution<double>      norm(0, 1);
  std::exponential_distribution<double> expd(20);

  arma::mat ticks(dim, n);
  for (int i = 0; i < n; ++i) {
    double base   = 100.0 + norm(rng) * 0.25;
    double spread = expd(rng);
    ticks(0, i)  = base - spread / 2;      // bid
    ticks(1, i)  = base + spread / 2;      // ask
    ticks(2, i)  = expd(rng) * 1000;       // volume
    ticks(3, i)  = expd(rng) * 0.01;       // inter-arrival
    for (int j = 4; j < dim; ++j)
      ticks(j, i) = norm(rng) * 0.02;      // micro-vol features
  }
  return ticks;
}

int main(int argc, char** argv) {
  const std::string node_id    = "cpp-node-01";
  const std::string grpc_addr  = argc > 1 ? argv[1] : "localhost:50051";
  const std::string zmq_ep     = "tcp://localhost:5570";
  const int         rounds     = 10;

  std::cout << "[CPP] Liquidity Blackout Storm Defense C++ Node starting\n";

  AnomalyDetector  detector(node_id);
  ZmqPublisher     publisher(zmq_ep);
  FederatedClient  client(grpc_addr);

  for (int rnd = 1; rnd <= rounds; ++rnd) {
    auto ticks = generate_ticks(512);
    detector.train(ticks);

    // Score batch
    auto results = detector.score_batch(ticks);
    int  n_anom  = 0;
    std::vector<float> first_feats;

    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i].is_anomaly) {
        ++n_anom;
        if (n_anom == 1) {
          first_feats.resize(ticks.n_rows);
          for (size_t j = 0; j < ticks.n_rows; ++j)
            first_feats[j] = static_cast<float>(ticks(j, i));
          publisher.publish_anomaly(node_id, results[i].score,
                                    detector.threshold(), first_feats);
        }
      }
    }

    std::cout << "[CPP] Round " << rnd
              << " | anomalies=" << n_anom << "\n";

    // Submit weights via gRPC
    auto weights = detector.serialise_weights();
    std::vector<float> wf(weights.size());
    for (size_t i = 0; i < weights.size(); ++i)
      wf[i] = static_cast<float>(weights[i]);

    client.submit_weights(node_id, std::to_string(rnd), wf, 0.01,
                          ticks.n_cols, weights);

    std::this_thread::sleep_for(500ms);
  }

  std::cout << "[CPP] Completed " << rounds << " rounds.\n";
  return 0;
}
