// Unit tests for StreamHijack Specter Defense C++ anomaly detector
#include "../src/anomaly_detector.hpp"
#include <armadillo>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace fspm;

void test_train_and_score() {
  AnomalyDetector det("test-node", 4, 0.9);
  arma::mat ticks = arma::randn(4, 200);
  det.train(ticks);

  arma::vec tick = ticks.col(0);
  auto res = det.score(tick);
  assert(res.score >= 0.0 && res.score <= 1.0);
  std::cout << "[PASS] train_and_score: score=" << res.score << "\n";
}

void test_serialise_roundtrip() {
  AnomalyDetector a("node-a", 4, 0.9), b("node-b", 4, 0.9);
  arma::mat ticks = arma::randn(4, 100);
  a.train(ticks);
  auto blob = a.serialise_weights();
  b.apply_global_weights(blob);
  std::cout << "[PASS] serialise_roundtrip: blob_size=" << blob.size() << "\n";
}

void test_batch_scoring() {
  AnomalyDetector det("batch-node", 4, 0.9);
  arma::mat ticks = arma::randn(4, 64);
  det.train(ticks);
  auto res = det.score_batch(ticks);
  assert(res.size() == 64);
  std::cout << "[PASS] batch_scoring: " << res.size() << " results\n";
}

int main() {
  test_train_and_score();
  test_serialise_roundtrip();
  test_batch_scoring();
  std::cout << "\n=== All C++ tests passed ===\n";
  return 0;
}
