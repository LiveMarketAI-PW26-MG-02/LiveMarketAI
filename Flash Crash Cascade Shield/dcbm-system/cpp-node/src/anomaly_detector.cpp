#include "anomaly_detector.hpp"
#include <armadillo>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dcbm {

AnomalyDetector::AnomalyDetector(const std::string& node_id,
                                 int                input_dim,
                                 double             threshold)
  : node_id_(node_id), input_dim_(input_dim), threshold_(threshold)
{}

void AnomalyDetector::train(const arma::mat& ticks) {
  // Normalise
  mean_   = arma::mean(ticks, 1);
  stddev_ = arma::stddev(ticks, 0, 1) + 1e-8;
  arma::mat norm = (ticks.each_col() - mean_).each_col() / stddev_;

  // Train DET (Density Estimation Tree) via MLpack
  det_ = std::make_unique<mlpack::DTree<>>();
  det_->Train(norm, /* useVolReg */ true);

  // Extract leaf volumes as federated weights
  fed_weights_.clear();
  // Simplified: use column variances as surrogate weights
  for (size_t i = 0; i < norm.n_rows; ++i) {
    fed_weights_.push_back(static_cast<float>(arma::var(norm.row(i))));
  }
  trained_ = true;
}

AnomalyResult AnomalyDetector::score(const arma::vec& tick) const {
  if (!trained_) throw std::runtime_error("Model not trained");
  arma::vec norm = (tick - mean_) / stddev_;
  // DET log-density: lower density ⇒ higher anomaly score
  double log_density = det_->LogDensity(norm);
  double score       = 1.0 - std::exp(log_density);
  score = std::clamp(score, 0.0, 1.0);

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  int64_t ns = static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;

  return {score, score >= threshold_, ns};
}

std::vector<AnomalyResult> AnomalyDetector::score_batch(const arma::mat& ticks) const {
  std::vector<AnomalyResult> results;
  results.reserve(ticks.n_cols);
  for (size_t i = 0; i < ticks.n_cols; ++i) {
    results.push_back(score(ticks.col(i)));
  }
  return results;
}

// Binary serialisation: 4-byte count + float array (little-endian)
std::vector<uint8_t> AnomalyDetector::serialise_weights() const {
  uint32_t n = static_cast<uint32_t>(fed_weights_.size());
  std::vector<uint8_t> blob(4 + n * sizeof(float));
  std::memcpy(blob.data(), &n, 4);
  std::memcpy(blob.data() + 4, fed_weights_.data(), n * sizeof(float));
  return blob;
}

void AnomalyDetector::apply_global_weights(const std::vector<uint8_t>& blob) {
  if (blob.size() < 4) throw std::runtime_error("Blob too small");
  uint32_t n;
  std::memcpy(&n, blob.data(), 4);
  if (blob.size() < 4 + n * sizeof(float))
    throw std::runtime_error("Blob truncated");
  fed_weights_.resize(n);
  std::memcpy(fed_weights_.data(), blob.data() + 4, n * sizeof(float));
}

} // namespace dcbm
