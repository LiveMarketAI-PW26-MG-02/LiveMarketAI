#pragma once
// ── Regulatory Singularity Breach Defense C++ Node: MLpack-based anomaly detector ──────────────────

#include <armadillo>
#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/local_outlier_factor/lof.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace fcig {

struct AnomalyResult {
  double  score;
  bool    is_anomaly;
  int64_t timestamp_ns;
};

// Federated weight packet (binary-safe, no JSON)
struct WeightPacket {
  std::vector<float> weights;
  float              loss;
  std::string        node_id;
  uint64_t           round_id;
};

class AnomalyDetector {
public:
  explicit AnomalyDetector(const std::string& node_id,
                           int                input_dim  = 16,
                           double             threshold  = 0.95);

  // Train local model on tick feature matrix (n × dim)
  void train(const arma::mat& ticks);

  // Score a single tick vector
  AnomalyResult score(const arma::vec& tick) const;

  // Batch scoring
  std::vector<AnomalyResult> score_batch(const arma::mat& ticks) const;

  // Serialise weights for federated exchange (binary blob)
  std::vector<uint8_t> serialise_weights() const;

  // Apply received global weights
  void apply_global_weights(const std::vector<uint8_t>& blob);

  // Access threshold
  double threshold() const { return threshold_; }

private:
  std::string node_id_;
  int         input_dim_;
  double      threshold_;

  // MLpack Density Estimation Tree (non-parametric anomaly detector)
  std::unique_ptr<mlpack::DTree<>> det_;

  // Running statistics for normalisation
  arma::vec mean_;
  arma::vec stddev_;
  bool      trained_ = false;

  // Federated weight vector (flattened DET leaf volumes)
  std::vector<float> fed_weights_;
};

} // namespace fcig
