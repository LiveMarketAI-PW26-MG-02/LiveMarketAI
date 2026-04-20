package com.ftcm.node;

/** Immutable anomaly event record (no JSON). */
public record AnomalyRecord(
    String nodeId,
    double score,
    double threshold,
    long   timestampNs
) {
  public boolean isAnomaly() { return score >= threshold; }
}
