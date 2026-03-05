#!/usr/bin/env Rscript
# Test suite for Latency Arbitrage Sentinel R node

source("r_statistical_node.R")

cat("--- Test: generate_ticks ---\n")
X <- generate_ticks(128L)
stopifnot(nrow(X) == 128L, ncol(X) == 16L)
cat("[PASS]\n")

cat("--- Test: fit_robust_model ---\n")
m <- fit_robust_model(X)
stopifnot(length(m$center) == 16L)
cat("[PASS]\n")

cat("--- Test: mahalanobis_scores ---\n")
s <- mahalanobis_scores(X, m)
stopifnot(all(s >= 0), length(s) == 128L)
cat("[PASS]\n")

cat("--- Test: spread_entropy ---\n")
ent <- spread_entropy(X[,1], X[,2])
stopifnot(ent > 0)
cat(sprintf("[PASS] entropy=%.4f\n", ent))

cat("--- Test: Arrow IPC round-trip ---\n")
buf <- pack_weights_arrow(m$center, m$cov, ent, 0.01)
g   <- apply_global_weights(buf)
stopifnot(abs(g$center[1] - m$center[1]) < 1e-6)
cat("[PASS]\n")

cat("\n=== All R tests passed ===\n")
