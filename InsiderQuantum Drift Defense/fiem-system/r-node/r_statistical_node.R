#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# InsiderQuantum Drift Defense — R Statistical Node
# Role   : Bayesian anomaly detection + federated weight exchange
# Transport: Apache Arrow IPC (primary), ZeroMQ MessagePack (pub/sub)
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(MASS)
  library(robust)
  library(mvtnorm)
  library(fitdistrplus)
}))

set.seed(42L)
NODE_ID <- paste0("r-stat-", Sys.getenv("HOSTNAME", "local"))
cat(sprintf("[R-NODE] Starting %s node: %s\n", "InsiderQuantum Drift Defense", NODE_ID))

# ── Simulated tick feature generator ─────────────────────────────────────────
generate_ticks <- function(n = 256L) {
  bid  <- 100 + rnorm(n, 0, 0.25)
  ask  <- bid + rexp(n, rate = 20)
  vol  <- rexp(n, rate = 1e-3)
  iat  <- rexp(n, rate = 100)
  micro <- matrix(rnorm(n * 12, 0, 0.02), nrow = n)
  cbind(bid, ask, vol, iat, micro)
}

# ── Robust Mahalanobis anomaly detector ───────────────────────────────────────
fit_robust_model <- function(X) {
  tryCatch({
    cov_rob <- cov.rob(X, method = "mcd", quantile.used = 0.75)
    list(center = cov_rob$center, cov = cov_rob$cov)
  }, error = function(e) {
    list(center = colMeans(X), cov = cov(X))
  })
}

mahalanobis_scores <- function(X, model) {
  mahalanobis(X, model$center, model$cov)
}

# ── Gaussian mixture anomaly baseline ────────────────────────────────────────
fit_gmm_baseline <- function(X, k = 3L) {
  # Simple k-means as GMM surrogate (no JSON serialisation)
  km <- kmeans(X, centers = k, nstart = 5L, iter.max = 50L)
  list(centers = km$centers, cluster = km$cluster, withinss = km$withinss)
}

# ── Entropy calculation on spread distribution ────────────────────────────────
spread_entropy <- function(bid, ask, bins = 30L) {
  spread <- ask - bid
  h      <- hist(spread, breaks = bins, plot = FALSE)
  p      <- h$counts / sum(h$counts)
  p      <- p[p > 0]
  -sum(p * log2(p))
}

# ── Federated weight vector (serialise as raw bytes via Arrow IPC) ────────────
pack_weights_arrow <- function(center, cov_diag, entropy, loss) {
  schema <- arrow::schema(
    center   = arrow::list_of(arrow::float64()),
    cov_diag = arrow::list_of(arrow::float64()),
    entropy  = arrow::float64(),
    loss     = arrow::float64(),
    node_id  = arrow::utf8()
  )
  tbl <- arrow::arrow_table(
    center   = list(center),
    cov_diag = list(diag(cov_diag)),
    entropy  = entropy,
    loss     = loss,
    node_id  = NODE_ID,
    schema   = schema
  )
  sink <- arrow::BufferOutputStream$create()
  writer <- arrow::RecordBatchStreamWriter$create(sink, tbl$schema)
  writer$write_table(tbl)
  writer$close()
  sink$finish()
}

# ── Apply incoming global weights ─────────────────────────────────────────────
apply_global_weights <- function(buf) {
  reader  <- arrow::RecordBatchStreamReader$create(buf)
  tbl     <- reader$read_all()
  center  <- as.numeric(tbl$center[[1]])
  cov_d   <- as.numeric(tbl$cov_diag[[1]])
  entropy <- as.numeric(tbl$entropy[[1]])
  list(center = center, cov_diag = cov_d, entropy = entropy)
}

# ── Main federated loop ───────────────────────────────────────────────────────
run_r_node <- function(rounds = 10L) {
  cat("[R-NODE] Beginning", rounds, "federated rounds\n")

  for (rnd in seq_len(rounds)) {
    X       <- generate_ticks(512L)
    model   <- fit_robust_model(X)
    scores  <- mahalanobis_scores(X, model)
    thresh  <- qchisq(0.975, df = ncol(X))
    anomaly_idx <- which(scores > thresh)
    ent     <- spread_entropy(X[,1], X[,2])

    cat(sprintf("[R-NODE] Round %2d | anomalies=%d | entropy=%.4f | threshold=%.4f\n",
                rnd, length(anomaly_idx), ent, thresh))

    # Pack weights as Arrow IPC
    buf <- pack_weights_arrow(
      center   = model$center,
      cov_diag = model$cov,
      entropy  = ent,
      loss     = mean(scores[anomaly_idx %||% 1])
    )

    # Simulate global weight application
    global <- apply_global_weights(buf)
    cat(sprintf("[R-NODE] Global center[1]=%.4f\n", global$center[1]))

    Sys.sleep(0.5)
  }

  cat("[R-NODE] Completed", rounds, "rounds.\n")
}

# Null-coalescing helper
`%||%` <- function(x, y) if (length(x) == 0) y else x

run_r_node(rounds = 10L)
