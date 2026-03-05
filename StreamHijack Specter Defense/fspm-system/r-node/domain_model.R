#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# StreamHijack Specter Defense — R Domain-Specific Statistical Model
# FSPM: Federated Session Physics Modeling (FSPM): Models session physics including interaction cadence entropy, propagation del...
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(MASS)
  library(robust)
  library(mvtnorm)
}))

NODE_ID <- paste0("fspm-r-domain-", Sys.getenv("HOSTNAME", "local"))
cat(sprintf("[R-DOMAIN] %s domain model node: %s\n", "StreamHijack Specter Defense", NODE_ID))

# ── Domain feature generator ──────────────────────────────────────────────────
# Simulates domain-specific observations (behaviour / fingerprint / session /
# gradient / intent / compliance depending on system)
generate_domain_features <- function(n = 256L) {
  set.seed(sample.int(1e6, 1))
  # 16-dimensional feature space matching the system's input dimensionality
  normal_part  <- matrix(rnorm(n * 12, 0, 1), nrow = n)
  entropy_part <- matrix(rexp(n * 2, rate = 5), nrow = n)
  graph_part   <- matrix(rbeta(n * 2, 2, 5), nrow = n)
  cbind(normal_part, entropy_part, graph_part)
}

# ── Anomalous injection (domain-specific attack simulation) ───────────────────
generate_attack_features <- function(n = 32L, attack_intensity = 3.0) {
  base  <- generate_domain_features(n)
  # Shift specific dimensions to simulate domain attack
  base[, 1:4] <- base[, 1:4] * attack_intensity + attack_intensity
  base[, 13]  <- rexp(n, rate = 0.1)  # extreme entropy
  base
}

# ── Bayesian anomaly scoring (Mahalanobis + chi-squared) ─────────────────────
fit_bayes_model <- function(X) {
  tryCatch({
    cr <- cov.rob(X, method = "mcd", quantile.used = 0.75)
    list(center = cr$center, cov = cr$cov, method = "mcd")
  }, error = function(e) {
    list(center = colMeans(X), cov = cov(X), method = "sample")
  })
}

domain_anomaly_score <- function(x_vec, model) {
  d2 <- mahalanobis(matrix(x_vec, nrow = 1), model$center, model$cov)
  # P-value under chi-squared(df=length(x_vec)): lower p = more anomalous
  p  <- 1 - pchisq(d2, df = length(x_vec))
  list(distance = d2, p_value = p, is_anomaly = (p < 0.025))
}

# ── Federated weight packing via Arrow IPC ────────────────────────────────────
pack_domain_weights <- function(model, entropy_score, n_anomalies) {
  schema <- arrow::schema(
    center       = arrow::list_of(arrow::float64()),
    cov_diagonal = arrow::list_of(arrow::float64()),
    entropy      = arrow::float64(),
    n_anomalies  = arrow::int32(),
    node_id      = arrow::utf8(),
    system       = arrow::utf8()
  )
  tbl <- arrow::arrow_table(
    center       = list(model$center),
    cov_diagonal = list(diag(model$cov)),
    entropy      = entropy_score,
    n_anomalies  = as.integer(n_anomalies),
    node_id      = NODE_ID,
    system       = "FSPM",
    schema       = schema
  )
  sink   <- arrow::BufferOutputStream$create()
  writer <- arrow::RecordBatchStreamWriter$create(sink, tbl$schema)
  writer$write_table(tbl)
  writer$close()
  sink$finish()
}

# ── Main domain model loop ────────────────────────────────────────────────────
run_domain_model <- function(rounds = 10L) {
  for (rnd in seq_len(rounds)) {
    X_normal <- generate_domain_features(512L)
    X_attack <- generate_attack_features(32L)
    X_all    <- rbind(X_normal, X_attack)

    model      <- fit_bayes_model(X_normal)

    # Score all samples
    scores <- sapply(seq_len(nrow(X_all)), function(i) {
      domain_anomaly_score(X_all[i, ], model)$is_anomaly
    })
    n_detected <- sum(scores)

    entropy <- -sum(apply(X_normal[,1:8], 2, function(col) {
      h <- hist(col, breaks = 20, plot = FALSE)
      p <- h$counts / sum(h$counts); p <- p[p > 0]
      -sum(p * log2(p))
    }))

    cat(sprintf("[R-DOMAIN] Round %2d | detected=%d | entropy=%.4f | method=%s\n",
                rnd, n_detected, entropy, model$method))

    buf <- pack_domain_weights(model, entropy, n_detected)
    cat(sprintf("[R-DOMAIN] Round %2d | Arrow IPC bytes=%d\n",
                rnd, length(buf)))

    Sys.sleep(0.3)
  }
  cat("[R-DOMAIN] Domain model completed.\n")
}

run_domain_model()
