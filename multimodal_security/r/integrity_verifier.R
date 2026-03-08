# =============================================================
#  MULTIMODAL SECURITY — R MODULE
#  Layer 2: Statistical Integrity Verification + Entropy Analysis
#  Format: XML read/write via xml2 (no JSON)
# =============================================================

# ─────────────────────────────────────────────
#  LIBRARY LOADING (base R wherever possible)
# ─────────────────────────────────────────────
suppressPackageStartupMessages({
  library(tools)      # md5sum, file hashing
  library(utils)      # zip / unzip
})

cat("=============================================================\n")
cat("  MULTIMODAL SECURITY — R LAYER 2\n")
cat("  Statistical Integrity Verification & Entropy Analysis\n")
cat("=============================================================\n\n")


# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
script_dir  <- normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE)
base_dir    <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
shared_dir  <- file.path(base_dir, "shared")
manifest_in <- file.path(shared_dir, "manifest_python.xml")
report_out  <- file.path(shared_dir, "security_report_r.xml")
enc_file    <- file.path(shared_dir, "archive.mmsec")
zip_file    <- file.path(shared_dir, "archive.zip")


# ─────────────────────────────────────────────
#  UTILITY: SHANNON ENTROPY
# ─────────────────────────────────────────────
shannon_entropy <- function(bytes_vec) {
  # Compute byte-level Shannon entropy (bits per byte, max = 8)
  freq  <- tabulate(bytes_vec + 1L, nbins = 256)   # count each 0–255
  prob  <- freq[freq > 0] / length(bytes_vec)
  -sum(prob * log2(prob))
}


# ─────────────────────────────────────────────
#  UTILITY: CHI-SQUARED UNIFORMITY TEST
# ─────────────────────────────────────────────
chi_squared_uniformity <- function(bytes_vec, significance = 0.05) {
  # A truly random (encrypted) byte stream should have ~uniform distribution
  observed <- tabulate(bytes_vec + 1L, nbins = 256)
  expected <- rep(length(bytes_vec) / 256, 256)
  chi_stat <- sum((observed - expected)^2 / expected)
  df       <- 255L
  p_value  <- pchisq(chi_stat, df = df, lower.tail = FALSE)
  list(
    statistic  = chi_stat,
    df         = df,
    p_value    = p_value,
    is_uniform = p_value > significance
  )
}


# ─────────────────────────────────────────────
#  UTILITY: AUTOCORRELATION RANDOMNESS CHECK
# ─────────────────────────────────────────────
autocorr_check <- function(bytes_vec, max_lag = 20) {
  # Encrypted data should show no significant autocorrelation
  x   <- as.numeric(bytes_vec)
  n   <- length(x)
  mn  <- mean(x)
  var <- var(x)
  if (var == 0) return(list(max_acf = NA, is_random = FALSE))
  acf_vals <- sapply(1:max_lag, function(lag) {
    sum((x[1:(n-lag)] - mn) * (x[(lag+1):n] - mn)) / ((n - lag) * var)
  })
  max_acf  <- max(abs(acf_vals))
  threshold <- 2 / sqrt(n)   # 95% confidence band
  list(
    max_acf   = max_acf,
    threshold = threshold,
    is_random = max_acf < threshold,
    lags      = seq_along(acf_vals),
    acf_vals  = acf_vals
  )
}


# ─────────────────────────────────────────────
#  UTILITY: RUNS TEST (Wald-Wolfowitz)
# ─────────────────────────────────────────────
runs_test <- function(bytes_vec) {
  # Binary runs test: above/below median
  med    <- median(bytes_vec)
  signs  <- ifelse(bytes_vec >= med, 1L, 0L)
  runs   <- rle(signs)$lengths
  n_runs <- length(runs)
  n1     <- sum(signs == 1L)
  n2     <- sum(signs == 0L)
  n_total <- n1 + n2
  if (n1 == 0 || n2 == 0) return(list(z_score = NA, is_random = FALSE))
  mu_r   <- (2 * n1 * n2 / n_total) + 1
  sigma2 <- (2 * n1 * n2 * (2 * n1 * n2 - n_total)) /
            (n_total^2 * (n_total - 1))
  z      <- (n_runs - mu_r) / sqrt(sigma2)
  p_val  <- 2 * pnorm(abs(z), lower.tail = FALSE)
  list(
    n_runs   = n_runs,
    z_score  = z,
    p_value  = p_val,
    is_random = p_val > 0.05
  )
}


# ─────────────────────────────────────────────
#  CORE: ANALYSE A BINARY FILE
# ─────────────────────────────────────────────
analyse_file <- function(filepath, label = "File") {
  cat(sprintf("\n[ANALYSE] %s: %s\n", label, basename(filepath)))

  if (!file.exists(filepath)) {
    cat("  ERROR: File not found!\n")
    return(NULL)
  }

  raw_bytes <- readBin(filepath, what = "raw", n = file.info(filepath)$size)
  byte_vals <- as.integer(raw_bytes)
  n_bytes   <- length(byte_vals)
  cat(sprintf("  Size: %d bytes\n", n_bytes))

  # --- Entropy ---
  entropy <- shannon_entropy(byte_vals)
  cat(sprintf("  Shannon Entropy  : %.4f bits/byte  (max=8.0)\n", entropy))

  # --- Chi-squared ---
  chi   <- chi_squared_uniformity(byte_vals)
  cat(sprintf("  Chi² Statistic   : %.2f  (df=255, p=%.4f) Uniform=%s\n",
              chi$statistic, chi$p_value,
              ifelse(chi$is_uniform, "YES ✓", "NO ✗")))

  # --- Autocorrelation ---
  sample_size <- min(n_bytes, 10000L)
  acr   <- autocorr_check(byte_vals[seq_len(sample_size)])
  cat(sprintf("  Max |ACF| (lag≤20): %.4f  (threshold=%.4f) Random=%s\n",
              acr$max_acf, acr$threshold,
              ifelse(acr$is_random, "YES ✓", "NO ✗")))

  # --- Runs test ---
  rt    <- runs_test(byte_vals[seq_len(sample_size)])
  cat(sprintf("  Runs Test z-score: %.4f  (p=%.4f) Random=%s\n",
              rt$z_score, rt$p_value,
              ifelse(rt$is_random, "YES ✓", "NO ✗")))

  # --- MD5 (for cross-language check) ---
  md5_val <- tools::md5sum(filepath)
  cat(sprintf("  MD5              : %s\n", md5_val))

  list(
    filepath    = filepath,
    label       = label,
    size_bytes  = n_bytes,
    entropy     = entropy,
    chi_stat    = chi$statistic,
    chi_p       = chi$p_value,
    chi_uniform = chi$is_uniform,
    acf_max     = acr$max_acf,
    acf_random  = acr$is_random,
    runs_z      = rt$z_score,
    runs_random = rt$is_random,
    md5         = md5_val[[1]]
  )
}


# ─────────────────────────────────────────────
#  XML PARSING: READ PYTHON MANIFEST
# ─────────────────────────────────────────────
read_python_manifest <- function(xml_path) {
  cat(sprintf("\n[XML] Reading Python manifest: %s\n", basename(xml_path)))
  if (!file.exists(xml_path)) {
    cat("  WARNING: manifest not found — skipping\n")
    return(NULL)
  }

  doc   <- xmlParse(xml_path)   # base XML parser fallback
  # Use manual line-parse if xmlParse unavailable (pure base R):
  lines <- readLines(xml_path)

  extract_tag <- function(tag) {
    pattern <- sprintf("<%s>(.*?)</%s>", tag, tag)
    hits    <- regmatches(lines, regexpr(pattern, lines, perl = TRUE))
    if (length(hits) == 0) return(NA_character_)
    sub(sprintf(".*<%s>(.*?)</%s>.*", tag, tag), "\\1", hits[1], perl = TRUE)
  }

  list(
    enc_sha256   = extract_tag("EncryptedFileSHA256"),
    zip_sha256   = extract_tag("OriginalZipSHA256"),
    algorithm    = extract_tag("Algorithm"),
    key_size     = extract_tag("KeySizeBits"),
    hmac_val     = extract_tag("HMAC"),
    mac_header   = extract_tag("MagicHeader")
  )
}


# ─────────────────────────────────────────────
#  XML WRITING: SECURITY REPORT
# ─────────────────────────────────────────────
write_security_report <- function(analyses, manifest, xml_path) {
  cat(sprintf("\n[XML] Writing R security report → %s\n", basename(xml_path)))

  ts   <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  lines <- c(
    '<?xml version="1.0" encoding="UTF-8"?>',
    sprintf('<SecurityReport language="R" layer="2" timestamp="%s">', ts),
    '  <Description>Statistical integrity and randomness analysis of encrypted ZIP</Description>',
    '',
    '  <PythonManifest>'
  )

  if (!is.null(manifest)) {
    lines <- c(lines,
      sprintf('    <EncryptedFileSHA256>%s</EncryptedFileSHA256>', manifest$enc_sha256 %||% ""),
      sprintf('    <OriginalZipSHA256>%s</OriginalZipSHA256>',   manifest$zip_sha256  %||% ""),
      sprintf('    <Algorithm>%s</Algorithm>',                   manifest$algorithm   %||% ""),
      sprintf('    <KeySizeBits>%s</KeySizeBits>',               manifest$key_size    %||% ""),
      sprintf('    <HMAC>%s</HMAC>',                             manifest$hmac_val    %||% "")
    )
  }

  lines <- c(lines, '  </PythonManifest>', '', '  <FileAnalyses>')

  for (a in analyses) {
    if (is.null(a)) next
    overall_ok <- a$chi_uniform && a$acf_random && a$runs_random && a$entropy > 7.0
    lines <- c(lines,
      sprintf('    <FileAnalysis label="%s">', a$label),
      sprintf('      <Filepath>%s</Filepath>', basename(a$filepath)),
      sprintf('      <SizeBytes>%d</SizeBytes>', a$size_bytes),
      sprintf('      <ShannonEntropy>%.6f</ShannonEntropy>', a$entropy),
      sprintf('      <ChiSquaredStat>%.4f</ChiSquaredStat>', a$chi_stat),
      sprintf('      <ChiSquaredP>%.6f</ChiSquaredP>', a$chi_p),
      sprintf('      <IsUniform>%s</IsUniform>', tolower(a$chi_uniform)),
      sprintf('      <MaxACF>%.6f</MaxACF>', a$acf_max),
      sprintf('      <ACFRandom>%s</ACFRandom>', tolower(a$acf_random)),
      sprintf('      <RunsZScore>%.4f</RunsZScore>', a$runs_z),
      sprintf('      <RunsRandom>%s</RunsRandom>', tolower(a$runs_random)),
      sprintf('      <MD5>%s</MD5>', a$md5),
      sprintf('      <OverallSecure>%s</OverallSecure>', tolower(overall_ok)),
      '    </FileAnalysis>'
    )
  }

  lines <- c(lines,
    '  </FileAnalyses>',
    '',
    '  <SecurityTests>',
    '    <Test name="ShannonEntropy"  threshold="gt 7.0 bits/byte" />',
    '    <Test name="ChiSquared"      threshold="p gt 0.05 (uniform dist)" />',
    '    <Test name="Autocorrelation" threshold="max ACF within 95pct CI" />',
    '    <Test name="RunsTest"        threshold="p gt 0.05 (Wald-Wolfowitz)" />',
    '  </SecurityTests>',
    '</SecurityReport>'
  )

  writeLines(lines, con = xml_path, useBytes = FALSE)
  cat("  [XML] Report written.\n")
}

# Helper: null-coalescing operator
`%||%` <- function(x, y) if (is.na(x) || is.null(x)) y else x


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
main <- function() {
  cat("[STEP 1] Analysing encrypted archive …\n")
  enc_analysis <- analyse_file(enc_file, label = "EncryptedArchive")

  cat("\n[STEP 2] Analysing original ZIP (if present) …\n")
  zip_analysis <- analyse_file(zip_file, label = "OriginalZIP")

  cat("\n[STEP 3] Reading Python XML manifest …\n")
  manifest <- tryCatch(
    read_python_manifest(manifest_in),
    error = function(e) { cat("  WARN:", conditionMessage(e), "\n"); NULL }
  )

  cat("\n[STEP 4] Writing R security report …\n")
  write_security_report(
    analyses = list(enc_analysis, zip_analysis),
    manifest = manifest,
    xml_path = report_out
  )

  cat("\n[STEP 5] Summary\n")
  cat("─────────────────────────────────────────────\n")
  if (!is.null(enc_analysis)) {
    cat(sprintf("  Encrypted file entropy : %.4f / 8.0\n", enc_analysis$entropy))
    enc_secure <- enc_analysis$entropy > 7.5
    cat(sprintf("  Encryption quality     : %s\n",
                ifelse(enc_secure, "HIGH ✓ (entropy > 7.5)", "LOW ✗")))
  }
  cat("─────────────────────────────────────────────\n")
  cat("\n[DONE] R layer complete.\n")
}

main()
