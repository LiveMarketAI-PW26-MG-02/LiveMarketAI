# CAPM & Factor Models

source("R/black_scholes.R")

#' Generate market and stock returns
simulate_returns <- function(n = 252, market_vol = 0.01, seed = 42) {
  set.seed(seed)
  market <- rnorm(n, 0.0003, market_vol)
  list(market = market)
}

#' Estimate CAPM beta (OLS)
#' @param stock_returns  Numeric vector
#' @param market_returns Numeric vector
capm_beta <- function(stock_returns, market_returns) {
  cov(stock_returns, market_returns) / var(market_returns)
}

#' CAPM expected return (annualised)
capm_expected_return <- function(beta, risk_free = 0.05, market_premium = 0.06) {
  risk_free + beta * market_premium
}

#' Simulate Fama-French 3-factor returns
#' @param n    Number of days
#' @param beta_mkt, beta_smb, beta_hml Factor loadings
ff3_returns <- function(n = 252, beta_mkt = 1.1, beta_smb = 0.3, beta_hml = -0.2, seed = 7) {
  set.seed(seed)
  mkt <- rnorm(n, 0.0004, 0.010)  # Market excess return
  smb <- rnorm(n, 0.0001, 0.005)  # Small minus big
  hml <- rnorm(n, 0.0001, 0.004)  # High minus low (value)
  alpha <- 0.0001
  epsilon <- rnorm(n, 0, 0.008)
  r <- alpha + beta_mkt * mkt + beta_smb * smb + beta_hml * hml + epsilon
  list(returns = r, mkt = mkt, smb = smb, hml = hml)
}

#' Full quant summary
quant_summary <- function(S = 178.5, K = 180, T = 0.25, r = 0.05, sigma = 0.28) {
  call_px <- bs_call(S, K, T, r, sigma)
  put_px  <- bs_put(S, K, T, r, sigma)
  greeks  <- bs_greeks(S, K, T, r, sigma, "call")
  iv      <- implied_vol(call_px * 0.95, S, K, T, r)

  cat("=== Black-Scholes Summary ===\n")
  cat(sprintf("Call Price:  $%.4f\n", call_px))
  cat(sprintf("Put  Price:  $%.4f\n", put_px))
  cat(sprintf("Delta:       %.4f\n",  greeks$delta))
  cat(sprintf("Gamma:       %.6f\n",  greeks$gamma))
  cat(sprintf("Theta:       %.4f\n",  greeks$theta))
  cat(sprintf("Vega:        %.4f\n",  greeks$vega))
  cat(sprintf("Impl Vol:    %.2f%%\n", iv * 100))

  ff <- ff3_returns()
  beta <- capm_beta(ff$returns, ff$mkt)
  exp_ret <- capm_expected_return(beta)
  cat("\n=== CAPM ===\n")
  cat(sprintf("Beta:            %.3f\n", beta))
  cat(sprintf("Expected Return: %.2f%%\n", exp_ret * 100))
}
