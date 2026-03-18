# Black-Scholes Option Pricing Model

#' Black-Scholes call price
#' @param S  Current stock price
#' @param K  Strike price
#' @param T  Time to expiry (years)
#' @param r  Risk-free rate (annual)
#' @param sigma Implied volatility (annual)
#' @return Call option price
bs_call <- function(S, K, T, r, sigma) {
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

#' Black-Scholes put price (via put-call parity)
bs_put <- function(S, K, T, r, sigma) {
  bs_call(S, K, T, r, sigma) - S + K * exp(-r * T)
}

#' Greeks
#' @return Named list: delta, gamma, theta, vega, rho
bs_greeks <- function(S, K, T, r, sigma, type = "call") {
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  phi_d1 <- dnorm(d1)   # standard normal PDF at d1

  delta <- if (type == "call") pnorm(d1) else pnorm(d1) - 1
  gamma <- phi_d1 / (S * sigma * sqrt(T))
  theta_call <- -(S * phi_d1 * sigma / (2 * sqrt(T))) - r * K * exp(-r * T) * pnorm(d2)
  theta <- if (type == "call") theta_call / 365
           else (theta_call + r * K * exp(-r * T)) / 365
  vega  <- S * phi_d1 * sqrt(T) / 100    # per 1% change in vol
  rho   <- if (type == "call")  K * T * exp(-r * T) * pnorm(d2)  / 100
           else                -K * T * exp(-r * T) * pnorm(-d2) / 100

  list(delta = delta, gamma = gamma, theta = theta, vega = vega, rho = rho)
}

#' Implied volatility via bisection
implied_vol <- function(market_price, S, K, T, r, type = "call",
                         tol = 1e-6, max_iter = 200) {
  price_fn <- if (type == "call") bs_call else bs_put
  lo <- 0.001; hi <- 5.0
  for (i in seq_len(max_iter)) {
    mid <- (lo + hi) / 2
    diff <- price_fn(S, K, T, r, mid) - market_price
    if (abs(diff) < tol) return(mid)
    if (diff > 0) hi <- mid else lo <- mid
  }
  (lo + hi) / 2
}

#' Option chain for a symbol
option_chain <- function(S = 178.5, r = 0.05, T = 0.25,
                          strikes = seq(160, 200, by = 5),
                          sigma = 0.28) {
  data.frame(
    strike = strikes,
    call   = sapply(strikes, function(K) round(bs_call(S, K, T, r, sigma), 2)),
    put    = sapply(strikes, function(K) round(bs_put (S, K, T, r, sigma), 2)),
    delta_call = sapply(strikes, function(K) round(bs_greeks(S,K,T,r,sigma,"call")$delta, 4)),
    delta_put  = sapply(strikes, function(K) round(bs_greeks(S,K,T,r,sigma,"put")$delta, 4))
  )
}
