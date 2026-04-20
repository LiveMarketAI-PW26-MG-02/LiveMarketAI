library(testthat)
source("../R/black_scholes.R")
source("../R/capm.R")

test_that("call > 0 when ITM", {
  expect_gt(bs_call(200, 180, 0.25, 0.05, 0.25), 0)
})

test_that("put-call parity holds", {
  S <- 178.5; K <- 180; T <- 0.25; r <- 0.05; sigma <- 0.28
  parity <- bs_call(S,K,T,r,sigma) - bs_put(S,K,T,r,sigma) - S + K*exp(-r*T)
  expect_lt(abs(parity), 1e-8)
})

test_that("delta in [0,1] for call", {
  g <- bs_greeks(178.5, 180, 0.25, 0.05, 0.28, "call")
  expect_gte(g$delta, 0)
  expect_lte(g$delta, 1)
})

test_that("capm expected return is numeric", {
  sim <- simulate_returns()
  stock <- sim$market + rnorm(252, 0, 0.005)
  b <- capm_beta(stock, sim$market)
  e <- capm_expected_return(b)
  expect_type(e, "double")
})
