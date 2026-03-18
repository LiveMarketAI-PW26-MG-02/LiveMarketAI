library(testthat)
source("../R/report_engine.R")

test_that("blotter has correct columns", {
  b <- generate_blotter(10)
  expect_true(all(c("trade_id","date","symbol","side","quantity","price","commission") %in% names(b)))
})

test_that("daily_summary has positive turnover", {
  s <- daily_summary()
  expect_true(all(s$gross_turnover > 0))
})

test_that("monthly_statement sums correctly", {
  b <- generate_blotter(100)
  ms <- monthly_statement(b)
  expect_equal(sum(ms$trades), nrow(b))
})

test_that("1099b realized gains are numeric", {
  result <- form_1099b()
  expect_type(result$realized_gain, "double")
})
