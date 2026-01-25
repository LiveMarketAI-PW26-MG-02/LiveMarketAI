# Simpler R service for demo
library(jsonlite)

# Suppress warnings and messages
options(warn = -1)
suppressMessages(suppressWarnings({
  library(quantmod)
}))

# Read input from command line argument instead of stdin
args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  ticker <- args[1]
} else {
  ticker <- "AAPL"  # Default
}

# Analyze stock
tryCatch({
  
  # Download data quietly
  invisible(capture.output(
    getSymbols(ticker, src = "yahoo", auto.assign = FALSE, warnings = FALSE) -> stock_data
  ))
  
  # Calculate statistics
  returns <- dailyReturn(Cl(stock_data))
  volatility <- round(sd(returns, na.rm = TRUE) * sqrt(252) * 100, 2)
  mean_return <- round(mean(returns, na.rm = TRUE) * 252 * 100, 2)
  sharpe <- round(mean_return / volatility, 2)
  
  # Trend analysis
  recent_prices <- tail(Cl(stock_data), 20)
  trend <- if(as.numeric(recent_prices[length(recent_prices)]) > as.numeric(recent_prices[1])) "UPWARD" else "DOWNWARD"
  
  # Prepare result
  result <- list(
    service = "R Statistical Analysis",
    ticker = ticker,
    statistics = list(
      annualized_return_pct = mean_return,
      annualized_volatility_pct = volatility,
      sharpe_ratio = sharpe,
      trend_20d = trend,
      data_points = nrow(stock_data)
    ),
    recommendation = if(sharpe > 1 && trend == "UPWARD") "FAVORABLE" else if(sharpe < 0.5) "RISKY" else "NEUTRAL"
  )
  
  # Output JSON
  cat(toJSON(result, auto_unbox = TRUE, pretty = FALSE))
  
}, error = function(e) {
  result <- list(
    service = "R Statistical Analysis",
    ticker = ticker,
    error = as.character(e$message)
  )
  cat(toJSON(result, auto_unbox = TRUE))
})