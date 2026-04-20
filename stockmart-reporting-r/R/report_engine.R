# StockMart Reporting Engine

library(dplyr)

#' Generate synthetic trade blotter
generate_blotter <- function(n = 50, seed = 42) {
  set.seed(seed)
  symbols  <- c("AAPL","GOOGL","TSLA","MSFT","NVDA","AMZN","META")
  sides    <- c("BUY","SELL")
  dates    <- seq(as.Date("2024-01-01"), as.Date("2024-03-31"), by = "day")
  dates    <- dates[!weekdays(dates) %in% c("Saturday","Sunday")]

  data.frame(
    trade_id   = paste0("T", sprintf("%05d", seq_len(n))),
    date       = sample(dates, n, replace = TRUE),
    symbol     = sample(symbols, n, replace = TRUE),
    side       = sample(sides, n, replace = TRUE),
    quantity   = sample(c(10,25,50,100,200), n, replace = TRUE),
    price      = round(runif(n, 80, 900), 2),
    commission = round(runif(n, 0.5, 5.0), 2),
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      gross_amount = quantity * price,
      net_amount   = ifelse(side == "BUY",
                            -(gross_amount + commission),
                             gross_amount - commission)
    ) %>%
    arrange(date, trade_id)
}

#' Daily trading summary (like ecommerce daily sales dashboard)
daily_summary <- function(blotter = generate_blotter()) {
  blotter %>%
    group_by(date) %>%
    summarise(
      total_trades  = n(),
      buy_volume    = sum(quantity[side == "BUY"],  na.rm = TRUE),
      sell_volume   = sum(quantity[side == "SELL"], na.rm = TRUE),
      gross_turnover = sum(gross_amount),
      total_commission = sum(commission),
      net_cash_flow = sum(net_amount),
      .groups = "drop"
    ) %>%
    arrange(date)
}

#' Monthly statement
monthly_statement <- function(blotter = generate_blotter()) {
  blotter %>%
    mutate(month = format(date, "%Y-%m")) %>%
    group_by(month) %>%
    summarise(
      trades = n(),
      symbols_traded = n_distinct(symbol),
      gross_turnover = round(sum(gross_amount), 2),
      total_commissions = round(sum(commission), 2),
      net_cash_flow = round(sum(net_amount), 2),
      .groups = "drop"
    )
}

#' Per-symbol breakdown (ecommerce: sales by product category)
symbol_breakdown <- function(blotter = generate_blotter()) {
  blotter %>%
    group_by(symbol) %>%
    summarise(
      total_trades  = n(),
      total_bought  = sum(quantity[side == "BUY"],  na.rm = TRUE),
      total_sold    = sum(quantity[side == "SELL"], na.rm = TRUE),
      avg_buy_price = round(mean(price[side == "BUY"],  na.rm = TRUE), 2),
      avg_sell_price= round(mean(price[side == "SELL"], na.rm = TRUE), 2),
      gross_turnover = round(sum(gross_amount), 2),
      .groups = "drop"
    ) %>%
    arrange(desc(gross_turnover))
}

#' 1099-B simulation (realized gains/losses)
form_1099b <- function(blotter = generate_blotter()) {
  buys  <- blotter %>% filter(side == "BUY")  %>% arrange(symbol, date)
  sells <- blotter %>% filter(side == "SELL") %>% arrange(symbol, date)

  # Simple FIFO matching (approximate)
  result <- sells %>%
    left_join(
      buys %>% group_by(symbol) %>% summarise(avg_buy = mean(price), .groups="drop"),
      by = "symbol"
    ) %>%
    mutate(
      cost_basis     = round(quantity * coalesce(avg_buy, price * 0.9), 2),
      proceeds       = gross_amount,
      realized_gain  = round(proceeds - cost_basis - commission, 2),
      holding_period = "SHORT"   # simplification
    ) %>%
    select(trade_id, date, symbol, quantity, proceeds, cost_basis, realized_gain, holding_period)

  cat("=== Form 1099-B Summary ===\n")
  cat(sprintf("Total Realized Gains:  $%s\n", format(round(sum(result$realized_gain)), big.mark=",")))
  cat(sprintf("Total Proceeds:        $%s\n", format(round(sum(result$proceeds)), big.mark=",")))
  cat(sprintf("Taxable Transactions:  %d\n",  nrow(result)))

  result
}

#' Full report to console
run_reports <- function() {
  blotter <- generate_blotter()

  cat("=== StockMart Daily Summary (first 5 days) ===\n")
  print(head(daily_summary(blotter), 5))

  cat("\n=== Monthly Statement ===\n")
  print(monthly_statement(blotter))

  cat("\n=== Symbol Breakdown ===\n")
  print(symbol_breakdown(blotter))

  invisible(blotter)
}
