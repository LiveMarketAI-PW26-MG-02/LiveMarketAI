# Trade confirmation slip generator
source("R/report_engine.R")

#' Generate a trade confirmation (like ecommerce order receipt)
trade_confirmation <- function(trade_id, blotter = generate_blotter()) {
  trade <- blotter[blotter$trade_id == trade_id, ]
  if (nrow(trade) == 0) stop("Trade not found: ", trade_id)

  t <- trade[1, ]
  cat("╔══════════════════════════════════════╗\n")
  cat("║      STOCKMART TRADE CONFIRMATION     ║\n")
  cat("╠══════════════════════════════════════╣\n")
  cat(sprintf("║ Trade ID:   %-25s ║\n", t$trade_id))
  cat(sprintf("║ Date:       %-25s ║\n", as.character(t$date)))
  cat(sprintf("║ Symbol:     %-25s ║\n", t$symbol))
  cat(sprintf("║ Side:       %-25s ║\n", t$side))
  cat(sprintf("║ Quantity:   %-25s ║\n", t$quantity))
  cat(sprintf("║ Price:      $%-24.2f ║\n", t$price))
  cat(sprintf("║ Commission: $%-24.2f ║\n", t$commission))
  cat(sprintf("║ Net Amount: $%-24.2f ║\n", t$net_amount))
  cat("╚══════════════════════════════════════╝\n")
  invisible(t)
}
