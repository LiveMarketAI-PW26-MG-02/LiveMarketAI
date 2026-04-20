#include "position_ledger.h"
#include <iostream>
#include <chrono>

void print_pnl_report(const std::vector<Position>& positions);

int main() {
    PositionLedger ledger("ACC-001");
    auto now = std::chrono::system_clock::now();

    // Buy 100 AAPL at 175
    ledger.apply_fill({"AAPL", 100, 175.0, now});
    // Buy 50 more AAPL at 180 (avg cost rises)
    ledger.apply_fill({"AAPL",  50, 180.0, now});
    // Buy 200 TSLA at 165
    ledger.apply_fill({"TSLA", 200, 165.0, now});
    // Sell 50 TSLA at 170
    ledger.apply_fill({"TSLA", -50, 170.0, now});
    // Short 30 MSFT at 425
    ledger.apply_fill({"MSFT", -30, 425.0, now});

    // Mark to market
    ledger.update_price("AAPL", 182.0);
    ledger.update_price("TSLA", 172.0);
    ledger.update_price("MSFT", 418.0);

    print_pnl_report(ledger.all_positions());

    std::cout << "\nTotal Mkt Value:   $" << ledger.total_market_value()   << "\n";
    std::cout << "Total Unreal P&L:  $" << ledger.total_unrealized_pnl() << "\n";
    std::cout << "Gross Exposure:    $" << ledger.gross_exposure()        << "\n";
    return 0;
}
