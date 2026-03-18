#include "position.h"
#include <iostream>
#include <vector>
#include <iomanip>

void print_pnl_report(const std::vector<Position>& positions) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Position P&L Report ===\n";
    std::cout << std::left
              << std::setw(8)  << "Symbol"
              << std::setw(10) << "Side"
              << std::setw(10) << "Qty"
              << std::setw(12) << "Avg Cost"
              << std::setw(12) << "Mkt Price"
              << std::setw(14) << "Mkt Value"
              << std::setw(12) << "Unreal P&L"
              << std::setw(8)  << "P&L%"
              << "\n" << std::string(86, '-') << "\n";

    for (auto& p : positions) {
        std::string side_str = p.side() == PositionSide::LONG ? "LONG" :
                               p.side() == PositionSide::SHORT ? "SHORT" : "FLAT";
        std::cout << std::setw(8)  << p.symbol
                  << std::setw(10) << side_str
                  << std::setw(10) << p.quantity
                  << std::setw(12) << p.avg_cost
                  << std::setw(12) << p.market_price
                  << std::setw(14) << p.market_value()
                  << std::setw(12) << p.unrealized_pnl()
                  << std::setw(8)  << p.pnl_pct() << "%\n";
    }
}
