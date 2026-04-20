#pragma once
#include <string>
#include <chrono>

enum class PositionSide { LONG, SHORT, FLAT };

struct Position {
    std::string account_id;
    std::string symbol;
    double quantity    = 0.0;    // positive=long, negative=short
    double avg_cost    = 0.0;
    double market_price = 0.0;

    PositionSide side() const {
        if (quantity > 0) return PositionSide::LONG;
        if (quantity < 0) return PositionSide::SHORT;
        return PositionSide::FLAT;
    }

    double market_value() const { return quantity * market_price; }
    double cost_basis()   const { return quantity * avg_cost; }
    double unrealized_pnl() const { return market_value() - cost_basis(); }
    double pnl_pct() const {
        if (avg_cost == 0.0) return 0.0;
        return (market_price - avg_cost) / avg_cost * 100.0;
    }
};

struct Fill {
    std::string symbol;
    double quantity;     // positive=buy, negative=sell
    double price;
    std::chrono::system_clock::time_point time;
};
