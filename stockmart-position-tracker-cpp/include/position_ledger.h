#pragma once
#include "position.h"
#include <unordered_map>
#include <vector>
#include <string>

class PositionLedger {
public:
    explicit PositionLedger(const std::string& account_id);

    void apply_fill(const Fill& fill);
    void update_price(const std::string& symbol, double price);

    const Position* get(const std::string& symbol) const;
    std::vector<Position> all_positions() const;
    double total_market_value() const;
    double total_unrealized_pnl() const;
    double gross_exposure() const;

private:
    std::string account_id_;
    std::unordered_map<std::string, Position> positions_;
};
