#include "position_ledger.h"
#include <numeric>
#include <cmath>

PositionLedger::PositionLedger(const std::string& id) : account_id_(id) {}

void PositionLedger::apply_fill(const Fill& fill) {
    auto& pos = positions_[fill.symbol];
    pos.account_id = account_id_;
    pos.symbol = fill.symbol;

    double old_qty = pos.quantity;
    double new_qty = old_qty + fill.quantity;

    // Weighted average cost (for buys; for sells we just reduce qty)
    if (fill.quantity > 0) {
        if (old_qty >= 0) {
            // Adding to long or opening from flat
            double total_cost = old_qty * pos.avg_cost + fill.quantity * fill.price;
            pos.avg_cost = total_cost / (old_qty + fill.quantity);
        } else {
            // Covering a short
            pos.avg_cost = fill.price;
        }
    }
    pos.quantity = new_qty;
    pos.market_price = fill.price;

    // Remove flat positions
    if (std::abs(pos.quantity) < 1e-9)
        positions_.erase(fill.symbol);
}

void PositionLedger::update_price(const std::string& symbol, double price) {
    auto it = positions_.find(symbol);
    if (it != positions_.end())
        it->second.market_price = price;
}

const Position* PositionLedger::get(const std::string& symbol) const {
    auto it = positions_.find(symbol);
    return it != positions_.end() ? &it->second : nullptr;
}

std::vector<Position> PositionLedger::all_positions() const {
    std::vector<Position> result;
    result.reserve(positions_.size());
    for (auto& [k, v] : positions_) result.push_back(v);
    return result;
}

double PositionLedger::total_market_value() const {
    double sum = 0;
    for (auto& [k, p] : positions_) sum += p.market_value();
    return sum;
}

double PositionLedger::total_unrealized_pnl() const {
    double sum = 0;
    for (auto& [k, p] : positions_) sum += p.unrealized_pnl();
    return sum;
}

double PositionLedger::gross_exposure() const {
    double sum = 0;
    for (auto& [k, p] : positions_) sum += std::abs(p.market_value());
    return sum;
}
