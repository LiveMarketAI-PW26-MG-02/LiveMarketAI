#include "position_ledger.h"
#include <cassert>
#include <iostream>
#include <chrono>
#include <cmath>

auto now = std::chrono::system_clock::now();

void test_long_position() {
    PositionLedger L("A1");
    L.apply_fill({"AAPL", 100, 175.0, now});
    const auto* p = L.get("AAPL");
    assert(p != nullptr);
    assert(p->quantity == 100.0);
    assert(p->avg_cost == 175.0);
    std::cout << "PASS: test_long_position\n";
}

void test_avg_cost() {
    PositionLedger L("A2");
    L.apply_fill({"AAPL", 100, 100.0, now});
    L.apply_fill({"AAPL", 100, 200.0, now});
    const auto* p = L.get("AAPL");
    assert(p->avg_cost == 150.0);
    std::cout << "PASS: test_avg_cost\n";
}

void test_flat_after_sell() {
    PositionLedger L("A3");
    L.apply_fill({"TSLA", 50, 170.0, now});
    L.apply_fill({"TSLA", -50, 175.0, now});
    assert(L.get("TSLA") == nullptr);
    std::cout << "PASS: test_flat_after_sell\n";
}

void test_unrealized_pnl() {
    PositionLedger L("A4");
    L.apply_fill({"NVDA", 10, 800.0, now});
    L.update_price("NVDA", 900.0);
    const auto* p = L.get("NVDA");
    assert(std::abs(p->unrealized_pnl() - 1000.0) < 0.01);
    std::cout << "PASS: test_unrealized_pnl\n";
}

int main() {
    test_long_position();
    test_avg_cost();
    test_flat_after_sell();
    test_unrealized_pnl();
    std::cout << "All position tracker tests passed.\n";
    return 0;
}
