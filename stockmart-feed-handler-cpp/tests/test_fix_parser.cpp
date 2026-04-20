#include "fix_message.h"
#include <cassert>
#include <iostream>
#include <string>

FIXMessage parse_fix(const std::string& raw);
NormalizedTick normalize(const FIXMessage& msg);

void test_parse_symbol() {
    auto msg = parse_fix("35=W|55=AAPL|270=178.50|132=178.45|133=178.55|");
    assert(msg.get(55) == "AAPL");
    assert(msg.get(270) == "178.50");
    std::cout << "PASS: test_parse_symbol\n";
}

void test_normalize_tick() {
    auto msg  = parse_fix("55=TSLA|270=172.80|132=172.70|133=172.90|1020=3000|");
    auto tick = normalize(msg);
    assert(tick.symbol     == "TSLA");
    assert(tick.last_price == 172.80);
    assert(tick.bid        == 172.70);
    assert(tick.volume     == 3000.0);
    std::cout << "PASS: test_normalize_tick\n";
}

void test_missing_field() {
    auto msg = parse_fix("55=NVDA|");
    auto tick = normalize(msg);
    assert(tick.last_price == 0.0);
    std::cout << "PASS: test_missing_field\n";
}

int main() {
    test_parse_symbol();
    test_normalize_tick();
    test_missing_field();
    std::cout << "All feed handler tests passed.\n";
    return 0;
}
