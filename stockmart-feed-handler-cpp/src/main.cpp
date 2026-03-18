#include "fix_message.h"
#include "feed_dispatcher.h"
#include <iostream>
#include <vector>
#include <string>

// Forward declarations
FIXMessage parse_fix(const std::string& raw);
NormalizedTick normalize(const FIXMessage& msg);

int main() {
    FeedDispatcher dispatcher;
    int tick_count = 0;

    // Subscribe to AAPL and TSLA
    dispatcher.subscribe("AAPL", [&](const NormalizedTick& t) {
        std::cout << "[AAPL] px=" << t.last_price
                  << " bid=" << t.bid << " ask=" << t.ask << "\n";
        tick_count++;
    });
    dispatcher.subscribe("TSLA", [&](const NormalizedTick& t) {
        std::cout << "[TSLA] px=" << t.last_price << " vol=" << t.volume << "\n";
        tick_count++;
    });

    // Simulate incoming FIX messages
    std::vector<std::string> feed = {
        "35=W|55=AAPL|270=178.50|132=178.45|133=178.55|",
        "35=W|55=TSLA|270=172.80|1020=5000|",
        "35=W|55=AAPL|270=179.10|132=179.05|133=179.15|",
        "35=W|55=MSFT|270=420.50|",   // no subscriber, silent
    };

    for (auto& raw : feed) {
        auto msg  = parse_fix(raw);
        auto tick = normalize(msg);
        dispatcher.dispatch(tick);
    }

    std::cout << "\nTotal ticks delivered: " << tick_count << "\n";
    std::cout << "Subscribers active:    " << dispatcher.subscriber_count() << "\n";
    return 0;
}
