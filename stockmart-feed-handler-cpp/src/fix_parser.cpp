#include "fix_message.h"
#include <sstream>
#include <stdexcept>

// Parse FIX tag=value|tag=value|... string
FIXMessage parse_fix(const std::string& raw) {
    FIXMessage msg;
    std::istringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, '|')) {
        if (token.empty()) continue;
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        try {
            int tag = std::stoi(token.substr(0, eq));
            msg.fields[tag] = token.substr(eq + 1);
        } catch (...) {}
    }
    return msg;
}

// Convert parsed FIX message to normalized tick
NormalizedTick normalize(const FIXMessage& msg) {
    NormalizedTick tick;
    tick.symbol     = msg.get(FIXMessage::TAG_SYMBOL);
    tick.last_price = msg.get(FIXMessage::TAG_PRICE).empty() ? 0.0 : std::stod(msg.get(FIXMessage::TAG_PRICE));
    tick.bid        = msg.get(FIXMessage::TAG_BID).empty()   ? 0.0 : std::stod(msg.get(FIXMessage::TAG_BID));
    tick.ask        = msg.get(FIXMessage::TAG_ASK).empty()   ? 0.0 : std::stod(msg.get(FIXMessage::TAG_ASK));
    tick.volume     = msg.get(FIXMessage::TAG_TRADE_VOL).empty() ? 0.0 : std::stod(msg.get(FIXMessage::TAG_TRADE_VOL));
    tick.timestamp  = std::chrono::system_clock::now();
    return tick;
}
