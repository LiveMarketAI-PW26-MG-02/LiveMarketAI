#pragma once
#include <string>
#include <unordered_map>
#include <chrono>

// Minimal FIX-like tag=value message representation
struct FIXMessage {
    std::unordered_map<int, std::string> fields;

    const std::string& get(int tag, const std::string& def = "") const {
        auto it = fields.find(tag);
        return it != fields.end() ? it->second : def;
    }

    // FIX tag constants (subset)
    static constexpr int TAG_MSG_TYPE  = 35;
    static constexpr int TAG_SYMBOL    = 55;
    static constexpr int TAG_PRICE     = 270;
    static constexpr int TAG_SIZE      = 271;
    static constexpr int TAG_BID       = 132;
    static constexpr int TAG_ASK       = 133;
    static constexpr int TAG_TRADE_VOL = 1020;
    static constexpr int TAG_SENDER    = 49;
};

struct NormalizedTick {
    std::string symbol;
    double last_price = 0.0;
    double bid        = 0.0;
    double ask        = 0.0;
    double volume     = 0.0;
    std::chrono::system_clock::time_point timestamp;
};
