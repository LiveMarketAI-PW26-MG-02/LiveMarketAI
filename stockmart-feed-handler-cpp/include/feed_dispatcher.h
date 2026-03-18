#pragma once
#include "fix_message.h"
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>

using TickCallback = std::function<void(const NormalizedTick&)>;

class FeedDispatcher {
public:
    void subscribe(const std::string& symbol, TickCallback cb);
    void unsubscribe(const std::string& symbol);
    void dispatch(const NormalizedTick& tick);
    size_t subscriber_count() const;

private:
    std::unordered_map<std::string, std::vector<TickCallback>> subs_;
};
