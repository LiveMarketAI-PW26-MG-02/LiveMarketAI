#include "feed_dispatcher.h"
#include <iostream>

void FeedDispatcher::subscribe(const std::string& symbol, TickCallback cb) {
    subs_[symbol].push_back(std::move(cb));
}

void FeedDispatcher::unsubscribe(const std::string& symbol) {
    subs_.erase(symbol);
}

void FeedDispatcher::dispatch(const NormalizedTick& tick) {
    // Dispatch to symbol subscribers + wildcard "*"
    for (auto* key : {&tick.symbol, (const std::string*)nullptr}) {
        std::string k = key ? *key : "*";
        auto it = subs_.find(k);
        if (it != subs_.end())
            for (auto& cb : it->second) cb(tick);
    }
}

size_t FeedDispatcher::subscriber_count() const {
    size_t n = 0;
    for (auto& [k, v] : subs_) n += v.size();
    return n;
}
