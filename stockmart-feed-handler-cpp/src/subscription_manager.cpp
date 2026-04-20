#include "feed_dispatcher.h"
#include <iostream>
#include <set>

class SubscriptionManager {
public:
    void add(const std::string& userId, const std::string& symbol) {
        subs_[userId].insert(symbol);
        std::cout << "[SUB] " << userId << " -> " << symbol << "\n";
    }

    void remove(const std::string& userId, const std::string& symbol) {
        subs_[userId].erase(symbol);
    }

    bool is_subscribed(const std::string& userId, const std::string& symbol) const {
        auto it = subs_.find(userId);
        return it != subs_.end() && it->second.count(symbol) > 0;
    }

    std::set<std::string> get_symbols(const std::string& userId) const {
        auto it = subs_.find(userId);
        return it != subs_.end() ? it->second : std::set<std::string>{};
    }

private:
    std::unordered_map<std::string, std::set<std::string>> subs_;
};
