// QuantumFlow - synthetic market-data feed.
//
// Real exchange feeds (e.g. Nasdaq ITCH) are streams of add / cancel /
// execute messages. We generate a statistically similar stream so the order
// book can be benchmarked without buying market data:
//  * a "fair price" does a random walk
//  * most adds are passive, a few ticks away from the fair price
//    (distance roughly geometric, as in real books)
//  * a few adds are aggressive and cross the spread (they trade)
//  * cancels and executions hit random previously added orders
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "order_book.hpp"

namespace qf::ob {

struct FeedConfig {
    std::uint64_t n_messages = 1'000'000;
    std::int32_t center_tick = 1 << 13;  // starting fair price, in ticks
    double p_add = 0.50, p_cancel = 0.42;  // remainder = execute
    double p_aggressive = 0.04;            // share of adds that cross the spread
    std::uint64_t seed = 7;
};

inline std::vector<Message> generate_feed(const FeedConfig& cfg, std::int32_t n_ticks = 1 << 14) {
    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::geometric_distribution<int> dist_away(0.25);  // mean ~3 ticks from fair
    std::uniform_int_distribution<int> lot(1, 10);
    std::vector<Message> out;
    out.reserve(cfg.n_messages);
    std::vector<std::uint64_t> live;
    live.reserve(1 << 16);
    std::uint64_t next_id = 1;
    double fair = cfg.center_tick;

    for (std::uint64_t k = 0; k < cfg.n_messages; ++k) {
        fair += (U(rng) - 0.5) * 0.2;  // slow random walk
        const double u = U(rng);
        if (u < cfg.p_add || live.empty()) {
            const Side side = U(rng) < 0.5 ? Side::kBuy : Side::kSell;
            const int away = 1 + dist_away(rng);
            const int f = static_cast<int>(fair);
            std::int32_t px;
            if (U(rng) < cfg.p_aggressive)
                px = side == Side::kBuy ? f + away : f - away;  // crosses the spread
            else
                px = side == Side::kBuy ? f - away : f + away;  // rests in the book
            if (px < 0) px = 0;
            if (px >= n_ticks) px = n_ticks - 1;
            const std::uint64_t id = next_id++;
            out.push_back({MsgType::kAdd, side, px, 100 * lot(rng), id});
            live.push_back(id);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, live.size() - 1);
            const std::size_t j = pick(rng);
            const std::uint64_t id = live[j];
            live[j] = live.back();
            live.pop_back();
            if (u < cfg.p_add + cfg.p_cancel)
                out.push_back({MsgType::kCancel, Side::kBuy, 0, 0, id});
            else
                out.push_back({MsgType::kExecute, Side::kBuy, 0, 100 * lot(rng), id});
        }
    }
    return out;
}

}  // namespace qf::ob
