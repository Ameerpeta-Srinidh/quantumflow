// Order book correctness tests: price-time priority, matching, cancels, the
// SIMD scanner against a scalar reference, and a randomized comparison with a
// slow but obviously-correct std::map book.
#include <map>
#include <random>
#include <thread>

#include "check.hpp"
#include "orderbook/feed_simulator.hpp"
#include "orderbook/order_book.hpp"
#include "orderbook/spsc_queue.hpp"

using namespace qf::ob;

static void test_basic() {
    OrderBook b(1000);
    CHECK(b.best_bid() == -1 && b.best_ask() == -1);
    b.add(1, Side::kBuy, 100, 10);
    b.add(2, Side::kBuy, 101, 5);
    b.add(3, Side::kSell, 105, 7);
    b.add(4, Side::kSell, 104, 3);
    CHECK(b.best_bid() == 101);
    CHECK(b.best_ask() == 104);
    CHECK_NEAR(b.mid_price(), 1.025, 1e-12);  // (101+104)/2 ticks * 0.01
    auto t = b.top();
    CHECK(t.bid_qty == 5 && t.ask_qty == 3);

    CHECK(b.cancel(2));
    CHECK(!b.cancel(2));  // already gone
    CHECK(b.best_bid() == 100);
    CHECK(b.execute(4, 3));  // fully executes -> level empties
    CHECK(b.best_ask() == 105);
    CHECK(b.execute(3, 2));  // partial
    CHECK(b.depth_at(Side::kSell, 105) == 5);
    CHECK(b.live_orders() == 2);
}

static void test_matching_priority() {
    OrderBook b(1000);
    b.add(1, Side::kSell, 200, 5);  // oldest at 200
    b.add(2, Side::kSell, 200, 5);
    b.add(3, Side::kSell, 201, 5);
    // Buy 12 up to 201: takes 5 (id 1) + 5 (id 2) at 200, then 2 of id 3 at 201.
    const int filled = b.add(10, Side::kBuy, 201, 12);
    CHECK(filled == 12);
    CHECK(b.best_ask() == 201);
    CHECK(b.depth_at(Side::kSell, 201) == 3);
    CHECK(!b.cancel(1) && !b.cancel(2));  // both fully filled
    CHECK(b.best_bid() == -1);           // nothing rested
    // A buy that crosses partially rests the remainder.
    CHECK(b.add(11, Side::kBuy, 205, 10) == 3);
    CHECK(b.best_bid() == 205 && b.best_ask() == -1);
    CHECK(b.depth_at(Side::kBuy, 205) == 7);
    CHECK(b.trades() == 4);
}

static void test_simd_scanner() {
    std::mt19937 rng(1);
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<std::int32_t> a(257, 0);
        const int k = rng() % 4;
        for (int j = 0; j < k; ++j) a[rng() % a.size()] = 1 + rng() % 9;
        const int from = rng() % a.size(), to = from + rng() % (a.size() - from + 1);
        int ref_f = -1, ref_b = -1;
        for (int i = from; i < to; ++i) if (a[i]) { ref_f = i; break; }
        for (int i = to - 1; i >= from; --i) if (a[i]) { ref_b = i; break; }
        CHECK(scan_forward_nonzero(a.data(), from, to) == ref_f);
        CHECK(scan_backward_nonzero(a.data(), from, to) == ref_b);
    }
}

// Reference book: std::map from price to total qty, plus id -> (side, price, qty).
static void test_against_reference() {
    const std::int32_t n_ticks = 1 << 14;
    FeedConfig cfg;
    cfg.n_messages = 200000;
    cfg.center_tick = n_ticks / 2;
    const auto feed = generate_feed(cfg, n_ticks);
    OrderBook b(n_ticks);

    struct RefOrder { Side side; int px; int qty; std::uint64_t seq; };
    std::map<std::uint64_t, RefOrder> orders;
    std::map<int, std::map<std::uint64_t, std::uint64_t>> bid_q, ask_q;  // px -> (seq -> id)
    std::uint64_t seq = 0;
    int mismatches = 0;

    auto ref_remove = [&](std::uint64_t id) {
        auto& o = orders[id];
        auto& q = (o.side == Side::kBuy ? bid_q : ask_q)[o.px];
        q.erase(o.seq);
        if (q.empty()) (o.side == Side::kBuy ? bid_q : ask_q).erase(o.px);
        orders.erase(id);
    };

    for (const auto& m : feed) {
        b.apply(m);
        if (m.type == MsgType::kAdd) {
            int rem = m.qty;
            if (m.side == Side::kBuy) {
                while (rem > 0 && !ask_q.empty() && ask_q.begin()->first <= m.price) {
                    const std::uint64_t id = ask_q.begin()->second.begin()->second;
                    auto& o = orders[id];
                    const int f = std::min(rem, o.qty);
                    rem -= f; o.qty -= f;
                    if (o.qty == 0) ref_remove(id);
                }
            } else {
                while (rem > 0 && !bid_q.empty() && bid_q.rbegin()->first >= m.price) {
                    const std::uint64_t id = bid_q.rbegin()->second.begin()->second;
                    auto& o = orders[id];
                    const int f = std::min(rem, o.qty);
                    rem -= f; o.qty -= f;
                    if (o.qty == 0) ref_remove(id);
                }
            }
            if (rem > 0) {
                orders[m.id] = {m.side, m.price, rem, seq};
                (m.side == Side::kBuy ? bid_q : ask_q)[m.price][seq] = m.id;
                ++seq;
            }
        } else if (orders.count(m.id)) {
            auto& o = orders[m.id];
            if (m.type == MsgType::kCancel || m.qty >= o.qty) ref_remove(m.id);
            else o.qty -= m.qty;
        }
        const int rb = bid_q.empty() ? -1 : bid_q.rbegin()->first;
        const int ra = ask_q.empty() ? -1 : ask_q.begin()->first;
        if (rb != b.best_bid() || ra != b.best_ask()) ++mismatches;
    }
    CHECK(mismatches == 0);
    CHECK(orders.size() == b.live_orders());
}

static void test_spsc() {
    SpscQueue<std::uint64_t> q(1024);
    const std::uint64_t N = 2'000'000;
    std::thread producer([&] {
        for (std::uint64_t i = 1; i <= N; ++i)
            while (!q.try_push(i)) {}
    });
    std::uint64_t expected = 1, v = 0;
    bool in_order = true;
    while (expected <= N) {
        if (q.try_pop(v)) {
            if (v != expected) in_order = false;
            ++expected;
        }
    }
    producer.join();
    CHECK(in_order);
}

static void test_concurrent_readers() {
    // Readers must always see a consistent, non-crossed top of book while the
    // writer is busy.
    const std::int32_t n_ticks = 1 << 14;
    FeedConfig cfg;
    cfg.n_messages = 500000;
    cfg.center_tick = n_ticks / 2;
    const auto feed = generate_feed(cfg, n_ticks);
    OrderBook b(n_ticks);
    std::atomic<bool> done{false};
    std::atomic<long> bad{0}, reads{0};
    std::thread reader([&] {
        while (!done.load(std::memory_order_relaxed)) {
            const auto t = b.top();
            if (t.bid_px >= 0 && t.ask_px >= 0 && t.bid_px >= t.ask_px) bad++;
            if ((t.bid_px >= 0) != (t.bid_qty > 0)) bad++;
            reads++;
        }
    });
    for (const auto& m : feed) b.apply(m);
    done = true;
    reader.join();
    CHECK(bad.load() == 0);
    CHECK(reads.load() > 0);
}

int main() {
    test_basic();
    test_matching_priority();
    test_simd_scanner();
    test_against_reference();
    test_spsc();
    test_concurrent_readers();
    return report("test_orderbook");
}
