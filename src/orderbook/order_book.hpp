// QuantumFlow - low-latency limit order book.
//
// Architecture: ONE writer thread owns the book and applies every message
// (add / cancel / execute) in order; any number of reader threads can watch
// the top of the book at the same time without locks. This "single writer"
// rule is how real matching engines are built: it removes contention on the
// hot path entirely, so no mutex and no compare-and-swap retry loops are
// needed for updates. Readers see a consistent top of book through a seqlock.
//
// Memory layout:
//  * prices are integer ticks; each side is a flat array of Level structs
//    indexed by tick (O(1) lookup, no tree, no hashing)
//  * every Level is alignas(64): one cache line each, so updating one level
//    never invalidates a neighbour's line in another core's cache
//  * a separate dense int32 quantity array per side is what the AVX2 scanner
//    walks to find the next best price
//  * order nodes come from an ObjectPool (no heap allocation after warm-up)
#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "memory_pool.hpp"
#include "simd_scanner.hpp"
#include "spsc_queue.hpp"

namespace qf::ob {

enum class Side : std::uint8_t { kBuy = 0, kSell = 1 };

struct Order {
    std::uint64_t id;
    std::int32_t price;
    std::int32_t qty;
    Side side;
    Order* prev = nullptr;  // FIFO queue at this price (time priority)
    Order* next = nullptr;
};

// One price level: one cache line.
struct alignas(kCacheLine) Level {
    Order* head = nullptr;  // oldest order (first to be filled)
    Order* tail = nullptr;
    std::atomic<std::int64_t> qty{0};    // published total quantity (readable by any thread)
    std::atomic<std::int32_t> orders{0}; // published number of orders
};
static_assert(sizeof(Level) == kCacheLine, "Level must be exactly one cache line");

// Consistent snapshot of the best bid and ask.
struct TopOfBook {
    std::int32_t bid_px = -1, ask_px = -1;  // -1 = side is empty
    std::int64_t bid_qty = 0, ask_qty = 0;
    std::uint64_t seq = 0;
};

// Message format of the simulated market-data feed.
enum class MsgType : std::uint8_t { kAdd = 0, kCancel = 1, kExecute = 2 };
struct Message {
    MsgType type;
    Side side;
    std::int32_t price;
    std::int32_t qty;
    std::uint64_t id;
};

class OrderBook {
public:
    explicit OrderBook(std::int32_t n_ticks = 1 << 14, double tick_size = 0.01)
        : n_ticks_(n_ticks), tick_size_(tick_size), bids_(n_ticks), asks_(n_ticks),
          bid_qty_(n_ticks, 0), ask_qty_(n_ticks, 0), pool_(1 << 16) {
        ids_.reserve(1 << 20);
    }

    // ---------------- writer-thread API ----------------

    // Adds a limit order. If it crosses the spread it trades immediately
    // against resting orders (price-time priority); any remainder rests.
    // Returns the quantity that traded.
    std::int32_t add(std::uint64_t id, Side side, std::int32_t price, std::int32_t qty) {
        if (price < 0 || price >= n_ticks_ || qty <= 0) return 0;
        seq_begin();
        std::int32_t filled = (side == Side::kBuy) ? match_buy(price, qty) : match_sell(price, qty);
        const std::int32_t rest = qty - filled;
        if (rest > 0) {
            Order* o = pool_.allocate(Order{id, price, rest, side});
            if (id >= ids_.size()) ids_.resize(id + 1 > ids_.size() * 2 ? id + 1 : ids_.size() * 2, nullptr);
            ids_[id] = o;
            link(o);
            if (side == Side::kBuy) { if (price > best_bid_) best_bid_ = price; }
            else if (best_ask_ < 0 || price < best_ask_) best_ask_ = price;
        }
        publish_top();
        seq_end();
        return filled;
    }

    // Removes a resting order. Returns false if the id is unknown (already gone).
    bool cancel(std::uint64_t id) {
        Order* o = find(id);
        if (!o) return false;
        seq_begin();
        remove(o, o->qty);
        publish_top();
        seq_end();
        return true;
    }

    // Reduces a resting order by qty (an execution reported by the exchange feed).
    bool execute(std::uint64_t id, std::int32_t qty) {
        Order* o = find(id);
        if (!o || qty <= 0) return false;
        seq_begin();
        if (qty >= o->qty) {
            remove(o, o->qty);
        } else {
            o->qty -= qty;
            level(o->side, o->price).qty.fetch_sub(qty, std::memory_order_relaxed);
            side_qty(o->side)[o->price] -= qty;
        }
        publish_top();
        seq_end();
        return true;
    }

    // Apply one feed message. Returns traded quantity for adds, 0/1 otherwise.
    std::int32_t apply(const Message& m) {
        switch (m.type) {
            case MsgType::kAdd: return add(m.id, m.side, m.price, m.qty);
            case MsgType::kCancel: return cancel(m.id) ? 1 : 0;
            case MsgType::kExecute: return execute(m.id, m.qty) ? 1 : 0;
        }
        return 0;
    }

    // ---------------- reader API (any thread, lock-free) ----------------

    std::int32_t best_bid() const { return pub_bid_px_.load(std::memory_order_acquire); }
    std::int32_t best_ask() const { return pub_ask_px_.load(std::memory_order_acquire); }

    // Seqlock read: retry if the writer was mid-update (odd sequence number)
    // or finished an update while we were reading (sequence changed).
    TopOfBook top() const {
        TopOfBook t;
        for (;;) {
            const std::uint64_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;
            t.bid_px = pub_bid_px_.load(std::memory_order_relaxed);
            t.ask_px = pub_ask_px_.load(std::memory_order_relaxed);
            t.bid_qty = pub_bid_qty_.load(std::memory_order_relaxed);
            t.ask_qty = pub_ask_qty_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s1) {
                t.seq = s1;
                return t;
            }
        }
    }

    // Mid price in currency units, or a negative number if either side is empty.
    double mid_price() const {
        const TopOfBook t = top();
        if (t.bid_px < 0 || t.ask_px < 0) return -1.0;
        return 0.5 * (t.bid_px + t.ask_px) * tick_size_;
    }
    double spread() const {
        const TopOfBook t = top();
        if (t.bid_px < 0 || t.ask_px < 0) return -1.0;
        return (t.ask_px - t.bid_px) * tick_size_;
    }

    std::int64_t depth_at(Side side, std::int32_t price) const {
        if (price < 0 || price >= n_ticks_) return 0;
        return (side == Side::kBuy ? bids_ : asks_)[price].qty.load(std::memory_order_relaxed);
    }

    // Pre-size the id -> order table so it never grows (and copies) on the hot path.
    void reserve_ids(std::uint64_t max_id) {
        if (max_id + 1 > ids_.size()) ids_.resize(max_id + 1, nullptr);
    }

    std::size_t live_orders() const { return pool_.live(); }
    std::uint64_t trades() const { return trades_; }
    std::uint64_t traded_volume() const { return traded_volume_; }
    std::int32_t n_ticks() const { return n_ticks_; }
    double tick_size() const { return tick_size_; }

private:
    Level& level(Side s, std::int32_t px) { return s == Side::kBuy ? bids_[px] : asks_[px]; }
    std::int32_t* side_qty(Side s) { return s == Side::kBuy ? bid_qty_.data() : ask_qty_.data(); }

    Order* find(std::uint64_t id) const { return id < ids_.size() ? ids_[id] : nullptr; }

    void link(Order* o) {
        Level& L = level(o->side, o->price);
        o->prev = L.tail;
        o->next = nullptr;
        if (L.tail) L.tail->next = o; else L.head = o;
        L.tail = o;
        L.qty.fetch_add(o->qty, std::memory_order_relaxed);
        L.orders.fetch_add(1, std::memory_order_relaxed);
        side_qty(o->side)[o->price] += o->qty;
    }

    // Unlinks and frees an order whose remaining quantity is qty, then
    // repairs the best price if that level became empty.
    void remove(Order* o, std::int32_t qty) {
        Level& L = level(o->side, o->price);
        if (o->prev) o->prev->next = o->next; else L.head = o->next;
        if (o->next) o->next->prev = o->prev; else L.tail = o->prev;
        L.qty.fetch_sub(qty, std::memory_order_relaxed);
        L.orders.fetch_sub(1, std::memory_order_relaxed);
        std::int32_t* q = side_qty(o->side);
        q[o->price] -= qty;
        if (q[o->price] == 0) {
            if (o->side == Side::kBuy && o->price == best_bid_)
                best_bid_ = scan_backward_nonzero(q, 0, o->price);
            else if (o->side == Side::kSell && o->price == best_ask_)
                best_ask_ = scan_forward_nonzero(q, o->price + 1, n_ticks_);
        }
        ids_[o->id] = nullptr;
        pool_.release(o);
    }

    // Fill an incoming buy against asks priced <= limit. Returns filled qty.
    std::int32_t match_buy(std::int32_t limit, std::int32_t qty) {
        std::int32_t filled = 0;
        while (filled < qty && best_ask_ >= 0 && best_ask_ <= limit) filled += take(asks_[best_ask_], qty - filled);
        return filled;
    }
    std::int32_t match_sell(std::int32_t limit, std::int32_t qty) {
        std::int32_t filled = 0;
        while (filled < qty && best_bid_ >= 0 && best_bid_ >= limit) filled += take(bids_[best_bid_], qty - filled);
        return filled;
    }
    // Trade against the oldest order at a level (time priority).
    std::int32_t take(Level& L, std::int32_t want) {
        Order* o = L.head;
        const std::int32_t f = o->qty < want ? o->qty : want;
        ++trades_;
        traded_volume_ += static_cast<std::uint64_t>(f);
        if (f == o->qty) {
            remove(o, f);
        } else {
            o->qty -= f;
            L.qty.fetch_sub(f, std::memory_order_relaxed);
            side_qty(o->side)[o->price] -= f;
        }
        return f;
    }

    void seq_begin() {
        seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);  // odd: writing
        std::atomic_thread_fence(std::memory_order_release);
    }
    void seq_end() { seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release); }  // even: done

    void publish_top() {
        pub_bid_px_.store(best_bid_, std::memory_order_relaxed);
        pub_ask_px_.store(best_ask_, std::memory_order_relaxed);
        pub_bid_qty_.store(best_bid_ >= 0 ? bid_qty_[best_bid_] : 0, std::memory_order_relaxed);
        pub_ask_qty_.store(best_ask_ >= 0 ? ask_qty_[best_ask_] : 0, std::memory_order_relaxed);
    }

    const std::int32_t n_ticks_;
    const double tick_size_;
    std::vector<Level> bids_, asks_;
    std::vector<std::int32_t> bid_qty_, ask_qty_;  // writer-private mirror, scanned with AVX2
    ObjectPool<Order> pool_;
    std::vector<Order*> ids_;
    std::int32_t best_bid_ = -1, best_ask_ = -1;  // writer-private
    std::uint64_t trades_ = 0, traded_volume_ = 0;

    // Published state, each on its own cache line away from writer-private data.
    alignas(kCacheLine) std::atomic<std::uint64_t> seq_{0};
    std::atomic<std::int32_t> pub_bid_px_{-1}, pub_ask_px_{-1};
    std::atomic<std::int64_t> pub_bid_qty_{0}, pub_ask_qty_{0};
};

}  // namespace qf::ob
