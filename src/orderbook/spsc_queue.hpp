// QuantumFlow - bounded lock-free single-producer / single-consumer ring.
//
// The market-data thread (producer) pushes messages; the book thread
// (consumer) pops them. With exactly one writer per index, no CAS is needed:
// each side only *stores* its own index and *loads* the other's, using
// release/acquire ordering so the message contents are visible before the
// index that publishes them. This is the classic Lamport ring as refined in
// Dmitry Vyukov's 1024cores articles, plus "cached" copies of the other
// side's index so the common case touches only the local cache line.
#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <vector>

namespace qf::ob {

inline constexpr std::size_t kCacheLine = 64;

template <class T>
class SpscQueue {
public:
    explicit SpscQueue(std::size_t capacity_pow2) : mask_(capacity_pow2 - 1), buf_(capacity_pow2) {
        // capacity must be a power of two so "index & mask" replaces modulo.
    }

    bool try_push(const T& v) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head - tail_cache_ > mask_) {  // looks full: refresh our view of the consumer
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head - tail_cache_ > mask_) return false;
        }
        buf_[head & mask_] = v;
        head_.store(head + 1, std::memory_order_release);  // publish
        return true;
    }

    bool try_pop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_cache_) {  // looks empty: refresh our view of the producer
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail == head_cache_) return false;
        }
        out = buf_[tail & mask_];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

private:
    const std::size_t mask_;
    std::vector<T> buf_;
    // Producer-owned and consumer-owned data on separate cache lines so the two
    // threads never fight over the same line ("false sharing").
    alignas(kCacheLine) std::atomic<std::size_t> head_{0};
    std::size_t tail_cache_ = 0;
    alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
    std::size_t head_cache_ = 0;
};

}  // namespace qf::ob
