// QuantumFlow - fixed-size object pool on top of std::pmr.
//
// Calling new/delete for every order would hit the general-purpose heap
// allocator on the hot path: unpredictable latency, fragmentation, and
// possibly a system call. Instead we carve order nodes out of large chunks
// obtained from a std::pmr::monotonic_buffer_resource (a "bump pointer"
// allocator: allocating is just advancing a pointer) and recycle freed nodes
// through an intrusive free list. After warm-up, allocate() and release() are
// a couple of pointer moves and never touch the system heap.
//
// Single-threaded by design: only the book's writer thread allocates orders.
#pragma once

#include <cstddef>
#include <memory_resource>
#include <new>
#include <utility>

namespace qf::ob {

template <class T>
class ObjectPool {
    union Slot {
        Slot* next;
        alignas(T) unsigned char storage[sizeof(T)];
    };

public:
    explicit ObjectPool(std::size_t initial_capacity = 1 << 16)
        : upstream_(initial_capacity * sizeof(Slot)), chunk_(initial_capacity) {}

    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    template <class... Args>
    T* allocate(Args&&... args) {
        if (!free_) refill();
        Slot* s = free_;
        free_ = s->next;
        ++live_;
        return new (s->storage) T(std::forward<Args>(args)...);
    }

    void release(T* obj) {
        obj->~T();
        Slot* s = reinterpret_cast<Slot*>(obj);
        s->next = free_;
        free_ = s;
        --live_;
    }

    std::size_t live() const { return live_; }

private:
    void refill() {
        void* mem = upstream_.allocate(chunk_ * sizeof(Slot), alignof(Slot));
        Slot* slots = static_cast<Slot*>(mem);
        for (std::size_t i = 0; i < chunk_; ++i) {
            slots[i].next = free_;
            free_ = &slots[i];
        }
    }

    std::pmr::monotonic_buffer_resource upstream_;
    std::size_t chunk_;
    Slot* free_ = nullptr;
    std::size_t live_ = 0;
};

}  // namespace qf::ob
