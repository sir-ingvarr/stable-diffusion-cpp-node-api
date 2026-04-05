#pragma once

#include <atomic>
#include <csetjmp>

namespace AbortHelper {

// Atomic flag set from JS main thread, checked from worker thread
inline std::atomic<bool> abort_requested{false};

// Per-thread jmp_buf for longjmp-based cancellation.
// Only valid when jmp_active is true on that thread.
inline thread_local std::jmp_buf jmp_buf_storage;
inline thread_local bool jmp_active{false};

inline void requestAbort() {
    abort_requested.store(true, std::memory_order_release);
}

inline void clearAbort() {
    abort_requested.store(false, std::memory_order_release);
}

// Called from progress/preview callbacks on the worker thread.
// If abort was requested and a jmp_buf is active, longjmp back to Execute().
inline void checkAbort() {
    if (jmp_active && abort_requested.load(std::memory_order_acquire)) {
        jmp_active = false;
        std::longjmp(jmp_buf_storage, 1);
    }
}

}  // namespace AbortHelper
