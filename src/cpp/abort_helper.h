#pragma once

#include <atomic>
#include <exception>

namespace AbortHelper {

// Per-ctx cancellation state. Owned by a StableDiffusionContext or
// UpscalerContext (one per native sd_ctx_t / upscaler_ctx_t) and held
// alongside the ctx by in-flight workers, so two contexts can be
// cancelled independently.
struct AbortState {
    std::atomic<bool> requested{false};
};

// Pointer to the AbortState the worker currently running on this thread
// is watching. Set by Scope ctor, cleared by Scope dtor.
inline thread_local AbortState* current = nullptr;

// Whether the current thread is inside an abortable operation. Disabled
// during context creation since model loading has not been audited for
// exception safety.
inline thread_local bool throw_on_abort{false};

class AbortException : public std::exception {
  public:
    const char* what() const noexcept override { return "Operation aborted"; }
};

inline void requestAbort(AbortState& state) {
    state.requested.store(true, std::memory_order_release);
}

inline void clearAbort(AbortState& state) {
    state.requested.store(false, std::memory_order_release);
}

// RAII guard that points the worker thread at a specific ctx's
// AbortState for the duration of one operation. Note we deliberately
// do NOT clear `state.requested` on entry — the JS thread is
// responsible for clearing it before queueing the worker, since
// clearing on the worker thread races with abort() calls landing
// between Queue() and Execute().
class Scope {
  public:
    explicit Scope(AbortState& state) {
        current = &state;
        throw_on_abort = true;
    }
    ~Scope() {
        throw_on_abort = false;
        if (current) {
            // Clear so a leftover abort from the run that just ended
            // doesn't trip the next op's first checkpoint. The JS
            // thread also clears before queueing the next op, so this
            // is belt-and-suspenders.
            clearAbort(*current);
            current = nullptr;
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// Called from the C progress callback on the worker thread. If abort
// was requested on this thread's current ctx, throws AbortException
// which unwinds through stable-diffusion.cpp back to the worker's
// try/catch in Execute().
inline void throwIfAborted() {
    if (throw_on_abort && current &&
        current->requested.load(std::memory_order_acquire)) {
        throw AbortException();
    }
}

}  // namespace AbortHelper
