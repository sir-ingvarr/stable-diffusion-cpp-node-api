#pragma once

#include <stable-diffusion.h>

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

// The sd_ctx_t the current worker is generating on, when the operation has
// a native cancel API (generate_image / generate_video). When set, an abort
// is promoted to sd_cancel_generation() instead of throwing — upstream's
// pipeline is no longer exception-safe under our progress-callback throw
// (std::terminate observed mid-generation), but it polls its own cancel
// flag densely and returns false from generate_*.
inline thread_local sd_ctx_t* current_sd_ctx = nullptr;

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
    explicit Scope(AbortState& state, sd_ctx_t* ctx = nullptr) {
        current = &state;
        current_sd_ctx = ctx;
        throw_on_abort = true;
    }
    ~Scope() {
        throw_on_abort = false;
        current_sd_ctx = nullptr;
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

// Called from the C progress callback on the worker thread. If abort was
// requested on this thread's current ctx:
//   - generate ops (current_sd_ctx set): promote the abort to the engine's
//     native cancel flag. The pipeline polls it at every step/segment
//     boundary and generate_* returns false; the worker maps that back to
//     an "Aborted" rejection.
//   - upscaler ops (no native cancel API): legacy behaviour — throw
//     AbortException, unwinding back to the worker's try/catch.
inline void throwIfAborted() {
    if (throw_on_abort && current &&
        current->requested.load(std::memory_order_acquire)) {
        if (current_sd_ctx) {
            sd_cancel_generation(current_sd_ctx, SD_CANCEL_ALL);
            return;
        }
        throw AbortException();
    }
}

}  // namespace AbortHelper
