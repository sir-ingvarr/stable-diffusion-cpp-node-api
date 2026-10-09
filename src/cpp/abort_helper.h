#pragma once

#include <stable-diffusion.h>

#include <atomic>
#include <exception>

namespace AbortHelper {

// Per-ctx cancellation state. Owned by a StableDiffusionContext or
// UpscalerContext (one per native sd_ctx_t / upscaler_ctx_t) and held
// alongside the ctx by in-flight workers, so two contexts can be
// cancelled independently.
//
// `mode` picks the engine-side cancel behaviour for generate ops:
// SD_CANCEL_ALL stops at the next step; SD_CANCEL_NEW_LATENTS finishes the
// current batch image, skips the remaining latents, and generate_* returns
// the completed images (partial success). Upscaler ops ignore the mode.
struct AbortState {
    std::atomic<bool> requested{false};
    std::atomic<int>  mode{SD_CANCEL_ALL};
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

inline void requestAbort(AbortState& state, sd_cancel_mode_t mode = SD_CANCEL_ALL) {
    // Mode must be visible before the flag flips; `requested` release-store
    // publishes it to the worker's acquire-load.
    state.mode.store(mode, std::memory_order_relaxed);
    state.requested.store(true, std::memory_order_release);
}

inline void clearAbort(AbortState& state) {
    state.requested.store(false, std::memory_order_release);
    state.mode.store(SD_CANCEL_ALL, std::memory_order_relaxed);
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
            sd_cancel_generation(current_sd_ctx,
                static_cast<sd_cancel_mode_t>(
                    current->mode.load(std::memory_order_relaxed)));
            return;
        }
        throw AbortException();
    }
}

}  // namespace AbortHelper
