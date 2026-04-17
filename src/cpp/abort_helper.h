#pragma once

#include <atomic>
#include <exception>

namespace AbortHelper {

// Global flag. Set from JS thread via native.abort(); checked from worker
// threads inside the progress callback. Since stable-diffusion.cpp
// processes one generation at a time per sd_ctx, a single global flag
// is sufficient — if two workers run concurrently both will abort.
inline std::atomic<bool> abort_requested{false};

// Enabled per-thread for operations where mid-run cancellation is safe
// (generate_image, generate_video, upscale). Left disabled during
// context creation since the model-loading path has not been audited
// for exception safety. When disabled, an abort request is silently
// ignored at the native layer (the JS Promise still rejects via
// withAbortSignal — soft-cancel fallback for that operation).
inline thread_local bool throw_on_abort{false};

class AbortException : public std::exception {
  public:
    const char* what() const noexcept override { return "Operation aborted"; }
};

inline void requestAbort() {
    abort_requested.store(true, std::memory_order_release);
}

inline void clearAbort() {
    abort_requested.store(false, std::memory_order_release);
}

// RAII guard that enables abort-throwing on the current thread for the
// scope of one worker operation, and clears the global flag on entry.
class Scope {
  public:
    Scope() {
        clearAbort();
        throw_on_abort = true;
    }
    ~Scope() {
        throw_on_abort = false;
        clearAbort();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// Called from the C progress callback on the worker thread. If abort
// was requested and we're inside a Scope, throws AbortException which
// unwinds through stable-diffusion.cpp (running destructors) back to
// the worker's try/catch in Execute().
inline void throwIfAborted() {
    if (throw_on_abort && abort_requested.load(std::memory_order_acquire)) {
        throw AbortException();
    }
}

}  // namespace AbortHelper
