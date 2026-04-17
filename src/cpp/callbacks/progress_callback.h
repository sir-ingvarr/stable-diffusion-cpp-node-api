#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include "../abort_helper.h"

namespace ProgressCallback {

inline Napi::ThreadSafeFunction tsfn;

struct ProgressData {
    int step;
    int steps;
    float time;
};

inline void CCallback(int step, int steps, float time, void* /*data*/) {
    // Cancellation point: throws AbortException if abort was requested
    // and the current worker has enabled throw_on_abort. This unwinds
    // through stable-diffusion.cpp back to the worker's try/catch.
    AbortHelper::throwIfAborted();

    if (!tsfn) return;

    auto* pd = new ProgressData{step, steps, time};

    tsfn.NonBlockingCall(pd, [](Napi::Env env, Napi::Function jsCallback, ProgressData* data) {
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("step", Napi::Number::New(env, data->step));
        obj.Set("steps", Napi::Number::New(env, data->steps));
        obj.Set("time", Napi::Number::New(env, data->time));
        delete data;
        jsCallback.Call({obj});
    });
}

// Register the C callback once at module init so the abort check is
// active even when no JS progress callback is set.
inline void Init() {
    sd_set_progress_callback(ProgressCallback::CCallback, nullptr);
}

inline Napi::Value Set(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (tsfn) {
        tsfn.Release();
        tsfn = Napi::ThreadSafeFunction();
        // Leave CCallback registered so the abort check stays active.
    }

    if (info.Length() < 1 || info[0].IsNull() || info[0].IsUndefined()) {
        return env.Undefined();
    }

    if (!info[0].IsFunction()) {
        Napi::TypeError::New(env, "Expected function or null").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    tsfn = Napi::ThreadSafeFunction::New(
        env,
        info[0].As<Napi::Function>(),
        "sd_progress_callback",
        0, 1);

    return env.Undefined();
}

}  // namespace ProgressCallback
