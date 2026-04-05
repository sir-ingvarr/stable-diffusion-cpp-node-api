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
    // Check abort before doing any work. If abort was requested and a
    // jmp_buf is active on this thread, longjmp back to the worker's Execute().
    AbortHelper::checkAbort();

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

// Register our CCallback once at module init. This ensures the abort
// check is always active, even when no JS progress callback is set.
inline void Init() {
    sd_set_progress_callback(ProgressCallback::CCallback, nullptr);
}

inline Napi::Value Set(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (tsfn) {
        tsfn.Release();
        tsfn = Napi::ThreadSafeFunction();
        // Don't clear the C callback — CCallback must stay registered
        // for the abort mechanism. It handles tsfn being null gracefully.
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
