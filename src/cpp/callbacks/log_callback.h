#pragma once

#include <napi.h>
#include <stable-diffusion.h>

namespace LogCallback {

// The TSFN instance — global since the C library uses a global callback.
inline Napi::ThreadSafeFunction tsfn;

inline void CCallback(sd_log_level_t level, const char* text, void* /*data*/) {
    if (!tsfn) return;

    // Copy the string since text may be freed after callback returns
    std::string* msg = new std::string(text ? text : "");
    int lvl = static_cast<int>(level);

    tsfn.NonBlockingCall(msg, [lvl](Napi::Env env, Napi::Function jsCallback, std::string* data) {
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("level", Napi::Number::New(env, lvl));
        obj.Set("text", Napi::String::New(env, *data));
        delete data;
        jsCallback.Call({obj});
    });
}

inline Napi::Value Set(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Release previous TSFN if any
    if (tsfn) {
        tsfn.Release();
        tsfn = Napi::ThreadSafeFunction();
        sd_set_log_callback(nullptr, nullptr);
    }

    // If null/undefined passed, just clear
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
        "sd_log_callback",
        0,   // unlimited queue
        1);  // one thread

    sd_set_log_callback(CCallback, nullptr);
    return env.Undefined();
}

}  // namespace LogCallback
