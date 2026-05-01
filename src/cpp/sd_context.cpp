#include "sd_context.h"

#include "abort_helper.h"
#include "workers/create_context_worker.h"
#include "workers/generate_image_worker.h"
#include "workers/generate_video_worker.h"

Napi::FunctionReference StableDiffusionContext::constructor_;

Napi::Object StableDiffusionContext::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "StableDiffusionContext", {
        InstanceMethod<&StableDiffusionContext::GenerateImage>("generateImage"),
        InstanceMethod<&StableDiffusionContext::GenerateVideo>("generateVideo"),
        InstanceMethod<&StableDiffusionContext::GetDefaultSampleMethod>("getDefaultSampleMethod"),
        InstanceMethod<&StableDiffusionContext::GetDefaultScheduler>("getDefaultScheduler"),
        InstanceMethod<&StableDiffusionContext::Abort>("abort"),
        InstanceMethod<&StableDiffusionContext::Close>("close"),
        InstanceAccessor<&StableDiffusionContext::IsClosed>("isClosed"),
        StaticMethod<&StableDiffusionContext::Create>("create"),
    });

    constructor_ = Napi::Persistent(func);
    constructor_.SuppressDestruct();

    exports.Set("StableDiffusionContext", func);
    return exports;
}

StableDiffusionContext::StableDiffusionContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<StableDiffusionContext>(info) {
    Napi::Env env = info.Env();

    // Internal construction from Create() — receives an External<sd_ctx_t>.
    if (info.Length() >= 1 && info[0].IsExternal()) {
        sd_ctx_t* raw = info[0].As<Napi::External<sd_ctx_t>>().Data();
        ctx_ = SdCtxPtr(raw, [](sd_ctx_t* c) {
            if (c) free_sd_ctx(c);
        });
        abort_state_ = std::make_shared<AbortHelper::AbortState>();
        return;
    }

    Napi::TypeError::New(env, "Use StableDiffusionContext.create() instead of new")
        .ThrowAsJavaScriptException();
}

Napi::Value StableDiffusionContext::Create(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();
    auto* worker = new CreateContextWorker(env, opts, constructor_);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

Napi::Value StableDiffusionContext::GenerateImage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();
    // Wipe stale abort state on the JS thread before queueing — doing it on
    // the worker thread inside Scope::Scope() races with native.abort() calls
    // landing between Queue() and Execute() and silently drops them.
    AbortHelper::clearAbort(*abort_state_);
    auto* worker = new GenerateImageWorker(env, ctx_, abort_state_, opts);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

Napi::Value StableDiffusionContext::GenerateVideo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();
    // See note in GenerateImage — clear on JS thread, not worker thread.
    AbortHelper::clearAbort(*abort_state_);
    auto* worker = new GenerateVideoWorker(env, ctx_, abort_state_, opts);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

Napi::Value StableDiffusionContext::GetDefaultSampleMethod(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    sample_method_t method = sd_get_default_sample_method(ctx_.get());
    return Napi::String::New(env, sd_sample_method_name(method));
}

Napi::Value StableDiffusionContext::GetDefaultScheduler(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    sample_method_t method = SAMPLE_METHOD_COUNT;
    if (info.Length() >= 1 && info[0].IsString()) {
        std::string name = info[0].As<Napi::String>().Utf8Value();
        method = str_to_sample_method(name.c_str());
    }

    scheduler_t scheduler = sd_get_default_scheduler(ctx_.get(), method);
    return Napi::String::New(env, sd_scheduler_name(scheduler));
}

void StableDiffusionContext::Abort(const Napi::CallbackInfo& info) {
    if (abort_state_) AbortHelper::requestAbort(*abort_state_);
}

void StableDiffusionContext::Close(const Napi::CallbackInfo& info) {
    // Signal in-flight workers to abort at the next checkpoint, then drop
    // the wrapper's ref. The worker holds its own shared_ptr to abort_state_
    // so the flag stays observable for it; the native ctx is freed when
    // the last ref (worker or wrapper) drops.
    if (abort_state_) AbortHelper::requestAbort(*abort_state_);
    ctx_.reset();
}

Napi::Value StableDiffusionContext::IsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !ctx_);
}
