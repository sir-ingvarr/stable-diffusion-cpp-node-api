#include "sd_context.h"

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
    : Napi::ObjectWrap<StableDiffusionContext>(info), ctx_(nullptr) {
    Napi::Env env = info.Env();

    // Internal construction from Create() — receives an External<sd_ctx_t>
    if (info.Length() >= 1 && info[0].IsExternal()) {
        ctx_ = info[0].As<Napi::External<sd_ctx_t>>().Data();
        return;
    }

    Napi::TypeError::New(env, "Use StableDiffusionContext.create() instead of new")
        .ThrowAsJavaScriptException();
}

StableDiffusionContext::~StableDiffusionContext() {
    if (ctx_) {
        free_sd_ctx(ctx_);
        ctx_ = nullptr;
    }
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
    auto* worker = new GenerateImageWorker(env, ctx_, opts);
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
    auto* worker = new GenerateVideoWorker(env, ctx_, opts);
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
    sample_method_t method = sd_get_default_sample_method(ctx_);
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

    scheduler_t scheduler = sd_get_default_scheduler(ctx_, method);
    return Napi::String::New(env, sd_scheduler_name(scheduler));
}

void StableDiffusionContext::Close(const Napi::CallbackInfo& info) {
    if (ctx_) {
        free_sd_ctx(ctx_);
        ctx_ = nullptr;
    }
}

Napi::Value StableDiffusionContext::IsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), ctx_ == nullptr);
}
