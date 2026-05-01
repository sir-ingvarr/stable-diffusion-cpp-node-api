#include "upscaler_context.h"

#include "abort_helper.h"
#include "helpers/params_converter.h"
#include "workers/upscale_worker.h"

Napi::FunctionReference UpscalerContext::constructor_;

Napi::Object UpscalerContext::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "UpscalerContext", {
        InstanceMethod<&UpscalerContext::Upscale>("upscale"),
        InstanceMethod<&UpscalerContext::GetUpscaleFactor>("getUpscaleFactor"),
        InstanceMethod<&UpscalerContext::Abort>("abort"),
        InstanceMethod<&UpscalerContext::Close>("close"),
        InstanceAccessor<&UpscalerContext::IsClosed>("isClosed"),
        StaticMethod<&UpscalerContext::Create>("create"),
    });

    constructor_ = Napi::Persistent(func);
    constructor_.SuppressDestruct();

    exports.Set("UpscalerContext", func);
    return exports;
}

UpscalerContext::UpscalerContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<UpscalerContext>(info) {
    if (info.Length() >= 1 && info[0].IsExternal()) {
        upscaler_ctx_t* raw = info[0].As<Napi::External<upscaler_ctx_t>>().Data();
        ctx_ = UpscalerCtxPtr(raw, [](upscaler_ctx_t* c) {
            if (c) free_upscaler_ctx(c);
        });
        abort_state_ = std::make_shared<AbortHelper::AbortState>();
        return;
    }
    Napi::TypeError::New(info.Env(), "Use UpscalerContext.create() instead of new")
        .ThrowAsJavaScriptException();
}

Napi::Value UpscalerContext::Create(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();
    std::string esrganPath;
    if (opts.Has("esrganPath") && opts.Get("esrganPath").IsString()) {
        esrganPath = opts.Get("esrganPath").As<Napi::String>().Utf8Value();
    }
    bool offload = ParamsConverter::GetBool(opts, "offloadParamsToCpu", false);
    bool direct = ParamsConverter::GetBool(opts, "direct", false);
    int nThreads = ParamsConverter::GetInt(opts, "nThreads", sd_get_num_physical_cores());
    int tileSize = ParamsConverter::GetInt(opts, "tileSize", 0);

    auto* worker = new CreateUpscalerWorker(env, esrganPath, offload, direct, nThreads, tileSize, constructor_);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

Napi::Value UpscalerContext::Upscale(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected image object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    uint32_t factor = 4;
    if (info.Length() >= 2 && info[1].IsNumber()) {
        factor = info[1].As<Napi::Number>().Uint32Value();
    }

    ArrayStore as;
    // See note in StableDiffusionContext::GenerateImage — clear on JS thread,
    // not worker thread, so an abort issued between Queue() and Execute()
    // isn't wiped by the worker's Scope ctor.
    AbortHelper::clearAbort(*abort_state_);
    auto* worker = new UpscaleWorker(env, ctx_, abort_state_, info[0].As<Napi::Object>(), factor, as);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

Napi::Value UpscalerContext::GetUpscaleFactor(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, get_upscale_factor(ctx_.get()));
}

void UpscalerContext::Abort(const Napi::CallbackInfo& info) {
    if (abort_state_) AbortHelper::requestAbort(*abort_state_);
}

void UpscalerContext::Close(const Napi::CallbackInfo& info) {
    // See note in StableDiffusionContext::Close — signal abort, then drop ref.
    if (abort_state_) AbortHelper::requestAbort(*abort_state_);
    ctx_.reset();
}

Napi::Value UpscalerContext::IsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !ctx_);
}
