#include "upscaler_context.h"

#include "helpers/params_converter.h"
#include "workers/upscale_worker.h"

Napi::FunctionReference UpscalerContext::constructor_;

Napi::Object UpscalerContext::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "UpscalerContext", {
        InstanceMethod<&UpscalerContext::Upscale>("upscale"),
        InstanceMethod<&UpscalerContext::GetUpscaleFactor>("getUpscaleFactor"),
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
    : Napi::ObjectWrap<UpscalerContext>(info), ctx_(nullptr) {
    if (info.Length() >= 1 && info[0].IsExternal()) {
        ctx_ = info[0].As<Napi::External<upscaler_ctx_t>>().Data();
        return;
    }
    Napi::TypeError::New(info.Env(), "Use UpscalerContext.create() instead of new")
        .ThrowAsJavaScriptException();
}

UpscalerContext::~UpscalerContext() {
    if (ctx_) {
        free_upscaler_ctx(ctx_);
        ctx_ = nullptr;
    }
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
    auto* worker = new UpscaleWorker(env, ctx_, info[0].As<Napi::Object>(), factor, as);
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
    return Napi::Number::New(env, get_upscale_factor(ctx_));
}

void UpscalerContext::Close(const Napi::CallbackInfo& info) {
    if (ctx_) {
        free_upscaler_ctx(ctx_);
        ctx_ = nullptr;
    }
}

Napi::Value UpscalerContext::IsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), ctx_ == nullptr);
}
