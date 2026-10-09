#include "adetailer_context.h"

#include <string>

#include "helpers/params_converter.h"
#include "sd_context.h"
#include "workers/adetailer_worker.h"

Napi::FunctionReference AdetailerContext::constructor_;

Napi::Object AdetailerContext::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "AdetailerContext", {
        InstanceMethod<&AdetailerContext::Adetail>("adetail"),
        InstanceMethod<&AdetailerContext::Close>("close"),
        // Runtime-pointer overload: the template form (InstanceAccessor<&T::IsClosed>)
        // hits an MSVC C1001 internal compiler error (constexpr.cpp) on current
        // VS 17.x toolsets.
        InstanceAccessor("isClosed", &AdetailerContext::IsClosed, nullptr),
        StaticMethod<&AdetailerContext::Create>("create"),
    });

    constructor_ = Napi::Persistent(func);
    constructor_.SuppressDestruct();

    exports.Set("AdetailerContext", func);
    return exports;
}

AdetailerContext::AdetailerContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<AdetailerContext>(info) {
    if (info.Length() >= 1 && info[0].IsExternal()) {
        adetailer_ctx_t* raw = info[0].As<Napi::External<adetailer_ctx_t>>().Data();
        ctx_ = AdetailerCtxPtr(raw, [](adetailer_ctx_t* c) {
            if (c) free_adetailer_ctx(c);
        });
        return;
    }
    Napi::TypeError::New(info.Env(), "Use AdetailerContext.create() instead of new")
        .ThrowAsJavaScriptException();
}

Napi::Value AdetailerContext::Create(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();
    std::string detectorPath;
    if (opts.Has("detectorPath") && opts.Get("detectorPath").IsString()) {
        detectorPath = opts.Get("detectorPath").As<Napi::String>().Utf8Value();
    }
    if (detectorPath.empty()) {
        Napi::TypeError::New(env, "detectorPath is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int nThreads = ParamsConverter::GetInt(opts, "nThreads", sd_get_num_physical_cores());
    std::string backend, paramsBackend;
    if (opts.Has("backend") && opts.Get("backend").IsString()) {
        backend = opts.Get("backend").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("paramsBackend") && opts.Get("paramsBackend").IsString()) {
        paramsBackend = opts.Get("paramsBackend").As<Napi::String>().Utf8Value();
    }

    auto* worker = new CreateAdetailerWorker(env, std::move(detectorPath), nThreads,
                                             std::move(backend), std::move(paramsBackend),
                                             constructor_);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

// adetail(sdCtx: native StableDiffusionContext, image, opts)
Napi::Value AdetailerContext::Adetail(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        Napi::Error::New(env, "Context is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsObject()) {
        Napi::TypeError::New(env, "Expected (sdContext, image, options?)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto* sdc = Napi::ObjectWrap<StableDiffusionContext>::Unwrap(info[0].As<Napi::Object>());
    if (!sdc || !sdc->shared_ctx()) {
        Napi::Error::New(env, "Target StableDiffusionContext is closed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = (info.Length() >= 3 && info[2].IsObject())
                            ? info[2].As<Napi::Object>()
                            : Napi::Object::New(env);

    // Same contract as generateImage: clear stale abort state on the JS
    // thread before queueing (see note in StableDiffusionContext).
    AbortHelper::clearAbort(*sdc->abort_state());
    auto* worker = new AdetailWorker(env, ctx_, sdc->shared_ctx(), sdc->abort_state(),
                                     info[1].As<Napi::Object>(), opts);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

void AdetailerContext::Close(const Napi::CallbackInfo& info) {
    // A running worker holds its own shared_ptr; the native ctx is freed when
    // the last ref drops. Aborting the generation half goes through the
    // target sd ctx's abort(), not here.
    ctx_.reset();
}

Napi::Value AdetailerContext::IsClosed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !ctx_);
}
