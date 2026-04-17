#include <napi.h>
#include <stable-diffusion.h>

#include "abort_helper.h"
#include "callbacks/log_callback.h"
#include "callbacks/preview_callback.h"
#include "callbacks/progress_callback.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"
#include "sd_context.h"
#include "upscaler_context.h"
#include "workers/convert_worker.h"

// --- Free functions ---

static Napi::Value GetSystemInfo(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), sd_get_system_info());
}

static Napi::Value GetNumPhysicalCores(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), sd_get_num_physical_cores());
}

static Napi::Value Version(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), sd_version());
}

static Napi::Value Commit(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), sd_commit());
}

static Napi::Value Convert(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();

    std::string inputPath, vaePath, outputPath, tensorTypeRules;
    if (opts.Has("inputPath") && opts.Get("inputPath").IsString())
        inputPath = opts.Get("inputPath").As<Napi::String>().Utf8Value();
    if (opts.Has("vaePath") && opts.Get("vaePath").IsString())
        vaePath = opts.Get("vaePath").As<Napi::String>().Utf8Value();
    if (opts.Has("outputPath") && opts.Get("outputPath").IsString())
        outputPath = opts.Get("outputPath").As<Napi::String>().Utf8Value();
    if (opts.Has("tensorTypeRules") && opts.Get("tensorTypeRules").IsString())
        tensorTypeRules = opts.Get("tensorTypeRules").As<Napi::String>().Utf8Value();

    sd_type_t outputType = SD_TYPE_COUNT;
    if (opts.Has("outputType") && opts.Get("outputType").IsString()) {
        std::string typeStr = opts.Get("outputType").As<Napi::String>().Utf8Value();
        outputType = str_to_sd_type(typeStr.c_str());
    }

    bool convertName = ParamsConverter::GetBool(opts, "convertName", false);

    auto* worker = new ConvertWorker(env, inputPath, vaePath, outputPath, outputType, tensorTypeRules, convertName);
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

static Napi::Value PreprocessCanny(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected image object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object imgObj = info[0].As<Napi::Object>();
    sd_image_t img = {};
    img.width = imgObj.Get("width").As<Napi::Number>().Uint32Value();
    img.height = imgObj.Get("height").As<Napi::Number>().Uint32Value();
    img.channel = imgObj.Get("channel").As<Napi::Number>().Uint32Value();
    // preprocess_canny works in-place, so use the Buffer directly
    Napi::Buffer<uint8_t> buf = imgObj.Get("data").As<Napi::Buffer<uint8_t>>();
    img.data = buf.Data();

    float highThreshold = 1.0f;
    float lowThreshold = 0.1f;
    float weak = 0.2f;
    float strong = 1.0f;
    bool inverse = false;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        highThreshold = static_cast<float>(ParamsConverter::GetDouble(opts, "highThreshold", highThreshold));
        lowThreshold = static_cast<float>(ParamsConverter::GetDouble(opts, "lowThreshold", lowThreshold));
        weak = static_cast<float>(ParamsConverter::GetDouble(opts, "weak", weak));
        strong = static_cast<float>(ParamsConverter::GetDouble(opts, "strong", strong));
        inverse = ParamsConverter::GetBool(opts, "inverse", inverse);
    }

    bool result = preprocess_canny(img, highThreshold, lowThreshold, weak, strong, inverse);
    return Napi::Boolean::New(env, result);
}

static Napi::Value Abort(const Napi::CallbackInfo& info) {
    AbortHelper::requestAbort();
    return info.Env().Undefined();
}

// --- Module init ---

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Register our progress callback once. It must stay registered even
    // when the user hasn't set a JS callback, because it's also our
    // cancellation point.
    ProgressCallback::Init();

    StableDiffusionContext::Init(env, exports);
    UpscalerContext::Init(env, exports);

    exports.Set("getSystemInfo", Napi::Function::New(env, GetSystemInfo));
    exports.Set("getNumPhysicalCores", Napi::Function::New(env, GetNumPhysicalCores));
    exports.Set("version", Napi::Function::New(env, Version));
    exports.Set("commit", Napi::Function::New(env, Commit));
    exports.Set("convert", Napi::Function::New(env, Convert));
    exports.Set("preprocessCanny", Napi::Function::New(env, PreprocessCanny));
    exports.Set("setLogCallback", Napi::Function::New(env, LogCallback::Set));
    exports.Set("setProgressCallback", Napi::Function::New(env, ProgressCallback::Set));
    exports.Set("setPreviewCallback", Napi::Function::New(env, PreviewCallback::Set));
    exports.Set("abort", Napi::Function::New(env, Abort));

    return exports;
}

NODE_API_MODULE(node_stable_diffusion, Init)
