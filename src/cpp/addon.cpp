#include <napi.h>
#include <stable-diffusion.h>

#include <string>

#include "abort_helper.h"
#include "adetailer_context.h"
#include "callbacks/log_callback.h"
#include "callbacks/preview_callback.h"
#include "callbacks/progress_callback.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"
#include "sd_context.h"
#include "upscaler_context.h"
#include "workers/convert_worker.h"
#include "workers/extract_metadata_worker.h"

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

static Napi::Value ExtractMetaData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model file path string").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string path = info[0].As<Napi::String>().Utf8Value();
    auto* worker = new ExtractMetadataWorker(env, std::move(path));
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

static Napi::Value Convert(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object opts = info[0].As<Napi::Object>();

    ConvertRequest req;
    auto getStr = [&](const char* key, std::string& out) {
        if (opts.Has(key) && opts.Get(key).IsString())
            out = opts.Get(key).As<Napi::String>().Utf8Value();
    };
    getStr("inputPath", req.input_path);
    getStr("clipLPath", req.clip_l_path);
    getStr("clipGPath", req.clip_g_path);
    getStr("t5xxlPath", req.t5xxl_path);
    getStr("diffusionModelPath", req.diffusion_model_path);
    getStr("vaePath", req.vae_path);
    getStr("outputPath", req.output_path);
    getStr("tensorTypeRules", req.tensor_type_rules);

    if (opts.Has("outputType") && opts.Get("outputType").IsString()) {
        std::string typeStr = opts.Get("outputType").As<Napi::String>().Utf8Value();
        req.output_type = str_to_sd_type(typeStr.c_str());
    }
    req.convert_name = ParamsConverter::GetBool(opts, "convertName", false);
    req.n_threads = ParamsConverter::GetInt(opts, "nThreads", sd_get_num_physical_cores());

    // Loras baked into the converted output (components API only).
    if (opts.Has("loras") && opts.Get("loras").IsArray()) {
        Napi::Array arr = opts.Get("loras").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); i++) {
            Napi::Value v = arr.Get(i);
            if (!v.IsObject()) continue;
            Napi::Object l = v.As<Napi::Object>();
            if (!l.Has("path") || !l.Get("path").IsString()) continue;
            req.lora_paths.push_back(l.Get("path").As<Napi::String>().Utf8Value());
            req.lora_multipliers.push_back(
                static_cast<float>(ParamsConverter::GetDouble(l, "multiplier", 1.0)));
        }
    }

    // Any component path / lora list switches to convert_with_components;
    // the classic single-file convert() path is byte-identical otherwise.
    req.use_components = !req.clip_l_path.empty() || !req.clip_g_path.empty() ||
                         !req.t5xxl_path.empty() || !req.diffusion_model_path.empty() ||
                         !req.lora_paths.empty();

    auto* worker = new ConvertWorker(env, std::move(req));
    auto promise = worker->Deferred().Promise();
    worker->Queue();
    return promise;
}

static Napi::Value GetUpscalerModelScale(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected model file path string").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path = info[0].As<Napi::String>().Utf8Value();
    // Reads model metadata only; 0 = not a recognized RGB ESRGAN model.
    return Napi::Number::New(env, get_upscaler_model_scale(path.c_str()));
}

// listDevices() → [{ name, description }] — ggml backend devices accepted by
// the backend / paramsBackend assignment specs.
static Napi::Value ListDevices(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    size_t needed = sd_list_devices(nullptr, 0);
    std::string buf(needed + 1, '\0');
    sd_list_devices(buf.data(), buf.size());
    buf.resize(needed);

    Napi::Array result = Napi::Array::New(env);
    uint32_t idx = 0;
    size_t pos = 0;
    while (pos < buf.size()) {
        size_t eol = buf.find('\n', pos);
        if (eol == std::string::npos) eol = buf.size();
        std::string line = buf.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty()) continue;
        size_t tab = line.find('\t');
        Napi::Object dev = Napi::Object::New(env);
        dev.Set("name", Napi::String::New(env, line.substr(0, tab)));
        dev.Set("description", Napi::String::New(env,
            tab == std::string::npos ? "" : line.substr(tab + 1)));
        result.Set(idx++, dev);
    }
    return result;
}

// --- imatrix (importance matrix) workflow for quantization quality ---

static Napi::Value LoadImatrix(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected imatrix file path string").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path = info[0].As<Napi::String>().Utf8Value();
    return Napi::Boolean::New(env, load_imatrix(path.c_str()));
}

static Napi::Value SaveImatrix(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected imatrix file path string").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path = info[0].As<Napi::String>().Utf8Value();
    save_imatrix(path.c_str());
    return env.Undefined();
}

static Napi::Value SetImatrixCollection(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    bool enable = info.Length() >= 1 && info[0].ToBoolean().Value();
    if (enable) enable_imatrix_collection();
    else        disable_imatrix_collection();
    return env.Undefined();
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

// --- Module init ---

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Register our progress callback once. It must stay registered even
    // when the user hasn't set a JS callback, because it's also our
    // cancellation point.
    ProgressCallback::Init();

    StableDiffusionContext::Init(env, exports);
    UpscalerContext::Init(env, exports);
    AdetailerContext::Init(env, exports);

    exports.Set("getSystemInfo", Napi::Function::New(env, GetSystemInfo));
    exports.Set("getNumPhysicalCores", Napi::Function::New(env, GetNumPhysicalCores));
    exports.Set("version", Napi::Function::New(env, Version));
    exports.Set("commit", Napi::Function::New(env, Commit));
    exports.Set("convert", Napi::Function::New(env, Convert));
    exports.Set("extractMetaData", Napi::Function::New(env, ExtractMetaData));
    exports.Set("preprocessCanny", Napi::Function::New(env, PreprocessCanny));
    exports.Set("listDevices", Napi::Function::New(env, ListDevices));
    exports.Set("getUpscalerModelScale", Napi::Function::New(env, GetUpscalerModelScale));
    exports.Set("loadImatrix", Napi::Function::New(env, LoadImatrix));
    exports.Set("saveImatrix", Napi::Function::New(env, SaveImatrix));
    exports.Set("setImatrixCollection", Napi::Function::New(env, SetImatrixCollection));
    exports.Set("setLogCallback", Napi::Function::New(env, LogCallback::Set));
    exports.Set("setProgressCallback", Napi::Function::New(env, ProgressCallback::Set));
    exports.Set("setPreviewCallback", Napi::Function::New(env, PreviewCallback::Set));

    return exports;
}

NODE_API_MODULE(node_stable_diffusion, Init)
