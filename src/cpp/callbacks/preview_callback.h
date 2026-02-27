#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <cstring>
#include <vector>

namespace PreviewCallback {

inline Napi::ThreadSafeFunction tsfn;

struct PreviewData {
    int step;
    bool is_noisy;
    // Deep-copied frame data (original may be freed after C callback returns)
    struct Frame {
        uint32_t width;
        uint32_t height;
        uint32_t channel;
        std::vector<uint8_t> data;
    };
    std::vector<Frame> frames;
};

inline void CCallback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* /*data*/) {
    if (!tsfn) return;

    auto* pd = new PreviewData();
    pd->step = step;
    pd->is_noisy = is_noisy;
    pd->frames.resize(frame_count);

    for (int i = 0; i < frame_count; i++) {
        pd->frames[i].width = frames[i].width;
        pd->frames[i].height = frames[i].height;
        pd->frames[i].channel = frames[i].channel;
        size_t sz = static_cast<size_t>(frames[i].width) * frames[i].height * frames[i].channel;
        if (frames[i].data && sz > 0) {
            pd->frames[i].data.assign(frames[i].data, frames[i].data + sz);
        }
    }

    tsfn.NonBlockingCall(pd, [](Napi::Env env, Napi::Function jsCallback, PreviewData* data) {
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("step", Napi::Number::New(env, data->step));
        obj.Set("isNoisy", Napi::Boolean::New(env, data->is_noisy));

        Napi::Array framesArr = Napi::Array::New(env, data->frames.size());
        for (size_t i = 0; i < data->frames.size(); i++) {
            auto& f = data->frames[i];
            Napi::Object fObj = Napi::Object::New(env);
            fObj.Set("width", Napi::Number::New(env, f.width));
            fObj.Set("height", Napi::Number::New(env, f.height));
            fObj.Set("channel", Napi::Number::New(env, f.channel));
            // Copy into a Node.js Buffer
            Napi::Buffer<uint8_t> buf = Napi::Buffer<uint8_t>::Copy(env, f.data.data(), f.data.size());
            fObj.Set("data", buf);
            framesArr.Set(static_cast<uint32_t>(i), fObj);
        }
        obj.Set("frames", framesArr);

        delete data;
        jsCallback.Call({obj});
    });
}

inline Napi::Value Set(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (tsfn) {
        tsfn.Release();
        tsfn = Napi::ThreadSafeFunction();
        sd_set_preview_callback(nullptr, PREVIEW_NONE, 0, false, false, nullptr);
    }

    if (info.Length() < 1 || info[0].IsNull() || info[0].IsUndefined()) {
        return env.Undefined();
    }

    if (!info[0].IsFunction()) {
        Napi::TypeError::New(env, "Expected function or null as first argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Optional options object as second argument
    preview_t mode = PREVIEW_PROJ;
    int interval = 1;
    bool denoised = true;
    bool noisy = false;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();

        if (opts.Has("mode") && opts.Get("mode").IsString()) {
            std::string modeStr = opts.Get("mode").As<Napi::String>().Utf8Value();
            mode = str_to_preview(modeStr.c_str());
        }
        if (opts.Has("interval") && opts.Get("interval").IsNumber()) {
            interval = opts.Get("interval").As<Napi::Number>().Int32Value();
        }
        if (opts.Has("denoised") && opts.Get("denoised").IsBoolean()) {
            denoised = opts.Get("denoised").As<Napi::Boolean>().Value();
        }
        if (opts.Has("noisy") && opts.Get("noisy").IsBoolean()) {
            noisy = opts.Get("noisy").As<Napi::Boolean>().Value();
        }
    }

    tsfn = Napi::ThreadSafeFunction::New(
        env,
        info[0].As<Napi::Function>(),
        "sd_preview_callback",
        0, 1);

    sd_set_preview_callback(PreviewCallback::CCallback, mode, interval, denoised, noisy, nullptr);
    return env.Undefined();
}

}  // namespace PreviewCallback
