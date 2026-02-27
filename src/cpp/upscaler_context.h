#pragma once

#include <napi.h>
#include <stable-diffusion.h>

class UpscalerContext : public Napi::ObjectWrap<UpscalerContext> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Value Create(const Napi::CallbackInfo& info);

    UpscalerContext(const Napi::CallbackInfo& info);
    ~UpscalerContext();

  private:
    static Napi::FunctionReference constructor_;

    Napi::Value Upscale(const Napi::CallbackInfo& info);
    Napi::Value GetUpscaleFactor(const Napi::CallbackInfo& info);
    void Close(const Napi::CallbackInfo& info);
    Napi::Value IsClosed(const Napi::CallbackInfo& info);

    upscaler_ctx_t* ctx_;
};
