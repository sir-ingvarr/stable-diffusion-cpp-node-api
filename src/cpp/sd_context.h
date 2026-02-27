#pragma once

#include <napi.h>
#include <stable-diffusion.h>

class StableDiffusionContext : public Napi::ObjectWrap<StableDiffusionContext> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Value Create(const Napi::CallbackInfo& info);

    StableDiffusionContext(const Napi::CallbackInfo& info);
    ~StableDiffusionContext();

  private:
    static Napi::FunctionReference constructor_;

    Napi::Value GenerateImage(const Napi::CallbackInfo& info);
    Napi::Value GenerateVideo(const Napi::CallbackInfo& info);
    Napi::Value GetDefaultSampleMethod(const Napi::CallbackInfo& info);
    Napi::Value GetDefaultScheduler(const Napi::CallbackInfo& info);
    void Close(const Napi::CallbackInfo& info);
    Napi::Value IsClosed(const Napi::CallbackInfo& info);

    sd_ctx_t* ctx_;
};
