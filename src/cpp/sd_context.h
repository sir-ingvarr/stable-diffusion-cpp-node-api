#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <memory>

#include "abort_helper.h"

using SdCtxPtr = std::shared_ptr<sd_ctx_t>;
using AbortStatePtr = std::shared_ptr<AbortHelper::AbortState>;

class StableDiffusionContext : public Napi::ObjectWrap<StableDiffusionContext> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Value Create(const Napi::CallbackInfo& info);

    StableDiffusionContext(const Napi::CallbackInfo& info);

    // Shared-ownership handles for workers driven by other wrappers
    // (AdetailerContext runs inpaint generations on this ctx).
    SdCtxPtr shared_ctx() const { return ctx_; }
    AbortStatePtr abort_state() const { return abort_state_; }

  private:
    static Napi::FunctionReference constructor_;

    Napi::Value GenerateImage(const Napi::CallbackInfo& info);
    Napi::Value GenerateVideo(const Napi::CallbackInfo& info);
    Napi::Value GetDefaultSampleMethod(const Napi::CallbackInfo& info);
    Napi::Value GetDefaultScheduler(const Napi::CallbackInfo& info);
    Napi::Value GetModelVersionName(const Napi::CallbackInfo& info);
    Napi::Value LoadControlNet(const Napi::CallbackInfo& info);
    Napi::Value UnloadControlNet(const Napi::CallbackInfo& info);
    Napi::Value HasControlNet(const Napi::CallbackInfo& info);
    void Abort(const Napi::CallbackInfo& info);
    void Close(const Napi::CallbackInfo& info);
    Napi::Value IsClosed(const Napi::CallbackInfo& info);

    // Shared with in-flight workers so close() / GC can't free the native
    // context while a worker thread is still using it. The underlying
    // sd_ctx_t is freed when the last ref (JS wrapper or worker) drops.
    SdCtxPtr ctx_;

    // Per-ctx cancellation flag. Held alongside ctx_ by in-flight workers
    // so they watch this ctx specifically and aborts on one ctx don't
    // bleed into another.
    AbortStatePtr abort_state_;
};
