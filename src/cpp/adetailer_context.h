#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <memory>

#include "abort_helper.h"

using AdetailerCtxPtr = std::shared_ptr<adetailer_ctx_t>;

// ADetailer: automatic face/hand detail-fix pass. Wraps new_adetailer_ctx /
// adetail_image. The detail pass runs an inpaint generation on a
// StableDiffusionContext, so adetail() takes that ctx as its target and the
// JS wrapper serializes the call on both contexts' queues.
class AdetailerContext : public Napi::ObjectWrap<AdetailerContext> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Value Create(const Napi::CallbackInfo& info);

    AdetailerContext(const Napi::CallbackInfo& info);

  private:
    static Napi::FunctionReference constructor_;

    Napi::Value Adetail(const Napi::CallbackInfo& info);
    void Close(const Napi::CallbackInfo& info);
    Napi::Value IsClosed(const Napi::CallbackInfo& info);

    // Shared with in-flight workers so close() can't free the native ctx
    // under a running detail pass.
    AdetailerCtxPtr ctx_;
};
