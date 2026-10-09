#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <string>
#include <utility>

#include "../sd_context.h"

// Loads or unloads a ControlNet on a live sd_ctx_t without reloading the base
// model. Upstream documents the hot-swap APIs as unsafe while a generation is
// in flight, so the JS wrapper serializes these calls through the same per-ctx
// queue as generateImage / generateVideo.
class ControlNetWorker : public Napi::AsyncWorker {
  public:
    // Empty path = unload.
    ControlNetWorker(Napi::Env env, SdCtxPtr ctx, std::string path)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(std::move(ctx)),
          path_(std::move(path)) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        if (path_.empty()) {
            if (!sd_ctx_unload_control_net(ctx_.get())) {
                SetError("Failed to unload ControlNet");
            }
        } else if (!sd_ctx_load_control_net(ctx_.get(), path_.c_str())) {
            SetError("Failed to load ControlNet: " + path_);
        }
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    SdCtxPtr ctx_;
    std::string path_;
};
