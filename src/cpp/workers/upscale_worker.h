#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <utility>

#include "../abort_helper.h"
#include "../upscaler_context.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class CreateUpscalerWorker : public Napi::AsyncWorker {
  public:
    CreateUpscalerWorker(Napi::Env env, const std::string& esrgan_path,
                         bool offload, bool direct, int n_threads, int tile_size,
                         Napi::FunctionReference& constructor)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          constructor_ref_(constructor),
          esrgan_path_(esrgan_path),
          offload_(offload),
          direct_(direct),
          n_threads_(n_threads),
          tile_size_(tile_size),
          ctx_(nullptr) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        // The old `offload` bool became backend assignment strings upstream;
        // map it the same way the CLI compat alias does (params on CPU).
        ctx_ = new_upscaler_ctx(esrgan_path_.c_str(), direct_, n_threads_, tile_size_,
                                nullptr, offload_ ? "*=cpu" : nullptr);
        if (!ctx_) {
            SetError("Failed to create upscaler context");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);
        Napi::External<upscaler_ctx_t> ext = Napi::External<upscaler_ctx_t>::New(env, ctx_);
        Napi::Object obj = constructor_ref_.New({ext});
        deferred_.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    Napi::FunctionReference& constructor_ref_;
    std::string esrgan_path_;
    bool offload_;
    bool direct_;
    int n_threads_;
    int tile_size_;
    upscaler_ctx_t* ctx_;
};

class UpscaleWorker : public Napi::AsyncWorker {
  public:
    UpscaleWorker(Napi::Env env, UpscalerCtxPtr ctx, std::shared_ptr<AbortHelper::AbortState> abort_state,
                  const Napi::Object& imgObj, uint32_t factor, ArrayStore& /*as*/)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(std::move(ctx)),
          abort_state_(std::move(abort_state)),
          factor_(factor) {
        input_ = ImageHelpers::ExtractImage(imgObj, as_);
        result_ = {};
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        try {
            AbortHelper::Scope abort_scope(*abort_state_);
            // upscale() now returns an image array via out-params (bool = success).
            sd_image_t* images_out = nullptr;
            int num_out = 0;
            if (!upscale(ctx_.get(), input_, factor_, &images_out, &num_out) ||
                num_out < 1 || !images_out) {
                if (images_out) free_sd_images(images_out, num_out);
                result_ = {};
                SetError("Upscaling failed");
                return;
            }
            // Single input image → first output is ours; free any extras.
            result_ = images_out[0];
            images_out[0].data = nullptr;
            free_sd_images(images_out, num_out);
        } catch (const AbortHelper::AbortException&) {
            result_ = {};
            SetError("Aborted");
        } catch (const std::exception& e) {
            result_ = {};
            SetError(e.what());
        } catch (...) {
            result_ = {};
            SetError("Upscaling failed (unknown exception)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);
        deferred_.Resolve(ImageHelpers::ImageToJS(env, result_));
    }

    void OnError(const Napi::Error& e) override {
        if (result_.data) free(result_.data);
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    UpscalerCtxPtr ctx_;
    std::shared_ptr<AbortHelper::AbortState> abort_state_;
    sd_image_t input_;
    uint32_t factor_;
    sd_image_t result_;
    ArrayStore as_;
};
