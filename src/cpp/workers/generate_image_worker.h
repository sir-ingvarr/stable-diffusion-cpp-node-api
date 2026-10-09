#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <utility>

#include "../abort_helper.h"
#include "../sd_context.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class GenerateImageWorker : public Napi::AsyncWorker {
  public:
    GenerateImageWorker(Napi::Env env, SdCtxPtr ctx, AbortStatePtr abort_state, const Napi::Object& opts)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(std::move(ctx)),
          abort_state_(std::move(abort_state)),
          result_images_(nullptr) {
        params_ = ParamsConverter::ToImgGenParams(opts, ss_, as_);
        batch_count_ = params_.batch_count;
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        try {
            AbortHelper::Scope abort_scope(*abort_state_, ctx_.get());
            if (abort_state_->requested.load(std::memory_order_acquire)) {
                SetError("Aborted");
                return;
            }
            // generate_image returns the actual image count via out-params
            // (can be < batch_count when a generation is cancelled with
            // SD_CANCEL_NEW_LATENTS).
            if (!generate_image(ctx_.get(), &params_, &result_images_, &num_images_)) {
                result_images_ = nullptr;
                // A mid-run abort is promoted to the engine's cancel flag and
                // surfaces as a false return; keep the AbortError contract.
                SetError(abort_state_->requested.load(std::memory_order_acquire)
                             ? "Aborted"
                             : "Image generation failed");
            }
        } catch (const AbortHelper::AbortException&) {
            result_images_ = nullptr;
            SetError("Aborted");
        } catch (const std::exception& e) {
            result_images_ = nullptr;
            SetError(e.what());
        } catch (...) {
            result_images_ = nullptr;
            SetError("Image generation failed (unknown exception)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        Napi::Array arr = Napi::Array::New(env, num_images_);
        for (int i = 0; i < num_images_; i++) {
            arr.Set(static_cast<uint32_t>(i), ImageHelpers::ImageToJS(env, result_images_[i]));
            // ImageToJS transfers ownership of result_images_[i].data via Buffer
            // free callback, so null it out to avoid double-free
            result_images_[i].data = nullptr;
        }
        free_sd_images(result_images_, num_images_);
        deferred_.Resolve(arr);
    }

    void OnError(const Napi::Error& e) override {
        // Clean up any allocated images on error
        if (result_images_) {
            free_sd_images(result_images_, num_images_);
        }
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    SdCtxPtr ctx_;
    AbortStatePtr abort_state_;
    sd_img_gen_params_t params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_images_;
    int num_images_ = 0;
    int batch_count_;
};
