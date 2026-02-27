#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class GenerateImageWorker : public Napi::AsyncWorker {
  public:
    GenerateImageWorker(Napi::Env env, sd_ctx_t* ctx, const Napi::Object& opts)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(ctx),
          result_images_(nullptr) {
        params_ = ParamsConverter::ToImgGenParams(opts, ss_, as_);
        batch_count_ = params_.batch_count;
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        result_images_ = generate_image(ctx_, &params_);
        if (!result_images_) {
            SetError("Image generation failed");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        Napi::Array arr = Napi::Array::New(env, batch_count_);
        for (int i = 0; i < batch_count_; i++) {
            arr.Set(static_cast<uint32_t>(i), ImageHelpers::ImageToJS(env, result_images_[i]));
            // ImageToJS transfers ownership of result_images_[i].data via Buffer
            // free callback, so null it out to avoid double-free
            result_images_[i].data = nullptr;
        }
        free(result_images_);
        deferred_.Resolve(arr);
    }

    void OnError(const Napi::Error& e) override {
        // Clean up any allocated images on error
        if (result_images_) {
            for (int i = 0; i < batch_count_; i++) {
                if (result_images_[i].data) {
                    free(result_images_[i].data);
                }
            }
            free(result_images_);
        }
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    sd_ctx_t* ctx_;
    sd_img_gen_params_t params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_images_;
    int batch_count_;
};
