#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include "../abort_helper.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class GenerateVideoWorker : public Napi::AsyncWorker {
  public:
    GenerateVideoWorker(Napi::Env env, sd_ctx_t* ctx, const Napi::Object& opts)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(ctx),
          result_frames_(nullptr),
          num_frames_(0) {
        params_ = ParamsConverter::ToVidGenParams(opts, ss_, as_);
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        AbortHelper::clearAbort();

        if (setjmp(AbortHelper::jmp_buf_storage) != 0) {
            AbortHelper::jmp_active = false;
            AbortHelper::clearAbort();
            SetError("Aborted");
            return;
        }
        AbortHelper::jmp_active = true;

        result_frames_ = generate_video(ctx_, &params_, &num_frames_);
        AbortHelper::jmp_active = false;

        if (!result_frames_) {
            SetError("Video generation failed");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        Napi::Array arr = Napi::Array::New(env, num_frames_);
        for (int i = 0; i < num_frames_; i++) {
            arr.Set(static_cast<uint32_t>(i), ImageHelpers::ImageToJS(env, result_frames_[i]));
            result_frames_[i].data = nullptr;
        }
        free(result_frames_);
        deferred_.Resolve(arr);
    }

    void OnError(const Napi::Error& e) override {
        if (result_frames_) {
            for (int i = 0; i < num_frames_; i++) {
                if (result_frames_[i].data) {
                    free(result_frames_[i].data);
                }
            }
            free(result_frames_);
        }
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    sd_ctx_t* ctx_;
    sd_vid_gen_params_t params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_frames_;
    int num_frames_;
};
