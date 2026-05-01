#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <utility>

#include "../abort_helper.h"
#include "../sd_context.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class GenerateVideoWorker : public Napi::AsyncWorker {
  public:
    GenerateVideoWorker(Napi::Env env, SdCtxPtr ctx, AbortStatePtr abort_state, const Napi::Object& opts)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(std::move(ctx)),
          abort_state_(std::move(abort_state)),
          result_frames_(nullptr),
          num_frames_(0) {
        params_ = ParamsConverter::ToVidGenParams(opts, ss_, as_);
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        try {
            AbortHelper::Scope abort_scope(*abort_state_);
            result_frames_ = generate_video(ctx_.get(), &params_, &num_frames_);
            if (!result_frames_) {
                SetError("Video generation failed");
            }
        } catch (const AbortHelper::AbortException&) {
            result_frames_ = nullptr;
            num_frames_ = 0;
            SetError("Aborted");
        } catch (const std::exception& e) {
            result_frames_ = nullptr;
            num_frames_ = 0;
            SetError(e.what());
        } catch (...) {
            result_frames_ = nullptr;
            num_frames_ = 0;
            SetError("Video generation failed (unknown exception)");
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
    SdCtxPtr ctx_;
    AbortStatePtr abort_state_;
    sd_vid_gen_params_t params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_frames_;
    int num_frames_;
};
