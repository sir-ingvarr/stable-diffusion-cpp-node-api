#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <cstring>
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
            AbortHelper::Scope abort_scope(*abort_state_, ctx_.get());
            if (abort_state_->requested.load(std::memory_order_acquire)) {
                SetError("Aborted");
                return;
            }
            // generate_video now returns success via bool, with frames, an
            // optional audio track, and the effective fps as out-params.
            if (!generate_video(ctx_.get(), &params_, &result_frames_, &num_frames_,
                                &result_audio_, &fps_out_)) {
                result_frames_ = nullptr;
                num_frames_ = 0;
                SetError(abort_state_->requested.load(std::memory_order_acquire)
                             ? "Aborted"
                             : "Video generation failed");
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
        free_sd_images(result_frames_, num_frames_);

        // Audio-capable models return a track; ride it (and the effective
        // fps) on the frames array as non-index properties so the resolve
        // shape stays backward compatible.
        arr.Set("fps", Napi::Number::New(env, fps_out_));
        if (result_audio_ && result_audio_->data && result_audio_->sample_count > 0) {
            Napi::Object audio = Napi::Object::New(env);
            audio.Set("sampleRate", Napi::Number::New(env, result_audio_->sample_rate));
            audio.Set("channels", Napi::Number::New(env, result_audio_->channels));
            const size_t n = static_cast<size_t>(result_audio_->sample_count) *
                             result_audio_->channels;
            auto buf = Napi::ArrayBuffer::New(env, n * sizeof(float));
            std::memcpy(buf.Data(), result_audio_->data, n * sizeof(float));
            audio.Set("data", Napi::Float32Array::New(env, n, buf, 0));
            arr.Set("audio", audio);
        }
        FreeAudio();
        deferred_.Resolve(arr);
    }

    void OnError(const Napi::Error& e) override {
        if (result_frames_) {
            free_sd_images(result_frames_, num_frames_);
        }
        FreeAudio();
        deferred_.Reject(e.Value());
    }

  private:
    // Audio output is not surfaced to JS yet; free it so audio-capable models
    // don't leak the returned track.
    void FreeAudio() {
        if (result_audio_) {
            free_sd_audio(result_audio_);
            result_audio_ = nullptr;
        }
    }

    Napi::Promise::Deferred deferred_;
    SdCtxPtr ctx_;
    AbortStatePtr abort_state_;
    sd_vid_gen_params_t params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_frames_;
    int num_frames_;
    sd_audio_t* result_audio_ = nullptr;
    int fps_out_ = 0;
};
