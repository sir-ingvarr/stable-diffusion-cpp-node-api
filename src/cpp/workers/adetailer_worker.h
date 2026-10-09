#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <string>
#include <utility>

#include "../abort_helper.h"
#include "../adetailer_context.h"
#include "../sd_context.h"
#include "helpers/image_helpers.h"
#include "helpers/params_converter.h"

class CreateAdetailerWorker : public Napi::AsyncWorker {
  public:
    CreateAdetailerWorker(Napi::Env env, std::string detector_path, int n_threads,
                          std::string backend, std::string params_backend,
                          Napi::FunctionReference& constructor)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          constructor_ref_(constructor),
          detector_path_(std::move(detector_path)),
          n_threads_(n_threads),
          backend_(std::move(backend)),
          params_backend_(std::move(params_backend)),
          ctx_(nullptr) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        ctx_ = new_adetailer_ctx(detector_path_.c_str(), n_threads_,
                                 backend_.empty() ? nullptr : backend_.c_str(),
                                 params_backend_.empty() ? nullptr : params_backend_.c_str());
        if (!ctx_) {
            SetError("Failed to create ADetailer context (detector loading failed)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);
        Napi::External<adetailer_ctx_t> ext = Napi::External<adetailer_ctx_t>::New(env, ctx_);
        Napi::Object obj = constructor_ref_.New({ext});
        deferred_.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    Napi::FunctionReference& constructor_ref_;
    std::string detector_path_;
    int n_threads_;
    std::string backend_;
    std::string params_backend_;
    adetailer_ctx_t* ctx_;
};

class AdetailWorker : public Napi::AsyncWorker {
  public:
    AdetailWorker(Napi::Env env, AdetailerCtxPtr ad_ctx, SdCtxPtr sd_ctx,
                  AbortStatePtr abort_state, const Napi::Object& imgObj,
                  const Napi::Object& opts)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ad_ctx_(std::move(ad_ctx)),
          sd_ctx_(std::move(sd_ctx)),
          abort_state_(std::move(abort_state)) {
        input_ = ImageHelpers::ExtractImage(imgObj, as_);

        ad_params_ = {};
        ad_params_.prompt          = ss_.add(opts, "prompt");
        ad_params_.negative_prompt = ss_.add(opts, "negativePrompt");
        ad_params_.extra_ad_args   = ss_.add(opts, "extraAdArgs");

        // The inpaint pass takes a full generation-params struct; reuse the
        // generateImage option surface under `inpaint` (defaults otherwise).
        Napi::Object inpaint = (opts.Has("inpaint") && opts.Get("inpaint").IsObject())
                                   ? opts.Get("inpaint").As<Napi::Object>()
                                   : Napi::Object::New(opts.Env());
        inpaint_params_ = ParamsConverter::ToImgGenParams(inpaint, ss_, as_);
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        try {
            // The detail pass generates on the target sd_ctx — register that
            // ctx so abort() on it promotes to the engine cancel flag.
            AbortHelper::Scope abort_scope(*abort_state_, sd_ctx_.get());
            if (abort_state_->requested.load(std::memory_order_acquire)) {
                SetError("Aborted");
                return;
            }
            if (!adetail_image(ad_ctx_.get(), sd_ctx_.get(), input_,
                               &ad_params_, &inpaint_params_,
                               &result_images_, &num_images_) ||
                num_images_ < 1 || !result_images_ || !result_images_[0].data) {
                if (result_images_) free_sd_images(result_images_, num_images_);
                result_images_ = nullptr;
                num_images_ = 0;
                SetError(abort_state_->requested.load(std::memory_order_acquire)
                             ? "Aborted"
                             : "ADetailer pass failed");
            }
        } catch (const AbortHelper::AbortException&) {
            result_images_ = nullptr;
            SetError("Aborted");
        } catch (const std::exception& e) {
            result_images_ = nullptr;
            SetError(e.what());
        } catch (...) {
            result_images_ = nullptr;
            SetError("ADetailer pass failed (unknown exception)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);
        // adetail_image returns the detailed image first; extras (if any)
        // follow the same ownership rules as generate_image.
        Napi::Object out = ImageHelpers::ImageToJS(env, result_images_[0]);
        result_images_[0].data = nullptr;
        free_sd_images(result_images_, num_images_);
        deferred_.Resolve(out);
    }

    void OnError(const Napi::Error& e) override {
        if (result_images_) {
            free_sd_images(result_images_, num_images_);
        }
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    AdetailerCtxPtr ad_ctx_;
    SdCtxPtr sd_ctx_;
    AbortStatePtr abort_state_;
    sd_image_t input_;
    sd_adetailer_params_t ad_params_;
    sd_img_gen_params_t inpaint_params_;
    StringStore ss_;
    ArrayStore as_;
    sd_image_t* result_images_ = nullptr;
    int num_images_ = 0;
};
