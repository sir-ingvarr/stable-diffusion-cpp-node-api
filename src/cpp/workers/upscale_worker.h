#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include "../abort_helper.h"
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
        AbortHelper::clearAbort();

        if (setjmp(AbortHelper::jmp_buf_storage) != 0) {
            AbortHelper::jmp_active = false;
            AbortHelper::clearAbort();
            ctx_ = nullptr;
            SetError("Aborted");
            return;
        }
        AbortHelper::jmp_active = true;

        ctx_ = new_upscaler_ctx(esrgan_path_.c_str(), offload_, direct_, n_threads_, tile_size_);
        AbortHelper::jmp_active = false;

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
    UpscaleWorker(Napi::Env env, upscaler_ctx_t* ctx, const Napi::Object& imgObj,
                  uint32_t factor, ArrayStore& as)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          ctx_(ctx),
          factor_(factor) {
        input_ = ImageHelpers::ExtractImage(imgObj, as_);
        result_ = {};
    }

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        AbortHelper::clearAbort();

        if (setjmp(AbortHelper::jmp_buf_storage) != 0) {
            AbortHelper::jmp_active = false;
            AbortHelper::clearAbort();
            result_ = {};
            SetError("Aborted");
            return;
        }
        AbortHelper::jmp_active = true;

        result_ = upscale(ctx_, input_, factor_);
        AbortHelper::jmp_active = false;

        if (!result_.data) {
            SetError("Upscaling failed");
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
    upscaler_ctx_t* ctx_;
    sd_image_t input_;
    uint32_t factor_;
    sd_image_t result_;
    ArrayStore as_;
};
