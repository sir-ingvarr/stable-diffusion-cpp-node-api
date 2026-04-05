#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include "../abort_helper.h"
#include "helpers/params_converter.h"

class CreateContextWorker : public Napi::AsyncWorker {
  public:
    CreateContextWorker(Napi::Env env, const Napi::Object& opts,
                        Napi::FunctionReference& constructor)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          constructor_ref_(constructor),
          ctx_(nullptr) {
        params_ = ParamsConverter::ToCtxParams(opts, ss_, as_);
    }

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

        ctx_ = new_sd_ctx(&params_);
        AbortHelper::jmp_active = false;

        if (!ctx_) {
            SetError("Failed to create StableDiffusion context (model loading failed)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        // Create the JS wrapper by calling the constructor with the raw pointer
        Napi::External<sd_ctx_t> ext = Napi::External<sd_ctx_t>::New(env, ctx_);
        Napi::Object obj = constructor_ref_.New({ext});
        deferred_.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    Napi::FunctionReference& constructor_ref_;
    sd_ctx_t* ctx_;
    sd_ctx_params_t params_;
    StringStore ss_;
    ArrayStore as_;
};
