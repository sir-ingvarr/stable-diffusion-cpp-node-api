#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <string>

#include "../abort_helper.h"
#include "helpers/params_converter.h"

class ConvertWorker : public Napi::AsyncWorker {
  public:
    ConvertWorker(Napi::Env env,
                  const std::string& input_path,
                  const std::string& vae_path,
                  const std::string& output_path,
                  sd_type_t output_type,
                  const std::string& tensor_type_rules,
                  bool convert_name)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          input_path_(input_path),
          vae_path_(vae_path),
          output_path_(output_path),
          output_type_(output_type),
          tensor_type_rules_(tensor_type_rules),
          convert_name_(convert_name),
          result_(false) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        AbortHelper::clearAbort();

        if (setjmp(AbortHelper::jmp_buf_storage) != 0) {
            AbortHelper::jmp_active = false;
            AbortHelper::clearAbort();
            result_ = false;
            SetError("Aborted");
            return;
        }
        AbortHelper::jmp_active = true;

        result_ = convert(
            input_path_.c_str(),
            vae_path_.empty() ? nullptr : vae_path_.c_str(),
            output_path_.c_str(),
            output_type_,
            tensor_type_rules_.empty() ? nullptr : tensor_type_rules_.c_str(),
            convert_name_);
        AbortHelper::jmp_active = false;
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), result_));
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    std::string input_path_;
    std::string vae_path_;
    std::string output_path_;
    sd_type_t output_type_;
    std::string tensor_type_rules_;
    bool convert_name_;
    bool result_;
};
