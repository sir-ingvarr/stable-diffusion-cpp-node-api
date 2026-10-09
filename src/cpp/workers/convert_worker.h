#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <string>
#include <utility>
#include <vector>

#include "helpers/params_converter.h"

// Inputs for a model conversion. When any component path (clipL / clipG /
// t5xxl / diffusionModel), thread count, or lora list is supplied, the
// multi-component API (convert_with_components) is used; otherwise the
// classic single-file convert().
struct ConvertRequest {
    std::string input_path;        // model_path for the components API
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string output_path;
    sd_type_t output_type = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    bool convert_name = false;
    int n_threads = -1;
    bool use_components = false;
    // Lora paths + multipliers baked into the converted model (components API).
    std::vector<std::string> lora_paths;
    std::vector<float> lora_multipliers;
};

class ConvertWorker : public Napi::AsyncWorker {
  public:
    ConvertWorker(Napi::Env env, ConvertRequest req)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          req_(std::move(req)),
          result_(false) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        auto cstr = [](const std::string& s) -> const char* {
            return s.empty() ? nullptr : s.c_str();
        };
        if (req_.use_components) {
            std::vector<sd_lora_t> loras;
            loras.reserve(req_.lora_paths.size());
            for (size_t i = 0; i < req_.lora_paths.size(); i++) {
                sd_lora_t l = {};
                l.path = req_.lora_paths[i].c_str();
                l.multiplier = req_.lora_multipliers[i];
                l.is_high_noise = false;
                loras.push_back(l);
            }
            result_ = convert_with_components(
                cstr(req_.input_path),
                cstr(req_.clip_l_path),
                cstr(req_.clip_g_path),
                cstr(req_.t5xxl_path),
                cstr(req_.diffusion_model_path),
                cstr(req_.vae_path),
                req_.output_path.c_str(),
                req_.output_type,
                cstr(req_.tensor_type_rules),
                req_.convert_name,
                req_.n_threads,
                loras.empty() ? nullptr : loras.data(),
                static_cast<int>(loras.size()));
        } else {
            result_ = convert(
                req_.input_path.c_str(),
                cstr(req_.vae_path),
                req_.output_path.c_str(),
                req_.output_type,
                cstr(req_.tensor_type_rules),
                req_.convert_name);
        }
    }

    void OnOK() override {
        deferred_.Resolve(Napi::Boolean::New(Env(), result_));
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    Napi::Promise::Deferred deferred_;
    ConvertRequest req_;
    bool result_;
};
