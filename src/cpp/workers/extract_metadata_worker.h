#pragma once

#include <napi.h>

#include <string>
#include <utility>
#include <vector>

#include "src/model.h"

class ExtractMetadataWorker : public Napi::AsyncWorker {
  public:
    ExtractMetadataWorker(Napi::Env env, std::string path)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          path_(std::move(path)) {}

    Napi::Promise::Deferred& Deferred() { return deferred_; }

    void Execute() override {
        try {
            ModelLoader loader;
            if (!loader.init_from_file(path_)) {
                SetError("Failed to read model metadata from: " + path_);
                return;
            }

            version_ = loader.get_sd_version();

            for (const auto& [name, ts] : loader.get_tensor_storage_map()) {
                tensor_count_++;

                if (!has_diffusion_model_ &&
                    (name.find("model.diffusion_model.") != std::string::npos ||
                     name.find("unet.") != std::string::npos)) {
                    has_diffusion_model_ = true;
                }
                if (!has_vae_ &&
                    (name.find("first_stage_model.") != std::string::npos ||
                     name.find("vae.") != std::string::npos)) {
                    has_vae_ = true;
                }
                if (!has_clip_l_ &&
                    (name.find("text_encoders.clip_l.") != std::string::npos ||
                     name.find("conditioner.embedders.0.") != std::string::npos ||
                     name.find("cond_stage_model.transformer.") != std::string::npos ||
                     name.find("te.text_model.") != std::string::npos)) {
                    has_clip_l_ = true;
                }
                if (!has_clip_g_ &&
                    (name.find("text_encoders.clip_g.") != std::string::npos ||
                     name.find("conditioner.embedders.1.") != std::string::npos ||
                     name.find("cond_stage_model.1.") != std::string::npos ||
                     name.find("te.1.") != std::string::npos)) {
                    has_clip_g_ = true;
                }
                if (!has_t5xxl_ &&
                    name.find("text_encoders.t5xxl.") != std::string::npos) {
                    has_t5xxl_ = true;
                }
                if (!has_control_net_ &&
                    name.find("control_model.") != std::string::npos) {
                    has_control_net_ = true;
                }
                if (!is_lora_ &&
                    (name.find(".lora_up.") != std::string::npos ||
                     name.find(".lora_down.") != std::string::npos ||
                     name.find(".lora_A.") != std::string::npos ||
                     name.find(".lora_B.") != std::string::npos)) {
                    is_lora_ = true;
                }
            }

            for (auto& kv : loader.get_wtype_stat()) {
                weight_types_.emplace_back(kv.first, kv.second);
            }
            for (auto& kv : loader.get_diffusion_model_wtype_stat()) {
                diffusion_weight_types_.emplace_back(kv.first, kv.second);
            }
            for (auto& kv : loader.get_vae_wtype_stat()) {
                vae_weight_types_.emplace_back(kv.first, kv.second);
            }
            for (auto& kv : loader.get_conditioner_wtype_stat()) {
                conditioner_weight_types_.emplace_back(kv.first, kv.second);
            }

            est_params_bytes_ = loader.get_params_mem_size(nullptr, GGML_TYPE_COUNT);
        } catch (const std::exception& e) {
            SetError(e.what());
        } catch (...) {
            SetError("Failed to read model metadata (unknown exception)");
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        auto out = Napi::Object::New(env);
        out.Set("version", Napi::String::New(env, versionSlug(version_)));
        out.Set("versionLabel", Napi::String::New(env, versionLabel(version_)));

        out.Set("isUnet", Napi::Boolean::New(env, sd_version_is_unet(version_)));
        out.Set("isDit", Napi::Boolean::New(env, sd_version_is_dit(version_)));
        out.Set("isSd1", Napi::Boolean::New(env, sd_version_is_sd1(version_)));
        out.Set("isSd2", Napi::Boolean::New(env, sd_version_is_sd2(version_)));
        out.Set("isSdxl", Napi::Boolean::New(env, sd_version_is_sdxl(version_)));
        out.Set("isSd3", Napi::Boolean::New(env, sd_version_is_sd3(version_)));
        out.Set("isFlux", Napi::Boolean::New(env, sd_version_is_flux(version_)));
        out.Set("isFlux2", Napi::Boolean::New(env, sd_version_is_flux2(version_)));
        out.Set("isWan", Napi::Boolean::New(env, sd_version_is_wan(version_)));
        out.Set("isQwenImage", Napi::Boolean::New(env, sd_version_is_qwen_image(version_)));
        out.Set("isZImage", Napi::Boolean::New(env, sd_version_is_z_image(version_)));
        out.Set("isInpaint", Napi::Boolean::New(env, sd_version_is_inpaint(version_)));
        out.Set("isControl", Napi::Boolean::New(env, sd_version_is_control(version_)));

        out.Set("hasDiffusionModel", Napi::Boolean::New(env, has_diffusion_model_));
        out.Set("hasVae", Napi::Boolean::New(env, has_vae_));
        out.Set("hasClipL", Napi::Boolean::New(env, has_clip_l_));
        out.Set("hasClipG", Napi::Boolean::New(env, has_clip_g_));
        out.Set("hasT5xxl", Napi::Boolean::New(env, has_t5xxl_));
        out.Set("hasControlNet", Napi::Boolean::New(env, has_control_net_));
        out.Set("isLora", Napi::Boolean::New(env, is_lora_));

        out.Set("tensorCount", Napi::Number::New(env, tensor_count_));
        out.Set("estParamsBytes", Napi::Number::New(env, static_cast<double>(est_params_bytes_)));

        out.Set("weightTypes", buildTypeMap(env, weight_types_));
        out.Set("diffusionWeightTypes", buildTypeMap(env, diffusion_weight_types_));
        out.Set("vaeWeightTypes", buildTypeMap(env, vae_weight_types_));
        out.Set("conditionerWeightTypes", buildTypeMap(env, conditioner_weight_types_));

        deferred_.Resolve(out);
    }

    void OnError(const Napi::Error& e) override {
        deferred_.Reject(e.Value());
    }

  private:
    static const char* versionSlug(SDVersion v) {
        switch (v) {
            case VERSION_SD1: return "sd1";
            case VERSION_SD1_INPAINT: return "sd1_inpaint";
            case VERSION_SD1_PIX2PIX: return "sd1_pix2pix";
            case VERSION_SD1_TINY_UNET: return "sd1_tiny_unet";
            case VERSION_SD2: return "sd2";
            case VERSION_SD2_INPAINT: return "sd2_inpaint";
            case VERSION_SD2_TINY_UNET: return "sd2_tiny_unet";
            case VERSION_SDXS: return "sdxs";
            case VERSION_SDXL: return "sdxl";
            case VERSION_SDXL_INPAINT: return "sdxl_inpaint";
            case VERSION_SDXL_PIX2PIX: return "sdxl_pix2pix";
            case VERSION_SDXL_VEGA: return "sdxl_vega";
            case VERSION_SDXL_SSD1B: return "sdxl_ssd1b";
            case VERSION_SVD: return "svd";
            case VERSION_SD3: return "sd3";
            case VERSION_FLUX: return "flux";
            case VERSION_FLUX_FILL: return "flux_fill";
            case VERSION_FLUX_CONTROLS: return "flux_controls";
            case VERSION_FLEX_2: return "flex2";
            case VERSION_CHROMA_RADIANCE: return "chroma_radiance";
            case VERSION_WAN2: return "wan2";
            case VERSION_WAN2_2_I2V: return "wan2_2_i2v";
            case VERSION_WAN2_2_TI2V: return "wan2_2_ti2v";
            case VERSION_QWEN_IMAGE: return "qwen_image";
            case VERSION_FLUX2: return "flux2";
            case VERSION_FLUX2_KLEIN: return "flux2_klein";
            case VERSION_Z_IMAGE: return "z_image";
            case VERSION_OVIS_IMAGE: return "ovis_image";
            default: return "unknown";
        }
    }

    static const char* versionLabel(SDVersion v) {
        switch (v) {
            case VERSION_SD1: return "SD 1.x";
            case VERSION_SD1_INPAINT: return "SD 1.x Inpaint";
            case VERSION_SD1_PIX2PIX: return "Instruct-Pix2Pix";
            case VERSION_SD1_TINY_UNET: return "SD 1.x Tiny UNet";
            case VERSION_SD2: return "SD 2.x";
            case VERSION_SD2_INPAINT: return "SD 2.x Inpaint";
            case VERSION_SD2_TINY_UNET: return "SD 2.x Tiny UNet";
            case VERSION_SDXS: return "SDXS";
            case VERSION_SDXL: return "SDXL";
            case VERSION_SDXL_INPAINT: return "SDXL Inpaint";
            case VERSION_SDXL_PIX2PIX: return "SDXL Instruct-Pix2Pix";
            case VERSION_SDXL_VEGA: return "SDXL (Vega)";
            case VERSION_SDXL_SSD1B: return "SDXL (SSD1B)";
            case VERSION_SVD: return "SVD";
            case VERSION_SD3: return "SD3.x";
            case VERSION_FLUX: return "Flux";
            case VERSION_FLUX_FILL: return "Flux Fill";
            case VERSION_FLUX_CONTROLS: return "Flux Control";
            case VERSION_FLEX_2: return "Flex.2";
            case VERSION_CHROMA_RADIANCE: return "Chroma Radiance";
            case VERSION_WAN2: return "Wan 2.x";
            case VERSION_WAN2_2_I2V: return "Wan 2.2 I2V";
            case VERSION_WAN2_2_TI2V: return "Wan 2.2 TI2V";
            case VERSION_QWEN_IMAGE: return "Qwen Image";
            case VERSION_FLUX2: return "Flux.2";
            case VERSION_FLUX2_KLEIN: return "Flux.2 klein";
            case VERSION_Z_IMAGE: return "Z-Image";
            case VERSION_OVIS_IMAGE: return "Ovis Image";
            default: return "Unknown";
        }
    }

    static Napi::Object buildTypeMap(Napi::Env env,
                                     const std::vector<std::pair<ggml_type, uint32_t>>& stats) {
        auto obj = Napi::Object::New(env);
        for (const auto& [t, c] : stats) {
            const char* name = ggml_type_name(t);
            obj.Set(name ? name : "unknown", Napi::Number::New(env, c));
        }
        return obj;
    }

    Napi::Promise::Deferred deferred_;
    std::string path_;

    SDVersion version_ = VERSION_COUNT;
    int tensor_count_ = 0;
    bool has_diffusion_model_ = false;
    bool has_vae_ = false;
    bool has_clip_l_ = false;
    bool has_clip_g_ = false;
    bool has_t5xxl_ = false;
    bool has_control_net_ = false;
    bool is_lora_ = false;
    int64_t est_params_bytes_ = 0;

    std::vector<std::pair<ggml_type, uint32_t>> weight_types_;
    std::vector<std::pair<ggml_type, uint32_t>> diffusion_weight_types_;
    std::vector<std::pair<ggml_type, uint32_t>> vae_weight_types_;
    std::vector<std::pair<ggml_type, uint32_t>> conditioner_weight_types_;
};
