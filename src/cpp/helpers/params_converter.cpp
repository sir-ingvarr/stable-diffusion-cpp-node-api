#include "params_converter.h"

#include "image_helpers.h"

#include <cmath>

namespace ParamsConverter {

static sd_tiling_params_t ToTilingParams(const Napi::Object& obj) {
    sd_tiling_params_t tp = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    if (!obj.Has("vaeTiling")) return tp;
    Napi::Value val = obj.Get("vaeTiling");
    if (!val.IsObject()) return tp;
    Napi::Object t = val.As<Napi::Object>();
    tp.enabled = GetBool(t, "enabled", false);
    tp.tile_size_x = GetInt(t, "tileSizeX", 0);
    tp.tile_size_y = GetInt(t, "tileSizeY", 0);
    tp.target_overlap = static_cast<float>(GetDouble(t, "targetOverlap", 0.5));
    tp.rel_size_x = static_cast<float>(GetDouble(t, "relSizeX", 0.0));
    tp.rel_size_y = static_cast<float>(GetDouble(t, "relSizeY", 0.0));
    return tp;
}

static sd_cache_params_t ToCacheParams(const Napi::Object& obj, StringStore& ss) {
    sd_cache_params_t cp;
    sd_cache_params_init(&cp);
    if (!obj.Has("cache")) return cp;
    Napi::Value val = obj.Get("cache");
    if (!val.IsObject()) return cp;
    Napi::Object c = val.As<Napi::Object>();

    // mode: string → enum
    if (c.Has("mode")) {
        Napi::Value mv = c.Get("mode");
        if (mv.IsString()) {
            std::string ms = mv.As<Napi::String>().Utf8Value();
            if (ms == "disabled") cp.mode = SD_CACHE_DISABLED;
            else if (ms == "easycache") cp.mode = SD_CACHE_EASYCACHE;
            else if (ms == "ucache") cp.mode = SD_CACHE_UCACHE;
            else if (ms == "dbcache") cp.mode = SD_CACHE_DBCACHE;
            else if (ms == "taylorseer") cp.mode = SD_CACHE_TAYLORSEER;
            else if (ms == "cachedit") cp.mode = SD_CACHE_CACHE_DIT;
            else if (ms == "spectrum") cp.mode = SD_CACHE_SPECTRUM;
        }
    }
    cp.reuse_threshold = static_cast<float>(GetDouble(c, "reuseThreshold", cp.reuse_threshold));
    cp.start_percent = static_cast<float>(GetDouble(c, "startPercent", cp.start_percent));
    cp.end_percent = static_cast<float>(GetDouble(c, "endPercent", cp.end_percent));
    cp.error_decay_rate = static_cast<float>(GetDouble(c, "errorDecayRate", cp.error_decay_rate));
    cp.use_relative_threshold = GetBool(c, "useRelativeThreshold", cp.use_relative_threshold);
    cp.reset_error_on_compute = GetBool(c, "resetErrorOnCompute", cp.reset_error_on_compute);
    cp.Fn_compute_blocks = GetInt(c, "fnComputeBlocks", cp.Fn_compute_blocks);
    cp.Bn_compute_blocks = GetInt(c, "bnComputeBlocks", cp.Bn_compute_blocks);
    cp.residual_diff_threshold = static_cast<float>(GetDouble(c, "residualDiffThreshold", cp.residual_diff_threshold));
    cp.max_warmup_steps = GetInt(c, "maxWarmupSteps", cp.max_warmup_steps);
    cp.max_cached_steps = GetInt(c, "maxCachedSteps", cp.max_cached_steps);
    cp.max_continuous_cached_steps = GetInt(c, "maxContinuousCachedSteps", cp.max_continuous_cached_steps);
    cp.taylorseer_n_derivatives = GetInt(c, "taylorseerNDerivatives", cp.taylorseer_n_derivatives);
    cp.taylorseer_skip_interval = GetInt(c, "taylorseerSkipInterval", cp.taylorseer_skip_interval);
    cp.scm_mask = ss.add(c, "scmMask");
    cp.scm_policy_dynamic = GetBool(c, "scmPolicyDynamic", cp.scm_policy_dynamic);
    cp.spectrum_w = static_cast<float>(GetDouble(c, "spectrumW", cp.spectrum_w));
    cp.spectrum_m = GetInt(c, "spectrumM", cp.spectrum_m);
    cp.spectrum_lam = static_cast<float>(GetDouble(c, "spectrumLam", cp.spectrum_lam));
    cp.spectrum_window_size = GetInt(c, "spectrumWindowSize", cp.spectrum_window_size);
    cp.spectrum_flex_window = static_cast<float>(GetDouble(c, "spectrumFlexWindow", cp.spectrum_flex_window));
    cp.spectrum_warmup_steps = GetInt(c, "spectrumWarmupSteps", cp.spectrum_warmup_steps);
    cp.spectrum_stop_percent = static_cast<float>(GetDouble(c, "spectrumStopPercent", cp.spectrum_stop_percent));
    return cp;
}

static sd_hires_params_t ToHiresParams(const Napi::Object& obj, StringStore& ss) {
    sd_hires_params_t hp;
    sd_hires_params_init(&hp);
    if (!obj.Has("hires")) return hp;
    Napi::Value val = obj.Get("hires");
    if (!val.IsObject()) return hp;
    Napi::Object h = val.As<Napi::Object>();

    hp.enabled = GetBool(h, "enabled", hp.enabled);
    // Upstream's str_to_sd_hires_upscaler matches against display names
    // ("Lanczos", "Latent (nearest)", ...) — not the snake_case tokens we
    // expose. Dispatch ourselves so JS users get the clean names.
    if (h.Has("upscaler")) {
        Napi::Value uv = h.Get("upscaler");
        if (uv.IsString()) {
            std::string us = uv.As<Napi::String>().Utf8Value();
            if (us == "none") hp.upscaler = SD_HIRES_UPSCALER_NONE;
            else if (us == "latent") hp.upscaler = SD_HIRES_UPSCALER_LATENT;
            else if (us == "latent_nearest") hp.upscaler = SD_HIRES_UPSCALER_LATENT_NEAREST;
            else if (us == "latent_nearest_exact") hp.upscaler = SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT;
            else if (us == "latent_antialiased") hp.upscaler = SD_HIRES_UPSCALER_LATENT_ANTIALIASED;
            else if (us == "latent_bicubic") hp.upscaler = SD_HIRES_UPSCALER_LATENT_BICUBIC;
            else if (us == "latent_bicubic_antialiased") hp.upscaler = SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED;
            else if (us == "lanczos") hp.upscaler = SD_HIRES_UPSCALER_LANCZOS;
            else if (us == "nearest") hp.upscaler = SD_HIRES_UPSCALER_NEAREST;
            else if (us == "model") hp.upscaler = SD_HIRES_UPSCALER_MODEL;
        }
    }
    hp.model_path = ss.add(h, "modelPath");
    hp.scale = static_cast<float>(GetDouble(h, "scale", hp.scale));
    hp.target_width = GetInt(h, "targetWidth", hp.target_width);
    hp.target_height = GetInt(h, "targetHeight", hp.target_height);
    hp.steps = GetInt(h, "steps", hp.steps);
    hp.denoising_strength = static_cast<float>(GetDouble(h, "denoisingStrength", hp.denoising_strength));
    hp.upscale_tile_size = GetInt(h, "upscaleTileSize", hp.upscale_tile_size);
    return hp;
}

static void ExtractLoras(const Napi::Object& obj, StringStore& ss, ArrayStore& as,
                         const sd_lora_t** out_loras, uint32_t* out_count) {
    *out_loras = nullptr;
    *out_count = 0;
    if (!obj.Has("loras")) return;
    Napi::Value val = obj.Get("loras");
    if (!val.IsArray()) return;
    Napi::Array arr = val.As<Napi::Array>();
    uint32_t len = arr.Length();
    if (len == 0) return;

    as.lora_arrays.emplace_back();
    auto& vec = as.lora_arrays.back();
    vec.resize(len);

    for (uint32_t i = 0; i < len; i++) {
        Napi::Object lo = arr.Get(i).As<Napi::Object>();
        vec[i].path = ss.add(lo, "path");
        vec[i].multiplier = static_cast<float>(GetDouble(lo, "multiplier", 1.0));
        vec[i].is_high_noise = GetBool(lo, "isHighNoise", false);
    }
    *out_loras = vec.data();
    *out_count = len;
}

static sd_slg_params_t ToSlgParams(const Napi::Object& obj, ArrayStore& as) {
    sd_slg_params_t slg = {};
    slg.layer_count = 0;
    slg.layer_start = 0.01f;
    slg.layer_end = 0.2f;
    slg.scale = 0.f;
    if (!obj.Has("slg")) return slg;
    Napi::Value val = obj.Get("slg");
    if (!val.IsObject()) return slg;
    Napi::Object s = val.As<Napi::Object>();

    slg.layer_start = static_cast<float>(GetDouble(s, "layerStart", slg.layer_start));
    slg.layer_end = static_cast<float>(GetDouble(s, "layerEnd", slg.layer_end));
    slg.scale = static_cast<float>(GetDouble(s, "scale", slg.scale));

    if (s.Has("layers") && s.Get("layers").IsArray()) {
        Napi::Array layers = s.Get("layers").As<Napi::Array>();
        uint32_t len = layers.Length();
        if (len > 0) {
            as.int_arrays.emplace_back();
            auto& vec = as.int_arrays.back();
            vec.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                vec[i] = layers.Get(i).As<Napi::Number>().Int32Value();
            }
            slg.layers = vec.data();
            slg.layer_count = len;
        }
    }
    return slg;
}

sd_sample_params_t ToSampleParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as) {
    sd_sample_params_t sp;
    sd_sample_params_init(&sp);

    if (!opts.Has("sampleParams")) return sp;
    Napi::Value val = opts.Get("sampleParams");
    if (!val.IsObject()) return sp;
    Napi::Object s = val.As<Napi::Object>();

    // guidance
    if (s.Has("guidance") && s.Get("guidance").IsObject()) {
        Napi::Object g = s.Get("guidance").As<Napi::Object>();
        sp.guidance.txt_cfg = static_cast<float>(GetDouble(g, "txtCfg", sp.guidance.txt_cfg));
        sp.guidance.img_cfg = static_cast<float>(GetDouble(g, "imgCfg", sp.guidance.img_cfg));
        sp.guidance.distilled_guidance = static_cast<float>(GetDouble(g, "distilledGuidance", sp.guidance.distilled_guidance));
        sp.guidance.slg = ToSlgParams(g, as);
    }

    sp.scheduler = GetEnum<scheduler_t>(s, "scheduler", sp.scheduler, str_to_scheduler);
    sp.sample_method = GetEnum<sample_method_t>(s, "sampleMethod", sp.sample_method, str_to_sample_method);
    sp.sample_steps = GetInt(s, "sampleSteps", sp.sample_steps);
    sp.eta = static_cast<float>(GetDouble(s, "eta", sp.eta));
    sp.shifted_timestep = GetInt(s, "shiftedTimestep", sp.shifted_timestep);
    sp.flow_shift = static_cast<float>(GetDouble(s, "flowShift", sp.flow_shift));

    // custom sigmas
    if (s.Has("customSigmas") && s.Get("customSigmas").IsArray()) {
        Napi::Array sigArr = s.Get("customSigmas").As<Napi::Array>();
        uint32_t len = sigArr.Length();
        if (len > 0) {
            as.float_arrays.emplace_back();
            auto& vec = as.float_arrays.back();
            vec.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                vec[i] = sigArr.Get(i).As<Napi::Number>().FloatValue();
            }
            sp.custom_sigmas = vec.data();
            sp.custom_sigmas_count = static_cast<int>(len);
        }
    }

    return sp;
}

static sd_pm_params_t ToPmParams(const Napi::Object& obj, StringStore& ss, ArrayStore& as) {
    sd_pm_params_t pm = {nullptr, 0, nullptr, 20.f};
    if (!obj.Has("photoMaker")) return pm;
    Napi::Value val = obj.Get("photoMaker");
    if (!val.IsObject()) return pm;
    Napi::Object p = val.As<Napi::Object>();

    pm.id_embed_path = ss.add(p, "idEmbedPath");
    pm.style_strength = static_cast<float>(GetDouble(p, "styleStrength", 20.f));

    if (p.Has("idImages") && p.Get("idImages").IsArray()) {
        Napi::Array arr = p.Get("idImages").As<Napi::Array>();
        uint32_t len = arr.Length();
        if (len > 0) {
            as.image_arrays.emplace_back();
            auto& images = as.image_arrays.back();
            images.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                images[i] = ImageHelpers::ExtractImage(arr.Get(i).As<Napi::Object>(), as);
            }
            pm.id_images = images.data();
            pm.id_images_count = static_cast<int>(len);
        }
    }
    return pm;
}

sd_ctx_params_t ToCtxParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as) {
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);

    p.model_path = ss.add(opts, "modelPath");
    p.clip_l_path = ss.add(opts, "clipLPath");
    p.clip_g_path = ss.add(opts, "clipGPath");
    p.clip_vision_path = ss.add(opts, "clipVisionPath");
    p.t5xxl_path = ss.add(opts, "t5xxlPath");
    p.llm_path = ss.add(opts, "llmPath");
    p.llm_vision_path = ss.add(opts, "llmVisionPath");
    p.diffusion_model_path = ss.add(opts, "diffusionModelPath");
    p.high_noise_diffusion_model_path = ss.add(opts, "highNoiseDiffusionModelPath");
    p.vae_path = ss.add(opts, "vaePath");
    p.taesd_path = ss.add(opts, "taesdPath");
    p.control_net_path = ss.add(opts, "controlNetPath");
    p.photo_maker_path = ss.add(opts, "photoMakerPath");
    p.tensor_type_rules = ss.add(opts, "tensorTypeRules");

    // embeddings
    if (opts.Has("embeddings") && opts.Get("embeddings").IsArray()) {
        Napi::Array arr = opts.Get("embeddings").As<Napi::Array>();
        uint32_t len = arr.Length();
        if (len > 0) {
            as.embedding_arrays.emplace_back();
            auto& vec = as.embedding_arrays.back();
            vec.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                Napi::Object e = arr.Get(i).As<Napi::Object>();
                vec[i].name = ss.add(e, "name");
                vec[i].path = ss.add(e, "path");
            }
            p.embeddings = vec.data();
            p.embedding_count = len;
        }
    }

    p.vae_decode_only = GetBool(opts, "vaeDecodeOnly", p.vae_decode_only);
    // Override upstream's default of true: we promise context reuse in API.md, and
    // true makes the second generate_image call assert on the Metal backend (sd.cpp #1298).
    p.free_params_immediately = GetBool(opts, "freeParamsImmediately", false);
    p.n_threads = GetInt(opts, "nThreads", p.n_threads);
    p.wtype = GetEnum<sd_type_t>(opts, "wtype", p.wtype, str_to_sd_type);
    p.rng_type = GetEnum<rng_type_t>(opts, "rngType", p.rng_type, str_to_rng_type);
    p.sampler_rng_type = GetEnum<rng_type_t>(opts, "samplerRngType", p.sampler_rng_type, str_to_rng_type);
    p.prediction = GetEnum<prediction_t>(opts, "prediction", p.prediction, str_to_prediction);
    p.lora_apply_mode = GetEnum<lora_apply_mode_t>(opts, "loraApplyMode", p.lora_apply_mode, str_to_lora_apply_mode);
    p.offload_params_to_cpu = GetBool(opts, "offloadParamsToCpu", p.offload_params_to_cpu);
    p.enable_mmap = GetBool(opts, "enableMmap", p.enable_mmap);
    p.keep_clip_on_cpu = GetBool(opts, "keepClipOnCpu", p.keep_clip_on_cpu);
    p.keep_control_net_on_cpu = GetBool(opts, "keepControlNetOnCpu", p.keep_control_net_on_cpu);
    p.keep_vae_on_cpu = GetBool(opts, "keepVaeOnCpu", p.keep_vae_on_cpu);
    p.flash_attn = GetBool(opts, "flashAttn", p.flash_attn);
    p.diffusion_flash_attn = GetBool(opts, "diffusionFlashAttn", p.diffusion_flash_attn);
    p.tae_preview_only = GetBool(opts, "taePreviewOnly", p.tae_preview_only);
    p.diffusion_conv_direct = GetBool(opts, "diffusionConvDirect", p.diffusion_conv_direct);
    p.vae_conv_direct = GetBool(opts, "vaeConvDirect", p.vae_conv_direct);
    p.circular_x = GetBool(opts, "circularX", p.circular_x);
    p.circular_y = GetBool(opts, "circularY", p.circular_y);
    p.force_sdxl_vae_conv_scale = GetBool(opts, "forceSdxlVaeConvScale", p.force_sdxl_vae_conv_scale);
    p.chroma_use_dit_mask = GetBool(opts, "chromaUseDitMask", p.chroma_use_dit_mask);
    p.chroma_use_t5_mask = GetBool(opts, "chromaUseT5Mask", p.chroma_use_t5_mask);
    p.chroma_t5_mask_pad = GetInt(opts, "chromaT5MaskPad", p.chroma_t5_mask_pad);
    p.qwen_image_zero_cond_t = GetBool(opts, "qwenImageZeroCondT", p.qwen_image_zero_cond_t);

    return p;
}

sd_img_gen_params_t ToImgGenParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as) {
    sd_img_gen_params_t p;
    sd_img_gen_params_init(&p);

    ExtractLoras(opts, ss, as, &p.loras, &p.lora_count);

    p.prompt = ss.add(opts, "prompt");
    p.negative_prompt = ss.add(opts, "negativePrompt");
    p.clip_skip = GetInt(opts, "clipSkip", p.clip_skip);
    p.width = GetInt(opts, "width", p.width);
    p.height = GetInt(opts, "height", p.height);
    p.strength = static_cast<float>(GetDouble(opts, "strength", p.strength));
    p.seed = GetInt64(opts, "seed", p.seed);
    p.batch_count = GetInt(opts, "batchCount", p.batch_count);
    p.control_strength = static_cast<float>(GetDouble(opts, "controlStrength", p.control_strength));
    p.auto_resize_ref_image = GetBool(opts, "autoResizeRefImage", p.auto_resize_ref_image);
    p.increase_ref_index = GetBool(opts, "increaseRefIndex", p.increase_ref_index);

    // init_image
    if (opts.Has("initImage") && opts.Get("initImage").IsObject()) {
        p.init_image = ImageHelpers::ExtractImage(opts.Get("initImage").As<Napi::Object>(), as);
    }
    // mask_image
    if (opts.Has("maskImage") && opts.Get("maskImage").IsObject()) {
        p.mask_image = ImageHelpers::ExtractImage(opts.Get("maskImage").As<Napi::Object>(), as);
    }
    // control_image
    if (opts.Has("controlImage") && opts.Get("controlImage").IsObject()) {
        p.control_image = ImageHelpers::ExtractImage(opts.Get("controlImage").As<Napi::Object>(), as);
    }
    // ref_images
    if (opts.Has("refImages") && opts.Get("refImages").IsArray()) {
        Napi::Array arr = opts.Get("refImages").As<Napi::Array>();
        uint32_t len = arr.Length();
        if (len > 0) {
            as.image_arrays.emplace_back();
            auto& images = as.image_arrays.back();
            images.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                images[i] = ImageHelpers::ExtractImage(arr.Get(i).As<Napi::Object>(), as);
            }
            p.ref_images = images.data();
            p.ref_images_count = static_cast<int>(len);
        }
    }

    p.sample_params = ToSampleParams(opts, ss, as);
    p.pm_params = ToPmParams(opts, ss, as);
    p.vae_tiling_params = ToTilingParams(opts);
    p.cache = ToCacheParams(opts, ss);
    p.hires = ToHiresParams(opts, ss);

    return p;
}

sd_vid_gen_params_t ToVidGenParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as) {
    sd_vid_gen_params_t p;
    sd_vid_gen_params_init(&p);

    ExtractLoras(opts, ss, as, &p.loras, &p.lora_count);

    p.prompt = ss.add(opts, "prompt");
    p.negative_prompt = ss.add(opts, "negativePrompt");
    p.clip_skip = GetInt(opts, "clipSkip", -1);
    p.width = GetInt(opts, "width", p.width);
    p.height = GetInt(opts, "height", p.height);
    p.strength = static_cast<float>(GetDouble(opts, "strength", p.strength));
    p.seed = GetInt64(opts, "seed", p.seed);
    p.video_frames = GetInt(opts, "videoFrames", p.video_frames);
    p.moe_boundary = static_cast<float>(GetDouble(opts, "moeBoundary", p.moe_boundary));
    p.vace_strength = static_cast<float>(GetDouble(opts, "vaceStrength", p.vace_strength));

    // init_image
    if (opts.Has("initImage") && opts.Get("initImage").IsObject()) {
        p.init_image = ImageHelpers::ExtractImage(opts.Get("initImage").As<Napi::Object>(), as);
    }
    // end_image
    if (opts.Has("endImage") && opts.Get("endImage").IsObject()) {
        p.end_image = ImageHelpers::ExtractImage(opts.Get("endImage").As<Napi::Object>(), as);
    }
    // control_frames
    if (opts.Has("controlFrames") && opts.Get("controlFrames").IsArray()) {
        Napi::Array arr = opts.Get("controlFrames").As<Napi::Array>();
        uint32_t len = arr.Length();
        if (len > 0) {
            as.image_arrays.emplace_back();
            auto& images = as.image_arrays.back();
            images.resize(len);
            for (uint32_t i = 0; i < len; i++) {
                images[i] = ImageHelpers::ExtractImage(arr.Get(i).As<Napi::Object>(), as);
            }
            p.control_frames = images.data();
            p.control_frames_size = static_cast<int>(len);
        }
    }

    p.sample_params = ToSampleParams(opts, ss, as);

    // high noise sample params
    if (opts.Has("highNoiseSampleParams") && opts.Get("highNoiseSampleParams").IsObject()) {
        Napi::Object hOpts = Napi::Object::New(opts.Env());
        hOpts.Set("sampleParams", opts.Get("highNoiseSampleParams"));
        p.high_noise_sample_params = ToSampleParams(hOpts, ss, as);
    }

    p.vae_tiling_params = ToTilingParams(opts);
    p.cache = ToCacheParams(opts, ss);

    return p;
}

}  // namespace ParamsConverter
