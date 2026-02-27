#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <string>
#include <vector>

// Holds std::string values so that const char* pointers into them remain valid
// as long as the StringStore is alive.
struct StringStore {
    std::vector<std::string> strings;

    const char* add(const Napi::Object& obj, const char* key) {
        if (obj.Has(key)) {
            Napi::Value val = obj.Get(key);
            if (val.IsString()) {
                strings.push_back(val.As<Napi::String>().Utf8Value());
                return strings.back().c_str();
            }
        }
        return nullptr;
    }

    const char* add(const std::string& s) {
        strings.push_back(s);
        return strings.back().c_str();
    }
};

// Holds vectors that back pointer+count fields in C structs.
struct ArrayStore {
    std::vector<std::vector<sd_lora_t>> lora_arrays;
    std::vector<std::vector<sd_image_t>> image_arrays;
    std::vector<std::vector<uint8_t>> image_data_buffers;
    std::vector<std::vector<int>> int_arrays;
    std::vector<std::vector<float>> float_arrays;
    std::vector<std::vector<sd_embedding_t>> embedding_arrays;
};

namespace ParamsConverter {

// Extract a bool property with a default value.
inline bool GetBool(const Napi::Object& obj, const char* key, bool defaultVal) {
    if (obj.Has(key)) {
        Napi::Value val = obj.Get(key);
        if (val.IsBoolean()) return val.As<Napi::Boolean>().Value();
    }
    return defaultVal;
}

// Extract an int32 property with a default value.
inline int32_t GetInt(const Napi::Object& obj, const char* key, int32_t defaultVal) {
    if (obj.Has(key)) {
        Napi::Value val = obj.Get(key);
        if (val.IsNumber()) return val.As<Napi::Number>().Int32Value();
    }
    return defaultVal;
}

// Extract a double/float property with a default value.
inline double GetDouble(const Napi::Object& obj, const char* key, double defaultVal) {
    if (obj.Has(key)) {
        Napi::Value val = obj.Get(key);
        if (val.IsNumber()) return val.As<Napi::Number>().DoubleValue();
    }
    return defaultVal;
}

// Extract an int64 property with a default value.
inline int64_t GetInt64(const Napi::Object& obj, const char* key, int64_t defaultVal) {
    if (obj.Has(key)) {
        Napi::Value val = obj.Get(key);
        if (val.IsNumber()) return val.As<Napi::Number>().Int64Value();
    }
    return defaultVal;
}

// Convert a string enum value using the C API str_to_* functions.
template <typename EnumT>
inline EnumT GetEnum(const Napi::Object& obj, const char* key, EnumT defaultVal,
                     EnumT (*strToEnum)(const char*)) {
    if (obj.Has(key)) {
        Napi::Value val = obj.Get(key);
        if (val.IsString()) {
            std::string s = val.As<Napi::String>().Utf8Value();
            return strToEnum(s.c_str());
        }
    }
    return defaultVal;
}

// Convert JS options → sd_ctx_params_t
sd_ctx_params_t ToCtxParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as);

// Convert JS options → sd_sample_params_t
sd_sample_params_t ToSampleParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as);

// Convert JS options → sd_img_gen_params_t
sd_img_gen_params_t ToImgGenParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as);

// Convert JS options → sd_vid_gen_params_t
sd_vid_gen_params_t ToVidGenParams(const Napi::Object& opts, StringStore& ss, ArrayStore& as);

}  // namespace ParamsConverter
