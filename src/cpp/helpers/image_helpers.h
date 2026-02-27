#pragma once

#include <napi.h>
#include <stable-diffusion.h>

#include <cstring>
#include <vector>

#include "params_converter.h"

namespace ImageHelpers {

// Extract an sd_image_t from a JS object { width, height, channel, data: Buffer }.
// Copies the buffer data into ArrayStore so it remains valid for async workers.
inline sd_image_t ExtractImage(const Napi::Object& obj, ArrayStore& as) {
    sd_image_t img = {};
    img.width = obj.Get("width").As<Napi::Number>().Uint32Value();
    img.height = obj.Get("height").As<Napi::Number>().Uint32Value();
    img.channel = obj.Get("channel").As<Napi::Number>().Uint32Value();

    Napi::Buffer<uint8_t> buf = obj.Get("data").As<Napi::Buffer<uint8_t>>();
    size_t len = buf.Length();

    // Store a copy in ArrayStore
    as.image_data_buffers.emplace_back(buf.Data(), buf.Data() + len);
    img.data = as.image_data_buffers.back().data();
    return img;
}

// Convert an sd_image_t to a JS object { width, height, channel, data: Buffer }.
// Uses Napi::Buffer::New with a free callback for zero-copy transfer.
inline Napi::Object ImageToJS(Napi::Env env, const sd_image_t& img) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("width", Napi::Number::New(env, img.width));
    obj.Set("height", Napi::Number::New(env, img.height));
    obj.Set("channel", Napi::Number::New(env, img.channel));

    size_t dataLen = static_cast<size_t>(img.width) * img.height * img.channel;
    // Copy into V8-managed memory — Electron disallows external (C-owned) buffers.
    // Free the C-library allocation immediately after copying.
    Napi::Buffer<uint8_t> buffer = Napi::Buffer<uint8_t>::Copy(env, img.data, dataLen);
    free(img.data);
    obj.Set("data", buffer);
    return obj;
}

}  // namespace ImageHelpers
