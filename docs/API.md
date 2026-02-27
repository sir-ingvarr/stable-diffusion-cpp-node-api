# API Reference

Complete API documentation for `node-stable-diffusion-cpp`.

All async functions return Promises. All heavy operations (model loading, generation, upscaling, conversion) run on background threads and do not block the Node.js event loop.

## Table of contents

- [Classes](#classes)
  - [StableDiffusionContext](#stablediffusioncontext)
  - [UpscalerContext](#upscalercontext)
- [Free functions](#free-functions)
  - [convert](#convertoptions)
  - [preprocessCanny](#preprocesscannyimage-options)
  - [setLogCallback](#setlogcallbackcallback)
  - [setProgressCallback](#setprogresscallbackcallback)
  - [setPreviewCallback](#setpreviewcallbackcallback-options)
  - [getSystemInfo](#getsysteminfo)
  - [getNumPhysicalCores](#getnumphysicalcores)
  - [version](#version)
  - [commit](#commit)
- [Types](#types)
  - [SdImage](#sdimage)
  - [ContextOptions](#contextoptions)
  - [ImageGenerationOptions](#imagegenerationoptions)
  - [VideoGenerationOptions](#videogenerationoptions)
  - [SampleParams](#sampleparams)
  - [GuidanceParams](#guidanceparams)
  - [CacheParams](#cacheparams)
  - [TilingParams](#tilingparams)
  - [PhotoMakerParams](#photomakerparams)
  - [LoraDefinition](#loradefinition)
  - [EmbeddingDefinition](#embeddingdefinition)
  - [UpscalerOptions](#upscaleroptions)
  - [ConvertOptions](#convertoptions-1)
  - [CannyOptions](#cannyoptions)
  - [PreviewOptions](#previewoptions)
- [Enums (string literal types)](#enums)

---

## Classes

### StableDiffusionContext

The main class for image and video generation. Wraps a loaded model and all its components (text encoders, diffusion model, VAE, etc.).

Contexts are expensive to create (model loading) and should be reused for multiple generations.

#### `StableDiffusionContext.create(options)`

Static async factory method. Loads a model on a background thread.

```javascript
const ctx = await sd.StableDiffusionContext.create({
    modelPath: 'model.gguf',           // single-file model (ckpt, safetensors, full gguf)
    vaePath: 'vae.gguf',              // optional separate VAE
    clipLPath: 'clip_l.gguf',         // optional separate CLIP-L
    nThreads: 8,
});
```

**Parameters:**
- `options` — [`ContextOptions`](#contextoptions) object

**Returns:** `Promise<StableDiffusionContext>`

**Throws:** Rejects if the model fails to load (file not found, corrupt, out of memory, etc.)

---

#### `ctx.generateImage(options)`

Generate one or more images. Supports txt2img, img2img, inpainting, ControlNet, LoRA, and PhotoMaker.

```javascript
const images = await ctx.generateImage({
    prompt: 'a painting of a sunset over the ocean',
    negativePrompt: 'blurry, bad quality',
    width: 768,
    height: 512,
    seed: 42,
    batchCount: 2,
    sampleParams: {
        sampleSteps: 25,
        guidance: { txtCfg: 7.5 },
        scheduler: 'karras',
        sampleMethod: 'euler_a',
    },
});

console.log(images.length);        // 2
console.log(images[0].width);      // 768
console.log(images[0].data.length); // 768 * 512 * 3
```

**Parameters:**
- `options` — [`ImageGenerationOptions`](#imagegenerationoptions) object

**Returns:** `Promise<SdImage[]>` — array of length `batchCount`

---

#### `ctx.generateVideo(options)`

Generate video frames. Works with Wan2.1/2.2 models.

```javascript
const frames = await ctx.generateVideo({
    prompt: 'a timelapse of clouds moving over mountains',
    width: 512,
    height: 512,
    videoFrames: 16,
    seed: 42,
    sampleParams: {
        sampleSteps: 30,
    },
});

console.log(frames.length); // number of generated frames
```

**Parameters:**
- `options` — [`VideoGenerationOptions`](#videogenerationoptions) object

**Returns:** `Promise<SdImage[]>` — array of frames

---

#### `ctx.getDefaultSampleMethod()`

Returns the model's preferred sampling method.

```javascript
const method = ctx.getDefaultSampleMethod(); // e.g. 'euler'
```

**Returns:** `string` — one of the [`SampleMethod`](#samplemethod) values

---

#### `ctx.getDefaultScheduler(sampleMethod?)`

Returns the model's preferred scheduler, optionally for a specific sampling method.

```javascript
const scheduler = ctx.getDefaultScheduler();           // for default method
const scheduler2 = ctx.getDefaultScheduler('euler_a');  // for specific method
```

**Parameters:**
- `sampleMethod` *(optional)* — a [`SampleMethod`](#samplemethod) string

**Returns:** `string` — one of the [`Scheduler`](#scheduler) values

---

#### `ctx.close()`

Free all native resources (model weights, GGML contexts, GPU memory). The context cannot be used after calling `close()`.

```javascript
ctx.close();
console.log(ctx.isClosed); // true
```

---

#### `ctx.isClosed`

Read-only boolean property. `true` after `close()` has been called.

---

### UpscalerContext

Wraps an ESRGAN upscaling model.

#### `UpscalerContext.create(options)`

Static async factory method. Loads an ESRGAN model.

```javascript
const upscaler = await sd.UpscalerContext.create({
    esrganPath: 'RealESRGAN_x4plus.gguf',
    nThreads: 4,
});
```

**Parameters:**
- `options` — [`UpscalerOptions`](#upscaleroptions) object

**Returns:** `Promise<UpscalerContext>`

---

#### `upscaler.upscale(image, upscaleFactor?)`

Upscale an image on a background thread.

```javascript
const result = await upscaler.upscale(inputImage, 4);
console.log(result.width);  // inputImage.width * 4
```

**Parameters:**
- `image` — [`SdImage`](#sdimage) to upscale
- `upscaleFactor` *(optional, default: 4)* — integer scale factor

**Returns:** `Promise<SdImage>`

---

#### `upscaler.getUpscaleFactor()`

Returns the natural upscale factor of the loaded model.

**Returns:** `number`

---

#### `upscaler.close()`

Free native resources.

---

#### `upscaler.isClosed`

Read-only boolean. `true` after `close()`.

---

## Free functions

### `convert(options)`

Convert a model file from one format/quantization to another. Runs on a background thread.

```javascript
const success = await sd.convert({
    inputPath: 'model.safetensors',
    outputPath: 'model-q4_0.gguf',
    outputType: 'q4_0',
});
```

**Parameters:**
- `options` — [`ConvertOptions`](#convertoptions-1) object

**Returns:** `Promise<boolean>` — `true` on success

---

### `preprocessCanny(image, options?)`

Apply Canny edge detection to an image. This is a **synchronous** operation that modifies the image buffer **in-place**.

```javascript
sd.preprocessCanny(image, {
    highThreshold: 1.0,
    lowThreshold: 0.1,
    weak: 0.2,
    strong: 1.0,
    inverse: false,
});
```

**Parameters:**
- `image` — [`SdImage`](#sdimage) (buffer is modified in-place)
- `options` *(optional)* — [`CannyOptions`](#cannyoptions) object

**Returns:** `boolean` — `true` on success

---

### `setLogCallback(callback)`

Set or clear the global log callback. The library uses a single global callback (not per-context).

```javascript
// Set
sd.setLogCallback(({ level, text }) => {
    const labels = ['DEBUG', 'INFO', 'WARN', 'ERROR'];
    console.log(`[${labels[level]}] ${text.trimEnd()}`);
});

// Clear
sd.setLogCallback(null);
```

**Parameters:**
- `callback` — `((data: { level: number, text: string }) => void) | null`

Log levels: 0 = DEBUG, 1 = INFO, 2 = WARN, 3 = ERROR.

---

### `setProgressCallback(callback)`

Set or clear the global progress callback. Called during sampling steps.

```javascript
sd.setProgressCallback(({ step, steps, time }) => {
    console.log(`Step ${step}/${steps} — ${time.toFixed(1)}s`);
});
```

**Parameters:**
- `callback` — `((data: { step: number, steps: number, time: number }) => void) | null`

---

### `setPreviewCallback(callback, options?)`

Set or clear the global preview callback. Called periodically during generation with intermediate frame previews.

```javascript
sd.setPreviewCallback(
    ({ step, isNoisy, frames }) => {
        console.log(`Preview at step ${step}, ${frames.length} frame(s), noisy=${isNoisy}`);
        // frames[i] is an SdImage
    },
    {
        mode: 'proj',      // 'none' | 'proj' | 'tae' | 'vae'
        interval: 2,       // callback every N steps
        denoised: true,     // include denoised preview
        noisy: false,       // include noisy preview
    },
);
```

**Parameters:**
- `callback` — function or `null`
- `options` *(optional)* — [`PreviewOptions`](#previewoptions) object

**Note:** The preview callback deep-copies frame pixel data before posting to the JS event loop, since the original data may be freed after the C callback returns.

---

### `getSystemInfo()`

Returns a string describing detected hardware features and backends.

```javascript
console.log(sd.getSystemInfo());
// "System Info: SSE3 = 0 | AVX = 0 | ... | NEON = 1 | ..."
```

**Returns:** `string`

---

### `getNumPhysicalCores()`

Returns the number of physical CPU cores.

```javascript
console.log(sd.getNumPhysicalCores()); // 10
```

**Returns:** `number`

---

### `version()`

Returns the stable-diffusion.cpp version string.

```javascript
console.log(sd.version()); // "master-507-b314d80"
```

**Returns:** `string`

---

### `commit()`

Returns the stable-diffusion.cpp git commit hash.

```javascript
console.log(sd.commit()); // "0752cc9"
```

**Returns:** `string`

---

## Types

### SdImage

The universal image type used for all input and output images.

```typescript
interface SdImage {
    width: number;     // image width in pixels
    height: number;    // image height in pixels
    channel: number;   // number of color channels (3 for RGB)
    data: Buffer;      // raw pixel data, length = width * height * channel
}
```

Pixel layout is row-major, top-to-bottom, RGB with no padding. Each pixel component is a `uint8` (0-255).

---

### ContextOptions

Options for `StableDiffusionContext.create()`.

All path fields are optional. At minimum, provide either `modelPath` (for single-file models containing all components) or `diffusionModelPath` (for standalone diffusion-only component files, e.g. when text encoders are loaded separately via `clipLPath`, `t5xxlPath`, etc.).

| Property | Type | Default | Description |
|---|---|---|---|
| `modelPath` | `string` | — | Path to a legacy .ckpt or .safetensors model |
| `clipLPath` | `string` | — | Separate CLIP-L text encoder |
| `clipGPath` | `string` | — | Separate CLIP-G text encoder |
| `clipVisionPath` | `string` | — | CLIP vision encoder (for IP-Adapter, etc.) |
| `t5xxlPath` | `string` | — | T5-XXL text encoder (for SD3, FLUX) |
| `llmPath` | `string` | — | LLM text encoder (for Qwen-Image) |
| `llmVisionPath` | `string` | — | LLM vision encoder |
| `diffusionModelPath` | `string` | — | Path to standalone diffusion-only model (loads with prefix filter — use `modelPath` for single-file models) |
| `highNoiseDiffusionModelPath` | `string` | — | High-noise diffusion model (for Wan MoE) |
| `vaePath` | `string` | — | Separate VAE model |
| `taesdPath` | `string` | — | Tiny AutoEncoder for previews |
| `controlNetPath` | `string` | — | ControlNet model |
| `photoMakerPath` | `string` | — | PhotoMaker model |
| `tensorTypeRules` | `string` | — | Custom tensor type override rules |
| `embeddings` | [`EmbeddingDefinition[]`](#embeddingdefinition) | — | Textual inversion embeddings |
| `vaeDecodeOnly` | `boolean` | `true` | Only load VAE decoder (saves memory) |
| `freeParamsImmediately` | `boolean` | `true` | Free model weights after first use |
| `nThreads` | `number` | CPU cores | Number of CPU threads |
| `wtype` | [`SdType`](#sdtype) | `'count'` (auto) | Weight quantization type override |
| `rngType` | [`RngType`](#rngtype) | `'cuda'` | Random number generator type |
| `samplerRngType` | [`RngType`](#rngtype) | auto | Sampler RNG type |
| `prediction` | [`Prediction`](#prediction) | auto | Noise prediction type |
| `loraApplyMode` | [`LoraApplyMode`](#loraapplymode) | `'auto'` | When to apply LoRA weights |
| `offloadParamsToCpu` | `boolean` | `false` | Offload params to CPU (saves GPU memory) |
| `enableMmap` | `boolean` | `false` | Use memory-mapped file I/O |
| `keepClipOnCpu` | `boolean` | `false` | Keep CLIP encoder on CPU |
| `keepControlNetOnCpu` | `boolean` | `false` | Keep ControlNet on CPU |
| `keepVaeOnCpu` | `boolean` | `false` | Keep VAE on CPU |
| `flashAttn` | `boolean` | `false` | Enable flash attention |
| `diffusionFlashAttn` | `boolean` | `false` | Enable flash attention for diffusion model |
| `taePreviewOnly` | `boolean` | `false` | Only load TAE for previews |
| `diffusionConvDirect` | `boolean` | `false` | Use direct convolution in diffusion model |
| `vaeConvDirect` | `boolean` | `false` | Use direct convolution in VAE |
| `circularX` | `boolean` | `false` | Circular padding in X (for seamless textures) |
| `circularY` | `boolean` | `false` | Circular padding in Y |
| `forceSdxlVaeConvScale` | `boolean` | `false` | Force SDXL VAE conv scale |
| `chromaUseDitMask` | `boolean` | `true` | Chroma DiT mask |
| `chromaUseT5Mask` | `boolean` | `false` | Chroma T5 mask |
| `chromaT5MaskPad` | `number` | `1` | Chroma T5 mask padding |
| `qwenImageZeroCondT` | `boolean` | `false` | Qwen-Image zero conditioning |

---

### ImageGenerationOptions

Options for `ctx.generateImage()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | — | Text prompt |
| `negativePrompt` | `string` | — | Negative prompt |
| `clipSkip` | `number` | `-1` (auto) | CLIP layers to skip |
| `initImage` | [`SdImage`](#sdimage) | — | Input image for img2img |
| `refImages` | [`SdImage[]`](#sdimage) | — | Reference images (IP-Adapter, etc.) |
| `autoResizeRefImage` | `boolean` | `false` | Auto-resize reference images |
| `increaseRefIndex` | `boolean` | `false` | Increment ref image index per batch |
| `maskImage` | [`SdImage`](#sdimage) | — | Inpainting mask (white = inpaint) |
| `width` | `number` | `512` | Output width in pixels |
| `height` | `number` | `512` | Output height in pixels |
| `sampleParams` | [`SampleParams`](#sampleparams) | defaults | Sampling parameters |
| `strength` | `number` | `0.75` | Denoising strength for img2img (0.0-1.0) |
| `seed` | `number` | `-1` (random) | RNG seed. `-1` for random. |
| `batchCount` | `number` | `1` | Number of images to generate |
| `controlImage` | [`SdImage`](#sdimage) | — | ControlNet conditioning image |
| `controlStrength` | `number` | `0.9` | ControlNet influence strength |
| `photoMaker` | [`PhotoMakerParams`](#photomakerparams) | — | PhotoMaker options |
| `vaeTiling` | [`TilingParams`](#tilingparams) | disabled | VAE tiling for large images |
| `cache` | [`CacheParams`](#cacheparams) | disabled | Step caching for faster generation |
| `loras` | [`LoraDefinition[]`](#loradefinition) | — | LoRA models to apply |

---

### VideoGenerationOptions

Options for `ctx.generateVideo()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | — | Text prompt |
| `negativePrompt` | `string` | — | Negative prompt |
| `clipSkip` | `number` | `-1` | CLIP layers to skip |
| `initImage` | [`SdImage`](#sdimage) | — | Starting frame (for img2vid) |
| `endImage` | [`SdImage`](#sdimage) | — | Ending frame (for FLF2V) |
| `controlFrames` | [`SdImage[]`](#sdimage) | — | Control frames |
| `width` | `number` | `512` | Frame width |
| `height` | `number` | `512` | Frame height |
| `sampleParams` | [`SampleParams`](#sampleparams) | defaults | Sampling parameters |
| `highNoiseSampleParams` | [`SampleParams`](#sampleparams) | defaults | High-noise model sampling params |
| `moeBoundary` | `number` | `0.875` | MoE boundary for Wan A14B |
| `strength` | `number` | `0.75` | Denoising strength |
| `seed` | `number` | `-1` | RNG seed |
| `videoFrames` | `number` | `6` | Number of frames to generate |
| `vaceStrength` | `number` | `1.0` | VACE conditioning strength |
| `vaeTiling` | [`TilingParams`](#tilingparams) | disabled | VAE tiling |
| `cache` | [`CacheParams`](#cacheparams) | disabled | Step caching |
| `loras` | [`LoraDefinition[]`](#loradefinition) | — | LoRA models |

---

### SampleParams

Controls the sampling/denoising process.

| Property | Type | Default | Description |
|---|---|---|---|
| `guidance` | [`GuidanceParams`](#guidanceparams) | — | Guidance/CFG parameters |
| `scheduler` | [`Scheduler`](#scheduler) | auto | Noise scheduler |
| `sampleMethod` | [`SampleMethod`](#samplemethod) | auto | Sampling algorithm |
| `sampleSteps` | `number` | `20` | Number of denoising steps |
| `eta` | `number` | `0` | Stochasticity (for DDIM-like samplers) |
| `shiftedTimestep` | `number` | `0` | Shifted timestep |
| `flowShift` | `number` | auto | Flow shift parameter |
| `customSigmas` | `number[]` | — | Custom sigma schedule (overrides steps/scheduler) |

---

### GuidanceParams

Classifier-free guidance settings.

| Property | Type | Default | Description |
|---|---|---|---|
| `txtCfg` | `number` | `7.0` | Text guidance scale (CFG) |
| `imgCfg` | `number` | same as txtCfg | Image guidance scale (for img2img edit models) |
| `distilledGuidance` | `number` | `3.5` | Distilled guidance (for FLUX) |
| `slg` | [`SlgParams`](#slgparams) | — | Skip-layer guidance |

#### SlgParams

| Property | Type | Default | Description |
|---|---|---|---|
| `layers` | `number[]` | — | Layer indices to skip |
| `layerStart` | `number` | `0.01` | Start fraction |
| `layerEnd` | `number` | `0.2` | End fraction |
| `scale` | `number` | `0` | SLG scale |

---

### CacheParams

Step caching accelerates generation by reusing intermediate results.

| Property | Type | Default | Description |
|---|---|---|---|
| `mode` | [`CacheMode`](#cachemode) | `'disabled'` | Cache strategy |
| `reuseThreshold` | `number` | `1.0` | Similarity threshold for reuse |
| `startPercent` | `number` | `0.15` | Start caching at this % of steps |
| `endPercent` | `number` | `0.95` | Stop caching at this % |
| `errorDecayRate` | `number` | `1.0` | Error decay rate |
| `useRelativeThreshold` | `boolean` | `true` | Use relative vs absolute threshold |
| `resetErrorOnCompute` | `boolean` | `true` | Reset error when computing |
| `fnComputeBlocks` | `number` | `8` | Forward compute blocks |
| `bnComputeBlocks` | `number` | `0` | Backward compute blocks |
| `residualDiffThreshold` | `number` | `0.08` | Residual diff threshold |
| `maxWarmupSteps` | `number` | `8` | Max warmup steps |
| `maxCachedSteps` | `number` | `-1` (unlimited) | Max steps to cache |
| `maxContinuousCachedSteps` | `number` | `-1` (unlimited) | Max continuous cached steps |
| `taylorseerNDerivatives` | `number` | `1` | TaylorSeer derivatives |
| `taylorseerSkipInterval` | `number` | `1` | TaylorSeer skip interval |
| `scmPolicyDynamic` | `boolean` | `true` | Dynamic SCM policy |

---

### TilingParams

VAE tiling for generating images larger than VRAM allows.

| Property | Type | Default | Description |
|---|---|---|---|
| `enabled` | `boolean` | `false` | Enable VAE tiling |
| `tileSizeX` | `number` | `0` (auto) | Tile width |
| `tileSizeY` | `number` | `0` (auto) | Tile height |
| `targetOverlap` | `number` | `0.5` | Overlap fraction between tiles |
| `relSizeX` | `number` | `0.0` | Relative tile size X |
| `relSizeY` | `number` | `0.0` | Relative tile size Y |

---

### PhotoMakerParams

PhotoMaker identity-preserving generation.

| Property | Type | Default | Description |
|---|---|---|---|
| `idImages` | [`SdImage[]`](#sdimage) | — | Identity reference images |
| `idEmbedPath` | `string` | — | Path to ID embedding file |
| `styleStrength` | `number` | `20` | Style strength (higher = more identity preservation) |

---

### LoraDefinition

A LoRA adapter to apply during generation.

```javascript
{
    path: 'lora-model.safetensors',
    multiplier: 0.8,        // weight multiplier (default: 1.0)
    isHighNoise: false,      // for high-noise diffusion models
}
```

| Property | Type | Default | Description |
|---|---|---|---|
| `path` | `string` | *required* | Path to LoRA file |
| `multiplier` | `number` | `1.0` | Weight multiplier |
| `isHighNoise` | `boolean` | `false` | Apply to high-noise model |

---

### EmbeddingDefinition

A textual inversion embedding.

| Property | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | *required* | Trigger token name |
| `path` | `string` | *required* | Path to embedding file |

---

### UpscalerOptions

Options for `UpscalerContext.create()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `esrganPath` | `string` | *required* | Path to ESRGAN model |
| `offloadParamsToCpu` | `boolean` | `false` | Offload parameters to CPU |
| `direct` | `boolean` | `false` | Use direct convolution |
| `nThreads` | `number` | CPU cores | Number of threads |
| `tileSize` | `number` | `0` (auto) | Tile size for tiled upscaling |

---

### ConvertOptions

Options for `convert()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `inputPath` | `string` | *required* | Input model path |
| `outputPath` | `string` | *required* | Output model path |
| `vaePath` | `string` | — | Separate VAE to bake in |
| `outputType` | [`SdType`](#sdtype) | auto | Output quantization type |
| `tensorTypeRules` | `string` | — | Custom tensor type rules |
| `convertName` | `boolean` | `false` | Convert tensor names |

---

### CannyOptions

Options for `preprocessCanny()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `highThreshold` | `number` | `1.0` | High threshold |
| `lowThreshold` | `number` | `0.1` | Low threshold |
| `weak` | `number` | `0.2` | Weak edge value |
| `strong` | `number` | `1.0` | Strong edge value |
| `inverse` | `boolean` | `false` | Invert the output |

---

### PreviewOptions

Options for `setPreviewCallback()`.

| Property | Type | Default | Description |
|---|---|---|---|
| `mode` | [`PreviewMode`](#previewmode) | `'proj'` | Preview decode method |
| `interval` | `number` | `1` | Callback every N steps |
| `denoised` | `boolean` | `true` | Include denoised preview |
| `noisy` | `boolean` | `false` | Include noisy preview |

---

## Enums

All enum values are passed as lowercase strings. The library uses the stable-diffusion.cpp `str_to_*` functions for conversion, so the exact accepted strings match the upstream CLI.

### SampleMethod

Sampling algorithms.

| Value | Description |
|---|---|
| `'euler'` | Euler method |
| `'euler_a'` | Euler Ancestral |
| `'heun'` | Heun's method |
| `'dpm2'` | DPM2 |
| `'dpm++2s_a'` | DPM++ 2S Ancestral |
| `'dpm++2m'` | DPM++ 2M |
| `'dpm++2mv2'` | DPM++ 2M v2 |
| `'ipndm'` | iPNDM |
| `'ipndm_v'` | iPNDM v |
| `'lcm'` | LCM |
| `'ddim_trailing'` | DDIM Trailing |
| `'tcd'` | TCD |
| `'res_multistep'` | Restart Multistep |
| `'res_2s'` | Restart 2S |

### Scheduler

Noise schedules.

| Value | Description |
|---|---|
| `'discrete'` | Discrete |
| `'karras'` | Karras |
| `'exponential'` | Exponential |
| `'ays'` | Align Your Steps |
| `'gits'` | GITS |
| `'sgm_uniform'` | SGM Uniform |
| `'simple'` | Simple |
| `'smoothstep'` | Smoothstep |
| `'kl_optimal'` | KL Optimal |
| `'lcm'` | LCM |
| `'bong_tangent'` | Bong Tangent |

### SdType

Quantization / data types.

`'f32'`, `'f16'`, `'bf16'`, `'q4_0'`, `'q4_1'`, `'q5_0'`, `'q5_1'`, `'q8_0'`, `'q8_1'`, `'q2_k'`, `'q3_k'`, `'q4_k'`, `'q5_k'`, `'q6_k'`, `'q8_k'`, `'iq2_xxs'`, `'iq2_xs'`, `'iq2_s'`, `'iq3_xxs'`, `'iq3_s'`, `'iq4_nl'`, `'iq4_xs'`, `'iq1_s'`, `'iq1_m'`, `'i8'`, `'i16'`, `'i32'`, `'i64'`, `'f64'`, `'tq1_0'`, `'tq2_0'`, `'mxfp4'`

### RngType

Random number generator backend.

| Value | Description |
|---|---|
| `'std_default'` | Standard library default |
| `'cuda'` | CUDA RNG |
| `'cpu'` | CPU RNG |

### Prediction

Noise prediction type (auto-detected from model in most cases).

`'eps'`, `'v'`, `'edm_v'`, `'flow'`, `'flux_flow'`, `'flux2_flow'`

### PreviewMode

Preview image decode method.

| Value | Description |
|---|---|
| `'none'` | No preview |
| `'proj'` | Linear projection (fastest, lowest quality) |
| `'tae'` | Tiny AutoEncoder (fast, decent quality) |
| `'vae'` | Full VAE decode (slow, best quality) |

### LoraApplyMode

When LoRA weights are applied.

| Value | Description |
|---|---|
| `'auto'` | Automatically choose |
| `'immediately'` | Apply during model loading |
| `'at_runtime'` | Apply at generation time |

### CacheMode

Step caching strategy.

| Value | Description |
|---|---|
| `'disabled'` | No caching |
| `'easycache'` | EasyCache |
| `'ucache'` | UCache |
| `'dbcache'` | DBCache |
| `'taylorseer'` | TaylorSeer |
| `'cachedit'` | Cache-DiT |
