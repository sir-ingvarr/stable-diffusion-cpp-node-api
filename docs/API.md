# API Reference

Complete API documentation for `node-stable-diffusion-cpp`.

All async functions return Promises. All heavy operations (model loading, generation, upscaling, conversion) run on background threads and do not block the Node.js event loop.

## Table of contents

- [Cancellation](#cancellation)
- [Classes](#classes)
  - [StableDiffusionContext](#stablediffusioncontext)
  - [UpscalerContext](#upscalercontext)
- [Free functions](#free-functions)
  - [convert](#convertoptions)
  - [extractMetaData](#extractmetadatapath)
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
  - [HiresParams](#hiresparams)
  - [TilingParams](#tilingparams)
  - [PhotoMakerParams](#photomakerparams)
  - [LoraDefinition](#loradefinition)
  - [EmbeddingDefinition](#embeddingdefinition)
  - [UpscalerOptions](#upscaleroptions)
  - [ConvertOptions](#convertoptions-1)
  - [CannyOptions](#cannyoptions)
  - [PreviewOptions](#previewoptions)
  - [ModelMetadata](#modelmetadata)
  - [SDVersionSlug](#sdversionslug)
- [Enums (string literal types)](#enums)

---

## Cancellation

Every async call accepts an optional `signal: AbortSignal` and every context exposes an `abort()` method. There are two flavors of cancellation depending on the call.

**Hard cancel** — `ctx.generateImage`, `ctx.generateVideo`, `upscaler.upscale`. Aborting reaches the native worker: it throws at the next sampling/compute checkpoint (typically within a fraction of a second), unwinds out of stable-diffusion.cpp, and the returned Promise rejects with `AbortError` only after the worker has fully finished — so when `await` returns, no native work is still running on this ctx.

**Soft cancel** — `StableDiffusionContext.create`, `UpscalerContext.create`, `convert`. The wrapper Promise rejects immediately, but the underlying load/conversion runs to completion in the background and its result is discarded. There's no way to hard-cancel these from the JS side; if you need to drop a partially-loaded model, let the load finish and call `ctx.close()`.

Cancellation is per-ctx — aborting one context does not affect any other context. Calls on the same ctx are serialized in JS, so a queued-but-not-yet-running call rejects immediately with `AbortError` without touching the running one.

```javascript
const ctrl = new AbortController();

// Pass via options.signal
const promise = ctx.generateImage({ prompt: '...', signal: ctrl.signal });

// Or via the per-ctx method (cancels whatever is running or queued)
ctx.abort();
```

Calling `ctx.close()` implicitly aborts any in-flight work on that ctx — see [`ctx.close()`](#ctxclose).

---

## Classes

### StableDiffusionContext

The main class for image and video generation. Wraps a loaded model and all its components (text encoders, diffusion model, VAE, etc.).

Contexts are expensive to create (model loading) and should be reused for multiple generations. This relies on `freeParamsImmediately` being `false` (our default — upstream sd.cpp defaults it to `true`, which makes the context single-use). See [`freeParamsImmediately`](#contextoptions) for the trade-off.

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

#### `ctx.abort()`

Cancel any in-flight or queued `generateImage` / `generateVideo` call on this context. Takes effect at the next cancellation checkpoint inside the running op (typically within a fraction of a second). Queued ops that have not yet started reject immediately with `AbortError`. Calls on other contexts are unaffected.

Equivalent to passing an `AbortSignal` via `options.signal` and calling `.abort()` on its controller — use whichever fits your call site. See [Cancellation](#cancellation).

```javascript
const promise = ctx.generateImage({ prompt: '...' });
ctx.abort();
await promise; // rejects with AbortError
```

---

#### `ctx.close()`

Cancel any in-flight work on this context, then release the native resources (model weights, GGML contexts, GPU memory). The context cannot be used after calling `close()`.

The cancel is asynchronous — `close()` returns immediately, but the running native worker observes the abort at its next checkpoint and unwinds. The native context is freed when the last reference drops (the wrapper or the in-flight worker, whichever finishes last).

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

#### `upscaler.upscale(image, upscaleFactor?, options?)`

Upscale an image on a background thread.

```javascript
const result = await upscaler.upscale(inputImage, 4);
console.log(result.width);  // inputImage.width * 4

// With abort signal
const ctrl = new AbortController();
const result = await upscaler.upscale(inputImage, 4, { signal: ctrl.signal });
```

**Parameters:**
- `image` — [`SdImage`](#sdimage) to upscale
- `upscaleFactor` *(optional, default: 4)* — integer scale factor
- `options` *(optional)* — object with `signal?: AbortSignal`. See [Cancellation](#cancellation).

**Returns:** `Promise<SdImage>`

---

#### `upscaler.getUpscaleFactor()`

Returns the natural upscale factor of the loaded model.

**Returns:** `number`

---

#### `upscaler.abort()`

Cancel any in-flight or queued `upscale` call on this upscaler. Same semantics as [`ctx.abort()`](#ctxabort) — see [Cancellation](#cancellation).

---

#### `upscaler.close()`

Cancel any in-flight work, then release native resources. Same semantics as [`ctx.close()`](#ctxclose).

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

### `extractMetaData(path)`

Read model metadata directly from the file header without loading any tensor data. Works on safetensors, gguf, and ckpt files. Runs in milliseconds even for multi-GB models.

Useful for preflight checks before paying the load cost — detect the model architecture, verify required components are bundled (e.g. SD3 needs separate CLIP-L/CLIP-G/T5-XXL when not), inspect weight quantization, or estimate parameter memory.

```javascript
const meta = await sd.extractMetaData('sd3.5_large.safetensors');

console.log(meta.version);            // 'sd3'
console.log(meta.versionLabel);       // 'SD3.x'
console.log(meta.isSd3, meta.isDit);  // true true
console.log(meta.hasClipL, meta.hasT5xxl); // false false — must supply separately
console.log(meta.diffusionWeightTypes);    // { f16: 923 }
console.log(meta.vaeWeightTypes);          // { bf16: 244 }
console.log((meta.estParamsBytes / 1024 / 1024 / 1024).toFixed(1) + ' GB');

// Common pattern: auto-tune options based on the model
const opts = { /* ... */ };
if (meta.isSd3 && !opts.vaeTiling) {
    // SD3 fp16 VAE can overflow on full-image decode → black images.
    // Tiling sidesteps the overflow.
    opts.vaeTiling = { enabled: true };
}
if (meta.isDit && !meta.hasClipL && !meta.hasT5xxl) {
    throw new Error(`${meta.versionLabel} model has no embedded text encoders — pass clipLPath / clipGPath / t5xxlPath`);
}
```

**Parameters:**
- `path` — absolute or relative path to the model file

**Returns:** `Promise<ModelMetadata>` — see [`ModelMetadata`](#modelmetadata)

**Throws:** Rejects if the file can't be opened or parsed (missing, corrupt, unsupported format).

**Note:** This is a header read — no tensors are loaded into memory. The returned `estParamsBytes` is computed from tensor shapes and dtypes recorded in the header, not from actually allocating memory.

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
| `vaeDecodeOnly` | `boolean` | `true` | Only load VAE decoder (saves memory). Set to `false` when using img2img (`initImage` / `strength`), inpainting, ControlNet with a control image, or pixel-space [hires upscalers](#hiresupscaler) (`'lanczos'`, `'nearest'`, `'model'`) — all of these need to encode pixels into latents. |
| `freeParamsImmediately` | `boolean` | `false` | Free each model component's weights as soon as its stage finishes within a `generateImage` call. Lowers peak VRAM (peak ≈ largest single stage instead of all weights resident) but makes the context **single-use** — a second `generateImage` call would need to reload weights from disk and currently asserts on the Metal backend ([sd.cpp #1298](https://github.com/leejet/stable-diffusion.cpp/issues/1298)). Defaults to `false` so context reuse works out of the box; flip to `true` only if you're memory-constrained and doing one-shot generation. |
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
| `signal` | `AbortSignal` | — | Soft-cancel the load; see [Cancellation](#cancellation) |

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
| `hires` | [`HiresParams`](#hiresparams) | disabled | Two-pass "Hires. fix" upscale-and-denoise |
| `loras` | [`LoraDefinition[]`](#loradefinition) | — | LoRA models to apply |
| `signal` | `AbortSignal` | — | Hard-cancel the generation; see [Cancellation](#cancellation) |

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
| `signal` | `AbortSignal` | — | Hard-cancel the generation; see [Cancellation](#cancellation) |

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
| `scmMask` | `string` | — | Dynamic SCM policy mask string (used when `scmPolicyDynamic` is on) |
| `scmPolicyDynamic` | `boolean` | `true` | Dynamic SCM policy |
| `spectrumW` | `number` | upstream | Spectrum cache: weight |
| `spectrumM` | `number` | upstream | Spectrum cache: M |
| `spectrumLam` | `number` | upstream | Spectrum cache: lambda |
| `spectrumWindowSize` | `number` | upstream | Spectrum cache: window size |
| `spectrumFlexWindow` | `number` | upstream | Spectrum cache: flex window |
| `spectrumWarmupSteps` | `number` | upstream | Spectrum cache: warmup steps |
| `spectrumStopPercent` | `number` | upstream | Spectrum cache: stop % |

---

### HiresParams

Two-pass "Hires. fix" — generate at a smaller resolution, then upscale and run additional denoising steps at the larger size. Tends to produce sharper composition than direct large-resolution generation, faster than direct + post-upscale.

| Property | Type | Default | Description |
|---|---|---|---|
| `enabled` | `boolean` | `false` | Enable hires fix |
| `upscaler` | [`HiresUpscaler`](#hiresupscaler) | `'latent'` | Upscaler to use between passes. Pixel-space modes (`'lanczos'`, `'nearest'`, `'model'`) require the context to be created with `vaeDecodeOnly: false`. |
| `modelPath` | `string` | — | Upscaler model path (required when `upscaler` is `'model'`, e.g. ESRGAN) |
| `scale` | `number` | `2.0` | Multiplier on width/height (ignored if `targetWidth`/`targetHeight` set) |
| `targetWidth` | `number` | `0` | Explicit target width (overrides `scale`) |
| `targetHeight` | `number` | `0` | Explicit target height |
| `steps` | `number` | `0` | Pass-2 sample steps (0 = derive from `denoisingStrength`) |
| `denoisingStrength` | `number` | `0.5` | Pass-2 strength (img2img-style) |
| `upscaleTileSize` | `number` | `0` | Tile size for the upscaler (0 = no tiling) |

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
| `signal` | `AbortSignal` | — | Soft-cancel the load; see [Cancellation](#cancellation) |

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
| `signal` | `AbortSignal` | — | Soft-cancel the conversion; see [Cancellation](#cancellation) |

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

### ModelMetadata

Returned by [`extractMetaData()`](#extractmetadatapath). All fields are derived from the file header alone — no tensors are loaded.

| Property | Type | Description |
|---|---|---|
| `version` | [`SDVersionSlug`](#sdversionslug) | Stable programmatic identifier for the architecture (e.g. `'sd3'`, `'sdxl_inpaint'`, `'flux2_klein'`). `'unknown'` if detection fails. |
| `versionLabel` | `string` | Human-readable label (e.g. `'SD3.x'`, `'SDXL Inpaint'`, `'Flux.2 klein'`). |
| `isUnet` | `boolean` | UNet-based architecture (SD1/SD2/SDXL family). |
| `isDit` | `boolean` | Diffusion-Transformer architecture (SD3, Flux, Flux2, Wan, Qwen-Image, Z-Image). |
| `isSd1` | `boolean` | Any SD1.x variant. |
| `isSd2` | `boolean` | Any SD2.x variant. |
| `isSdxl` | `boolean` | Any SDXL variant. |
| `isSd3` | `boolean` | SD3 / SD3.5. |
| `isFlux` | `boolean` | Flux.1 family (incl. Fill, Controls, Flex.2, Chroma Radiance, Ovis). |
| `isFlux2` | `boolean` | Flux.2 family (incl. klein). |
| `isWan` | `boolean` | Wan 2.x family. |
| `isQwenImage` | `boolean` | Qwen-Image. |
| `isZImage` | `boolean` | Z-Image. |
| `isInpaint` | `boolean` | Inpaint variant of any of the above. |
| `isControl` | `boolean` | Control variant (Flux Controls, Flex.2). |
| `hasDiffusionModel` | `boolean` | File contains diffusion model weights (`model.diffusion_model.*` or `unet.*`). |
| `hasVae` | `boolean` | File contains VAE weights (`first_stage_model.*` or `vae.*`). |
| `hasClipL` | `boolean` | File contains a bundled CLIP-L text encoder. |
| `hasClipG` | `boolean` | File contains a bundled CLIP-G text encoder. |
| `hasT5xxl` | `boolean` | File contains a bundled T5-XXL text encoder. |
| `hasControlNet` | `boolean` | File contains ControlNet weights. |
| `isLora` | `boolean` | File looks like a LoRA adapter (presence of `.lora_up`/`.lora_down`/`.lora_A`/`.lora_B`). |
| `tensorCount` | `number` | Total number of tensors in the file. |
| `estParamsBytes` | `number` | Estimated bytes occupied by parameters once loaded (file dtypes, no conversion, 128-byte alignment per tensor). |
| `weightTypes` | `Record<string, number>` | Tensor count grouped by ggml type name across the whole file (e.g. `{ f16: 1234, f32: 56 }`). Keys are the ggml type names: `'f32'`, `'f16'`, `'bf16'`, `'q4_0'`, `'q8_0'`, etc. |
| `diffusionWeightTypes` | `Record<string, number>` | Same shape as `weightTypes`, restricted to diffusion-model tensors. |
| `vaeWeightTypes` | `Record<string, number>` | Same shape as `weightTypes`, restricted to VAE tensors. |
| `conditionerWeightTypes` | `Record<string, number>` | Same shape as `weightTypes`, restricted to text encoder / conditioner tensors. |

---

### SDVersionSlug

Stable programmatic identifier for a detected model architecture. Returned in [`ModelMetadata.version`](#modelmetadata).

`'sd1'`, `'sd1_inpaint'`, `'sd1_pix2pix'`, `'sd1_tiny_unet'`, `'sd2'`, `'sd2_inpaint'`, `'sd2_tiny_unet'`, `'sdxs'`, `'sdxl'`, `'sdxl_inpaint'`, `'sdxl_pix2pix'`, `'sdxl_vega'`, `'sdxl_ssd1b'`, `'svd'`, `'sd3'`, `'flux'`, `'flux_fill'`, `'flux_controls'`, `'flex2'`, `'chroma_radiance'`, `'wan2'`, `'wan2_2_i2v'`, `'wan2_2_ti2v'`, `'qwen_image'`, `'flux2'`, `'flux2_klein'`, `'z_image'`, `'ovis_image'`, `'unknown'`

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
| `'spectrum'` | Spectrum cache |

### HiresUpscaler

Upscaler to use for the inter-pass step in [`HiresParams`](#hiresparams).

| Value | Description |
|---|---|
| `'none'` | No upscaler |
| `'latent'` | Latent upscale (default linear) |
| `'latent_nearest'` | Latent upscale, nearest |
| `'latent_nearest_exact'` | Latent upscale, nearest exact |
| `'latent_antialiased'` | Latent upscale, antialiased |
| `'latent_bicubic'` | Latent upscale, bicubic |
| `'latent_bicubic_antialiased'` | Latent upscale, bicubic + antialiased |
| `'lanczos'` | Pixel-space Lanczos |
| `'nearest'` | Pixel-space nearest |
| `'model'` | External upscaler model (`modelPath` required, e.g. ESRGAN) |
