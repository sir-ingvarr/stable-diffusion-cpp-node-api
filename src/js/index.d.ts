/// <reference types="node" />

// --- Enum string literal types ---

export type SampleMethod =
    | 'euler'
    | 'euler_a'
    | 'heun'
    | 'dpm2'
    | 'dpm++2s_a'
    | 'dpm++2m'
    | 'dpm++2mv2'
    | 'ipndm'
    | 'ipndm_v'
    | 'lcm'
    | 'ddim_trailing'
    | 'tcd'
    | 'res_multistep'
    | 'res_2s';

export type Scheduler =
    | 'discrete'
    | 'karras'
    | 'exponential'
    | 'ays'
    | 'gits'
    | 'sgm_uniform'
    | 'simple'
    | 'smoothstep'
    | 'kl_optimal'
    | 'lcm'
    | 'bong_tangent';

export type SdType =
    | 'f32'
    | 'f16'
    | 'q4_0'
    | 'q4_1'
    | 'q5_0'
    | 'q5_1'
    | 'q8_0'
    | 'q8_1'
    | 'q2_k'
    | 'q3_k'
    | 'q4_k'
    | 'q5_k'
    | 'q6_k'
    | 'q8_k'
    | 'iq2_xxs'
    | 'iq2_xs'
    | 'iq3_xxs'
    | 'iq1_s'
    | 'iq4_nl'
    | 'iq3_s'
    | 'iq2_s'
    | 'iq4_xs'
    | 'i8'
    | 'i16'
    | 'i32'
    | 'i64'
    | 'f64'
    | 'iq1_m'
    | 'bf16'
    | 'tq1_0'
    | 'tq2_0'
    | 'mxfp4';

export type RngType = 'std_default' | 'cuda' | 'cpu';

export type Prediction = 'eps' | 'v' | 'edm_v' | 'flow' | 'flux_flow' | 'flux2_flow';

export type PreviewMode = 'none' | 'proj' | 'tae' | 'vae';

export type LogLevel = 0 | 1 | 2 | 3; // debug, info, warn, error

export type LoraApplyMode = 'auto' | 'immediately' | 'at_runtime';

export type CacheMode = 'disabled' | 'easycache' | 'ucache' | 'dbcache' | 'taylorseer' | 'cachedit' | 'spectrum';

export type HiresUpscaler =
    | 'none'
    | 'latent'
    | 'latent_nearest'
    | 'latent_nearest_exact'
    | 'latent_antialiased'
    | 'latent_bicubic'
    | 'latent_bicubic_antialiased'
    | 'lanczos'
    | 'nearest'
    | 'model';

// --- Data interfaces ---

export interface SdImage {
    width: number;
    height: number;
    channel: number;
    data: Buffer;
}

export interface LoraDefinition {
    path: string;
    multiplier?: number;
    isHighNoise?: boolean;
}

export interface EmbeddingDefinition {
    name: string;
    path: string;
}

export interface SlgParams {
    layers?: number[];
    layerStart?: number;
    layerEnd?: number;
    scale?: number;
}

export interface GuidanceParams {
    txtCfg?: number;
    imgCfg?: number;
    distilledGuidance?: number;
    slg?: SlgParams;
}

export interface SampleParams {
    guidance?: GuidanceParams;
    scheduler?: Scheduler;
    sampleMethod?: SampleMethod;
    sampleSteps?: number;
    eta?: number;
    shiftedTimestep?: number;
    flowShift?: number;
    customSigmas?: number[];
}

export interface CacheParams {
    mode?: CacheMode;
    reuseThreshold?: number;
    startPercent?: number;
    endPercent?: number;
    errorDecayRate?: number;
    useRelativeThreshold?: boolean;
    resetErrorOnCompute?: boolean;
    fnComputeBlocks?: number;
    bnComputeBlocks?: number;
    residualDiffThreshold?: number;
    maxWarmupSteps?: number;
    maxCachedSteps?: number;
    maxContinuousCachedSteps?: number;
    taylorseerNDerivatives?: number;
    taylorseerSkipInterval?: number;
    scmMask?: string;
    scmPolicyDynamic?: boolean;
    spectrumW?: number;
    spectrumM?: number;
    spectrumLam?: number;
    spectrumWindowSize?: number;
    spectrumFlexWindow?: number;
    spectrumWarmupSteps?: number;
    spectrumStopPercent?: number;
}

export interface HiresParams {
    enabled?: boolean;
    upscaler?: HiresUpscaler;
    modelPath?: string;
    scale?: number;
    targetWidth?: number;
    targetHeight?: number;
    steps?: number;
    denoisingStrength?: number;
    upscaleTileSize?: number;
}

export interface TilingParams {
    enabled?: boolean;
    tileSizeX?: number;
    tileSizeY?: number;
    targetOverlap?: number;
    relSizeX?: number;
    relSizeY?: number;
    /** Tile along the temporal axis too (video models). */
    temporalTiling?: boolean;
    /** Extra upstream tiling args as a key=value list. */
    extraTilingArgs?: string;
}

export interface PhotoMakerParams {
    idImages?: SdImage[];
    idEmbedPath?: string;
    styleStrength?: number;
}

// --- Abort support ---

export interface AbortableOptions {
    signal?: AbortSignal;
}

// --- Context creation options ---

export interface ContextOptions extends AbortableOptions {
    modelPath?: string;
    clipLPath?: string;
    clipGPath?: string;
    clipVisionPath?: string;
    t5xxlPath?: string;
    llmPath?: string;
    llmVisionPath?: string;
    diffusionModelPath?: string;
    highNoiseDiffusionModelPath?: string;
    vaePath?: string;
    taesdPath?: string;
    controlNetPath?: string;
    photoMakerPath?: string;
    tensorTypeRules?: string;
    embeddings?: EmbeddingDefinition[];
    /** @deprecated No-op since the 2026-10 upstream bump (the device residency manager owns param lifetime). */
    vaeDecodeOnly?: boolean;
    /** @deprecated No-op since the 2026-10 upstream bump (the device residency manager owns param lifetime). */
    freeParamsImmediately?: boolean;
    nThreads?: number;
    wtype?: SdType;
    rngType?: RngType;
    samplerRngType?: RngType;
    prediction?: Prediction;
    loraApplyMode?: LoraApplyMode;
    /** @deprecated Maps to `paramsBackend: "*=cpu"`; prefer `paramsBackend`. */
    offloadParamsToCpu?: boolean;
    enableMmap?: boolean;
    /** @deprecated Maps to a `te=cpu` entry in `backend`; prefer `backend`. */
    keepClipOnCpu?: boolean;
    /** @deprecated Maps to a `controlnet=cpu` entry in `backend`; prefer `backend`. */
    keepControlNetOnCpu?: boolean;
    /** @deprecated Maps to a `vae=cpu` entry in `backend`; prefer `backend`. */
    keepVaeOnCpu?: boolean;
    /**
     * Runtime backend assignments per module, e.g. "te=cpu,vae=cpu" or
     * "diffusion=cuda0". See upstream docs/backend.md.
     */
    backend?: string;
    /**
     * Parameter (weight) placement assignments, e.g. "*=cpu" or "te=disk".
     * A nonempty value disables autoFit.
     */
    paramsBackend?: string;
    /** Weight distribution for multi-device modules: "layer" (default), "row", or per-module, e.g. "diffusion=row". */
    splitMode?: string;
    /** Optional per-device GiB budget for managed weights/buffers. */
    maxVram?: string;
    /** Automatic compute placement (default true upstream). */
    autoFit?: boolean;
    /** Comma-separated ggml RPC server addresses. */
    rpcServers?: string;
    /** tokenizer.json path or per-encoder assignments; required for PiD and Lens. */
    tokenizer?: string;
    /** Load all params at model-load time instead of lazily on first use. */
    eagerLoad?: boolean;
    /** Disable asynchronous next-segment weight prefetch. */
    disablePrefetch?: boolean;
    /** Max cached conditioning entries per context; 0 disables (upstream default 4). */
    conditioningCacheSize?: number;
    flashAttn?: boolean;
    /** SageAttention (quantized attention). */
    sageAttn?: boolean;
    diffusionFlashAttn?: boolean;
    taePreviewOnly?: boolean;
    diffusionConvDirect?: boolean;
    vaeConvDirect?: boolean;
    /** @deprecated Moved to per-generation options (ImageGenerationOptions / VideoGenerationOptions). */
    circularX?: boolean;
    /** @deprecated Moved to per-generation options (ImageGenerationOptions / VideoGenerationOptions). */
    circularY?: boolean;
    forceSdxlVaeConvScale?: boolean;
    /** Maps to model_args `chroma_use_dit_mask=...`; prefer `modelArgs`. */
    chromaUseDitMask?: boolean;
    /** Maps to model_args `chroma_use_t5_mask=...`; prefer `modelArgs`. */
    chromaUseT5Mask?: boolean;
    /** Maps to model_args `chroma_t5_mask_pad=...`; prefer `modelArgs`. */
    chromaT5MaskPad?: number;
    /** Maps to model_args `qwen_image_zero_cond_t=...`; prefer `modelArgs`. */
    qwenImageZeroCondT?: boolean;
    /**
     * Extra model args as a key=value list, e.g.
     * "qwen_image_zero_cond_t=true,chroma_use_dit_mask=false".
     */
    modelArgs?: string;
}

// --- Generation options ---

export interface ImageGenerationOptions extends AbortableOptions {
    prompt?: string;
    negativePrompt?: string;
    clipSkip?: number;
    initImage?: SdImage;
    refImages?: SdImage[];
    /** @deprecated Maps to refImageArgs `resize_before_vae=false` when false; prefer `refImageArgs`. */
    autoResizeRefImage?: boolean;
    /** @deprecated Maps to refImageArgs `ref_index_mode=increase`; prefer `refImageArgs`. */
    increaseRefIndex?: boolean;
    /** Reference-image args as a key=value list, e.g. "resize_before_vae=false,ref_index_mode=increase". */
    refImageArgs?: string;
    maskImage?: SdImage;
    width?: number;
    height?: number;
    sampleParams?: SampleParams;
    strength?: number;
    seed?: number;
    batchCount?: number;
    controlImage?: SdImage;
    controlStrength?: number;
    /** Seamless tiling along x (moved here from ContextOptions upstream). */
    circularX?: boolean;
    /** Seamless tiling along y (moved here from ContextOptions upstream). */
    circularY?: boolean;
    photoMaker?: PhotoMakerParams;
    vaeTiling?: TilingParams;
    cache?: CacheParams;
    hires?: HiresParams;
    loras?: LoraDefinition[];
}

export interface VideoGenerationOptions extends AbortableOptions {
    prompt?: string;
    negativePrompt?: string;
    clipSkip?: number;
    initImage?: SdImage;
    endImage?: SdImage;
    controlFrames?: SdImage[];
    width?: number;
    height?: number;
    sampleParams?: SampleParams;
    highNoiseSampleParams?: SampleParams;
    moeBoundary?: number;
    strength?: number;
    seed?: number;
    videoFrames?: number;
    /** Target frames per second (model-dependent). */
    fps?: number;
    vaceStrength?: number;
    /** Seamless tiling along x. */
    circularX?: boolean;
    /** Seamless tiling along y. */
    circularY?: boolean;
    vaeTiling?: TilingParams;
    cache?: CacheParams;
    loras?: LoraDefinition[];
}

export interface UpscalerOptions extends AbortableOptions {
    esrganPath: string;
    offloadParamsToCpu?: boolean;
    direct?: boolean;
    nThreads?: number;
    tileSize?: number;
}

export interface UpscaleOptions extends AbortableOptions {}

export interface ConvertOptions extends AbortableOptions {
    inputPath?: string;
    outputPath: string;
    vaePath?: string;
    outputType?: SdType;
    tensorTypeRules?: string;
    convertName?: boolean;
    /**
     * Multi-component conversion (upstream `convert_with_components`):
     * supplying any of the component paths below — or `loras` / explicit
     * `nThreads` behaviour — assembles the output from separate component
     * files instead of a single input file. `inputPath` is then the optional
     * base model.
     */
    clipLPath?: string;
    clipGPath?: string;
    t5xxlPath?: string;
    diffusionModelPath?: string;
    /** Threads for the components conversion. Default: physical core count. */
    nThreads?: number;
    /** LoRAs baked into the converted output (components conversion only). */
    loras?: LoraDefinition[];
}

export type SDVersionSlug =
    | 'sd1' | 'sd1_inpaint' | 'sd1_pix2pix' | 'sd1_tiny_unet'
    | 'sd2' | 'sd2_inpaint' | 'sd2_tiny_unet'
    | 'sdxs_512_ds' | 'sdxs_09'
    | 'sdxl' | 'sdxl_inpaint' | 'sdxl_pix2pix' | 'sdxl_vega' | 'sdxl_ssd1b'
    | 'svd'
    | 'sd3'
    | 'flux' | 'flux_fill' | 'flux_controls' | 'flex2'
    | 'chroma_radiance'
    | 'wan2' | 'wan2_2_i2v' | 'wan2_2_ti2v'
    | 'qwen_image'
    | 'flux2' | 'flux2_klein'
    | 'z_image'
    | 'ovis_image'
    | 'unknown';

export interface ModelMetadata {
    /** Detected version as a stable programmatic slug. */
    version: SDVersionSlug;
    /** Human-readable version label (e.g. "SD3.x", "Flux", "SDXL Inpaint"). */
    versionLabel: string;

    isUnet: boolean;
    isDit: boolean;
    isSd1: boolean;
    isSd2: boolean;
    isSdxl: boolean;
    isSd3: boolean;
    isFlux: boolean;
    isFlux2: boolean;
    isWan: boolean;
    isQwenImage: boolean;
    isZImage: boolean;
    isInpaint: boolean;
    isControl: boolean;

    hasDiffusionModel: boolean;
    hasVae: boolean;
    hasClipL: boolean;
    hasClipG: boolean;
    hasT5xxl: boolean;
    hasControlNet: boolean;
    isLora: boolean;

    tensorCount: number;
    /** Estimated bytes occupied by parameters once loaded (file dtypes, no conversion). */
    estParamsBytes: number;

    /** Tensor count grouped by ggml type name (e.g. { f16: 1234, f32: 56 }). */
    weightTypes: Record<string, number>;
    diffusionWeightTypes: Record<string, number>;
    vaeWeightTypes: Record<string, number>;
    conditionerWeightTypes: Record<string, number>;
}

export interface CannyOptions {
    highThreshold?: number;
    lowThreshold?: number;
    weak?: number;
    strong?: number;
    inverse?: boolean;
}

export interface PreviewOptions {
    mode?: PreviewMode;
    interval?: number;
    denoised?: boolean;
    noisy?: boolean;
}

// --- Callback data types ---

export interface LogCallbackData {
    level: LogLevel;
    text: string;
}

export interface ProgressCallbackData {
    step: number;
    steps: number;
    time: number;
}

export interface PreviewCallbackData {
    step: number;
    isNoisy: boolean;
    /**
     * Sampling pass this preview belongs to, numbered from 1 since the last
     * setPreviewCallback call (multi-pass flows like hires fix run several).
     * 0 before sampling starts.
     */
    samplePass: number;
    /** Actual step count of that pass (0 before sampling starts). */
    totalSteps: number;
    frames: SdImage[];
}

// --- Video / audio results ---

export interface SdAudio {
    sampleRate: number;
    channels: number;
    /** Interleaved samples, `sampleCount * channels` floats. */
    data: Float32Array;
}

/**
 * generateVideo resolves with the frames array; audio-capable models also
 * attach the decoded `audio` track, and `fps` carries the effective encoding
 * frame rate reported by the engine.
 */
export type VideoGenerationResult = SdImage[] & { audio?: SdAudio; fps: number };

// --- ADetailer ---

export interface AdetailerOptions extends AbortableOptions {
    /** Path to the detector model (YOLO-style face/hand detector). */
    detectorPath: string;
    nThreads?: number;
    /** Runtime backend assignments, same syntax as ContextOptions.backend. */
    backend?: string;
    /** Parameter placement assignments, same syntax as ContextOptions.paramsBackend. */
    paramsBackend?: string;
}

export interface AdetailOptions extends AbortableOptions {
    /** Prompt for the detail inpaint pass. */
    prompt?: string;
    negativePrompt?: string;
    /** Extra upstream ADetailer args as a key=value list. */
    extraAdArgs?: string;
    /**
     * Generation parameters for the inpaint pass (same surface as
     * generateImage: strength, sampleParams, seed, ...). Defaults otherwise.
     */
    inpaint?: Omit<ImageGenerationOptions, 'signal'>;
}

// --- Classes ---

export class StableDiffusionContext {
    private constructor();
    static create(options: ContextOptions): Promise<StableDiffusionContext>;
    generateImage(options: ImageGenerationOptions): Promise<SdImage[]>;
    generateVideo(options: VideoGenerationOptions): Promise<VideoGenerationResult>;
    getDefaultSampleMethod(): SampleMethod;
    getDefaultScheduler(sampleMethod?: SampleMethod): Scheduler;
    /**
     * Friendly model-version string for the loaded model (e.g. "SDXL",
     * "SD 3.5 Large", "Flux"), or "Unknown".
     */
    getModelVersionName(): string;
    /**
     * Attach a ControlNet to this live context without reloading the base
     * model (replaces any currently attached one). Serialized on the same
     * per-ctx queue as generateImage / generateVideo, because the hot-swap
     * is unsafe while a generation is in flight.
     */
    loadControlNet(path: string): Promise<void>;
    /** Detach and free the currently attached ControlNet. Serialized like loadControlNet. */
    unloadControlNet(): Promise<void>;
    /** Whether a ControlNet is currently attached. */
    hasControlNet(): boolean;
    /**
     * Cancel any in-flight or queued generateImage / generateVideo
     * call on this context. Concurrent calls on a different context
     * are unaffected.
     *
     * `{ mode: 'skip-pending' }` is a batch soft-cancel: the engine
     * finishes the image currently being sampled, skips the remaining
     * batch latents, and the generate call resolves with the completed
     * images (no AbortError). The default `'all'` stops at the next
     * step and rejects with AbortError.
     */
    abort(options?: { mode?: 'all' | 'skip-pending' }): void;
    /**
     * Cancel any in-flight work on this context, then release the native
     * resources. Returns immediately; the worker observes the abort at its
     * next checkpoint and the native ctx is freed when the last reference
     * (wrapper or worker) drops.
     */
    close(): void;
    readonly isClosed: boolean;
}

export class AdetailerContext {
    private constructor();
    /** Create an ADetailer context (loads the detector model on a background thread). */
    static create(options: AdetailerOptions): Promise<AdetailerContext>;
    /**
     * Run an automatic detail-fix pass (face/hand inpaint) over an image.
     * The inpaint generation runs on `sdCtx`; the call is serialized on both
     * this adetailer's queue and that context's generate queue, and an abort
     * maps to `sdCtx.abort()`.
     */
    adetail(sdCtx: StableDiffusionContext, image: SdImage, options?: AdetailOptions): Promise<SdImage>;
    /** Release the detector context. See {@link StableDiffusionContext#close}. */
    close(): void;
    readonly isClosed: boolean;
}

export class UpscalerContext {
    private constructor();
    static create(options: UpscalerOptions): Promise<UpscalerContext>;
    upscale(image: SdImage, upscaleFactor?: number, options?: UpscaleOptions): Promise<SdImage>;
    getUpscaleFactor(): number;
    /**
     * Cancel any in-flight or queued upscale call on this context.
     * Concurrent calls on a different context are unaffected.
     */
    abort(): void;
    /**
     * Cancel any in-flight work, then release native resources. See
     * {@link StableDiffusionContext#close}.
     */
    close(): void;
    readonly isClosed: boolean;
}

// --- Free functions ---

export function convert(options: ConvertOptions): Promise<boolean>;
/**
 * Read model metadata (version, components, weight type stats, estimated
 * memory) directly from the file header without loading any tensor data.
 * Works on safetensors, gguf, and ckpt files.
 */
export function extractMetaData(path: string): Promise<ModelMetadata>;
export function preprocessCanny(image: SdImage, options?: CannyOptions): boolean;
export function setLogCallback(callback: ((data: LogCallbackData) => void) | null): void;
export function setProgressCallback(callback: ((data: ProgressCallbackData) => void) | null): void;
export function setPreviewCallback(
    callback: ((data: PreviewCallbackData) => void) | null,
    options?: PreviewOptions,
): void;
export function getSystemInfo(): string;
export function getNumPhysicalCores(): number;
export function version(): string;
export function commit(): string;

export interface DeviceInfo {
    /** Device name accepted by the backend / paramsBackend assignment specs (e.g. "MTL0", "CUDA0", "CPU"). */
    name: string;
    /** Human-readable description (e.g. "Apple M3 Max"). */
    description: string;
}

/** List available ggml backend devices for backend / paramsBackend assignments. */
export function listDevices(): DeviceInfo[];

/**
 * Read an ESRGAN model's native upscale factor from its metadata without
 * creating an upscaler context. Returns 0 if the file is not a recognized
 * RGB ESRGAN model.
 */
export function getUpscalerModelScale(path: string): number;

// --- Importance-matrix (imatrix) workflow for quantization quality ---

/**
 * Load a previously saved importance matrix so a following convert() weighs
 * tensors by importance. Returns false if the file could not be loaded.
 */
export function loadImatrix(path: string): boolean;
/** Save the importance matrix collected since enableImatrixCollection(). */
export function saveImatrix(path: string): void;
/** Start collecting importance data during generations. */
export function enableImatrixCollection(): void;
/** Stop collecting importance data. */
export function disableImatrixCollection(): void;
