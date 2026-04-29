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
    vaeDecodeOnly?: boolean;
    freeParamsImmediately?: boolean;
    nThreads?: number;
    wtype?: SdType;
    rngType?: RngType;
    samplerRngType?: RngType;
    prediction?: Prediction;
    loraApplyMode?: LoraApplyMode;
    offloadParamsToCpu?: boolean;
    enableMmap?: boolean;
    keepClipOnCpu?: boolean;
    keepControlNetOnCpu?: boolean;
    keepVaeOnCpu?: boolean;
    flashAttn?: boolean;
    diffusionFlashAttn?: boolean;
    taePreviewOnly?: boolean;
    diffusionConvDirect?: boolean;
    vaeConvDirect?: boolean;
    circularX?: boolean;
    circularY?: boolean;
    forceSdxlVaeConvScale?: boolean;
    chromaUseDitMask?: boolean;
    chromaUseT5Mask?: boolean;
    chromaT5MaskPad?: number;
    qwenImageZeroCondT?: boolean;
}

// --- Generation options ---

export interface ImageGenerationOptions extends AbortableOptions {
    prompt?: string;
    negativePrompt?: string;
    clipSkip?: number;
    initImage?: SdImage;
    refImages?: SdImage[];
    autoResizeRefImage?: boolean;
    increaseRefIndex?: boolean;
    maskImage?: SdImage;
    width?: number;
    height?: number;
    sampleParams?: SampleParams;
    strength?: number;
    seed?: number;
    batchCount?: number;
    controlImage?: SdImage;
    controlStrength?: number;
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
    vaceStrength?: number;
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
    inputPath: string;
    outputPath: string;
    vaePath?: string;
    outputType?: SdType;
    tensorTypeRules?: string;
    convertName?: boolean;
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
    frames: SdImage[];
}

// --- Classes ---

export class StableDiffusionContext {
    private constructor();
    static create(options: ContextOptions): Promise<StableDiffusionContext>;
    generateImage(options: ImageGenerationOptions): Promise<SdImage[]>;
    generateVideo(options: VideoGenerationOptions): Promise<SdImage[]>;
    getDefaultSampleMethod(): SampleMethod;
    getDefaultScheduler(sampleMethod?: SampleMethod): Scheduler;
    close(): void;
    readonly isClosed: boolean;
}

export class UpscalerContext {
    private constructor();
    static create(options: UpscalerOptions): Promise<UpscalerContext>;
    upscale(image: SdImage, upscaleFactor?: number, options?: UpscaleOptions): Promise<SdImage>;
    getUpscaleFactor(): number;
    close(): void;
    readonly isClosed: boolean;
}

// --- Free functions ---

/**
 * Cancel any in-flight generate/upscale operation. Takes effect at the
 * next sampling-step boundary (typically within a fraction of a second
 * to a few seconds, depending on step duration). Normally invoked
 * automatically via AbortSignal — call this directly only for the
 * process-wide escape hatch.
 */
export function abort(): void;
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
