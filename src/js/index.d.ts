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

export type CacheMode = 'disabled' | 'easycache' | 'ucache' | 'dbcache' | 'taylorseer' | 'cachedit';

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
    scmPolicyDynamic?: boolean;
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

// --- Context creation options ---

export interface ContextOptions {
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

export interface ImageGenerationOptions {
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
    loras?: LoraDefinition[];
}

export interface VideoGenerationOptions {
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

export interface UpscalerOptions {
    esrganPath: string;
    offloadParamsToCpu?: boolean;
    direct?: boolean;
    nThreads?: number;
    tileSize?: number;
}

export interface ConvertOptions {
    inputPath: string;
    outputPath: string;
    vaePath?: string;
    outputType?: SdType;
    tensorTypeRules?: string;
    convertName?: boolean;
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
    upscale(image: SdImage, upscaleFactor?: number): Promise<SdImage>;
    getUpscaleFactor(): number;
    close(): void;
    readonly isClosed: boolean;
}

// --- Free functions ---

export function convert(options: ConvertOptions): Promise<boolean>;
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
