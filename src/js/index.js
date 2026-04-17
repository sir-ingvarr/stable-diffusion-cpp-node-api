const loadBinding = require('./load-bindings');

const native = loadBinding();

// generateImage / generateVideo / upscale: hard cancel at the next
// sampling-step boundary. native.abort() sets a global flag that the
// native progress callback checks; when set, it throws a C++
// exception that unwinds through stable-diffusion.cpp back to the
// worker. Cancellation takes effect at the next step, not instantly.
//
// create() / convert(): abort has no native effect (soft cancel). The
// outer Promise still rejects immediately, but the underlying load /
// conversion continues to completion and its result is discarded.
function withAbortSignal(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) {
        native.abort();
        return Promise.reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
    }
    return new Promise((resolve, reject) => {
        const onAbort = () => {
            native.abort();
            reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
        };
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            (v) => { signal.removeEventListener('abort', onAbort); resolve(v); },
            (e) => { signal.removeEventListener('abort', onAbort); reject(e); },
        );
    });
}

class StableDiffusionContext {
    #native;

    /** @internal — use StableDiffusionContext.create() */
    constructor(nativeCtx) {
        this.#native = nativeCtx;
    }

    /**
     * Create a new StableDiffusion context asynchronously.
     * Model loading happens on a background thread.
     * @param {object} options
     * @returns {Promise<StableDiffusionContext>}
     */
    static async create(options) {
        const { signal, ...opts } = options || {};
        const ctx = await withAbortSignal(native.StableDiffusionContext.create(opts), signal);
        return new StableDiffusionContext(ctx);
    }

    /**
     * Generate images from a text prompt.
     * @param {object} options
     * @returns {Promise<Array<{width: number, height: number, channel: number, data: Buffer}>>}
     */
    async generateImage(options) {
        const { signal, ...opts } = options || {};
        return withAbortSignal(this.#native.generateImage(opts), signal);
    }

    /**
     * Generate video frames from a text prompt.
     * @param {object} options
     * @returns {Promise<Array<{width: number, height: number, channel: number, data: Buffer}>>}
     */
    async generateVideo(options) {
        const { signal, ...opts } = options || {};
        return withAbortSignal(this.#native.generateVideo(opts), signal);
    }

    /**
     * Get the default sample method for this model.
     * @returns {string}
     */
    getDefaultSampleMethod() {
        return this.#native.getDefaultSampleMethod();
    }

    /**
     * Get the default scheduler for the given sample method.
     * @param {string} [sampleMethod]
     * @returns {string}
     */
    getDefaultScheduler(sampleMethod) {
        return this.#native.getDefaultScheduler(sampleMethod);
    }

    /**
     * Release the context. Any in-flight work continues to completion
     * on its own and releases the native context when it finishes.
     */
    close() {
        this.#native.close();
    }

    /**
     * Whether close() has been called on this wrapper.
     * @returns {boolean}
     */
    get isClosed() {
        return this.#native.isClosed;
    }
}

class UpscalerContext {
    #native;

    /** @internal — use UpscalerContext.create() */
    constructor(nativeCtx) {
        this.#native = nativeCtx;
    }

    /**
     * Create a new upscaler context asynchronously.
     * @param {object} options
     * @returns {Promise<UpscalerContext>}
     */
    static async create(options) {
        const { signal, ...opts } = options || {};
        const ctx = await withAbortSignal(native.UpscalerContext.create(opts), signal);
        return new UpscalerContext(ctx);
    }

    /**
     * Upscale an image.
     * @param {{width: number, height: number, channel: number, data: Buffer}} image
     * @param {number} [upscaleFactor=4]
     * @param {{ signal?: AbortSignal }} [options]
     * @returns {Promise<{width: number, height: number, channel: number, data: Buffer}>}
     */
    async upscale(image, upscaleFactor = 4, options) {
        const { signal } = options || {};
        return withAbortSignal(this.#native.upscale(image, upscaleFactor), signal);
    }

    /**
     * Get the upscale factor for this model.
     * @returns {number}
     */
    getUpscaleFactor() {
        return this.#native.getUpscaleFactor();
    }

    /**
     * Release the context. See StableDiffusionContext#close.
     */
    close() {
        this.#native.close();
    }

    /**
     * Whether close() has been called on this wrapper.
     * @returns {boolean}
     */
    get isClosed() {
        return this.#native.isClosed;
    }
}

module.exports = {
    StableDiffusionContext,
    UpscalerContext,

    // Free functions
    convert(options) {
        const { signal, ...opts } = options || {};
        return withAbortSignal(native.convert(opts), signal);
    },
    preprocessCanny: native.preprocessCanny,
    setLogCallback: native.setLogCallback,
    setProgressCallback: native.setProgressCallback,
    setPreviewCallback: native.setPreviewCallback,
    abort: native.abort,
    getSystemInfo: native.getSystemInfo,
    getNumPhysicalCores: native.getNumPhysicalCores,
    version: native.version,
    commit: native.commit,
};
