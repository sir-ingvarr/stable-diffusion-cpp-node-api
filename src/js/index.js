const loadBinding = require('./load-bindings');

const native = loadBinding();

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
        const ctx = await native.StableDiffusionContext.create(options);
        return new StableDiffusionContext(ctx);
    }

    /**
     * Generate images from a text prompt.
     * @param {object} options
     * @returns {Promise<Array<{width: number, height: number, channel: number, data: Buffer}>>}
     */
    async generateImage(options) {
        return this.#native.generateImage(options);
    }

    /**
     * Generate video frames from a text prompt.
     * @param {object} options
     * @returns {Promise<Array<{width: number, height: number, channel: number, data: Buffer}>>}
     */
    async generateVideo(options) {
        return this.#native.generateVideo(options);
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
     * Free the native context resources.
     */
    close() {
        this.#native.close();
    }

    /**
     * Whether the context has been closed.
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
        const ctx = await native.UpscalerContext.create(options);
        return new UpscalerContext(ctx);
    }

    /**
     * Upscale an image.
     * @param {{width: number, height: number, channel: number, data: Buffer}} image
     * @param {number} [upscaleFactor=4]
     * @returns {Promise<{width: number, height: number, channel: number, data: Buffer}>}
     */
    async upscale(image, upscaleFactor = 4) {
        return this.#native.upscale(image, upscaleFactor);
    }

    /**
     * Get the upscale factor for this model.
     * @returns {number}
     */
    getUpscaleFactor() {
        return this.#native.getUpscaleFactor();
    }

    /**
     * Free the native context resources.
     */
    close() {
        this.#native.close();
    }

    /**
     * Whether the context has been closed.
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
    convert: native.convert,
    preprocessCanny: native.preprocessCanny,
    setLogCallback: native.setLogCallback,
    setProgressCallback: native.setProgressCallback,
    setPreviewCallback: native.setPreviewCallback,
    getSystemInfo: native.getSystemInfo,
    getNumPhysicalCores: native.getNumPhysicalCores,
    version: native.version,
    commit: native.commit,
};
