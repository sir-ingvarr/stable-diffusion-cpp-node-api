const loadBinding = require('./load-bindings');

const native = loadBinding();

// withAbortSignal: wraps a native promise with abort handling.
//
// abortFn is the cancellation entry point — typically the per-ctx
// `nativeCtx.abort()` method. Omit it for soft-cancel callers
// (create / convert) that have no per-ctx target to signal.
//
// hardCancel:
//   true  — generateImage / generateVideo / upscale. The wrapper waits
//           for the native worker to fully unwind before settling, so
//           when `await` returns, no native work is in flight on this
//           ctx. (stable-diffusion.cpp is not thread-safe per ctx, so
//           this matters for chaining the next op.)
//   false — create / convert. Soft cancel: reject the wrapper Promise
//           immediately and let the native work continue in the
//           background; its result is discarded.
function withAbortSignal(promise, signal, hardCancel = false, abortFn = () => {}) {
    if (!signal) return promise;

    const abortReason = () => signal.reason ?? new DOMException('The operation was aborted.', 'AbortError');

    if (signal.aborted) {
        abortFn();
        if (!hardCancel) return Promise.reject(abortReason());
        return promise.then(
            () => { throw abortReason(); },
            () => { throw abortReason(); },
        );
    }

    return new Promise((resolve, reject) => {
        let aborted = false;
        const onAbort = () => {
            aborted = true;
            abortFn();
            if (!hardCancel) {
                signal.removeEventListener('abort', onAbort);
                reject(abortReason());
            }
        };
        signal.addEventListener('abort', onAbort, { once: true });
        promise.then(
            (v) => {
                signal.removeEventListener('abort', onAbort);
                if (aborted) reject(abortReason());
                else resolve(v);
            },
            (e) => {
                signal.removeEventListener('abort', onAbort);
                if (aborted) reject(abortReason());
                else reject(e);
            },
        );
    });
}

// Per-ctx serialization. stable-diffusion.cpp is not thread-safe within
// one sd_ctx_t, so two concurrent generateImage / generateVideo / upscale
// calls on the same ctx would corrupt internal state. We chain each
// abortable call through a per-ctx Promise queue so they run one at a
// time, even if the caller doesn't await between them.
//
// Aborting a queued-but-not-yet-running op short-circuits with an
// AbortError before any native work is queued — leaving previously-
// running ops alone.
function chainAbortable(ctx, signal, runOnce) {
    const abortReason = () => signal?.reason ?? new DOMException('The operation was aborted.', 'AbortError');
    const next = ctx._queue.then(() => {
        if (signal?.aborted) return Promise.reject(abortReason());
        return runOnce();
    });
    // Anchor the queue on the settled outcome regardless of resolve/reject,
    // so a thrown op doesn't poison subsequent queued ops.
    ctx._queue = next.then(() => {}, () => {});
    return next;
}

class StableDiffusionContext {
    #native;
    _queue = Promise.resolve();

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
    generateImage(options) {
        const { signal, ...opts } = options || {};
        const n = this.#native;
        return chainAbortable(this, signal, () =>
            withAbortSignal(n.generateImage(opts), signal, true, () => n.abort()),
        );
    }

    /**
     * Generate video frames from a text prompt.
     * @param {object} options
     * @returns {Promise<Array<{width: number, height: number, channel: number, data: Buffer}>>}
     */
    generateVideo(options) {
        const { signal, ...opts } = options || {};
        const n = this.#native;
        return chainAbortable(this, signal, () =>
            withAbortSignal(n.generateVideo(opts), signal, true, () => n.abort()),
        );
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
     * Cancel any in-flight or queued generateImage / generateVideo
     * call on this context. Takes effect at the next cancellation
     * checkpoint inside the running op (typically within a fraction of
     * a second). Queued ops that have not yet started reject
     * immediately with AbortError.
     */
    abort() {
        this.#native.abort();
    }

    /**
     * Release the context. Signals any in-flight generateImage /
     * generateVideo on this ctx to abort at the next cancellation
     * checkpoint, then drops the wrapper's reference. The native ctx
     * is freed once the last reference (worker or wrapper) drops.
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
    _queue = Promise.resolve();

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
    upscale(image, upscaleFactor = 4, options) {
        const { signal } = options || {};
        const n = this.#native;
        return chainAbortable(this, signal, () =>
            withAbortSignal(n.upscale(image, upscaleFactor), signal, true, () => n.abort()),
        );
    }

    /**
     * Get the upscale factor for this model.
     * @returns {number}
     */
    getUpscaleFactor() {
        return this.#native.getUpscaleFactor();
    }

    /**
     * Cancel any in-flight or queued upscale call on this context.
     */
    abort() {
        this.#native.abort();
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
    extractMetaData: native.extractMetaData,
    preprocessCanny: native.preprocessCanny,
    setLogCallback: native.setLogCallback,
    setProgressCallback: native.setProgressCallback,
    setPreviewCallback: native.setPreviewCallback,
    getSystemInfo: native.getSystemInfo,
    getNumPhysicalCores: native.getNumPhysicalCores,
    version: native.version,
    commit: native.commit,
};
