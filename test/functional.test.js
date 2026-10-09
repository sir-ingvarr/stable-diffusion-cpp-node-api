// Functional tests — run against a real diffusion model.
//
//   SD_NODE_MODEL_PATH=/path/to/model.safetensors npm test
//
// The whole suite self-skips when SD_NODE_MODEL_PATH is not set, so unit
// tests stay green in environments without a model. Any single-file SD
// model works (SD1.x/SDXL/...); generations run at 256x256 with few steps
// to keep the suite quick.
//
// Optional:
//   SD_NODE_ESRGAN_MODEL_PATH — an ESRGAN upscaler model; enables the
//                               upscaler tests.
'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');

const sd = require('../src/js/index');

const MODEL = process.env.SD_NODE_MODEL_PATH;
const ESRGAN = process.env.SD_NODE_ESRGAN_MODEL_PATH;

const describeModel = MODEL ? describe : describe.skip;

if (!MODEL) {
    it('functional suite (set SD_NODE_MODEL_PATH to enable)', (t) => t.skip());
}

const GEN_DEFAULTS = {
    width: 256,
    height: 256,
    seed: 42,
    sampleParams: { sampleSteps: 4, guidance: { txtCfg: 7 } },
};

function assertImage(img, width, height) {
    assert.equal(img.width, width);
    assert.equal(img.height, height);
    assert.equal(img.channel, 3);
    assert.ok(Buffer.isBuffer(img.data));
    assert.equal(img.data.length, width * height * 3);
}

// Sanity-check that an image looks like an actual render rather than a
// failed/degenerate one: not all-black, not all-white, and with a healthy
// variety of distinct colors (a real diffusion output at 256x256 has
// thousands; a solid fill or stripe artifact has a handful).
function assertLooksGenerated(img, width, height, label) {
    assertImage(img, width, height);
    const { data } = img;
    let sum = 0;
    const colors = new Set();
    for (let i = 0; i < data.length; i += 3) {
        sum += data[i] + data[i + 1] + data[i + 2];
        colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
    }
    const mean = sum / data.length;
    assert.ok(mean > 8, `${label}: image is (near-)black, mean=${mean.toFixed(1)}`);
    assert.ok(mean < 247, `${label}: image is (near-)white, mean=${mean.toFixed(1)}`);
    // A degenerate render (solid fill, stripes, NaN latents) has a handful
    // of colors; even a flat-looking real render at low step counts has
    // hundreds. 64 keeps the check model- and resolution-agnostic.
    assert.ok(colors.size >= 64,
        `${label}: only ${colors.size} unique colors — looks like a failed render`);
}

describeModel('StableDiffusionContext functional', () => {
    /** @type {import('../src/js').StableDiffusionContext} */
    let ctx;

    before(async () => {
        ctx = await sd.StableDiffusionContext.create({ modelPath: MODEL });
    });

    after(() => {
        sd.setProgressCallback(null);
        sd.setPreviewCallback(null);
        ctx?.close();
    });

    describe('model info', () => {
        it('extractMetaData reads the header', async () => {
            const meta = await sd.extractMetaData(MODEL);
            assert.equal(typeof meta.version, 'string');
            assert.equal(typeof meta.versionLabel, 'string');
            assert.ok(meta.tensorCount > 0);
            assert.ok(meta.estParamsBytes > 0);
        });

        it('getModelVersionName returns a non-empty string', () => {
            const name = ctx.getModelVersionName();
            assert.equal(typeof name, 'string');
            assert.ok(name.length > 0);
        });

        it('default sample method and scheduler are reported', () => {
            assert.ok(ctx.getDefaultSampleMethod().length > 0);
            assert.ok(ctx.getDefaultScheduler().length > 0);
            assert.ok(ctx.getDefaultScheduler(ctx.getDefaultSampleMethod()).length > 0);
        });
    });

    describe('ControlNet hot-swap surface', () => {
        it('reports no ControlNet on a plain context', () => {
            assert.equal(ctx.hasControlNet(), false);
        });

        it('loadControlNet rejects a bad path and leaves the ctx clean', async () => {
            await assert.rejects(ctx.loadControlNet('/nonexistent/controlnet.safetensors'),
                /Failed to load ControlNet/);
            assert.equal(ctx.hasControlNet(), false);
        });
    });

    describe('generateImage', () => {
        it('produces an image with progress and preview pass info', async () => {
            let progressEvents = 0;
            let preview = null;
            sd.setProgressCallback(() => { progressEvents++; });
            sd.setPreviewCallback((data) => {
                preview = { samplePass: data.samplePass, totalSteps: data.totalSteps };
            }, { mode: 'proj', interval: 2 });

            const images = await ctx.generateImage({
                ...GEN_DEFAULTS,
                prompt: 'a red apple on a wooden table',
            });
            sd.setProgressCallback(null);
            sd.setPreviewCallback(null);

            assert.equal(images.length, 1);
            assertLooksGenerated(images[0], 256, 256, 'txt2img');
            assert.ok(progressEvents > 0, 'no progress events seen');
            assert.ok(preview, 'no preview events seen');
            assert.ok(preview.samplePass >= 1);
            assert.equal(preview.totalSteps, GEN_DEFAULTS.sampleParams.sampleSteps);
        });

        it('a fixed seed is deterministic', async () => {
            const opts = { ...GEN_DEFAULTS, prompt: 'a lighthouse at dusk' };
            const [a] = await ctx.generateImage(opts);
            const [b] = await ctx.generateImage(opts);
            assert.ok(a.data.equals(b.data), 'same seed produced different images');
        });

        it('a full batch yields distinct images (per-image seed increment)', async () => {
            const images = await ctx.generateImage({
                ...GEN_DEFAULTS, prompt: 'a small boat', batchCount: 2,
            });
            assert.equal(images.length, 2);
            assertLooksGenerated(images[0], 256, 256, 'batch[0]');
            assertLooksGenerated(images[1], 256, 256, 'batch[1]');
            assert.ok(!images[0].data.equals(images[1].data),
                'batch images are identical — seed did not advance');
        });

        it('img2img transforms an init image', async () => {
            const [init] = await ctx.generateImage({
                ...GEN_DEFAULTS, prompt: 'a green meadow',
            });
            const [out] = await ctx.generateImage({
                ...GEN_DEFAULTS,
                prompt: 'a green meadow under a stormy sky',
                initImage: init,
                strength: 0.6,
                seed: 7,
            });
            assertLooksGenerated(out, 256, 256, 'img2img');
            assert.ok(!out.data.equals(init.data), 'img2img returned the init image unchanged');
        });

        it('VAE tiling produces a valid image', async () => {
            const [img] = await ctx.generateImage({
                ...GEN_DEFAULTS,
                prompt: 'a desert at noon',
                vaeTiling: { enabled: true, tileSizeX: 128, tileSizeY: 128 },
            });
            assertLooksGenerated(img, 256, 256, 'vae-tiling');
        });

        it('concurrent calls serialize on the per-ctx queue and both resolve', async () => {
            const [a, b] = await Promise.all([
                ctx.generateImage({ ...GEN_DEFAULTS, prompt: 'alpha' }),
                ctx.generateImage({ ...GEN_DEFAULTS, prompt: 'bravo', seed: 43 }),
            ]);
            assertLooksGenerated(a[0], 256, 256, 'queued A');
            assertLooksGenerated(b[0], 256, 256, 'queued B');
        });
    });

    describe('cancellation', () => {
        it('abort({ mode: "skip-pending" }) resolves with a partial batch', async () => {
            const BATCH = 3;
            let cancelled = false;
            sd.setProgressCallback(({ step, steps }) => {
                // Soft-cancel as soon as the first image's sampling completes.
                if (!cancelled && steps === GEN_DEFAULTS.sampleParams.sampleSteps && step >= steps) {
                    cancelled = true;
                    ctx.abort({ mode: 'skip-pending' });
                }
            });
            const images = await ctx.generateImage({
                ...GEN_DEFAULTS,
                prompt: 'a bowl of fruit',
                batchCount: BATCH,
            });
            sd.setProgressCallback(null);

            assert.ok(cancelled, 'progress callback never saw the end of image 1');
            assert.ok(images.length >= 1 && images.length < BATCH,
                `expected a partial batch, got ${images.length}/${BATCH}`);
            assertLooksGenerated(images[0], 256, 256, 'soft-cancel partial');
        });

        it('hard abort() rejects with Aborted and the ctx stays usable', async () => {
            const pending = ctx.generateImage({
                ...GEN_DEFAULTS,
                sampleParams: { sampleSteps: 20, guidance: { txtCfg: 7 } },
                prompt: 'a blue bird',
            });
            setTimeout(() => ctx.abort(), 500);
            await assert.rejects(pending, /Aborted/);

            const images = await ctx.generateImage({ ...GEN_DEFAULTS, prompt: 'ok again' });
            assertLooksGenerated(images[0], 256, 256, 'post-abort');
        });

        it('opts.signal rejects with AbortError', async () => {
            const ac = new AbortController();
            const pending = ctx.generateImage({
                ...GEN_DEFAULTS,
                sampleParams: { sampleSteps: 20, guidance: { txtCfg: 7 } },
                prompt: 'a long render',
                signal: ac.signal,
            });
            setTimeout(() => ac.abort(), 500);
            await assert.rejects(pending, (err) => err.name === 'AbortError');
        });

        it('an already-aborted signal short-circuits before native work', async () => {
            const ac = new AbortController();
            ac.abort();
            await assert.rejects(
                ctx.generateImage({ ...GEN_DEFAULTS, prompt: 'never runs', signal: ac.signal }),
                (err) => err.name === 'AbortError');
        });
    });

    describe('lifecycle', () => {
        it('close() marks the ctx closed and later calls reject', async () => {
            const scratch = await sd.StableDiffusionContext.create({ modelPath: MODEL });
            assert.equal(scratch.isClosed, false);
            scratch.close();
            assert.equal(scratch.isClosed, true);
            await assert.rejects(
                scratch.generateImage({ ...GEN_DEFAULTS, prompt: 'x' }),
                /Context is closed/);
        });
    });
});

const describeUpscaler = (MODEL && ESRGAN) ? describe : describe.skip;

describeUpscaler('UpscalerContext functional', () => {
    let upscaler;

    before(async () => {
        upscaler = await sd.UpscalerContext.create({ esrganPath: ESRGAN });
    });

    after(() => {
        upscaler?.close();
    });

    it('getUpscaleFactor matches the file-level metadata query', () => {
        const factor = upscaler.getUpscaleFactor();
        assert.ok(factor >= 2);
        assert.equal(sd.getUpscalerModelScale(ESRGAN), factor);
    });

    it('upscales a synthetic image by the model factor', async () => {
        const w = 64, h = 64;
        const data = Buffer.alloc(w * h * 3);
        for (let i = 0; i < data.length; i += 3) {
            data[i] = (i / 3) % 256; data[i + 1] = 128; data[i + 2] = 255 - ((i / 3) % 256);
        }
        const factor = upscaler.getUpscaleFactor();
        const out = await upscaler.upscale({ width: w, height: h, channel: 3, data }, factor);
        assertLooksGenerated(out, w * factor, h * factor, 'upscale');
    });
});
