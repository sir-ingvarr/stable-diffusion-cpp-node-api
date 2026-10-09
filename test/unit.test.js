// Unit tests — no model file required. API surface, option validation, and
// error paths that fail fast without loading weights. Supersedes the old
// test/basic.js.
//
// Run: npm test            (runs unit + functional; functional self-skips
//                           when SD_NODE_MODEL_PATH is not set)
'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const os = require('os');
const path = require('path');

const sd = require('../src/js/index');

describe('module surface', () => {
    it('exports the documented API', () => {
        assert.equal(typeof sd.StableDiffusionContext, 'function');
        assert.equal(typeof sd.StableDiffusionContext.create, 'function');
        assert.equal(typeof sd.UpscalerContext, 'function');
        assert.equal(typeof sd.UpscalerContext.create, 'function');
        assert.equal(typeof sd.AdetailerContext, 'function');
        assert.equal(typeof sd.AdetailerContext.create, 'function');
        for (const fn of ['convert', 'extractMetaData', 'preprocessCanny',
                          'setLogCallback', 'setProgressCallback', 'setPreviewCallback',
                          'getSystemInfo', 'getNumPhysicalCores', 'version', 'commit',
                          'listDevices', 'getUpscalerModelScale',
                          'loadImatrix', 'saveImatrix',
                          'enableImatrixCollection', 'disableImatrixCollection']) {
            assert.equal(typeof sd[fn], 'function', `missing export ${fn}`);
        }
    });

    it('version() / commit() return non-empty strings', () => {
        assert.ok(sd.version().length > 0);
        assert.equal(typeof sd.commit(), 'string');
    });

    it('getSystemInfo() / getNumPhysicalCores() report the host', () => {
        assert.ok(sd.getSystemInfo().length > 0);
        assert.ok(sd.getNumPhysicalCores() > 0);
    });

    it('listDevices() reports at least one backend device', () => {
        const devices = sd.listDevices();
        assert.ok(Array.isArray(devices) && devices.length > 0);
        for (const d of devices) {
            assert.equal(typeof d.name, 'string');
            assert.ok(d.name.length > 0);
            assert.equal(typeof d.description, 'string');
        }
        // The CPU backend always exists.
        assert.ok(devices.some((d) => d.name === 'CPU'));
    });

    it('callbacks can be set and cleared', () => {
        sd.setLogCallback(() => {});
        sd.setLogCallback(null);
        sd.setProgressCallback(() => {});
        sd.setProgressCallback(null);
        sd.setPreviewCallback(() => {}, { mode: 'proj', interval: 2 });
        sd.setPreviewCallback(null);
    });
});

describe('fast error paths', () => {
    it('getUpscalerModelScale returns 0 for a non-ESRGAN / missing file', () => {
        assert.equal(sd.getUpscalerModelScale('/nonexistent/upscaler.pth'), 0);
    });

    it('loadImatrix returns false for a missing file', () => {
        assert.equal(sd.loadImatrix('/nonexistent/imatrix.gguf'), false);
    });

    it('imatrix collection toggles without error', () => {
        sd.enableImatrixCollection();
        sd.disableImatrixCollection();
    });

    it('extractMetaData rejects a missing file', async () => {
        await assert.rejects(sd.extractMetaData('/nonexistent/model.safetensors'));
    });

    it('extractMetaData rejects a non-model file', async () => {
        await assert.rejects(sd.extractMetaData(path.join(__dirname, '..', 'package.json')));
    });

    it('convert (single-file) resolves false on a bad input', async () => {
        const ok = await sd.convert({
            inputPath: '/nonexistent/model.safetensors',
            outputPath: path.join(os.tmpdir(), `sd-node-test-conv-${process.pid}.gguf`),
            outputType: 'q8_0',
        });
        assert.equal(ok, false);
    });

    it('convert (components branch) resolves false on a bad input', async () => {
        const ok = await sd.convert({
            diffusionModelPath: '/nonexistent/unet.safetensors',
            outputPath: path.join(os.tmpdir(), `sd-node-test-convc-${process.pid}.gguf`),
            outputType: 'q8_0',
        });
        assert.equal(ok, false);
    });

    it('AdetailerContext.create rejects a bad detector path', async () => {
        await assert.rejects(
            sd.AdetailerContext.create({ detectorPath: '/nonexistent/detector.pt' }),
            /ADetailer/);
    });

    it('AdetailerContext.create requires detectorPath', async () => {
        await assert.rejects(sd.AdetailerContext.create({}), /detectorPath/);
    });

    it('UpscalerContext.create rejects a bad model path', async () => {
        await assert.rejects(
            sd.UpscalerContext.create({ esrganPath: '/nonexistent/esrgan.pth' }));
    });
});

describe('preprocessCanny', () => {
    it('runs in-place on a synthetic RGB image', () => {
        const w = 16, h = 16, c = 3;
        const data = Buffer.alloc(w * h * c);
        // A vertical edge: left half black, right half white.
        for (let y = 0; y < h; y++) {
            for (let x = w / 2; x < w; x++) {
                const o = (y * w + x) * c;
                data[o] = data[o + 1] = data[o + 2] = 255;
            }
        }
        const ok = sd.preprocessCanny({ width: w, height: h, channel: c, data });
        assert.equal(typeof ok, 'boolean');
    });
});
