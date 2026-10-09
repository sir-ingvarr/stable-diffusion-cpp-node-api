#!/usr/bin/env node
//
// Smoke test for the 0.14.0 feature batch: batch soft-cancel
// (abort({ mode: 'skip-pending' })), ControlNet hot-swap surface,
// getModelVersionName, listDevices, getUpscalerModelScale, preview pass
// info, ADetailer plumbing, imatrix workflow, and the convert components
// branch.
//
// Usage: node smoke_features.js <path-to-model> [path-to-esrgan-model]
//
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const sd = require('./src/js/index');

async function main() {
    const modelPath = process.argv[2];
    const esrganPath = process.argv[3];
    if (!modelPath) {
        console.error('Usage: node smoke_features.js <path-to-model> [path-to-esrgan-model]');
        process.exit(1);
    }

    console.log(`node-stable-diffusion-cpp v${sd.version()} (${sd.commit()})`);

    console.log('\n=== listDevices ===');
    const devices = sd.listDevices();
    console.log(' ', devices.map(d => `${d.name} (${d.description})`).join(', '));
    if (!devices.length) throw new Error('listDevices returned no devices');

    if (esrganPath) {
        console.log('\n=== getUpscalerModelScale ===');
        const scale = sd.getUpscalerModelScale(esrganPath);
        console.log(`  ${path.basename(esrganPath)} → x${scale}`);
    }

    console.log('\n=== imatrix plumbing ===');
    if (sd.loadImatrix('/nonexistent/imatrix.gguf') !== false) {
        throw new Error('loadImatrix on a missing file should return false');
    }
    sd.enableImatrixCollection();
    sd.disableImatrixCollection();
    console.log('  ok (bad load returns false; enable/disable cycle fine)');

    console.log('\n=== convert components branch (bad input resolves false) ===');
    const convOut = path.join(os.tmpdir(), `sd-node-conv-${process.pid}.gguf`);
    const convOk = await sd.convert({
        diffusionModelPath: '/nonexistent/unet.safetensors',
        outputPath: convOut,
        outputType: 'q8_0',
    });
    if (convOk !== false) throw new Error('components convert with bad input should resolve false');
    console.log('  ok');

    console.log('\n=== AdetailerContext error plumbing (bad detector rejects) ===');
    let adErr = null;
    try {
        await sd.AdetailerContext.create({ detectorPath: '/nonexistent/detector.pt' });
    } catch (e) { adErr = e; }
    if (!adErr) throw new Error('expected AdetailerContext.create to reject');
    console.log(`  ok (${adErr.message})`);

    console.log('\n=== loading model ===');
    const t0 = Date.now();
    const ctx = await sd.StableDiffusionContext.create({ modelPath });
    console.log(`  loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

    console.log('\n=== getModelVersionName ===');
    const versionName = ctx.getModelVersionName();
    console.log(`  ${versionName}`);
    if (typeof versionName !== 'string' || !versionName) {
        throw new Error('getModelVersionName returned an empty value');
    }

    console.log('\n=== ControlNet surface ===');
    if (ctx.hasControlNet() !== false) throw new Error('hasControlNet should be false');
    let cnErr = null;
    try { await ctx.loadControlNet('/nonexistent/controlnet.safetensors'); }
    catch (e) { cnErr = e; }
    if (!cnErr) throw new Error('loadControlNet with a bad path should reject');
    if (ctx.hasControlNet() !== false) throw new Error('failed load must not attach a ControlNet');
    console.log(`  ok (no controlnet; bad path rejected: ${cnErr.message})`);

    console.log('\n=== preview pass info ===');
    let previewSeen = null;
    sd.setPreviewCallback((data) => {
        previewSeen = { samplePass: data.samplePass, totalSteps: data.totalSteps };
    }, { mode: 'proj', interval: 2 });

    console.log('\n=== batch soft-cancel (skip-pending) ===');
    const BATCH = 3;
    const STEPS = 6;
    let firstImageDone = false;
    sd.setProgressCallback(({ step, steps }) => {
        // Fire the soft-cancel once the first image's sampling completes.
        if (!firstImageDone && steps === STEPS && step >= steps) {
            firstImageDone = true;
            ctx.abort({ mode: 'skip-pending' });
            process.stderr.write('\n  [soft-cancel requested]\n');
        }
    });
    const images = await ctx.generateImage({
        prompt: 'a red apple on a table',
        width: 512,
        height: 512,
        batchCount: BATCH,
        seed: 42,
        sampleParams: { sampleSteps: STEPS, guidance: { txtCfg: 7 } },
    });
    sd.setProgressCallback(null);
    sd.setPreviewCallback(null);
    console.log(`  resolved with ${images.length}/${BATCH} images (soft-cancel ${firstImageDone ? 'fired' : 'NOT fired'})`);
    if (!firstImageDone) throw new Error('progress callback never saw the end of image 1');
    if (images.length < 1 || images.length >= BATCH) {
        throw new Error(`expected partial batch (1..${BATCH - 1} images), got ${images.length}`);
    }
    if (!previewSeen || previewSeen.samplePass < 1 || previewSeen.totalSteps !== STEPS) {
        throw new Error(`preview pass info wrong: ${JSON.stringify(previewSeen)}`);
    }
    console.log(`  preview info ok (samplePass=${previewSeen.samplePass}, totalSteps=${previewSeen.totalSteps})`);

    console.log('\n=== hard abort still rejects ===');
    const pending = ctx.generateImage({
        prompt: 'a blue bird',
        width: 512,
        height: 512,
        sampleParams: { sampleSteps: 10 },
    });
    setTimeout(() => ctx.abort(), 1500);
    let hardErr = null;
    try { await pending; } catch (e) { hardErr = e; }
    if (!hardErr || !/abort/i.test(hardErr.message)) {
        throw new Error(`expected Aborted rejection, got: ${hardErr}`);
    }
    console.log(`  ok (${hardErr.message})`);

    ctx.close();
    console.log('\nAll feature checks passed.');
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
