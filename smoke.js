#!/usr/bin/env node
//
// Smoke test for node-stable-diffusion-cpp.
//
// Usage:  node smoke.js <path-to-model>
//
// Generates a single image and writes it as a PPM file (viewable in any image viewer).
// Edit the constants below to customize.
//

const fs = require('fs');
const path = require('path');
const sd = require('./src/js/index');

// ── Configuration ──────────────────────────────────────────────────────────────

const PROMPT          = 'a photograph of a cat sitting on a windowsill, golden hour lighting, 8k';
const NEGATIVE_PROMPT = 'blurry, low quality, distorted';
const WIDTH           = 512;
const HEIGHT          = 512;
const SAMPLE_STEPS    = 20;
const CFG_SCALE       = 7.0;
const SEED            = 42;
const BATCH_COUNT     = 1;
const OUTPUT_PATH     = 'smoke_output.ppm';

// ── Helpers ────────────────────────────────────────────────────────────────────

// Write an SdImage as a PPM file (binary P6 format — opens in any image viewer).
function writePPM(filePath, img) {
    const header = `P6\n${img.width} ${img.height}\n255\n`;
    const headerBuf = Buffer.from(header, 'ascii');
    const out = Buffer.concat([headerBuf, img.data]);
    fs.writeFileSync(filePath, out);
}

// ── Main ───────────────────────────────────────────────────────────────────────

async function main() {
    const modelPath = process.argv[2];
    if (!modelPath) {
        console.error('Usage: node smoke.js <path-to-model>');
        console.error('');
        console.error('Provide a path to a Stable Diffusion model file (.gguf, .safetensors, .ckpt).');
        console.error('You can download quantized models from https://huggingface.co');
        process.exit(1);
    }

    if (!fs.existsSync(modelPath)) {
        console.error(`Error: model file not found: ${modelPath}`);
        process.exit(1);
    }

    // Print library info
    console.log(`node-stable-diffusion-cpp v${sd.version()} (${sd.commit()})`);
    console.log(`System: ${sd.getSystemInfo().trim()}`);
    console.log(`Physical cores: ${sd.getNumPhysicalCores()}`);
    console.log('');

    // Set up logging (info and above)
    sd.setLogCallback(({ level, text }) => {
        if (level >= 1) process.stderr.write(text);
    });

    // Set up progress tracking
    sd.setProgressCallback(({ step, steps, time }) => {
        const pct = Math.round((step / steps) * 100);
        process.stderr.write(`\r  Progress: step ${step}/${steps} (${pct}%) [${time.toFixed(1)}s]`);
        if (step === steps) process.stderr.write('\n');
    });

    // Inspect model header (no weights loaded — just safetensors/gguf metadata).
    const meta = await sd.extractMetaData(path.resolve(modelPath));
    const gb = (b) => (b / 1024 / 1024 / 1024).toFixed(2);
    const fmtTypes = (m) => Object.entries(m).map(([k, v]) => `${k}:${v}`).join(' ') || '—';
    console.log(`Model: ${meta.versionLabel} [${meta.version}], ${meta.tensorCount} tensors, ~${gb(meta.estParamsBytes)} GB`);
    console.log(`  diffusion: ${fmtTypes(meta.diffusionWeightTypes)}`);
    console.log(`  vae:       ${fmtTypes(meta.vaeWeightTypes)}`);
    console.log(`  text-enc:  ${fmtTypes(meta.conditionerWeightTypes)}`);
    console.log(`  bundled:   diffusion=${meta.hasDiffusionModel} vae=${meta.hasVae} clipL=${meta.hasClipL} clipG=${meta.hasClipG} t5xxl=${meta.hasT5xxl}`);
    console.log('');

    // Auto-tune options based on what we just learned about the file.
    const genOptions = {};

    // SD3.5 has a documented fp16 VAE overflow on full-image decode that
    // produces all-black output. VAE tiling sidesteps it. Enable by default
    // unless the caller already configured tiling.
    if (meta.isSd3 && !genOptions.vaeTiling) {
        console.log('SD3 detected — auto-enabling VAE tiling to avoid black-image overflow.');
        genOptions.vaeTiling = { enabled: true };
    }

    // Loud warning when the file isn't self-contained: SD3/Flux/Wan/etc.
    // typically need clip_l + clip_g + t5xxl supplied separately.
    if (meta.isDit && meta.hasDiffusionModel && !meta.hasClipL && !meta.hasT5xxl) {
        console.warn(`WARNING: ${meta.versionLabel} model has no embedded text encoders.`);
        console.warn('         Pass clipLPath / clipGPath / t5xxlPath when creating the context.');
    }

    // Load model
    console.log(`Loading model: ${modelPath}`);
    const startLoad = Date.now();
    // Use modelPath for single-file models (ckpt, safetensors, full gguf).
    // Use diffusionModelPath only for standalone diffusion-only component files.
    const ctx = await sd.StableDiffusionContext.create({
        modelPath: path.resolve(modelPath),
    });
    console.log(`Model loaded in ${((Date.now() - startLoad) / 1000).toFixed(1)}s`);

    // Query model defaults
    const defaultMethod = ctx.getDefaultSampleMethod();
    const defaultScheduler = ctx.getDefaultScheduler();
    console.log(`Default sampler: ${defaultMethod}, scheduler: ${defaultScheduler}`);
    console.log('');

    // Generate
    console.log(`Generating ${BATCH_COUNT} image(s) at ${WIDTH}x${HEIGHT}, ${SAMPLE_STEPS} steps, seed=${SEED}`);
    console.log(`Prompt: "${PROMPT}"`);
    console.log('');

    const startGen = Date.now();
    const images = await ctx.generateImage({
        prompt: PROMPT,
        negativePrompt: NEGATIVE_PROMPT,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        batchCount: BATCH_COUNT,
        sampleParams: {
            sampleSteps: SAMPLE_STEPS,
            guidance: {
                txtCfg: CFG_SCALE,
            },
        },
        ...genOptions,
    });
    const genTime = ((Date.now() - startGen) / 1000).toFixed(1);
    console.log(`Generation completed in ${genTime}s`);

    // Save output as PPM (viewable in any image viewer)
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const outPath = images.length === 1
            ? OUTPUT_PATH
            : OUTPUT_PATH.replace('.ppm', `_${i}.ppm`);

        // Verify data size matches dimensions
        const expected = img.width * img.height * img.channel;
        if (img.data.length !== expected) {
            console.warn(`WARNING: data size mismatch! Expected ${expected} bytes (${img.width}x${img.height}x${img.channel}), got ${img.data.length}`);
        }

        writePPM(outPath, img);
        console.log(`Saved ${img.width}x${img.height}x${img.channel} image to ${outPath} (${img.data.length} bytes)`);
    }

    console.log('');
    console.log('Tip: convert PPM to PNG with ImageMagick:');
    console.log(`  magick ${OUTPUT_PATH} output.png`);
    console.log('');
    console.log('Or with sharp in Node.js (from raw RGB):');
    console.log('  const sharp = require("sharp");');
    console.log(`  sharp(img.data, { raw: { width: img.width, height: img.height, channels: img.channel } }).png().toFile("output.png");`);

    // Cleanup
    ctx.close();
    sd.setLogCallback(null);
    sd.setProgressCallback(null);

    console.log('\nDone.');
}

main().catch((err) => {
    console.error('Error:', err.message || err);
    process.exit(1);
});
