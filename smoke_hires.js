#!/usr/bin/env node
//
// Smoke test for the hires.fix path.
//
// Usage:  node smoke_hires.js <path-to-model>
//
// Generates the same image (same prompt, same seed) twice on one context:
//   1. Plain generation at WIDTH x HEIGHT.
//   2. Hires.fix: pass-1 at WIDTH x HEIGHT, then upscale by HIRES_SCALE and run
//      pass-2 denoising at the larger size.
//
// Outputs two PPMs so you can A/B the result. The seed is the same so the only
// difference is the hires pass.
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
const CFG_SCALE       = 5.0;
const SEED            = 42;
const BATCH_COUNT     = 1;

// Hires.fix settings for the second run.
const HIRES_SCALE              = 2;        // 512x512 → 768x768
const HIRES_UPSCALER           = 'lanczos';  // pixel-space, no external model needed
const HIRES_DENOISING_STRENGTH = 0.5;
const HIRES_STEPS              = 10;

const RUNS = [
    { label: 'plain', outputPath: 'smoke_hires_plain.ppm', hires: undefined },
    {
        label: 'hires-pixel',
        outputPath: 'smoke_hires_fixed.ppm',
        hires: {
            enabled: true,
            upscaler: HIRES_UPSCALER,
            scale: HIRES_SCALE,
            steps: HIRES_STEPS,
            denoisingStrength: HIRES_DENOISING_STRENGTH,
        },
    },
    {
        // Same hires pass but a latent-space upscaler. Skips the VAE
        // decode→encode round-trip — typically much faster, slightly softer.
        label: 'hires-latent',
        outputPath: 'smoke_hires_latent.ppm',
        hires: {
            enabled: true,
            upscaler: 'latent_bicubic',
            scale: HIRES_SCALE,
            steps: HIRES_STEPS,
            denoisingStrength: HIRES_DENOISING_STRENGTH,
        },
    },
];

// ── Helpers ────────────────────────────────────────────────────────────────────

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
        console.error('Usage: node smoke_hires.js <path-to-model>');
        process.exit(1);
    }
    if (!fs.existsSync(modelPath)) {
        console.error(`Error: model file not found: ${modelPath}`);
        process.exit(1);
    }

    console.log(`node-stable-diffusion-cpp v${sd.version()} (${sd.commit()})`);
    console.log(`System: ${sd.getSystemInfo().trim()}`);
    console.log('');

    // Quiet sd.cpp INFO/DEBUG chatter, but keep WARN+ so silent fallbacks
    // (e.g. "hires upscaler invalid, disabling") are still visible.
    sd.setLogCallback(({ level, text }) => {
        if (level >= 2) process.stderr.write(text);
    });

    const meta = await sd.extractMetaData(path.resolve(modelPath));
    console.log(`Model: ${meta.versionLabel} [${meta.version}]`);
    console.log('');

    const genOptions = {};
    if (meta.isSd3 && !genOptions.vaeTiling) {
        console.log('SD3 detected — auto-enabling VAE tiling.');
        genOptions.vaeTiling = { enabled: true };
    }

    console.log(`Loading model: ${modelPath}`);
    const startLoad = Date.now();
    const ctx = await sd.StableDiffusionContext.create({
        modelPath: path.resolve(modelPath),
        // Pixel-space hires upscalers (lanczos / nearest / model) decode the
        // latent, upscale the pixels, then re-encode back — so they need VAE
        // encoder weights. The default vaeDecodeOnly=true skips them.
        vaeDecodeOnly: false,
    });
    console.log(`Model loaded in ${((Date.now() - startLoad) / 1000).toFixed(1)}s`);
    console.log('');

    const timings = [];
    for (let r = 0; r < RUNS.length; r++) {
        const run = RUNS[r];
        console.log(`── Run ${r + 1}/${RUNS.length} (${run.label}) ──────────────────────`);
        if (run.hires) {
            const targetW = Math.round(WIDTH * HIRES_SCALE);
            const targetH = Math.round(HEIGHT * HIRES_SCALE);
            console.log(`Pass 1: ${WIDTH}x${HEIGHT}, ${SAMPLE_STEPS} steps`);
            console.log(`Pass 2: ${targetW}x${targetH} via ${run.hires.upscaler}, ${run.hires.steps} steps, denoise=${run.hires.denoisingStrength}`);
        } else {
            console.log(`${WIDTH}x${HEIGHT}, ${SAMPLE_STEPS} steps`);
        }

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
                guidance: { txtCfg: CFG_SCALE },
            },
            ...(run.hires ? { hires: run.hires } : {}),
            ...genOptions,
        });
        const elapsedMs = Date.now() - startGen;
        timings.push({ label: run.label, ms: elapsedMs, outputPath: run.outputPath, dims: `${images[0].width}x${images[0].height}` });

        writePPM(run.outputPath, images[0]);
        console.log(`  ${(elapsedMs / 1000).toFixed(2)}s  →  ${run.outputPath} (${images[0].width}x${images[0].height})`);
        console.log('');
    }

    ctx.close();
    sd.setLogCallback(null);

    console.log('── Summary ──────────────────────────────────────');
    for (const t of timings) {
        console.log(`  ${t.label.padEnd(10)}  ${(t.ms / 1000).toFixed(2)}s   ${t.dims.padEnd(9)}  ${t.outputPath}`);
    }
}

main().catch((err) => {
    console.error('Error:', err.message || err);
    process.exit(1);
});
