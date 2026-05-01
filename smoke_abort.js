#!/usr/bin/env node
//
// Smoke test for cancellation. Verifies that aborting generateImage()
// actually cancels the native run instead of letting it complete and
// silently discarding the result.
//
// The regression we're guarding against: Scope::Scope() used to call
// clearAbort() on the worker thread, which raced with native.abort()
// calls landing between Queue() and Execute() and dropped them. Pre-
// aborted signals were the worst case — the JS Promise rejected
// immediately, but the native run kept going until done.
//
// Usage: node smoke_abort.js <path-to-model>
//

const fs = require('fs');
const path = require('path');
const sd = require('./src/js/index');

const WIDTH        = 256;
const HEIGHT       = 256;
const SAMPLE_STEPS = 20;     // enough room that progress callbacks fire mid-run
const PROMPT       = 'a cat sitting on a windowsill';
const SEED         = 42;

let progressCount = 0;
let onProgress = null;

function reset() {
    progressCount = 0;
    onProgress = null;
}

async function expectReject(promise, label) {
    try {
        await promise;
    } catch (e) {
        return e;
    }
    throw new Error(`FAIL [${label}]: expected promise to reject, but it resolved`);
}

async function main() {
    const modelPath = process.argv[2];
    if (!modelPath) {
        console.error('Usage: node smoke_abort.js <path-to-model>');
        process.exit(1);
    }
    if (!fs.existsSync(modelPath)) {
        console.error(`Error: model file not found: ${modelPath}`);
        process.exit(1);
    }

    // Quiet logs — only warnings/errors. Per-step progress goes through
    // our counter below.
    sd.setLogCallback(({ level, text }) => {
        if (level >= 2) process.stderr.write(text);
    });
    sd.setProgressCallback((data) => {
        progressCount++;
        if (onProgress) onProgress(data);
    });

    console.log(`Loading model: ${modelPath}`);
    const ctx = await sd.StableDiffusionContext.create({
        modelPath: path.resolve(modelPath),
    });

    const baseGenOpts = {
        prompt: PROMPT,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        sampleParams: { sampleSteps: SAMPLE_STEPS },
    };

    // ── Test 1: pre-aborted signal ────────────────────────────────────────
    // Before the fix, withAbortSignal called native.abort() and rejected the
    // wrapped promise immediately, but Scope::Scope() then wiped the flag on
    // the worker thread and the run completed. We'd see SAMPLE_STEPS progress
    // callbacks. After the fix: native side throws at the first step's
    // progress checkpoint, so we see ≤1 callback (the first pretty_progress
    // is consumed by throwIfAborted before the JS tsfn dispatch).
    {
        reset();
        const ctrl = new AbortController();
        ctrl.abort();

        const t0 = Date.now();
        const err = await expectReject(
            ctx.generateImage({ ...baseGenOpts, signal: ctrl.signal }),
            'pre-aborted',
        );
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`pre-aborted:     rejected in ${elapsed}s, progress events=${progressCount}, reason=${err.name}`);

        if (progressCount > 2) {
            throw new Error(
                `FAIL [pre-aborted]: native run was not cancelled — ` +
                `saw ${progressCount} progress events (expected ≤2). ` +
                `Likely regression of the Scope::Scope() clearAbort race.`,
            );
        }
        // Wait a moment for the native worker to fully finish unwinding
        // before we touch the context again.
        await new Promise((r) => setTimeout(r, 50));
    }

    // ── Test 2: abort fired on the same JS tick as generateImage() ────────
    // Same race window as test 1, but with a non-pre-aborted signal — the
    // user calls .abort() right after kicking off the generation.
    {
        reset();
        const ctrl = new AbortController();

        const t0 = Date.now();
        const p = ctx.generateImage({ ...baseGenOpts, signal: ctrl.signal });
        ctrl.abort();

        const err = await expectReject(p, 'immediate');
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`immediate-abort: rejected in ${elapsed}s, progress events=${progressCount}, reason=${err.name}`);

        if (progressCount > 2) {
            throw new Error(
                `FAIL [immediate]: native run was not cancelled — ` +
                `saw ${progressCount} progress events (expected ≤2).`,
            );
        }
        await new Promise((r) => setTimeout(r, 50));
    }

    // ── Test 3: abort partway through sampling ────────────────────────────
    // This path always worked (the Scope ctor race only applies before
    // Execute() starts). Worth a sanity check that we didn't break it.
    {
        reset();
        const ctrl = new AbortController();
        const ABORT_AT_STEP = 3;
        onProgress = ({ step }) => {
            if (step === ABORT_AT_STEP) ctrl.abort();
        };

        const t0 = Date.now();
        const err = await expectReject(
            ctx.generateImage({ ...baseGenOpts, signal: ctrl.signal }),
            'mid-run',
        );
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`mid-run-abort:   rejected in ${elapsed}s, progress events=${progressCount}, reason=${err.name}`);

        if (progressCount >= SAMPLE_STEPS) {
            throw new Error(
                `FAIL [mid-run]: ran to completion despite abort — ` +
                `saw ${progressCount} progress events (expected < ${SAMPLE_STEPS}).`,
            );
        }
        await new Promise((r) => setTimeout(r, 50));
    }

    // ── Test 4: a normal generation still works after aborts ──────────────
    // Confirms the abort flag isn't sticky across operations.
    {
        reset();
        const t0 = Date.now();
        const images = await ctx.generateImage(baseGenOpts);
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`post-abort run:  resolved in ${elapsed}s, progress events=${progressCount}, images=${images.length}`);

        if (images.length === 0 || !images[0].data || images[0].data.length === 0) {
            throw new Error('FAIL [post-abort]: normal generation produced no image data');
        }
    }

    ctx.close();
    sd.setLogCallback(null);
    sd.setProgressCallback(null);

    console.log('\nAll cancellation smoke tests passed.');
}

main().catch((err) => {
    console.error('\n', err.message || err);
    process.exit(1);
});
