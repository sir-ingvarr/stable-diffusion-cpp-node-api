#!/usr/bin/env node
//
// Apply patches in patches/*.patch to the deps/stable-diffusion.cpp working tree.
//
// Each .patch is a unified diff (output of `git diff` from inside the submodule).
// The submodule itself is never modified in git — patches live as plain text in
// this repo and are re-applied to the working tree before each build.
//
// Per-patch state machine:
//
//   reverse-check succeeds + working tree clean wrt pinned commit
//     → OBSOLETE  (upstream pin already contains the fix; nothing for us to do
//                  and the patch file can be removed)
//
//   reverse-check succeeds + working tree dirty
//     → APPLIED   (we already applied it earlier; idempotent skip)
//
//   forward-check succeeds
//     → PENDING   (apply it)
//
//   neither succeeds
//     → STALE     (surrounding code drifted; patch needs to be regenerated)
//
// Flags:
//   --check    Dry-run. Report each patch's state without modifying anything.
//              Exit non-zero if any patch is OBSOLETE or STALE.
//   --strict   In default-mode runs, treat OBSOLETE as a hard error (default
//              warns and continues so contributors can keep building).
//
// Usage: node scripts/apply-patches.js [--check] [--strict]
//

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const submodule = path.join(repoRoot, 'deps', 'stable-diffusion.cpp');
const patchesDir = path.join(repoRoot, 'patches');

const args = new Set(process.argv.slice(2));
const dryRun = args.has('--check');
const strict = args.has('--strict');

function runGit(gitArgs, opts = {}) {
    return execFileSync('git', ['-C', submodule, ...gitArgs], {
        stdio: opts.stdio ?? 'pipe',
        encoding: 'utf8',
    });
}

function tryGit(gitArgs) {
    try {
        runGit(gitArgs, { stdio: 'pipe' });
        return true;
    } catch {
        return false;
    }
}

function filesTouchedByPatch(patchPath) {
    const text = fs.readFileSync(patchPath, 'utf8');
    const files = new Set();
    for (const line of text.split('\n')) {
        const m = line.match(/^diff --git a\/(\S+) b\/\S+/);
        if (m) files.add(m[1]);
    }
    return [...files];
}

function workingTreeCleanFor(files) {
    // True if every listed file matches its pinned HEAD content (no working-tree edits).
    for (const f of files) {
        if (!tryGit(['diff', '--quiet', 'HEAD', '--', f])) return false;
    }
    return true;
}

function classifyPatch(patchPath) {
    const reverseOk = tryGit(['apply', '--reverse', '--check', patchPath]);
    if (reverseOk) {
        const clean = workingTreeCleanFor(filesTouchedByPatch(patchPath));
        return clean ? 'obsolete' : 'applied';
    }
    if (tryGit(['apply', '--check', patchPath])) return 'pending';
    return 'stale';
}

function applyPatch(patchPath) {
    runGit(['apply', patchPath], { stdio: 'inherit' });
}

function main() {
    if (!fs.existsSync(patchesDir)) {
        console.log('apply-patches: no patches/ directory; nothing to do.');
        return 0;
    }
    if (!fs.existsSync(path.join(submodule, '.git'))) {
        console.error(`apply-patches: ${submodule} is not initialized. Run 'git submodule update --init --recursive' first.`);
        return 1;
    }

    const patches = fs.readdirSync(patchesDir)
        .filter((f) => f.endsWith('.patch'))
        .sort();

    if (patches.length === 0) {
        console.log('apply-patches: no patches to apply.');
        return 0;
    }

    let exitCode = 0;
    for (const file of patches) {
        const full = path.join(patchesDir, file);
        const state = classifyPatch(full);

        switch (state) {
            case 'applied':
                console.log(`  ✓ ${file} (already applied)`);
                break;
            case 'pending':
                if (dryRun) {
                    console.log(`  · ${file} (would apply)`);
                } else {
                    applyPatch(full);
                    console.log(`  + ${file} (applied)`);
                }
                break;
            case 'obsolete':
                console.warn(`  ⊘ ${file} (OBSOLETE — upstream pin already contains this fix)`);
                console.warn(`      → drop this file: rm patches/${file}`);
                if (strict || dryRun) exitCode = 1;
                break;
            case 'stale':
                console.error(`  ✗ ${file} (STALE — surrounding code drifted; cannot apply)`);
                console.error(`      → inspect deps/stable-diffusion.cpp and either regenerate the patch`);
                console.error(`        against the new upstream code or remove patches/${file} if the`);
                console.error(`        bug is gone.`);
                exitCode = 1;
                break;
        }
    }

    if (dryRun && exitCode === 0) console.log('apply-patches: all patches healthy.');
    return exitCode;
}

process.exit(main());
