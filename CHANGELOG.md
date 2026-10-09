# Changelog

All notable changes to `stable-diffusion-cpp-node-api` are documented in this
file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changelog entries start at the 2026-10-09 upstream bump; earlier history lives
in the git log.

## [0.13.0] - 2026-10-09

### Security

- **Bumped `deps/stable-diffusion.cpp` from `3d6064b` (master-593, April 2026)
  to `228c707` (master-948, 2026-10-09)** — 355 upstream commits, including two
  fixes the old pin was missing:
  - heap out-of-bounds write in `load_imatrix()` on duplicate tensor names
    (upstream `4fcc6fe`, #1750);
  - out-of-bounds read from invalid SafeTensors `data_offsets`
    (upstream `e22272e`, #1754).

### Changed

- **Abort now uses upstream's native cancellation** (`sd_cancel_generation`)
  for `generateImage` / `generateVideo` instead of throwing an exception
  through upstream frames — the old mechanism hit `std::terminate`
  mid-generation on the new upstream. The engine polls its cancel flag every
  sampling step; measured abort latency is ~1 step. The `AbortError` contract
  and post-abort context reuse are unchanged. The upscaler keeps the legacy
  throw path (no native cancel API upstream).
- **Progress callback semantics**: upstream now reports lazy tensor-loading
  and other long phases through the same progress callback (with their own
  `steps` totals). To track sampling only, filter on `steps` matching your
  `sampleSteps`, or use the preview callback for step-accurate UI.
- CPU-placement options are deprecated in favour of upstream's backend
  assignment strings and map onto them transparently:
  `keepClipOnCpu` → `backend: "te=cpu"`, `keepVaeOnCpu` → `"vae=cpu"`,
  `keepControlNetOnCpu` → `"controlnet=cpu"`,
  `offloadParamsToCpu` → `paramsBackend: "*=cpu"`.
- `chromaUseDitMask` / `chromaUseT5Mask` / `chromaT5MaskPad` /
  `qwenImageZeroCondT` now fold into upstream's `model_args` key=value list
  (prefer the new `modelArgs` option).
- `autoResizeRefImage` / `increaseRefIndex` now fold into upstream's
  `ref_image_args` list (prefer the new `refImageArgs` option).
- `circularX` / `circularY` moved from context options to per-generation
  options (`generateImage` / `generateVideo`), following upstream. Setting
  them on the context is now a no-op.
- `vaeDecodeOnly` and `freeParamsImmediately` are **no-ops**: the upstream
  device residency manager now loads components on demand and manages weight
  lifetime (see `maxVram` / `paramsBackend`). Both are still accepted for
  back-compat. Pixel-space hires upscalers no longer require
  `vaeDecodeOnly: false`.
- `generateImage` can now return fewer than `batchCount` images when a
  generation is cancelled with upstream's `SD_CANCEL_NEW_LATENTS` mode.

### Added

- New context options passed through to upstream: `backend`, `paramsBackend`,
  `splitMode`, `maxVram`, `autoFit`, `sageAttn`, `rpcServers`, `tokenizer`,
  `eagerLoad`, `disablePrefetch`, `conditioningCacheSize`, `modelArgs`.
- New generation options: `refImageArgs` (images), `fps` (video),
  `circularX` / `circularY` (both).
- New `vaeTiling` options: `temporalTiling`, `extraTilingArgs`.

### Fixed

- Windows build: worked around an MSVC C1001 internal compiler error on
  current VS 17.x toolsets by switching the two `isClosed` accessors from the
  templated `InstanceAccessor<&T::IsClosed>` form to the runtime-pointer
  overload (`napi-inl.h` constexpr instantiation was the crash site). No
  behaviour change.
- Pointer-stability hazard in the options string store: short strings could
  dangle after internal reallocation (vector → deque).
- Video generation no longer leaks the audio track returned by audio-capable
  models (freed until audio output is surfaced to JS).

### Removed

- `patches/0002-conditioner-sd3-chunk-overrun.patch` — fixed upstream by
  [leejet/stable-diffusion.cpp#2111](https://github.com/leejet/stable-diffusion.cpp/pull/2111)
  with the same approach. Patches 0001 (SD3 pooled-rank fallback), 0003
  (AYS SD2 crash) and 0004 (cancellation checkpoint) were regenerated against
  the reorganised upstream tree; the underlying upstream bugs are still
  present (see `docs/UPSTREAM_BUGS.md`).

### Notes

- Verified on macOS ARM64 (Metal, SDXL SafeTensors): `npm test`, `smoke.js`,
  `smoke_abort.js` (pre/immediate/mid-run abort + post-abort reuse),
  `smoke_hires.js` (pixel and latent hires at 1024²).
- Known cosmetic issue: exiting the process without disposing a live context
  can hit a ggml assert during `exit()` teardown on the new upstream. Dispose
  contexts before exit.
