# Upstream stable-diffusion.cpp bugs fixed by local patches

This document describes the four upstream bugs that `patches/` carries fixes
for. Each entry is self-contained so the issue can be filed upstream verbatim.

---

## 1. SD3 conditioner: rank-2 zero fallback for missing `clip_g` pooled output

**Patch:** `patches/0001-conditioner-sd3-pooled-rank-mismatch.patch`
**File:** `src/conditioning/conditioner.hpp` — `SD3CLIPEmbedder::get_learned_condition_common`

### Expected behaviour
When `clip_g` is absent (e.g. SD3 medium loaded without the larger CLIP), the
zero-fallback `pooled_g` should match the rank/shape of the real path so the
subsequent `sd::ops::concat(pooled_l, pooled_g, axis=0)` succeeds.

### Actual behaviour
`sd::ops::concat` aborts with:

```
Tensor concat requires same rank: lhs_dim=1, rhs_dim=2
```

…on any SD3 generation where `clip_l` is real but `clip_g` is missing. The
real `clip_l` path produces a rank-1 pooled output of shape `[1280]`, but the
fallback constructs a rank-2 zero of shape `[1280, 1]`.

### Tech details
The `sd::Tensor` migration in upstream commit `f16a110` ("refactor: migrate
generation pipeline to sd::Tensor") inflated the rank of the fallback:

| Version | Construction | Rank |
| --- | --- | --- |
| pre-refactor (`faabc5ad`) | `ggml_new_tensor_1d(... 1280)` | 1 |
| post-refactor (`f16a110`) | `sd::Tensor<float>::zeros({1280, 1})` | 2 |

The working path (real `clip_g`) still returns rank-1 `[1280]`, so concat
along axis 0 hits a rank mismatch and the whole conditioning step throws.

### Fix
Restore the rank-1 zero fallback:

```cpp
- pooled_g = sd::Tensor<float>::zeros({1280, 1});
+ pooled_g = sd::Tensor<float>::zeros({1280});
```

### Encountered in node-stable-diffusion-cpp
SD3 / SD3.5 medium loaded without `clip_g` (clip_l-only configurations,
common on lower-VRAM machines) failed to generate at all — the engine
threw on the conditioning step, before any sampling. From the JS side
this surfaced as a hard error on `ctx.txt2img(...)` for any SD3 prompt
on a clip_l-only context, with no obvious clue that the issue was the
zero-fallback shape rather than the user's model setup.

---

## 2. SD3 conditioner: chunk-loop overruns shorter tokenizer vectors — FIXED UPSTREAM

**Patch:** removed (was `patches/0002-conditioner-sd3-chunk-overrun.patch`)
**File:** `src/conditioning/conditioner.hpp` — `SD3CLIPEmbedder::get_learned_condition_common`
**Status:** fixed upstream by https://github.com/leejet/stable-diffusion.cpp/pull/2111
(merged 2026-10-08, commit `e16d26a`) with the same approach as our patch:
each shorter tokens vector is padded out to `chunk_count * chunk_len` with
synthesized empty chunks via the tokenizer's own `pad_tokens`. The local
patch was dropped when the submodule moved past that commit. The historical
analysis below is retained for reference.

### Expected behaviour
On a moderately long prompt where T5 tokenizes into more chunks than CLIP, all
three tokenizer streams (`clip_l`, `clip_g`, `t5`) should be consumed for the
same number of chunks without reading past the end of any vector — and output
should be deterministic across runs with the same seed and prompt.

### Actual behaviour
- **CPU backend:** `GGML_ASSERT(i01 < ne01)` failure inside
  `ggml_compute_forward_get_rows` when an out-of-bounds garbage `int32` lands
  outside `[0, vocab_size)`. Process aborts.
- **GPU backends (Metal / Vulkan / CUDA):** no bounds check, so garbage
  indices return arbitrary embedding rows. Output is silently non-deterministic
  on long prompts: same seed + same prompt produces visually-similar but
  not-identical images across runs, depending on heap state at call time.

### Tech details
`get_learned_condition_common` derives a single `chunk_count` from the
**longest** of the three token vectors:

```cpp
size_t chunk_count = std::max(std::max(
    clip_l_tokens.size(), clip_g_tokens.size()), t5_tokens.size()) / chunk_len;
```

Each tokenizer's `pad_tokens` had run independently before this point and
T5XXL routinely tokenizes longer than CLIP (different vocabulary granularity).
A moderately long prompt easily produces e.g.:

| Tokenizer | `tokens.size()` | chunks needed |
| --- | --- | --- |
| `clip_l` | 77  | 1 |
| `clip_g` | 77  | 1 |
| `t5`     | 154 | 2 |

`chunk_count` becomes 2 and the second loop iteration slices `clip_l_tokens`
with `begin + 2 * chunk_len` — past `.end()`. That is undefined behaviour
reading whatever happens to be in heap memory beyond the vector's storage,
which is then fed to `ggml_get_rows` as token indices.

### Fix
Before the loop, extend each shorter tokens vector out to
`chunk_count * chunk_len` by appending synthesized empty chunks via the
tokenizer's own `pad_tokens` (with empty input). Each appended chunk has the
model's normal empty-prompt structure (BOS + EOS + PAD for CLIP, EOS + PAD
for T5), so the diffusion model sees real, in-distribution embeddings for
the missing-chunk positions instead of literal zero vectors — same trick
sd-webui's chunked prompt path uses.

Verified deterministic on M3 Max + Metal + SD3.5 medium with prompts up to
5 T5 chunks (~385 tokens) where CLIP is at 4 chunks: previously 4-of-5
runs differed with `seed=42`; with the patch all 5 runs are byte-identical.
Short prompts produce bit-identical output to the pre-patch behaviour.

### Encountered in node-stable-diffusion-cpp
The pain showed up on the Metal backend (M3 Max) as a **reproducibility
bug**: the same SD3.5 medium context, same `seed: 42`, same long prompt
produced visibly different images across calls. Initially looked like a
node-stable-diffusion-cpp seeding bug — turned out the seed was being
honoured, but the conditioning vectors fed to the diffusion model were
randomly different per call because of the heap garbage being read past
`clip_l_tokens.end()`. This is the worst flavour of bug: silent, looks
like a user-side issue, only triggers on prompts long enough to push T5
into a second chunk while CLIP stays in one. Short prompts worked fine,
so it didn't show up in basic smoke tests (`smoke.js`).

CPU backend additionally crashed outright with the `GGML_ASSERT` above
on the same long prompts — that's the path that finally pinned the bug
to conditioning.

---

## 3. AYS scheduler: broken SD2.x fallthrough leaves `inputs` empty

**Patch:** `patches/0003-denoiser-ays-sd2-broken-fallthrough.patch`
**File:** `src/runtime/denoiser.hpp` — `AYSScheduler::get_sigmas`

### Expected behaviour
Calling the AYS scheduler on an SD2.x context should warn the user that AYS
isn't tuned for SD2.x and then proceed using the closest available reference
schedule (SD1.5's noise levels) — that is the original author's stated intent
based on the literal `/* fallthrough */` comment in the source.

### Actual behaviour
The host process crashes before the first sampling step on any SD2.x +
`schedule: 'ays'` request. Reproducible from `node-stable-diffusion-cpp` by
loading an SD2.1 model and setting `schedule: 'ays'` on the context.

### Tech details
The author wrote the dispatch as `if / else if`, not a `switch`, so the
literal `/* fallthrough */` comment between the SD2 and SD1 cases is not
honoured by the language:

```cpp
if (sd_version_is_sd2((SDVersion)version)) {
    LOG_WARN("AYS_SCHEDULER not designed for SD2.X models");
} /* fallthrough */
else if (sd_version_is_sd1((SDVersion)version)) {
    // never reached when SD2 branch fired
    inputs = noise_levels[0];
}
```

When the SD2 branch fires the warning is logged and the chain exits with
`inputs` still default-constructed (size 0). Two lines later:

```cpp
if ((n + 1) != inputs.size()) {
    results = log_linear_interpolation(inputs, n + 1);
}
```

`log_linear_interpolation` runs `linear_space(0.f, 1.f, 0)` — which computes
`(1.f - 0.f) / (size_t)(0 - 1)`, i.e. `1.f / SIZE_MAX` cast to float — and
feeds an empty `ref_x` / `ref_y` into `linear_interp`, whose binary-search
precondition (`ref_x[j+1]` access) reads past the end of an empty vector.
Crash.

### Fix
Collapse the SD1 and SD2 cases into a single branch that uses the SD1.5
noise levels and still emits a warning when the model is SD2 — honouring
the original author's stated intent. SD2.x shares SD1.5's noise schedule
shape (same v-pred-or-eps UNet, same training `sigma_max ≈ 14.6`), so the
SD1.5 levels are the closest available reference and the warning still
flags this as best-effort:

```cpp
if (sd_version_is_sd1((SDVersion)version) || sd_version_is_sd2((SDVersion)version)) {
    if (sd_version_is_sd2((SDVersion)version)) {
        LOG_WARN("AYS_SCHEDULER not designed for SD2.X models, "
                 "falling back to SD1.5 noise levels");
    }
    LOG_INFO("AYS_SCHEDULER using SD1.5 noise levels");
    inputs = noise_levels[0];
}
```

### Encountered in node-stable-diffusion-cpp
Setting `schedule: 'ays'` on an SD2.1 context took the entire Node
process down — not a JS exception, a native segfault from the empty-vector
read in `linear_interp`. The host couldn't catch it, so for any user
combining SD2.x + AYS the Node host died silently mid-call. There was no
warning that AYS isn't supported on SD2.x — just a crash on the first
`txt2img` call after context creation.

---

## 4. GGMLRunner: cancellation point only at sampling-step boundary

**Patch:** `patches/0004-ggml-extend-cancel-checkpoint.patch`
**File:** `src/core/ggml_runner.cpp` (pre-reorg: `src/ggml_extend.hpp`) — `GGMLRunner::compute()` template

### Expected behaviour
Aborting a generation from the host (e.g. `node-stable-diffusion-cpp`'s abort
flag) should take effect within "one in-flight ggml graph compute" everywhere
in the pipeline — conditioning, tokenizer/embedder forwards, VAE encode for
img2img, upscaler/ESRGAN, etc. — not just at the diffusion sampling-step
boundary.

### Actual behaviour
Upstream only invokes `pretty_progress` from sampler step boundaries on the
UNet/DiT, so abort checks only fire there. A heavy CLIP/T5 conditioning pass,
a long VAE encode, or an ESRGAN upscale runs uninterruptibly to completion
even after the host has asked to cancel. From a UX perspective the engine
appears unresponsive to abort for several seconds at a time on large models.

### Tech details
Every model runner (CLIP, T5, UNet, VAE, ControlNet, ESRGAN, ...) goes
through the `GGMLRunner::compute()` template. There is no host-driven
checkpoint at that template's entry, so the only cancellation surface is
whatever calls the upstream `pretty_progress` from inside a particular
sampler.

### Fix
Add a single `pretty_progress(0, 0, 0)` call at the entry of `compute()`,
turning the entire heavy-compute path into a cancellation point. This works
because:

- `pretty_progress(0, 0, 0)` with no progress callback registered bails on
  the `step == 0` early-return inside `pretty_progress`. No spurious stdout.
- With a callback registered, the callback is invoked with
  `(step=0, steps=0, time=0)`. `node-stable-diffusion-cpp`'s `CCallback`
  runs its abort check unconditionally and then drops the `(0,0,0)` sentinel
  before dispatching to JS, so the JS-side progress feed stays clean —
  only real sampler/VAE-tile events surface.

```cpp
+ // Cancellation checkpoint. With the host's progress callback
+ // registered, this drives the abort check inside the cb without
+ // surfacing as a real progress event (host filters steps==0).
+ // Without a callback, pretty_progress short-circuits on step==0.
+ pretty_progress(0, 0, 0);
  if (!offload_params_to_runtime_backend()) { ... }
```

Trade-off: cancellation latency becomes "one in-flight ggml graph compute"
everywhere instead of "one sampling step on the diffusion model" only —
strictly an improvement for every non-sampler path.

### Encountered in node-stable-diffusion-cpp
`ctx.abort()` (per-context cancellation, see commit `15f02f5`) appeared
broken on several user-facing flows even though it worked fine during
sampling:

- Calling `abort()` during the **T5 conditioning pass** of an SD3.5
  prompt did nothing for several seconds — T5XXL on CPU is the longest
  single graph compute in the pipeline and it has no internal cancellation
  point.
- **img2img**: aborting before sampling started (during VAE encode of
  the init image) waited for VAE encode to finish.
- **ESRGAN upscale**: completely uninterruptible mid-tile.
- **hires.fix latent-upscale path** (`smoke_hires.js`): the second
  conditioning + VAE pass was uncancellable.

From the JS side this looked like `abort()` being lossy / racy — actual
cause was that the only cancellation surface upstream exposes is the
sampler's `pretty_progress` call, so anything outside the sampler ran
to completion regardless of host abort state. After the patch, every
ggml graph compute in the engine becomes an abort checkpoint, and
`abort()` takes effect within "one in-flight graph" everywhere.
