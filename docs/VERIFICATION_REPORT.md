# LoRA and sampler verification report

This report records the historical LoRA, stable sampler, and multi-rate sampler
verification checkpoint. For the current plugin version, node inventory, and
Ref2VA still-image status, also read the project-root `README.md` and
`features.json`.

Verified on 2026-08-06 against ComfyUI `0.30.0`, source commit
`563b98eefbe643a4cd510ee7f0b43e79880d5a3f`.

## Artifacts

| File | Size | SHA-256 |
|---|---:|---|
| `minimax_h3_turbo_4步加速.safetensors` | 779,849,991 | `9344cd958f8d354da03dd00b7d462933eb5d0cbf11e56a25d8e9911bb971160e` |
| `minimax_h3_turbo_4步加速_comfyui.safetensors` | 779,858,752 | `35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b` |
| `minimax_h3_turbo_4步加速ema.safetensors` | 779,849,991 | `8a1265e81e5368ab0e52cbb990aee3cb59b28b91fdfa415ef8dbabf81aef890e` |
| `minimax_h3_turbo_4步加速ema_comfyui.safetensors` | 779,858,752 | `b07ab477437c6a525dfdaf11107722aad609975ac172f3b577a7a87b228ff7b3` |

## Checks passed

1. Both sources contain exactly 259 paired LoRA modules / 518 BF16 tensors:
   50 main transformer blocks, two token-refiner blocks, and the final AdaLN.
2. Every expected name and exact H3 shape was checked before conversion.
3. Every output key has the `diffusion_model.` prefix required by ComfyUI's
   generic diffusion-model LoRA mapping.
4. All 518 output tensors were read back and compared with their source tensor
   using exact `torch.equal`; no tensor value changed.
5. Both outputs were passed through current `comfy.lora.model_lora_keys_unet()`
   and `comfy.lora.load_lora()`:
   - target stems found: 259
   - adapters parsed: 259 `LoRAAdapter` objects
   - source tensors consumed: 518/518
   - unloaded-key warnings: 0
6. A representative AdaLN adapter was executed through ComfyUI's
   `BypassForwardHook`; its output exactly equaled `B(A(x))`.
7. Official Comfy-Org checkpoint headers were inspected for FL2VA and REF2VA,
   BF16 and INT8 ConvRot, pruned and non-pruned variants:
   - non-pruned base: 259/259 adapter module shapes match
   - pruned base: 208/259 match; all 51 AdaLN inputs are 8 instead of 2688

## Scope limit

An end-to-end video render was not run because no MiniMax-H3 base diffusion
model, text encoder, or VAEs were present in the supplied directory. The
conversion itself, ComfyUI key resolution, adapter parsing, and bypass math were
validated. For a render test, use the non-pruned base files listed in
`README_ComfyUI.md`; using a pruned base would not be a full or safe test of
these LoRAs.

## Dual-clock sampler validation

`minimax-h3-audio-T8` 1.2.0 was installed and validated against the user's
ComfyUI `0.30.0` tree at commit `6f7cd7fce`:

- the four-step video sigma grid is exactly
  `[1, 36/37, 12/13, 4/5, 0]`;
- mapping the same base times to audio shift 3 gives
  `[1, 0.9, 0.75, 0.5, 0]`;
- a synthetic joint H3 velocity test integrates both streams to their exact
  Euler endpoints on their own clocks;
- audio denoise-mask 0 retains ComfyUI's flat-clock inpaint endpoint behavior;
- all 40 plugin tests and Ruff checks pass;
- a CUDA tensor/device regression passes on an NVIDIA GeForce RTX 4060 Ti;
- ComfyUI `--quick-test-for-ci` loads the installed custom node successfully.

No full H3 render was run as part of this local sampler test, because the base
model stack was not placed in this workspace. The user-provided installation
can now run the included `examples/dual_clock_4step_api.json` workflow after its
placeholder model names are replaced.

## Experimental multi-rate sampler validation

The new `MiniMaxH3MultiRateSamplerEXPT8` is isolated in
`nodes_multirate_exp.py` and `sampling_multirate_exp.py`. The stable
`sampling.py` remained byte-for-byte unchanged with SHA-256
`26A3E6BAB2DEBB1519570D28165F682968F97FE828E3AA1541C834B190705CDB`.

Validated properties:

- 4/8 uses microstep counts `[2, 2, 2, 2]` and 4/10 uses
  `[2, 3, 2, 3]`;
- both schedules preserve the exact four video macro boundaries of the stable
  4-step sigma grid;
- `audio_steps` exactly equals the number of complete joint H3 model calls;
- video commits only one frozen-derivative Euler update per macro interval;
- audio is integrated on its shift-3 clock, while denoise-mask 0 still follows
  ComfyUI's flat inpaint clock and lands on the locked endpoint;
- the installed plugin passes 40 tests, Ruff, and ComfyUI whitelist import;
- a real CUDA 4/10 synthetic integration test passes on the NVIDIA GeForce
  RTX 4060 Ti with exactly 10 model calls.

The whitelist startup also reported an existing lock on `user/comfyui.db`
because another ComfyUI process was running; the custom node itself imported
successfully in 0.0 seconds. No full H3 render was run by this automated test,
so 4/8 versus 4/10 perceptual audio quality should be compared in the supplied
workflow using identical seed, prompt, and inputs.

## Frontend workflow validation

Three complete ComfyUI 0.4 frontend workflows were added for stable 4/4, EXP
4/8, and EXP 4/10. Each contains 12 nodes and 18 links, uses the installed
non-pruned H3 INT8 base, NVFP4 H3 text encoder, both H3 VAEs, EMA Turbo LoRA,
and `LoraLoaderBypassModelOnly`. Every node type, input type, and output type was
checked against the live ComfyUI `/object_info` endpoint; all links were also
checked bidirectionally in the plugin test suite. Copies were installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/`.

A fourth ComfyUI 0.4 frontend workflow,
`H3_Still_Edit_22Frames_EXP.json`, covers the experimental Ref2VA still-image
path. It uses the locally available pruned Ref2VA INT8 checkpoint without Turbo
LoRA, the H3 text encoder and video VAE, a 512x512/22-frame/20-step setup, Still
Preflight reporting, middle-frame Still Decode, and PNG output. Twenty-two
frames are on the native `17k+5` grid and map to video latent T=7 and audio
latent T=37, but remain below the approximate 124-frame training range.
The installed copy was listed by the live `/userdata` endpoint; all 13 nodes,
19 links, and serialized input/output types produced zero contract errors
against an isolated current-code `/object_info` server.

## VRAM validation harness

Added `tools/validate_h3_vram.py` as a diagnostic-only harness. It does not modify the stable or
experimental sampler implementations. The tool can inspect API prompts, build a controlled stock
Euler versus dual-clock pair, submit runs through the native ComfyUI API, correlate `/system_stats`
VRAM samples with WebSocket node/progress events, preserve OOM tracebacks, and reject comparisons
whose non-sampling controls differ.

Validated locally against the running ComfyUI `0.30.0` server at commit `2eb609766`:

- live `/system_stats` inspection identified comfy-aimdo `0.4.13`;
- the startup log supplied explicit `DynamicVRAM support detected and enabled` evidence;
- a lightweight API prompt completed through the WebSocket collector and produced node/progress
  events plus baseline samples;
- static analysis identified the stable 4-step setup and an intentionally constructed 12-step
  mismatch;
- unit tests cover API/frontend format detection, DynamicVRAM evidence, A/B rewiring, telemetry
  peak attribution, and controlled-input comparison.

### Real H3 VRAM checkpoint (2026-08-07)

After the model stack became available, the harness was run against the user's known-working
frontend workflow translated to the equivalent API graph. The active path used the non-pruned
FL2VA INT8 ConvRot model, SageAttention patch, Standard bypass Turbo LoRA, H3 text encoder and H3
video/audio VAEs. The muted reference-image node was correctly excluded from execution.

The reported stress scale was reproduced with `0.6M`, 15 seconds aligned to 362 frames, no preview,
and a 2,037.5 MiB pre-run device baseline:

| Treatment | Steps | Status | Duration | Device peak | PyTorch peak | Peak node |
|---|---:|---|---:|---:|---:|---|
| stock Euler + stock scheduler | 4 | success | 1,210.9 s | 16,213.5 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |
| T8 dual clock | 4 | success | 1,631.4 s | 16,182.2 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |
| T8 dual clock stress run | 12 | success | 3,280.2 s | 16,245.5 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |

The generated 4-step pair retained identical non-sampling controls. Its comparison verdict was
`no_material_peak_difference` at a 128 MiB threshold: dual-clock minus stock peak was -31.3 MiB,
and their measured PyTorch peaks were exactly equal. This run therefore does **not** support the
hypothesis that `MiniMaxH3DualClockSamplerT8` bypasses DynamicVRAM/VBAR and causes a material model
residency increase. Both paths are nevertheless extremely close to the 16 GiB device limit, so
small differences in other CUDA users, previews, allocator fragmentation, model cache state, or
workflow wiring can still decide whether an individual run OOMs.

This is one warm-cache A/B sequence on one RTX 4060 Ti 16 GiB environment, not a universal proof.
A cold-start, order-swapped repeat and the affected user's exact API-format official/modified pair
remain the next tests before considering a production sampler change. The 4-step stock control is
for memory attribution only; its audio integration is not numerically equivalent to dual-clock H3.

## ComfyUI FLOW_AV compatibility regression (2026-08-07)

ComfyUI commit `bdcb886a4` introduced `ModelType.FLOW_AV` / `ModelSamplingAV`, required
`model_sampling.audio_scale`, and changed MiniMax H3 from slope-scaled audio velocity to raw audio
velocity. Commit `a464ac335` is the validation HEAD. A property-only workaround would remove the
`AttributeError` but would retain the wrong audio integration math, so version 1.3.1 detects the
active H3 base-model protocol and selects the matching update rule. Its custom samplers expose a
neutral `audio_scale=1.0` because they already own the separate audio clock.

Validation evidence:

- all 63 Audio T8 tests pass, including legacy/current constant-velocity endpoints, mask and
  callback behavior, exact current `MiniMaxH3.audio_scale()` access, stable setup, and EXP setup;
- Ruff passes for Audio T8 and the companion H3 Block Cache project;
- a whitelist cold start imports Audio T8, H3 Block Cache, and H3 Prompt Enhancer together;
- live `/object_info` exposes stable, EXP, conditioning, still-image, Block Cache, and Prompt
  Enhancer nodes;
- real FL2VA INT8 / Qwen3-VL / H3 VAE probes at 512x512, 22 frames and one step completed both
  stable and EXP sampling; the deliberate core `SaveLatent` sink then failed because ComfyUI's
  `SaveLatent` does not support packed `NestedTensor`, after sampler execution had completed;
- a real one-step H3 forward with Block Cache attached also completed, reporting `cached 0/1` and
  a 19.1 MiB CPU cache before the same deliberate post-sampling sink error;
- all 14 Block Cache tests cover current raw audio velocity and simulated legacy slope-scaled
  velocity; all 74 Prompt Enhancer tests pass. The disabled EasyCache directory and RH H3 directory
  contain no active sampling implementation.

## Version 1.3.2 media, VAE, and 2.0MP regression (2026-08-07)

Three independent issues were reproduced and fixed without changing either stable or experimental
sampling mathematics:

- VideoHelperSuite returns its audio as a lazy `Mapping`, not necessarily a concrete `dict`.
  The shared audio validator now accepts the mapping protocol while preserving the same waveform,
  sample-rate, rank, and finite-value checks. A live `VHS_LoadVideo` output from `1.mp4` was connected
  directly to `ref_video_audios.ref_video_audio_0`; conditioning completed and mapped the media as
  `Video 1` plus `Audio 1`.
- Current ComfyUI initializes a generic `audio_sample_rate` attribute on both H3 VAE wrappers, so
  attribute presence cannot distinguish video from audio VAEs. Preflight now identifies the native
  H3 VAE contract from the underlying class or the latent geometry (`24/3D` video, `32/2D` audio).
  Live main and still-image preflights both classified the installed video VAE as `video`; the main
  preflight also classified the installed audio VAE as `audio` and returned `ready=true`.
- The accepted canvas-area envelope was raised from 1,032,192 pixels to 2,088,960 pixels, with
  `1920x1088` accepted exactly and larger test input `1952x1088` rejected. Canvases above the old
  0.98M threshold remain allowed but produce a high-VRAM warning.

Validation evidence:

- 65 project tests pass and Ruff reports no findings;
- isolated ComfyUI whitelist import succeeds against ComfyUI `0.30.0` at `a464ac335`;
- a live `1920x1088`, 22-frame, one-step stable dual-clock run completed a real joint H3 forward
  using the FL2VA INT8 ConvRot model, Qwen3-VL NVFP4 encoder, and both native H3 VAEs;
- that run completed in 30.4 seconds in the then-warm process, and coarse `/system_stats` polling
  observed a minimum of about 1,212 MiB free VRAM on the RTX 4060 Ti 16GB.

The real-model probe stopped at the generated joint latent and did not decode or assess perceptual
quality. It proves that the new boundary is executable for this short one-step case, not that a
2.0MP 124- or 362-frame workflow will fit every 16GB environment. Resolution, frame count, steps,
reference-media size, previews, allocator state, and other loaded models can still determine OOM.

## Version 1.3.3 selectable sampler/scheduler regression (2026-08-08)

The stable `MiniMaxH3DualClockSamplerT8` now appends two optional controls after the existing
`steps`, `shift_video`, and `shift_audio` widgets. `dual_clock_euler + native_flow` remains the
default and executes the same explicit dual-clock sampler and shifted-uniform sigma construction as
the previous five-argument setup. Existing API prompts may omit both new inputs.

Alternative sampler execution is deliberately separated from the custom default. When current
ComfyUI exposes `ModelSamplingAV`, a selected built-in sampler receives a newly patched native
FLOW_AV sampling object with coherent video/audio shifts and audio carry scale. Legacy H3 builds keep
the explicit T8 Euler default but do not expose built-in sampler alternatives. Alternative schedulers
use `comfy.samplers.calculate_sigmas`; changing that time grid is supported plumbing, not a claim of
better Turbo quality.

Validation evidence:

- all 71 project tests pass and Ruff reports no findings;
- implicit defaults and explicit `dual_clock_euler + native_flow` produce identical sampling type,
  sampler function, and sigma tensors;
- current-protocol built-in Euler setup produces native `ModelSamplingAV` with `audio_scale=4.0`
  for shifts 12/3; a simulated legacy protocol rejects that path with a clear FLOW_AV error;
- a non-default `normal` scheduler matches current ComfyUI's scheduler output while retaining the
  explicit T8 Euler audio protocol;
- the supplied eight-step frontend workflow retains its original `[8, 12, 3]` widget array;
- an isolated whitelist import succeeds, and isolated `/object_info` reports the original five
  inputs in required order followed by optional `sampler_name` and `scheduler`, defaulting to
  `dual_clock_euler` and `native_flow`.

No full perceptual H3 comparison across the additional sampler/scheduler matrix was run for this
change. The regression proves routing, backward compatibility, and protocol selection; users should
compare alternative numerical methods against the preserved default with controlled seeds before
adopting them for production.
