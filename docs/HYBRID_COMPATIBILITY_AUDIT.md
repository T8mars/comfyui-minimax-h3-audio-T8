# MiniMax H3 Hybrid Compatibility Audit (Advanced)

Version 1.17.0 adds an isolated, opt-in compatibility checkpoint for the
experimental FL2VA×Ref2VA Hybrid model path. It does not modify model weights,
Conditioning, sampler mathematics, caches, global H3 code, or the connected
MODEL object. The first output is the exact same Python MODEL object received
by the node.

## Placement

Place the node at the end of the MODEL patch chain and before `BasicGuider`:

```text
Hybrid Loader
  -> optional LoRA
  -> optional SageAttention
  -> optional Block Cache / Long Video / MultiKeyframe
  -> stable or EXP sampler setup
  -> Hybrid Compatibility Audit (Advanced)
  -> BasicGuider
```

Connect the final H3 `positive` Conditioning as well when using Long Video,
MultiKeyframe, or reference media. The Conditioning input is optional so a
model-only preflight remains possible.

The default `report_only` mode always passes the MODEL through. It can report
`compatible=false`, but it does not stop generation. The opt-in
`block_hard_conflicts` mode raises before the guider executes when a mechanical
contract failure is proven.

## What is audited

- Hybrid attachment schema, exact validated FL2VA/Ref2VA hashes, curve hashes,
  canonical recipe, identity fingerprint, operation count, and payload size.
- Every expected AdaLN `offset-set` entry, including offset and target shape.
- Hybrid-before-LoRA ordering. A patch before the Hybrid set is invalid;
  a later patch overlapping selected AdaLN rows is mechanically ordered but is
  rejected as uncalibrated. Attention/MLP-only patches outside selected AdaLN
  tensors remain mechanically compatible.
- MiniMax H3 Block Cache prototype, first/last `double_block` replacements, and
  both wrapper groups. Partial or foreign double-block replacement fails closed.
- MiniMax SageAttention replacement across every H3 DiT block. Partial coverage
  or an unknown attention-forward replacement fails closed.
- Long Video and MultiKeyframe scoped MODEL patches, matching Conditioning
  markers, patch version, and mutual exclusion.
- Stock/unpatched, stable dual-clock/native AV, experimental multi-rate, or
  unknown custom `model_sampling` routes.
- Loader-recorded VRAM-policy provenance, ModelPatcher DynamicVRAM state,
  current whole-device free VRAM, ComfyUI/AIMDO state, and host commit headroom.

The report uses stable issue codes. Examples include
`patch_precedes_hybrid_set`, `adaln_patch_overlaps_hybrid`,
`block_cache_contract_incomplete`, `sage_attention_contract_incomplete`,
`long_video_multikeyframe_conflict`,
`multikeyframe_conditioning_model_mismatch`,
`vram_policy_required_not_applied`,
`current_gpu_headroom_below_gate`, and
`host_commit_headroom_below_gate`.

## VRAM and VBAR boundary

When a T8 VRAM policy is connected to the Hybrid Loader, the Loader stores a
small immutable provenance attachment on the returned MODEL. It records whether
the policy was actually applied, the policy fingerprint and mode, the reserve
target, the ComfyUI/AIMDO setter routes, cleanup scope, and gate results. It
does not embed the large before/after telemetry tree.

`require_applied_vram_policy=true` rejects a missing or `report_only` policy.
The defaults of 512 MiB current whole-device headroom and 16 GiB host commit
headroom are preflight gates only. They are deliberately configurable because
hardware and workflows differ.

The audit always reports `memory_safe_claim=false`. VBAR pages model weights;
it does not provide an upper bound for activations, attention workspaces,
VAE/CLIP allocations, CUDA context, pinned host buffers, another process,
driver allocations, or host commit. A large page file prevents one class of
host-backing failure but does not prove that a future denoising peak cannot OOM.

## 736x416 / 124-frame Stock20 cache-hit matrix

On 2026-08-13 the exact local RTX 4060 Ti 16 GiB profile was tested with the
FL2VA pruned INT8 base, 27.69 MiB blocks-25-to-49 video+audio Hybrid artifact,
Qwen3-VL NVFP4, both H3 VAEs, `10A.jpg`, KJ all-block H3 SageAttention,
T8 H3 Block Cache at its default `0.12` threshold with a CPU cache, stable
dual-clock Stock20, 4 GiB applied policy, and strict compatibility audit.

| Run | Process state | Duration | Cache | Device peak | Headroom |
|---|---|---:|---:|---:|---:|
| cold1 | fresh process | 177.765 s | 6/20 | 15386.70 MiB | 992.80 MiB |
| cold2 | fresh process | 181.422 s | 6/20 | 15572.80 MiB | 806.70 MiB |
| cold3 | fresh process | 195.656 s | 6/20 | 15613.12 MiB | 766.38 MiB |
| warm1 | cold3 process | 128.219 s | 6/20 | 15426.05 MiB | 953.45 MiB |
| warm2 | cold3 process | 128.344 s | 6/20 | 15374.61 MiB | 1004.89 MiB |
| warm3 | cold3 process | 128.610 s | 6/20 | 15588.17 MiB | 791.33 MiB |

Every run used a 117.7 MiB CPU residual cache. Warm whole-device baselines were
15157.41, 15060.25, and 15143.19 MiB; the maximum positive consecutive change
was 82.94 MiB, below the project's 256 MiB staircase gate. All six runs saved
124 RGB24 frames at 736x416 and one 32 kHz stereo 5.175-second FLAC. The six
frame-sequence byte hashes collapsed to one SHA-256
`0a7064affac5d5edc71e5040ca3ce9528ef2668e4108d54b9ce50b6369efdb43`;
all FLAC files shared
`c372e1c637f535f5d5f8bf0e697be212b5db19bdd32e30f15cf939e511a54939`.

This passes the exact local profile's mechanical, repeatability, cache-hit, and
512 MiB headroom gates. The machine-readable local evidence is ignored from Git at
`artifacts/hybrid-compatibility-blockcache-validation/summary.json`.

### Same-stack Cache OFF control

A follow-up control removed only `MiniMaxH3BlockCacheT8`, connecting the same
Sage-patched MODEL directly to the same stable Stock20 sampler. Hybrid artifact,
reference, prompt, seed, VAEs, Qwen encoder, Sage, 4 GiB policy, strict audit,
resolution, frame count, sampler, shifts, scheduler, decoder and output path
remained controlled. Three fresh-process OFF runs and three warm OFF runs all
completed. In the third process, three Cache ON prompts were interleaved after
the matching OFF prompts.

| Warm pair | OFF total | ON total | Saving | OFF sampler | ON sampler |
|---|---:|---:|---:|---:|---:|
| 1 | 169.360 s | 127.828 s | 24.52% | 145.562 s | 104.562 s |
| 2 | 168.422 s | 128.937 s | 23.44% | 145.172 s | 104.531 s |
| 3 | 172.015 s | 130.797 s | 23.96% | 148.000 s | 106.844 s |

Mean full-workflow saving was 23.98%; mean sampler saving was 27.99%. Every ON
run again reported 6/20 cached forwards and a 117.7 MiB CPU cache. All six OFF
runs were exactly repeatable in decoded pixels and audio, all three paired ON
runs were exactly repeatable, and paired ON decoded media exactly matched the
earlier ON matrix. This establishes a real performance benefit for this exact
profile rather than relying only on a cache-hit counter.

The treatment is not numerically lossless. Across the three identical fixed-seed
pairs, OFF versus ON measured mean frame SSIM 0.8432, minimum-frame SSIM 0.7577,
8-bit RGB MAE 10.37, audio correlation 0.9207 and audio SNR 7.99 dB. These
proxies quantify a material trajectory change but cannot decide perceptual
acceptability. A local blinded A/B package is stored under the ignored
`artifacts/hybrid-compatibility-blockcache-off-validation/blind_review` folder.

OFF whole-device headroom was 239.40/1143.39/1288.00 MiB cold and at least
2154.58 MiB warm. Because one cold run failed the 512 MiB gate, the combined
control does not qualify as a general 16 GiB safe tier. The complete local
machine-readable record is
`artifacts/hybrid-compatibility-blockcache-off-validation/summary.json`.

### Three-material, two-seed extension

The same 736x416, 124-frame Stock20 profile was extended to portrait,
high-frequency mechanical-dragon, and rooftop-superhero references, with two
seeds per material. Five new pairs were run in one process with alternating
OFF/ON order; the earlier portrait seed supplies the sixth quality pair. One
new portrait OFF run was the process cold start while its ON run was warm, so
that pair is excluded from controlled performance averages but retained for
quality comparison.

All ten new prompts completed without OOM, NaN, or telemetry errors. Across
the five valid warm performance pairs (including the earlier warm portrait
seed), end-to-end saving was 22.05-28.47% (24.35% mean), sampler saving was
27.94-33.05% (29.07% mean), and Cache ON skipped 6-7 of 20 forwards. The order-
reversed pairs were also faster with Cache ON, so the result is not explained
by ON always running second.

The quality trajectory remained material- and seed-dependent. Across all six
pairs, mean video SSIM was 0.7020, pair means ranged 0.5192-0.9373, minimum
single-frame SSIM was 0.4774, mean audio correlation was 0.9329, and its range
was 0.8792-0.9806. None was bit-exact. These are difference measurements, not
perceptual scores. The randomized six-pair package was scored by one human
reviewer before reveal: video 6 ties and audio 6 ties. Every B clip was described
as slightly lighter with no other visible difference. The assignment was
balanced by side rather than treatment: B was Cache OFF for portrait s1/s2 and
dragon s1, then Cache ON for dragon s2 and superhero s1/s2. Framewise FFmpeg
signalstats found B higher in YAVG only once and lower in SATAVG four times, so
the B-side observation is not a treatment-consistent threshold-0.12 effect.
The frozen scorecard, reveal report and color check are stored under the ignored
`artifacts/hybrid-compatibility-blockcache-multimaterial-validation/
blind_review_six_pairs` directory; `summary.json` holds the matrix metrics.

The new matrix's minimum observed headroom was 1333.64 MiB, but the preceding
cold OFF matrix already recorded 239.40 MiB. Neither result licenses a general
16 GiB safety claim, and the wider quality spread prevents calling the current
0.12 threshold universally lossless or recommended.

### Conservative-threshold calibration

Thresholds 0.08 and 0.10 were first run on the previously most divergent
mechanical-dragon and rooftop-superhero seed-2 pairs. Both completed without
OOM or NaN. Threshold 0.08 skipped 3/20 forwards and averaged 12.04% end-to-end
and 13.29% sampler saving across those two warm runs. Threshold 0.10 skipped
5/20 and averaged 19.83%/23.23%, but superhero audio correlation fell to
0.6678. This non-monotonic result rejects the simple assumption that a lower
threshold always produces progressively closer decoded audio and prevents
using 0.10 as recommendation evidence.

Threshold 0.08 was then extended to all three materials and both seeds. All six
runs succeeded, with 3-4/20 hits. Excluding portrait seed 2 because its matching
historical OFF run was the process cold start, five controlled warm comparisons
saved 9.05-14.38% end-to-end (12.35% mean) and 12.20-18.39% sampler time
(15.10% mean). Minimum warm whole-device headroom was 2434.52 MiB.

Across six quality comparisons, mean video SSIM was 0.8598, pair means ranged
0.6013-0.9840, minimum single-frame SSIM was 0.5294, mean audio correlation was
0.9635, and its range was 0.8883-0.9927. Both video SSIM and audio correlation
were closer to Cache OFF than threshold 0.12 in all six pairs. The result still
is not bit-exact, and the hardest superhero pair remains materially different.
Proxy improvement is not perceptual non-inferiority.

Machine-readable evidence is under the ignored
`artifacts/hybrid-compatibility-blockcache-threshold-validation` directory.
Its `blind_review_t08_six_pairs` subdirectory contains 12 verified H.264/AAC
clips, the frozen modality scorecard and reveal report. One human reviewer
scored all audio and video preferences before the assignment key was opened.
After reveal, video was threshold-0.08 1 win / 5 ties / 0 losses. The only
explicit visual preference was portrait seed 1; the reviewer reported slight
differences in real-person material and no discernible difference in animated
material. Audio was Cache OFF 1 win / 5 ties / 0 losses, but its sole win was
low-confidence and occurred in the effectively silent superhero seed-2 pair.
This passes a single-reviewer subjective smoke screen for the exact profile,
not statistical non-inferiority or perceptual losslessness. Existing examples
and node defaults remain unchanged; no universal threshold or 16 GiB claim is licensed.

### Anonymous model-side screen

Without opening the private assignment key, a follow-up sampled frames
0/24/49/74/99/123 from every A/B clip and inspected each pair's lowest-SSIM
frame. No black frame or gross sampled-frame face/body/mechanical collapse was
observed. All six visual preferences were recorded as ties with low confidence;
the superhero pairs showed the largest trajectory divergence and still require
full-motion review. Objective audio checks found zero clipping on all 12 tracks.
They did not include listening, cannot judge semantic/music/effect quality, and
both superhero-seed-2 tracks were effectively below -60 dB throughout.

The report and contact sheets are stored under the ignored
`model_blind_review_t08` evidence directory. This screen only rules out some
gross failures. The later human screen above closes only the exact single-reviewer
smoke gate and does not prove general perceptual non-inferiority.

## Quality boundary

Passing this node means that the observed patch stack satisfies the known
mechanical contracts. It does not prove that a Hybrid recipe is better than
stock FL2VA or Ref2VA, removes waxy/oily rendering, improves identity, preserves
audio, or is safe on every 16 GiB GPU.

Current H3 has three AdaLN tags: video, text, and audio. Reference and target
rows share their video/audio tags. Static AdaLN fusion therefore affects target
and reference rows together and is not reference-only routing. An AdaLN LoRA
that overlaps selected Hybrid rows remains blocked until that exact base,
artifact, LoRA, order, sampler, and quality matrix is calibrated.

## Examples

- API: `tests/fixtures/api/hybrid_compatibility_audit_api.json`
- Frontend: `examples/workflows/H3_Hybrid_Compatibility_Audit_Stock20_EXP.json`

The frontend example keeps the validated 4 GiB reserve starting point for the
exact RTX 4060 Ti 16 GiB, 736×416, 124-frame Stock20 experiment, requires an
applied policy, connects final Conditioning, and uses `report_only`. It is not a
general memory-safe preset.
