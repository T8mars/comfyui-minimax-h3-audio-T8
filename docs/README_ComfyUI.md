# MiniMax-H3 Turbo 4-step LoRA — ComfyUI conversion

The converter lives in the project-local `tools/` directory. Model weights are
kept outside this code repository and installed through ComfyUI's standard
model directories. Conversion adds the required `diffusion_model.` prefix; it
does not merge, transpose, rescale, or otherwise modify tensor values.

## Requirements

- ComfyUI **0.30.0 or newer** with native MiniMax-H3 support.
- A **non-pruned** MiniMax-H3 diffusion model:
  - `minimax_h3_fl2va_bf16.safetensors`, or
  - `minimax_h3_fl2va_int8_convrot.safetensors`.
- For R2V, use the corresponding non-pruned `ref2va_bf16` or
  `ref2va_int8_convrot` model.

Do not use a `*_pruned_*` diffusion model for a complete application of this
LoRA. Pruned checkpoints replace each AdaLN input with an 8-dimensional curve
basis, while this LoRA was trained against the original 2688-dimensional
AdaLN input. The other 208 adapter modules match, but 51 AdaLN adapters do not.
The bypass loader can therefore fail at runtime on a pruned model.

One exact-checkpoint-specific Experimental exception was completed on 2026-08-10 for
`10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors` with SHA-256
`f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062`. Its dedicated
`curveproj1025` LoRA converts all 51 AdaLN adapters to the target's 8-dimensional curve basis and
adds the required bias deltas. This does not make the original 518-tensor LoRA generally compatible
with pruned models, and the converted file must not be used on a different checkpoint merely because
its filename also contains `pruned`.

## Version 1.18 Advanced Studio and diagnostic routes

Version 1.18 appends Advanced/Experimental routes without changing the existing H3 conditioning or
stable dual-clock sampler. Start with `MiniMaxH3EnvironmentAuditT8Advanced`; it is read-only and a
pass means only that no known blocker was found. Qwen prefix caching and MLP activation chunking
default to report-only. Context IR uses local validation by default, and an external visual provider
requires explicit upload confirmation while never receiving raw audio.
The audit reports cumulative process I/O/page-fault and pinned-memory/GPU-health state; use
`tools/validate_h3_vram.py run` for before/after workload deltas and the conservative
`fits/fits_with_thrashing/unsafe/unknown` classification. A single audit snapshot never proves
that the current workflow caused the observed cumulative reads.

The stable default sampler has also been run against the H3-era legacy ComfyUI
`0.30.0@563b98eef` and current `cbbc9dab1` using the same plugin, model files, workflow and seed.
Both runs produced 22/22 byte-identical PNG frames; their 32kHz stereo audio correlation was
0.999688 with 36.12dB SNR and a one-int16-LSB maximum difference. This is evidence for the stable
`dual_clock_euler` legacy velocity branch only, not a blanket claim for every Advanced route or
every historical ComfyUI release.

All file mutations are opt-in: repair acceptance, Reel Delivery composition and trajectory
checkpoint saving default false. Scheduled drive-audio injection defaults to bypass because its
first real A/B did not stop ASR-detected extra tail speech. AV Decode Safety defaults to preflight
only, and its current-headroom report is not a future VAE peak prediction. Current H3 regular decode
also uses internal 256-pixel tiles on larger canvases, while explicit tile controls are ignored by
the H3 first-stage alias; missing global tile coordinates are therefore high-risk in either mode.
A validation-only direct full-canvas spatial-coordinate substitution was then tested on three
736x416 source reconstructions. The 256x256 one-tile control was bit-exact, but all three tiled
cases lost SSIM/PSNR and worsened seam ratios with visible grid/ghost artifacts, so that direct
remedy was rejected and was not merged.
The supplied workflows in
`examples/workflows` retain these safe defaults.

The bounded verification record is in `VERIFICATION_REPORT.md`. A 736x416x124 controlled A/B rejects
activation chunking as a memory optimization for the current fused TensorWise INT8 path. The final
Trajectory v2 contract uses Load.resume_noise for direct internal-x-sigma transport, not DisableNoise.
Its 736x416x124 and 256x256x362 full-versus-2+2 final AV latents were bit-exact; the 124-frame three-cold/
three-warm matrix completed 18/18 prompts and 6/6 paired comparisons. The 362-frame full run left only
520.51MiB, and paired warm split+resume was not faster than full, so this still does not establish a
universal 16GiB safety or throughput benefit. AV Decode likewise has no tiled-equivalence claim.
The Qwen prefix cache now has three fresh-process cold pairs and three same-process warm pairs:
every hit arm was faster (paired mean 11.97%/11.01%), but outputs remained non-bit-exact, one warm
audio pair dropped to 0.2323 correlation, and minimum headroom was only 75.63/168.08MiB. It stays
report-only EXP and is not a lossless, VRAM-saving, or 16GiB-safe feature.
The exact short multi-reference and video-reference mechanics are now also exercised. Two image
references produced real cold and warm hits from a 60.70MiB entry and reduced elapsed time by
6.04%/6.87%, but video SSIM mean/minimum were 0.91869/0.91130, audio correlation was 0.95956 and
minimum headroom was 311.85MiB. A native 48-frame, 2-second, 24fps video-reference full A/V pair hit
a 110.74MiB entry and reduced elapsed time by 13.81% and peak device use by 166.31MiB, while leaving
only 145.15/311.46MiB OFF/HIT headroom; its video SSIM mean/minimum were 0.95093/0.94463 and audio
correlation was 0.95303. Both paths are non-exact one-step probes, not perceptual, fixed-speedup or
16GiB-safety evidence.
A further same-process warm matrix covered three two-image material combinations and two seeds each.
All 6/6 pairs produced real hits and every HIT arm was faster, with a mean elapsed-time change of
-11.09%. Video SSIM averaged 0.9314 across pairs with a 0.8531 minimum frame; audio correlation
averaged 0.9771. Post-pair process private memory had no 256MiB upward staircase, but whole-device
headroom fell to 111.93MiB. The one-step contact sheet is unsuitable for perceptual acceptance, so
human non-inferiority at a useful generation profile remains open.
A Stock20 follow-up used one seed from each of the same three material combinations and
conditioning-only primes. All 3/3 full HIT arms were real and faster, averaging -5.00% elapsed time,
but full diffusion amplified the numerical difference: video SSIM averaged 0.8227 across pairs
(pair range 0.6790-0.9073, minimum frame 0.6052), while audio correlation averaged 0.7188 and ranged
from 0.2603 to 0.9894. Minimum headroom was 190.68MiB. These automated quality and safety results do
not pass promotion; the cache remains report-only pending blind review and independent hardware.

Version 1.18.1 adds no node or schema. It hardens Reel Delivery around external termination:
the mixed PCM stage is validated in a temporary file before atomic replacement, one OS advisory
lock serializes a project root, and the next run removes only matching orphan temporary files before
reusing hash-verified phases. A Windows/NTFS 30-minute soak completed 50 independently addressed
clip paths, dialogue/music/ambience/SFX lanes, 43,200 frames and 86,400,000 48kHz samples. Recovery
also completed after killing the audio FFmpeg child, final-mux FFmpeg child and parent Python process.
The 50 clip paths were hardlinks to one small fixture, so that soak alone did not prove codec diversity.
A separate Windows/NTFS composition then mixed synthetic H.264/AAC, HEVC/MP3 and VP9/Opus 128x96
24fps sources plus WAV/FLAC/Opus/AAC lanes. It produced exactly 132 frames, a 264,000-sample plan and
5.500-second output stream; source hashes stayed unchanged, the repeat reused both phases and the
output hash stayed stable. This closes local synthetic codec mechanics, not real-H3 content diversity,
high-resolution throughput or non-Windows behavior.

Selective Repair additionally survived six hard-kill boundaries on an isolated 14-segment,
60-second accepted chain without changing the base manifest or 27 accepted assets. A real H3
segment-7 replacement composed to exact 1,440-frame/1,920,000-sample outputs, but the outgoing
boundary regressed because segment 8 still depended on the original segment 7. Two kill points also
left retry-safe orphan temporary files. Cascading dependent-segment regeneration, crash-clean temp
cleanup, blind quality review and cross-platform filesystems remain open.

## Install and connect

1. Copy either converted `*_comfyui.safetensors` file to
   `ComfyUI/models/loras/`.
2. Update ComfyUI and restart it.
3. Add **Load LoRA (Bypass, Model Only) (for debugging)** after **Load Diffusion
   Model**. Connect its model output wherever the diffusion model was connected.
4. Start with LoRA strength `1.0`. The upstream discussion reports that values
   around `1.5–2.2` can look stronger, but that is an empirical preference, not
   part of the trained LoRA math.
5. Use **MiniMax H3 Dual-Clock Sampler (T8)** from
   `custom_nodes/minimax-h3-audio-T8`, with `steps=4`, video shift `12`, and
   audio shift `3`. Keep `sampler=dual_clock_euler` and
   `scheduler=native_flow` for the original verified path. Its `model` output goes to the guider; its `sampler` and
   `sigmas` outputs go to `SamplerCustomAdvanced`. Connect the same H3 AV latent
   to both the dual-clock node and `SamplerCustomAdvanced.latent_image`.
6. To test more audio integration steps without changing the stable workflow,
   use the separate **MiniMax H3 Multi-Rate Sampler (EXP/T8)**. Start with
   `video_steps=4`, `audio_steps=8`; then compare 4/10 with the same seed.
   `audio_steps` is the number of full joint H3 DiT calls, so 4/10 costs about
   2.5 times as much as stable 4/4. The Turbo LoRA is still trained for four
   steps, so extra audio microsteps are experimental and not guaranteed to win.

Do not combine the dual-clock node with `MiniMax H3 Sigma Shift`,
`KSamplerSelect`, or an external scheduler node. The node replaces all three.
Version 1.7.0 keeps the 1.3.3 internal sampler and scheduler dropdowns while preserving
the original defaults. Alternative ComfyUI samplers use native `ModelSamplingAV`
and are exposed only when the installed ComfyUI has FLOW_AV support; alternative
schedulers change the sigma grid and are not a quality guarantee for a four-step
Turbo LoRA. Old workflow/API JSON may omit both new fields and retains the
original behavior.
The same no-extra-scheduler rule applies to the EXP node.

The bypass loader is recommended because it computes the author's intended
runtime expression `base(x) + B(A(x))`. A regular LoRA loader may round small
updates when it materializes them into BF16 weights, and it cannot faithfully
patch quantized weights in the same way.

## Ready-to-import workflows

Version 1.17.0 retains all 61 Version 1.16.0 node IDs and appends one isolated
`MiniMaxH3HybridCompatibilityAuditT8Advanced` node. Put it after every MODEL-changing node and
sampler setup, then route its passthrough MODEL to `BasicGuider`. Connecting final H3 Conditioning
also verifies Long Video/MultiKeyframe pairing and actual reference modalities.

The default `report_only` mode never blocks and returns the exact same MODEL object. The optional
`block_hard_conflicts` mode rejects invalid Hybrid offset-set identity, Hybrid/LoRA order or AdaLN
overlap, incomplete Block Cache/Sage contracts, Long Video/MultiKeyframe conflicts, mismatched
Conditioning, and configured current-VRAM/host-commit gate failures. It recognizes stock, stable
dual-clock/native AV and EXP multi-rate sampling without changing sampler mathematics.

When a T8 VRAM policy is connected to the Hybrid Loader, a small policy-application provenance
attachment now follows MODEL clones. `require_applied_vram_policy=true` distinguishes a real fixed/
auto reserve from missing or report-only policy. Current 512 MiB VRAM and 16 GiB host-commit gates
are not peak predictions. Passing the audit is mechanical compatibility only; quality, de-waxing,
reference identity and universal 16 GiB safety remain unproven, and `memory_safe_claim=false`.
See `docs/HYBRID_COMPATIBILITY_AUDIT.md` and import
`H3_Hybrid_Compatibility_Audit_Stock20_EXP.json` or
`hybrid_compatibility_audit_api.json`.

A follow-up exact-profile matrix on 2026-08-13 used 736x416, 124 frames,
Stock20, the 27.69 MiB Hybrid artifact, KJ H3 Sage, default-threshold T8 H3
Block Cache, the 4 GiB policy, and strict audit. Three fresh-process cold runs
and three same-process warm runs all succeeded and each cached 6/20 forwards.
Worst whole-device headroom was 766.38 MiB, maximum positive warm-baseline
movement was 82.94 MiB, and all six 124-frame PNG sequences plus FLAC files
were byte-identical. This passes only the exact local mechanical/repeatability
gate; cache-off quality, multiple materials/seeds, other GPUs, and universal
memory safety remain unproven.

A same-stack Cache OFF control then completed three cold and three warm runs,
with three interleaved warm OFF/ON pairs in one process. Mean end-to-end time
fell from 169.93 s OFF to 129.19 s ON (23.98%); sampler time fell from 146.24 s
to 105.31 s (27.99%), with 6/20 cache hits each time. The treatment was not
bit-exact: mean video SSIM was 0.8432 (minimum frame 0.7577), uint8 MAE was
10.37, and audio correlation/SNR were 0.9207/7.99 dB. One OFF cold run left
only 239.40 MiB, below the 512 MiB gate. This proves a repeatable performance
benefit only for this exact local profile. The later six-pair human screen is
single-reviewer evidence only, and `memory_safe_claim=false` remains unchanged.

The follow-up was extended to three visual material classes with two seeds each:
portrait, high-frequency mechanical dragon, and a rooftop superhero scene. Five
warm controlled pairs saved 22.05-28.47% end-to-end (24.35% mean) and
27.94-33.05% sampler time (29.07% mean), with 6-7/20 hits. Across all six
quality pairs, video SSIM ranged from 0.5192 to 0.9373 (0.7020 mean; minimum
single frame 0.4774), while audio correlation ranged from 0.8792 to 0.9806
(0.9329 mean). No pair was bit-exact. One human reviewer then scored the
randomized package as six video ties and six audio ties. The reviewer saw every
B side as slightly lighter, but B mapped to Cache OFF in the first three pairs
and Cache ON in the last three. Decoded signal statistics also found B brighter
in only 1/6 pairs and slightly less saturated in 4/6, so the observation is not
attributable to threshold 0.12. This is a single-reviewer smoke screen, not
statistical perceptual non-inferiority or a universal default recommendation.

A follow-up calibrated 0.08 and 0.10 on the two most divergent pairs, then
extended the more conservative 0.08 setting to the complete three-material,
two-seed matrix. Threshold 0.10 showed a non-monotonic superhero-audio
regression and is not recommendation evidence. At 0.08, five valid warm pairs
saved 9.05-14.38% end-to-end (12.35% mean) and 12.20-18.39% sampler time
(15.10% mean), with 3-4/20 hits. Six-pair video SSIM averaged 0.8598 (pair
range 0.6013-0.9840; minimum frame 0.5294), and audio correlation averaged
0.9635 (range 0.8883-0.9927). Both proxies improved over 0.12 in all six pairs,
but difficult cases remain materially different. The randomized OFF-versus-0.08
package was then scored by one human reviewer before reveal. Video favored 0.08
once with five ties and no Cache-OFF win; the reviewer saw a slight difference
in real-person material and no discernible difference in animated material.
Audio produced five ties and one low-confidence Cache-OFF preference in the
effectively silent sixth pair. This passes only a single-reviewer subjective
smoke screen. No node or legacy workflow default changed.

An additional anonymous model-side review sampled six timestamps per side plus
each pair's maximum-difference frame. It found no black frame or sampled-frame
structural collapse and recorded six visual ties. Objective checks found no
clipping in 12 audio tracks, but no human listening was performed and both
superhero-seed-2 tracks were effectively silent. This low-confidence screen
did not by itself establish motion/audio non-inferiority. The subsequent human
screen above still does not prove statistical non-inferiority or losslessness.

Version 1.16.0 retains all 60 Version 1.15.1 node IDs and appends one isolated
Hybrid Artifact Maintenance Advanced output node. Its API and frontend examples
default to side-effect-free inspection. Mutating actions require explicit confirmation and a
positive operation epoch; verified files are moved to a recoverable same-volume quarantine, not
permanently deleted. Exact path derivation, atomic fsynced journals, per-file SHA-256, stale-owner
checks, tampered-journal refusal, and a real worker-process kill/recovery test guard the feature.
It never scans source checkpoints, unloads a MODEL, or releases VRAM. See
`docs/HYBRID_ARTIFACT_MAINTENANCE.md`.

Version 1.15.1 retains all 60 Version 1.15.0 node IDs and updates the opt-in
`H3_Hybrid_Model_VBAR_Headroom_Stock20_EXP.json` / `hybrid_model_vbar_headroom_api.json` pair. It
connects a reportable 4.0 GiB total-reserve policy directly to the Hybrid Loader, guaranteeing that
ComfyUI reserve and AIMDO simple headroom are set before the stock diffusion-model load. The policy
uses a direct AIMDO setter, does not reinitialize devices or alter startup `--vram-headroom`, and
does not globally unload models in this fixed-policy example.

On the exact RTX 4060 Ti 16 GiB, 736x416, 124-frame Hybrid Stock20 validation graph, the 4.0 GiB
setting passed three cold and three warm runs with at least 1028.117 MiB and 1401.415 MiB headroom,
respectively. Decoded same-seed video and PCM matched the no-policy baseline bit-for-bit. This does
not generalize to other resolutions, frame counts, GPUs, concurrent CUDA users, or host-commit
conditions; `memory_safe` and `never_oom` remain false.

Version 1.14.0 added the opt-in
`examples/workflows/H3_Hybrid_Model_Advanced_Stock20_EXP.json` workflow and
`examples/hybrid_model_advanced_api.json`. Audio-only and mixed-reference variants are provided as
`H3_Hybrid_Model_Audio_Reference_Stock20_EXP.json`,
`H3_Hybrid_Model_Mixed_Reference_Stock20_EXP.json`,
`hybrid_model_audio_reference_api.json`, and `hybrid_model_mixed_reference_api.json`.
The graph strictly hashes the exact validated FL2VA/
Ref2VA pruned pair, builds or reuses a 27.69 MiB curve-rebased target-slice artifact under
`ComfyUI/models/h3_hybrid_artifacts`, and then applies it to a MODEL loaded through ComfyUI's stock
diffusion loader. It does not create a second full fused checkpoint. Keep the order Hybrid Loader →
optional LoRA. `auto_match_reference_modalities_exp` reads the connected Conditioning and selects the
smallest video/audio modality-row recipe for actual extra references; this is not a best-quality selector.
The resumable sequential matrix tool writes blind-review media and `matrix_summary.json/csv`, with optional
local-only ASR, face and speaker signals. Fifteen real Stock20 follow-up runs completed, but the minimum
whole-device headroom was only 41.34 MiB; no recipe is yet a proven quality winner, reference-only route,
de-wax fix, or universal 16 GB safe tier. See
`docs/HYBRID_MODEL_ADVANCED_VALIDATION.md` for the exact hashes and current pilot limits.

Three base sampler-comparison frontend workflows are installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/`: stable 4/4, experimental 4/8,
and experimental 4/10. They share the same seed, prompt, EMA LoRA, loaders, and
MP4 settings for direct comparison. Drag a JSON file into the ComfyUI canvas or
open it from the Workflows menu.

Three stable 4/4 source-audio workflows are installed in the same directory:
`H3_Audio_Lock_Source_Stable_4V4A.json`,
`H3_Audio_Remix_Source_Stable_4V4A.json`, and
`H3_Audio_Reference_Only_Stable_4V4A.json`. Each uses a 5-second Audio Window,
736x416 canvas, 124-frame legal H3 context, exact synchronized Output Trim, and
the explicit dual-clock Euler/native-flow defaults. Lock mode routes the clean
Conditioning `mux_audio` to the final MP4; remix and reference-only route decoded
model audio instead. Upload or select a source file in `Load Audio` before queuing.

Version 1.12.0 also installs two opt-in dialogue-safe audio workflows without changing any old
workflow: `H3_Dialogue_Safe_Master_EXP.json` accepts already independent speech/music/ambience/SFX
stems and keeps the background running after verified speech ends;
`H3_Dialogue_Timed_Background_Bed_Lock_EXP.json` is a two-pass H3 graph that locks an independent
dialogue-free background bed after an explicit 40Hz latent boundary. The first path is sample-exact
stem assembly. The second path is not source separation or a sample-exact cut: the real H3 Audio
VAE encoded a standard 124-frame window to 206 steps against the 207-step AV clock, so the example
explicitly selects `fit_reported`, and its decoder showed roughly 0.3 seconds of temporal influence
after the latent boundary. Both workflows use placeholders that must be replaced before queuing.

The project also includes the isolated experimental long-video workflow
`examples/workflows/H3_Long_Video_22F_EXP.json` and API graph
`examples/long_video_segment_api.json`. They plan and execute one bounded segment
at a time, store only a checksummed AV latent tail, and use a cloned-MODEL object
patch rather than a process-global MiniMax H3 monkey patch. Intermediate segments
must preserve the sampled tail; only a Planner-marked final segment may trim it
and automatically disables the next context checkpoint. The examples use core
`CreateVideo -> SaveVideo`; this avoids VideoHelperSuite's `apad + -shortest`
ending the AAC stream roughly 79-90ms before the video in the tested standalone
segments. A real four-step, three-segment run produced exact A/V stream durations.
A controlled single-case comparison found a single last frame materially worse
than 22-frame context, while video-VAE re-encoding did not beat the direct sampler
latent route and took longer. Audio boundary click risk, long-chain degradation,
and VRAM safety tiers are still unresolved, so the feature remains experimental.

Version 1.5.0 adds a safer accepted-state route without removing that P1 example:

- `examples/long_video_candidate_accept_api.json` loads only an accepted parent,
  saves a non-mutating candidate, previews it with acceptance off by default, and
  promotes it through a locked, checksummed, atomic manifest only after review;
- `examples/workflows/H3_Long_Video_Accepted_22F_EXP.json` provides the same
  review-first route as a drag-and-drop frontend workflow;
- replacing segment N is explicit and invalidates every accepted segment after N,
  because those outputs were conditioned on the old parent chain;
- `examples/long_video_compose_api.json` verifies the contiguous final manifest and
  composes it while holding only one decoded video frame and one segment PCM buffer;
- audio sample boundaries are absolute over the 24fps timeline. The optional cosine
  bridge preserves the exact sample count and reduces an instantaneous value step,
  but it is not proof of perceptually seamless, phase-continuous, or lossless audio.

The Long Video Conditioning node now exposes a default-off identity-reference experiment through
three inputs appended after the old schema. `first_frame_reuse=segment0_only` is the unchanged
default. `persistent_identity_reference` adds non-timeline image references only on continuation
segments while segment 0 remains controlled by the exact `first_frame`. The optional
`persistent_identity_image` accepts a continuation-only face or upper-body crop. The compatible
`single_reference` strategy prefers that crop and falls back to the full first frame;
`scene_plus_identity` sends the full scene and crop as two separate image references and fails
closed when the crop is missing. Persistent and user references together remain capped at nine;
use `task_type=auto` or `Hybrid`.

Old API JSON and widget prefixes remain valid because every new input is appended with a default.
No model or full video history is added, and stable sampling is untouched. A continuation does gain
one or two reference blocks and VAE encodes, so sequence length, runtime, and VRAM can rise and the
references can compete with motion context. The original full-first-frame strategy failed its
32-second identity-depth gate: source cosine fell from 0.613 in continuation 1 to 0.134 in
continuation 7.

A motion-rich three-seed crop-only probe improved legacy by pooled mean/median
+0.08272/+0.09845 on 54/59 paired detected frames, but one seed regressed against the full-scene
strategy. The subsequent scene-plus-crop strategy completed all six two-segment cold chains and
12/12 prompts with byte-identical segment 0 inside each comparison. It improved legacy by
+0.11278/+0.11628 on 56/59 paired frames and the full-scene strategy by +0.08454/+0.09455 on
52/60; every seed had a positive median delta and contact sheets showed active football/arm motion.

One independent scene-plus-crop 32-second/eight-segment chain then completed 8/8 prompts without
OOM, retry, or cache reuse and produced exact 768-frame/32.000-second video and audio. Continuation
identity medians were 0.699/0.639/0.644/0.609/0.737/0.601/0.574, for a last/first retention ratio
of 0.821. It improved legacy by pooled +0.42945/+0.46670 mean/median and the full-scene strategy by
+0.33771/+0.35802. Minimum free VRAM was 3906.07 MiB and post-15-second occupancy returned to
1231.63 MiB in this fixed local profile only. The predeclared motion gate still failed: continuation
2 and 5 flow-P90 ratios were 0.546/0.538 versus legacy, and continuation 5 temporal MAD was 0.647.
Two-fps strips show continuing ball, arm, and pose changes rather than a freeze, but the
action-amplitude/trajectory warning is real. The feature therefore remains EXP and default-off;
unseen-seed/multi-source intermediate validation and a motion-regression remedy are required before
any three-seed 60-second matrix or identity-lock/action-safe/memory-safe claim.

The accepted-state implementation has synthetic MP4 tests and one real 124/102/102-frame
H3 three-segment compose A/B. The 5ms bridge reduced the two post-AAC boundary jumps from
about 0.04226/0.03509 to 0.00434/0.00704 while preserving the 328-frame timeline. This is
an amplitude-discontinuity metric, not a listening test; the later 14-segment run below adds
one long-chain case, while multi-material validation is still required before any seamless,
arbitrary-length, or no-OOM claim.

Version 1.6.0 adds a human-reviewed total-duration resume route:

- `examples/workflows/H3_Long_Video_Auto_Resume_22F_EXP.json` is the recommended
  frontend graph and is installed in the same ComfyUI workflow directory;
- `examples/long_video_auto_resume_api.json` is the equivalent API graph;
- one `MiniMaxH3LongVideoOrchestratorT8` input defines the total duration, a fixed
  legal H3 render window, overlap context, global/per-segment prompts, a seed policy,
  steps, video/audio shifts, sampler, and scheduler;
- the same sampling outputs drive `MiniMaxH3DualClockSamplerT8` and the candidate's
  machine-generated `sampling_summary`, so changing a real sampler parameter cannot silently
  leave the accepted-state identity at an old manually entered value;
- total duration is quantized once at 24fps. With the 124-frame window and 22-frame
  context, 60 seconds is exactly `124 + 12*102 + 92 = 1440` output frames while every
  internal sampling window remains 124 frames;
- the accepted manifest length selects the next segment. Conflicting accepted timeline
  settings are rejected, and a complete final manifest blocks downstream sampling;
- the workflow deliberately remains review-first: queue a candidate with acceptance off,
  preview it, accept it in a separate queue, reset acceptance to false, then queue the next
  segment. It is not a background auto-queue, pause/cancel, or automatic model-unload system.

Version 1.7.0 adds a separate, explicitly enabled background route without changing the
review-first workflow:

- `H3_Long_Video_Background_22F_EXP.json` and `long_video_background_api.json` connect
  `Background Start` before expensive work and `Auto Accept & Continue` as the only terminal;
- `H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json` is the ready-to-import
  two-image variant: the full scene drives exact segment 0, while a same-subject face or
  upper-body crop joins the full scene as two continuation-only identity references;
- `review_only` is the safe default. `auto_accept_and_continue` accepts every successful
  candidate without human review and validates/queues exactly one next prompt at a time;
- node buttons and REST routes expose status, pause-after-current, resume, and targeted cancel;
- retry reuses the exact API prompt. It never silently lowers resolution, frame count, context,
  sampler settings, steps, or seed, and stops after the configured additional attempts;
- the selected policy is requested after every durable acceptance, including continue, pause,
  and final. `clear_execution_cache` sets `free_memory=true` with `unload_models=false`;
  `unload_all_models` is a stronger global ComfyUI unload, not an H3-only release; `keep_loaded`
  requests neither. A release failure preserves the accepted manifest and stops without regeneration;
- `background_job.json` persists state and a prompt hash, not the prompt body. Error persistence
  excludes `current_inputs/current_outputs`, media tensors, and prompts. After a server restart,
  status reconciles a stale active prompt to `detached` and reports whether to queue the workflow
  once or compose an already complete manifest. Queue once to reattach the in-memory prompt
  snapshot to an incomplete accepted manifest;
- automatic final composition is optional. A post-accept composition error stops the job and
  leaves the complete manifest for the standalone Compose Accepted node instead of retrying from
  an already advanced manifest;
- prompt snapshots remove ComfyUI's runtime `is_changed` cache fingerprints before requeueing.
  Without this sanitization, `keep_loaded` can incorrectly cache the entire next segment instead
  of rerunning the orchestrator, sampler, save, and terminal nodes.

Live model-free checks completed two-segment auto queue/composition, pause-after-current then
resume, targeted cancellation with no accepted manifest, and one exact-prompt retry followed by
bounded failure. A real H3 executor probe used FL2VA INT8, Standard Turbo EMA LoRA, NVFP4 CLIP,
both H3 VAEs, 256x256, a 124-frame window, 22-frame context, one step, DynamicVRAM headroom 2GiB,
and `unload_all_models`. Two distinct prompt IDs succeeded; manifest revision 2 contains
`124+20=144` frames, and the final H.264/AAC video and audio streams are both exactly 6.000s.

A repeated release-policy matrix used three paired seeds at 736x416, four steps, a 124-frame
window, 22-frame context, and two segments. Each policy ran in three fresh-process cold trials,
then one unmeasured same-process primer plus three measured warm trials. All 21 chains succeeded
without retry/OOM; the 18 measured chains were bit-identical per seed across segment videos,
accepted AV-tail tensor payloads, and final H.264/AAC files.

| Policy | Cold runtime mean | Warm runtime mean | Cold/warm device-peak mean | Cold/warm post-15s mean |
|---|---:|---:|---:|---:|
| `keep_loaded` | 170.89s | 153.10s | 13,449.95 / 13,467.30MiB | 8,083.22 / 7,987.22MiB |
| `clear_execution_cache` | 188.28s | 185.08s | 13,434.52 / 13,408.94MiB | 1,229.63 / 1,229.63MiB |
| `unload_all_models` | 189.08s | 197.13s | 13,421.52 / 13,384.03MiB | 1,229.63 / 1,229.63MiB |

Every paired peak difference was below the project's 128MiB material-difference threshold.
`keep_loaded` gained 17.39s cold and 31.97s warm versus the default, but retained about
6.85/6.76GiB more device memory. Global unload gained no repeatable peak advantage and was
12.05s slower than the default in warm trials. `clear_execution_cache` therefore remains the
balanced default. The result remains specific to this local model/GPU/profile.
This is a mechanical background/reload result, not a quality benchmark or a general four-step,
high-resolution, cross-GPU `memory_safe` claim.

A real hard-kill probe terminated ComfyUI after segment 0 was durably accepted at manifest
revision 1 and the next prompt was active. On restart, status reported `detached`, one accepted
segment, and `queue_workflow_once`. Requeueing the same workflow resumed at segment 1 under a new
job linked to the previous job, without changing or rewriting segment 0's candidate, video, or AV
tail tensors. Revision 2 completed with 144 frames and exact 6.000s A/V/container durations;
whole-device recovery peak was 13,537.02MiB in the native-v2 repeat. Two independent real H3 chains then interleaved on
the same ComfyUI queue as `A0 -> B0 -> A1 -> B1`; both completed under isolated prompt/job IDs,
parents, manifests, output roots, and final files, with a 13,511.44MiB device peak and no OOM.

The manifest commit lock is now an OS-owned `manifest.lock.v2`, automatically released on process
death. Same-host Windows/NTFS tests passed one-winner same-slot acceptance, four processes times
25 protected updates with all 100 retained, and forced owner termination followed by acquisition
within two seconds. A chain-wide background lease rejects a second ComfyUI process before it can
generate; after killing the first owner, a third process reattached with the old `previous_job_id`.
Live legacy locks are respected, dead legacy residue remains rollback-compatible, unknown schemas
fail closed without backup rollback, same-schema additive fields survive a later write, and corrupt
auxiliary background state is quarantined before manifest-led recovery.

Accepted manifests and background states now use separate schema-2 format markers. A valid
schema-1 file is normalized to schema 2 in memory without changing the raw file on read. The next
protected manifest write atomically upgrades the primary while preserving the raw schema-1
backup; background reattachment writes schema 2 and records the previous schema. Read-only
validation against an existing real H3 schema-1 chain preserved both original hashes, and the new
hard-kill plus dual-chain real probe wrote native schema 2 before and after recovery. Unknown
future schemas still fail closed.

Eight deterministic acceptance fault-injection cases now cover missing/corrupt accepted-asset
repair, candidate-id and archived-path collision refusal, context-copy failure, failure after the
backup write but before primary replacement, and missing-primary recovery. A valid backup is now
authoritative when the primary is absent even if the caller permits a new chain; an unknown or
corrupt backup cannot be replaced by an empty manifest. Retrying the same reviewed candidate after
a copy/commit failure completes without losing the prior revision. These tests cover named program
boundaries, not arbitrary CPU instructions or storage power loss.

The two highest-risk points also passed real Windows/NTFS process termination. A worker held the
actual `manifest.lock.v2` and was killed after either (a) the accepted MP4 was fully copied but the
context and manifest were not, or (b) revision 1 was written to the backup but revision 2 had not
replaced the primary. Three independent rounds per pair produced 6/6 successful recoveries: the
OS lock released automatically, the old/no manifest remained authoritative, and the same
candidate recommitted in under two seconds with valid hashes and revisions. No worker remained.
This used small test media rather than a live H3 CUDA generation and does not emulate machine power
loss or a network filesystem.

This validates the local on-disk schema-1 to schema-2 contract. It does not validate a complete
upgrade/downgrade matrix across separately released plugin/ComfyUI builds,
network/shared-filesystem locking, simultaneous CUDA execution, or multi-GPU parallelism.

A representative four-step background chain was then completed on the local RTX 4060 Ti 16GB
with FL2VA INT8, Standard Turbo LoRA, 736x416, a 124-frame window, 22-frame AV context,
DynamicVRAM headroom 2GiB, and global `unload_all_models` between segments. All 14 distinct
prompts succeeded once without retry or OOM. Manifest revision 14 is exactly
`124 + 12*102 + 92 = 1440` frames and 1,920,000 audio samples; the automatic H.264/AAC video,
audio, and container streams are all exactly 60.000 seconds. Half-second polling observed a
12,823MiB whole-device peak and 3,556MiB minimum free margin. The running maximum was essentially
flat after segment 3, with only a further 39MiB increase at segment 9 and no later staircase.
All 13 visual contact pairs avoided an obvious hard cut in this walking-shot sample. Audio
adjacent-window level change still reached 13.75dB; the 5ms bridge reduced median single-sample
jump about 96.2% but cannot repair level, semantics, rhythm, or lip sync. This is one local
prompt/seed profile, not a cross-GPU or universal memory-safe claim.

A follow-up real 256x256 one-step/two-segment probe verified the corrected final-release timing.
At completion, whole-device use was about 8,124MiB and state recorded
`last_release_policy=unload_all_models`; within 15 seconds it automatically fell to about 1,230MiB,
a 6,894MiB drop without a manual `/free` call. This validates the final release request, not
universal side-effect-free reload behavior for every third-party model.

All 146 current project tests, Ruff, ComfyUI whitelist import, and live `/object_info`/route probes pass
for this checkpoint. The live instance registered 25 T8 nodes. The earlier 1.6.0 checkpoint exposed the
auto-resume workflow through `/userdata`. A real DynamicVRAM probe using non-pruned FL2VA
INT8, Standard Turbo LoRA, NVFP4 H3 CLIP, both H3 VAEs, 736x416, a 124-frame internal
window, one step, and a one-second request also completed candidate generation, acceptance,
complete-chain blocking, and accepted-file composition. Candidate and final outputs both
contain 24 frames at 24fps with exactly 1.0-second video, audio, and container streams.
This is execution-path evidence, not a four-step/multi-segment quality or VRAM safety result.
The post-probe single-source sampling hardening was also rerun against the real model: changing
only Orchestrator `steps` to 1 drove the sampler and produced candidate metadata
`1-step dual_clock_euler/native_flow shift12/3`; acceptance, complete-chain blocking, and
composition succeeded again.

A later real four-step auto-resume probe requested six seconds and generated two accepted
segments: 124 frames followed by a 20-frame final segment conditioned on the accepted 22-frame
AV tail. The final manifest covers exactly 144 frames, and both unbridged and 5ms-bridge outputs
contain 24fps/144 frames with 6.000-second video, audio, and container streams. The post-AAC
single-sample boundary jump fell by about 80.2%, but the adjacent audio windows still differed
by about 33.3dB in level. The video contact sheet has no obvious identity or composition cut,
yet boundary MAD and SSIM discontinuity were the largest among 16 nearby intra-segment
transitions. Device peaks were about 15,461.4 and 16,181.5MiB; the latter leaves only about
198MiB, below the 512MiB safety gate. This is a successful bounded two-segment execution, not
a seamless-audio, long-chain, or 16GB-safe result.

A subsequent uninterrupted four-step DynamicVRAM run completed the full 60-second plan in
14 accepted segments: `124 + 12*102 + 92 = 1440` frames. Both the unbridged and 5ms-bridge
assemblies report 24fps/1440 frames and exactly 60.000-second video, audio, and container
streams. No explicit `/free` request was issued between segments. The 14 measured device peaks
ranged from about 15,480.0 to 16,228.2MiB; the descriptive warm-peak slope was about
+28.0MiB/segment and the baselines did not form a monotonic staircase. This single run therefore
does not show a cumulative VRAM leak, but five segments left less than 512MiB and the worst left
only about 151.3MiB. It is not a validated 16GB safety tier, and 0.25-second polling may miss
shorter spikes.

Across all 13 video seams, median/max MAD were about 0.01618/0.01906 and median/min SSIM were
about 0.96374/0.92868. Contact sheets do not show a hard subject/background cut at the worst
seams, but the 14-segment timeline shows gradual appearance and exposure drift; these metrics do
not prove identity preservation. Audio degradation is material: the median adjacent half-second
level change was about -9.51dB, the largest absolute change was about 40.83dB, and the final
segment's above-8kHz energy ratio was about 36.30dB below the first segment. The bridge reduced
the median post-AAC single-sample boundary jump by about 97.23%, but cannot restore level,
timbre, speech semantics, or lip sync. This proves the bounded/resumable 60-second execution
path, not seamless or lossless long-form quality. The later fixed-prompt three-base-seed cold
gate closes only mechanical/memory repeatability; different prompts/materials, same-seed
whole-chain warm repeats, dialogue/lip-sync, fast motion, rhythmic music, blind listening,
ASR/speaker checks, and cross-configuration VRAM profiles remain open.

After accepting the final segment, the first uncached full re-queue can terminate with the
expected ComfyUI `ExecutionBlocked` status (empty traceback, execution stops at the
Orchestrator). A cached re-queue may instead report success while running only the review node.
Both paths leave the candidate count at 14 and perform no extra sampling; the former is a safe
completion signal rather than a generation failure.

The 5-frame versus 22-frame comparison now includes repeated 0.3M and 0.6M matrices rather than
only single probes. Each resolution used two accepted-segment-0 baselines whose MP4 and video/audio
tail tensors were bit-identical. Three paired seeds were run in alternating order with a fresh
isolated ComfyUI process for every cold trial, then again after a same-process primer for the warm
matrix. Every matching context+seed cold/warm candidate was bit-identical, and VRAM was polled at
0.10-second intervals.

At 736x416, the cold 5/22-frame absolute device-peak means were 15,279.5/15,224.0MiB, while paired
`22-5` differences were +96.6, -78.3, and -184.9MiB. In contrast, the sampler PyTorch-pool means
were repeatably 3,189.9/3,495.3MiB: 5 frames saved about 305MiB locally. Cold runtime means were
86.53/93.08 seconds and warm means were 69.27/78.01 seconds. The warm process reached only 97.6MiB
free, with five of six measured trials below the 512MiB gate. Across these three seeds, the 22-frame
route had better mean video MAD/SSIM and audio level/NCC evidence.

At 1056x608, cold 5/22-frame absolute peak means were 15,739.0/15,724.2MiB, but paired differences
swung from -752.0 to +696.7MiB. Sampler-pool means remained stable at 5,753.4/6,381.2MiB, so 5 frames
saved about 628MiB of local pool and reduced cold/warm runtime means from 230.38/218.40 seconds to
200.29/187.89 seconds. All six warm trials failed the 512MiB margin gate; the minimum margin was
33.6MiB. Five frames had lower mean MAD/higher SSIM here, which may also reflect suppressed motion;
22 frames had much better mean audio level/NCC, and one of its three contacts showed a clear
front-to-profile boundary change.

The 39-frame treatment then reused the 736x416 baselines and the same three seeds for three fresh
cold starts and three post-primer warm runs. All six runs succeeded, matching cold/warm outputs
were bit-identical, and the 5/22/39 segment-0 MP4 plus AV tail tensors were identical. The
39-frame sampler-pool mean was about 3,799MiB, a repeatable 303-304MiB increase over 22 frames;
cold/warm runtime means were 101.65/87.38 seconds. All three warm trials failed the 512MiB gate,
with only 77.35MiB free at worst. Manual inspection found one acceptably continuous boundary,
one visible pose/framing jump, and one severe identity/shot discontinuity.

Five frames therefore remains `fast_context_5_experimental`: its compute and sampler-pool savings
are repeatable, but an absolute device-peak advantage is not. Twenty-two frames remains the current
balanced default candidate. Thirty-nine frames is now `context_39_high_risk_experimental`, not a
quality or safety tier. The 1056x608/39-frame treatment was denied by the predefined gate rather
than forced: its 22-frame warm control already had all six trials below 512MiB and only 33.6MiB
free at worst. This is not evidence that 39 frames must OOM; it is evidence that the unchanged
configuration cannot establish a safety tier and carries material OOM risk. No configuration
receives a `memory_safe` label before a hardware/model/resolution/plugin-specific gate passes.

A later controlled memory-policy matrix fixed DynamicVRAM headroom at 2.0GiB. Stock and Sage
each completed three cold and three warm trials at 736x416 and 1056x608 with every trial above
512MiB and matching strategy+seed cold/warm outputs bit-identical. Default Block Cache hit 0 of
4 forwards and cannot skip the first full forward. Sage was faster but produced a higher
whole-device peak than Stock at equal headroom and material shot/pose/trajectory divergence in
two of three 1056x608 seeds, so it remains a high-risk approximate speed experiment.

Stock+headroom-2.0 then completed a second uninterrupted 60-second/14-segment 736x416 chain with
2739.41MiB minimum free margin. Both assemblies are exact 24fps/1440-frame, 60.000-second AV
streams. Relative to the previous same-prompt/same-seed Stock headroom-0.5 chain, all 14 segment
MP4 hashes and all 13 continuation AV tensor payloads were identical; median peak fell by about
2635MiB and total generation time increased about 1.63%. This is a validated local conservative
profile for the exact RTX 4060 Ti 16GiB/model/resolution/window/context/plugin contract, not a
general `memory_safe` tier or never-OOM promise.

The same conservative profile was then repeated as three independent ComfyUI cold starts with
base seeds `2608082000`, `2608083101`, and `2608083202`, while prompt, model, canvas, render window,
context, and sampling remained fixed. All 42/42 segments completed once without OOM, retry, or
candidate reuse. Manifest/parent/revision continuity, candidate and accepted video/context
SHA-256 values, 1440 frames, 1,920,000 samples, completion blocking, and six exact 60.000-second
assemblies were independently verified. Per-chain maximum peaks were 13,640.09, 13,414.01, and
13,426.72MiB; the worst free margin was 2739.41MiB and no segment fell below 512MiB. This closes
the fixed local profile's cross-base-seed cold-start mechanical/memory gate, not same-seed
whole-chain warm repeats, cross-prompt/material coverage, other GPUs, or desktop-load profiles.

The long-term quality gate failed. All three 14-segment middle-frame timelines accumulate facial
age and identity drift, most severely for seed `2608083101`. Across the three chains, maximum
adjacent half-second audio level gaps were 23.59-48.06dB, descriptive NCC medians were only
0.127-0.206, and the final segment's above-8kHz energy ratio was 9.66-36.30dB below the first.
The 5ms bridge reduced median post-AAC single-sample jumps by 94.93%-97.33%, but cannot repair
level, timbre, semantics, or recursive dulling. The local report is
`artifacts/long-video-generation-check/stock-headroom2-60s-multiseed/analysis/REPORT.md`.

On the same ComfyUI commit, a core-only `EmptyMiniMaxH3LatentAV -> VAEDecodeAudio`
graph independently reproduces a CUDA-input/CPU-filter mismatch under `--novram` at the
MiniMax H3 audio VAE upsampler. The T8 AV Decode node is therefore not the source of that
failure. The DynamicVRAM route succeeds; `--novram` H3 audio decode is not advertised as
compatible until ComfyUI or a separately validated local workaround resolves the buffer move.

## Reproduce the conversion

```powershell
$sourceDir = '<path-to-source-loras>'
$outputDir = '<path-to-converted-loras>'
python .\tools\convert_minimax_h3_lora_for_comfyui.py `
  "$sourceDir\minimax_h3_turbo_4步加速.safetensors" `
  "$sourceDir\minimax_h3_turbo_4步加速ema.safetensors" `
  --output-dir $outputDir
```

The converter is strict: it checks the MiniMax-H3 metadata, all 259 expected
adapter modules, all 518 tensor names/shapes/dtypes, and bitwise tensor equality
after saving. It writes through a temporary file and never changes the sources.

For the exact 10Eros curve-pruned checkpoint, use the separate no-overwrite tool rather than the
generic prefix converter:

```powershell
$sourceLora = '<path-to-converted-LightX2V-LoRA>'
$targetModel = '<path-to-exact-pruned-model>'
$timeReference = '<path-to-full-FL2VA-time-reference>'
$outputLora = '<new-output-path>'
$coreAblation = '<optional-new-core208-output-path>'
python .\tools\convert_minimax_h3_turbo_for_pruned_curve.py `
  --lora $sourceLora `
  --pruned-model $targetModel `
  --time-embedder-reference $timeReference `
  --output $outputLora `
  --core208-output $coreAblation `
  --expected-lora-sha256 35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b `
  --expected-pruned-model-sha256 f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062 `
  --expected-time-reference-sha256 7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5 `
  --expected-table-sha256 ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e
```

The main output has 259 A/B adapters plus 51 FP32 `.diff_b` tensors (569 tensors total). The tool
keeps the 208 directly compatible adapters and every AdaLN B bit-identical, fits the 51 A tensors
over the 1025-point time curve with an affine intercept, refuses existing outputs, validates readback,
and re-hashes all three inputs after publication. The generated SHA-256 is
`6c2f38d45dfa3fc282a48de3171b6946a5e6d46e13f832c43b93734f6d12edf5`. Use the bypass loader at
strength 1.0 first. The current evidence is one 256x256/124-frame four-step AV smoke plus static
validation, not a high-resolution or multi-seed quality release. Generated model files and local
sidecars remain outside Git; see `VERIFICATION_REPORT.md` for the durable public summary.

## Variant choice

- Standard: usually sharper on fast motion.
- EMA: time-averaged; usually smoother.

These descriptions come from the upstream model card. Compare them with the
same prompt and seed.

## Verified mapping

| Source module | ComfyUI target | Count | Full/non-pruned | Pruned |
|---|---|---:|---:|---:|
| `blocks.*.{attn,mlp}` | `diffusion_model.blocks.*.{attn,mlp}` | 200 | 200 | 200 |
| `blocks.*.adaln_proj.linear` | prefixed same path | 50 | 50 | 0 |
| `token_refiner.blocks.*.{attn,mlp}` | prefixed same path | 8 | 8 | 8 |
| `final_layer.adaln_proj.linear` | prefixed same path | 1 | 1 | 0 |
| **Total** |  | **259** | **259** | **208** |

The files target the generic ComfyUI LoRA convention recognized by
`comfy.lora.model_lora_keys_unet()` and preserve the upstream scale semantics:
no `alpha` tensor means scale `1.0`, matching `W_eff = W + B @ A`.

## Sources checked

- [Original MiniMax-H3 Turbo LoRA repository](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [Upstream ComfyUI conversion discussion #1](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/1)
- [Upstream sampler/loading discussion #6](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/6)
- [Official ComfyUI MiniMax-H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI LoRA key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py)
- [ComfyUI bypass loader](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_lora_debug.py)
