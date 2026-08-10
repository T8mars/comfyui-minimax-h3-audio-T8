# LoRA and sampler verification report

This report records the historical LoRA, stable sampler, and multi-rate sampler
verification checkpoint. For the current plugin version, node inventory, and
Ref2VA still-image status, also read the project-root `README.md` and
`features.json`.

The current 1.12.0 checkpoint was verified on 2026-08-10 against ComfyUI `0.31.0` at
`cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`. Historical LoRA conversion evidence below was
originally recorded on 2026-08-06 against source commit
`563b98eefbe643a4cd510ee7f0b43e79880d5a3f`.

## 1.12.0 experimental dialogue-safe audio checkpoint

Three nodes were appended after the prior 51-node inventory. No old node ID, order, default,
schema prefix, or execution path changed:

- `MiniMaxH3DialogueBoundaryAnalyzerT8` performs read-only local CPU faster-whisper analysis. It
  returns a boundary only for exactly one contiguous exact normalized target sequence; zero or
  multiple exact spans are rejected. Signal energy after the boundary is reported as activity,
  not classified as speech.
- `MiniMaxH3DialogueSafeMasterT8` requires an upstream accepted boolean and independent stems. It
  places verified speech on an exact sample timeline and preserves full-duration music, ambience,
  and SFX after speech ends. Strict is the default fit policy; loop/pad/trim only occur when the
  user explicitly selects and receives a report for them.
- `MiniMaxH3TimedAudioBedLockT8` is a two-pass H3 helper. It encodes an independent dialogue-free
  background bed, preserves the input video stream/mask, and applies a 40Hz audio mask with a
  default zero-denoise tail. Existing audio masks are caps, so the node never increases generation
  freedom in an already constrained interval.

Model-free boundary, mixing, mask, immutability, strict-fit, explicit-fit, and registration tests
passed. A real FL2VA pruned INT8, Qwen3-VL NVFP4, H3 Audio VAE, 256x256, 124-frame, stable
dual-clock four-step forward also passed. The standard 124-frame Audio Window encoded to 206 audio
latent steps against the AV clock's 207, so strict mode deliberately failed and `fit_reported`
explicitly zero-padded one step. Comparing saved audio latents before and after sampling at the
2.0-second/step-80 lock boundary produced:

| Region | Mean absolute change | Maximum absolute change |
|---|---:|---:|
| editable head, steps 0–79 | 0.502230 | 2.402396 |
| locked tail, steps 80–206 | 1.81e-8 | 2.38e-7 |

The locked tail therefore remained within `1e-6` absolute tolerance across four sampler steps,
while the head materially changed. This establishes the latent-mask endpoint mechanically, not
decoded perceptual quality. The same Audio VAE decoded both latents to 165,600 samples at 32kHz.
The first 100ms after the 2.0-second boundary still had a 0.34177 maximum difference, and decoder
influence decayed through roughly 2.3 seconds; later 100ms windows were approximately `3.97e-4` or
lower in maximum difference. A latent boundary is therefore not a sample-exact audio cut and is
not advertised as seamless.

The read-only analyzer was also run against two prior real Joint-dialogue failures using the local
multilingual small model. When unwanted words interrupted the expected sequence, it returned
`target_not_found`. When 17 unwanted units preceded one contiguous exact target, it returned
7.00–9.72 seconds with `clean_exact=false`; it did not auto-trim or accept the mixed result.

Automatic source separation remains deliberately absent. The installed `audio_separator` Python
package has no selected local model, and common vocal/music separators are not evidence of safe
target-dialogue removal from a master containing intentional singing, music, ambience, and SFX.
Synthetic known-stem leakage/damage tests, real H3 mixes, and listening gates are required before
any separator can become an opt-in experiment. No speech-stop, mouth-stop, seamless-tail,
source-separation, or 16GiB `memory_safe` claim is made.

## 1.9.0 experimental visual-reference strength checkpoint

`MiniMaxH3VisualReferenceStrengthEXPT8` was added as node 36, after all 35 existing nodes. It is a
post-conditioning node that calls `node_helpers.conditioning_set_values()` with
`minimax_visual_cond_noise_aug`; it does not receive or patch MODEL, latent, VAE, sampler,
scheduler, sigmas, shifts, or steps. It rejects missing visual conditions and audio-only refs,
allows keyframes with an explicit global-scope warning, and reports values at or below 0.950 as
aggressive. The current H3 core maps this field to `visual_cond_noise_aug` and uses 0.999 when the
field is absent.

The live matrix used ComfyUI `0.31.0` at
`cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`, Windows, an RTX 4060 Ti 16GiB, DynamicVRAM, the full
`minimax_h3_ref2va_int8_convrot.safetensors`, Qwen3-VL NVFP4 CLIP, FP16 video VAE, FP32 audio VAE,
and no LoRA. Every treatment fixed one reference image, prompt, seed `2608102201`, 736x416,
124 frames, 20 steps, `dual_clock_euler + native_flow`, and shifts 12/3.

| Treatment | Runtime | Whole-device peak | Minimum free | Result |
|---|---:|---:|---:|---|
| no post node | 254.375s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.999 | 303.984s | 16,344.617MiB | 34.883MiB | success |
| explicit 0.995 | 276.812s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.990 | 276.563s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.980 | 264.703s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.950 | 378.860s; repeat 482.731s | 16,017.422MiB on repeat | 362.078MiB | both success |

The timing variation includes dynamic-loading stalls and is not attributed to the scalar itself;
the graph retains 20 DiT steps and the new node adds no model call. All observed free margins are
below the project's 512MiB safety gate, so this matrix does not establish a 16GiB memory-safe tier.

The critical compatibility gate passed exactly. Decoded no-node and explicit-0.999 video both
contained 124 RGB frames and 113,897,472 bytes with identical SHA-256
`59b0cff4408b2656f7c42c9e2c5430649e25b8899047fe7d54cf45e69b5763df`; their decoded float32 audio
both contained 325,632 samples and had identical SHA-256
`82796fa54b165f0ad9c86bf00777d47f68d11637328c567bb991e1f05ec8477f`. Both maximum absolute errors
were zero. Two explicit-0.950 runs were also decoded-video and decoded-audio identical, with maximum
absolute errors zero. This proves deterministic routing for these fixed controls; it is not a
cross-hardware bitwise guarantee.

Whole-video objective proxies were computed over all 124 decoded frames:

| Strength | Mean RGB MAD vs 0.999 | Temporal MAD | Temporal gray SSIM | Face high-pass std |
|---:|---:|---:|---:|---:|
| 0.999 | 0.0000 | 4.9202 | 0.89021 | 8.1772 |
| 0.995 | 24.8277 | 3.6117 | 0.91423 | 8.1577 |
| 0.990 | 25.5275 | 4.1108 | 0.90307 | 8.1552 |
| 0.980 | 13.5682 | 4.9253 | 0.88860 | 8.2397 |
| 0.950 | 33.5909 | 3.4078 | 0.92725 | 7.6752 |

MAD/SSIM only measure change, and Haar-face plus high-pass/Laplacian values are composition and
sharpness proxies, not identity or skin-realism scores. Higher temporal SSIM can also mean less
motion. Manual first/middle/last-frame review found that 0.995 through 0.950 changed pose,
expression, motion trajectory, and/or composition; 0.950 introduced the largest background and
framing shift. Its full-frame edge variance rose while its face high-pass proxy fell. This one
case therefore proves that the control is effective, but it does not establish a winning
"de-wax" value or monotonic visual improvement.

Regression evidence:

- 188 project tests passed; Ruff reported no findings;
- isolated live `/object_info` exposed the exact two inputs, 0.999/0..1/0.001 numeric contract,
  two named outputs, EXP flag, and category;
- the API and ComfyUI 0.4 frontend examples passed structural/link checks and the frontend workflow
  passed live object-info validation;
- `sampling.py` SHA-256 remains
  `111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`;
- `conditioning.py`, `sampling_multirate_exp.py`, and `still_image.py` also remain byte-for-byte
  unchanged, with SHA-256 `E15D95454FFD60076FFADECA5C205B9608AE225606ED955A09AAD95F0212C9E4`,
  `BADCFA055938FF2AB0E0B8BD8C2FD789B6FAB33CC312F891E5226E8419BD4D5F`, and
  `B154E3E154FD4DB1927A7E52BE96AC05EA827BE0A8CE6B5C2A27529016B23CE8`.

Local detailed telemetry, decoded-equivalence files, objective metrics, and the contact sheet are
under `artifacts/ref2va-visual-strength-check/`. That directory is intentionally excluded from Git.
Remaining quality work is a representative multi-reference, multi-seed, image/video/keyframe and
human-preference matrix; until then the node remains Experimental and must not be described as a
Ref2VA oiliness fix.

## 1.7.0 explicit background executor checkpoint

Implemented locally on 2026-08-09 as two nodes appended after the prior 23. Stable sampling
math and every old node position remain unchanged.

- `Background Start` executes before model work, captures the complete API prompt only in
  process memory, associates it with the current ComfyUI prompt ID, and starts history
  monitoring so errors before the terminal node are observable;
- `Auto Accept & Continue` is the explicit output boundary. In background mode it validates and
  accepts one candidate, requests the configured release policy, validates one copied prompt,
  and queues exactly one next segment. Its safe default remains non-mutating `review_only`;
- persistent `background_job.json` contains lifecycle state, counters, accepted paths, and a
  prompt SHA-256, but not the prompt body. The error serializer allowlists node/error/traceback
  fields and excludes ComfyUI `current_inputs/current_outputs`, media tensors, and prompt text;
- controls are status, pause-after-current, resume, and prompt-ID-targeted cancel. Pause retains
  the accepted manifest. Cancel does not wipe unrelated ComfyUI queue items;
- retries reuse the exact prompt and never silently change size, frames, context, seed, sampler,
  scheduler, or steps. The default is one additional attempt; exhaustion leaves `failed`;
- release policies are `keep_loaded`, `clear_execution_cache`, and `unload_all_models`.
  The middle option explicitly sets `unload_models=false` plus `free_memory=true`; the strong
  option is correctly described as a global ComfyUI model unload, not an H3-only operation.
  The selected policy is requested after every durable acceptance, including pause and final;
- copied prompts are sanitized of ComfyUI runtime `is_changed` fingerprints before queueing.
  This prevents `keep_loaded` from treating an advanced segment as a whole-graph cache hit;
- final composition may run automatically. Any failure after the manifest acceptance boundary
  stops without retrying a prompt whose Orchestrator would already resolve to a later segment.

Validation evidence:

- 139 project unit/structure tests and Ruff pass; the original 23 IDs keep their order and the
  two background IDs are appended;
- isolated `--quick-test-for-ci` import succeeds; live `/object_info` reports both schemas and
  the status/control routes respond with expected 200/409 semantics;
- a model-free live graph accepted and composed two segments under two prompt IDs;
- another graph paused while segment 0 was running, reached `paused` after accepting exactly one
  segment, resumed through a new prompt ID, and completed segment 1;
- a targeted running-prompt cancel signalled an interrupt, ended with error history, accepted
  zero segments, and created no manifest;
- a deterministic upstream error produced exactly two history records with `max_retries=1`,
  retained the same prompt/settings, then stopped in `failed`. This probe exposed that raw ComfyUI
  errors contain complete `current_inputs`; the allowlist fix above was added before release;
- the first live resume route test exposed a same-event-loop wait and returned a 60-second timeout.
  Route controls were moved through `asyncio.to_thread`; the complete pause/resume test then
  passed in 5.7 seconds with an immediate `running` resume response.

A final real-model mechanical probe used non-pruned FL2VA INT8, Standard Turbo EMA LoRA,
Qwen3-VL NVFP4, both H3 VAEs, 256x256, a 124-frame window, 22-frame AV context, one sampling
step, DynamicVRAM with 2.0GiB headroom, and `unload_all_models` between segments. Both distinct
prompt IDs completed successfully. Manifest revision 2 contains 124+20=144 contiguous frames
and 192,000 absolute audio samples; the automatically composed H.264/AAC file reports 24fps,
144 frames, and exact 6.000-second video, audio, and container streams. Runtime was 79.41 seconds;
whole-device polling observed 7,129MiB baseline and 14,254MiB peak. The final SHA-256 is
`e24acdc57996ae15a15a1590f3066c738f3ce39ba1949ffd5b42f2743e75eb7b`.

This one-step 256x256 result validates executor mechanics and strong-release reload only. It does
not establish four-step perceptual quality, high-resolution memory safety, cross-GPU behavior,
multi-reference safety, or any universal no-OOM guarantee. A separate, bounded crash-recovery
probe is recorded below.

The crash-recovery probe hard-killed the owned ComfyUI process after segment 0 was durably
accepted at manifest revision 1 while the next prompt was active. After restart, status converted
the stale persisted active state to `detached`, retained `accepted_count=1`, and returned
`recovery_action=queue_workflow_once`. Requeueing the workflow once created a new job linked to
the old job and generated segment 1 only. Segment 0's candidate ID, MP4 SHA-256, AV-tail tensor
SHA-256, and file modification time were unchanged. Revision 2 produced a 256x256, 144-frame file
whose video, audio, and container durations are exactly 6.000 seconds. The post-lease repeat's
native-v2 recovery whole-device peak was 13,537.02MiB and the final SHA-256 is
`5a2d59d69c8ff56549a76a0d274d8ce61c194bb1ebac2298f8e2803ba21461d8`.

The same probe then queued two independent real H3 background chains. The single ComfyUI prompt
queue interleaved them as `A0, B0, A1, B1`; both reached revision 2 with isolated jobs, prompts,
parent chains, roots, manifests, and final files. The native-v2 repeat peaked at 13,511.44MiB and
no OOM occurred.

The local multi-process gate now uses an OS-owned `manifest.lock.v2`. Two processes competing for
one chain/index/revision produced exactly one accepted revision and one pre-copy rejection; four
processes retained all 100 of 100 protected updates; and a killed lock owner was replaced within
two seconds. A chain-wide background lease rejected a second ComfyUI process before generation,
then allowed a third process to attach with `previous_job_id` after the first was killed. Tests
also prove live legacy locks are respected, dead legacy residue remains in place but does not
block v2, unknown schemas cannot roll back to an older backup, same-schema optional metadata is
preserved through a later commit, invalid primaries do not poison valid backups, unreadable
auxiliary state is quarantined, and newer background state is not overwritten.

Accepted manifests and background states now use explicit schema-2 format markers. Fixture tests
prove read-only in-memory normalization of schema 1, next-write atomic manifest upgrade with the
raw schema-1 primary preserved as backup, schema-1 background reattachment to schema 2, recovery
through a schema-1 backup, and fail-closed handling of unknown future schemas. An existing real H3
schema-1 chain was read without changing either raw file hash, while a new hard-kill plus dual-chain
probe wrote and retained native schema 2. In that repeat, all 13 crash assertions and all 9
isolation assertions passed.

An acceptance-transaction audit added eight deterministic fault-injection cases. Exact
idempotent re-accept now repairs missing/corrupt accepted assets from a hash-verified candidate
without changing manifest revision. Reusing a canonical candidate id for different bytes, or
reusing an invalidated id to collide with an archived path, is rejected before any overwrite.
Context-copy failure and failure after the backup write but before primary replacement leave the
old manifest authoritative and are recoverable by retrying the same candidate. A missing primary
now loads a valid backup even under `allow_new=True`; unknown-schema or corrupt backups fail closed
instead of resetting the chain to empty.

The accepted-media/manifest split and backup/primary split were then tested with actual
subprocess termination on Windows/NTFS. The worker held `manifest.lock.v2`; the parent killed it
after the accepted MP4 copy but before context/manifest, and after the revision-1 backup write but
before revision-2 primary replacement. Three independent rounds exercised both breakpoints, for
6/6 successful recoveries. The OS lock released automatically, the pre-kill manifest state stayed
authoritative, and retrying the same candidate completed in under two seconds with the expected
hashes and revisions. These were small-media transaction tests, not live H3 CUDA kills or power-loss
tests.

This closes the tested post-acceptance hard-kill boundary, same-queue multi-chain isolation, local
schema-1 to schema-2 migration contract, and same-host Windows/NTFS same-chain ownership/manifest
serialization. It does not validate a complete upgrade/downgrade matrix across independently
released plugin/ComfyUI builds, network/shared-filesystem locking, arbitrary-instruction recovery,
simultaneous GPU execution, or multi-GPU parallelism.

The representative four-step gate was subsequently run with non-pruned FL2VA INT8, Standard
Turbo LoRA, 736x416, a 124-frame render window, 22-frame AV context, DynamicVRAM headroom 2GiB,
and global `unload_all_models` between segments. A 60-second request produced 14 distinct prompt
IDs and every history record ended in success without retry or OOM. Accepted manifest revision
14 is exactly `124 + 12*102 + 92 = 1440` frames and 1,920,000 audio samples. Every accepted media
and continuation-context hash matches the manifest, and the candidate-parent chain is contiguous.
The final H.264/AAC file has 736x416, 24fps, 1440 decoded frames, and exact 60.000-second video,
audio, and container streams. Its SHA-256 is
`cb3bdf5bae847c6f0fe708d991bd35a736fdc012e44cb21ef612d7c0f2f83ed0`.

Half-second `/system_stats` polling measured 1,229.63MiB baseline, 12,823.13MiB whole-device
peak, 3,556.37MiB minimum free margin, and 3,496.30MiB PyTorch peak. The running maximum reached
12,784.24MiB at segment 3, increased only 38.89MiB at segment 9, then remained flat through
segment 14; this one sequence shows no staircase leak. Total runtime was 1,478.83 seconds.
The same prompt/seed's segment-0 MP4 is bit-identical to the six-second preflight; context file
metadata differs, while `video_tail` and `audio_tail` tensor payloads remain bit-identical.

Across 13 visual seams, MAD median/max was 0.04696/0.07420 and SSIM median/min was
0.73139/0.64589. Only two seam MAD values were locally highest; inspection of all contact pairs
found no obvious scene cut, identity reappearance, or camera reversal in this walking shot.
Audio half-second level change had -0.42dB median and 13.75dB maximum absolute value. The 5ms
bridge reduced median single-sample jump from 0.08247 in source segment contacts to 0.00222 in
the final encode, about 96.16%, but does not solve level, speech, music, timbre, or lip-sync
continuity. Therefore this is a passed local fixed-profile/single-case gate, not a universal
`memory_safe`, seamless, or never-OOM result. Later evidence closes fixed-prompt cross-base-seed
cold mechanical repeatability only; same-seed whole-chain warm repeats, different materials, other GPUs,
high resolutions/references, actual cross-version migration, network/shared-filesystem locking,
and true GPU parallelism remain open; the separate post-acceptance hard-kill, local multiprocess,
and same-queue dual-chain probes above do not close those broader gates.

The 60-second run also exposed that the original release call was only reached when another
prompt was queued, so pause/final states retained model VRAM. The state machine was changed to
request the selected policy immediately after every durable acceptance and before branching to
continue, pause, or final. A real 256x256 one-step/two-segment follow-up completed with
`last_release_policy=unload_all_models`; whole-device use fell from about 8,124MiB at completion
to 1,230MiB within 15 seconds without calling `/free`, a 6,894MiB drop. Unit coverage also locks
pause/final release and stops without queueing if a release request fails after acceptance.

A controlled release-policy matrix then used the same FL2VA INT8 checkpoint, Standard four-step
Turbo LoRA, prompt, 736x416 canvas, 124-frame window, 22-frame AV context, and two-segment
six-second plan. Three paired seeds were run for each policy in three fresh-process cold trials.
For warm trials, each policy received one unmeasured same-process primer followed by the same
three measured seeds. All 21 chains succeeded without retry or OOM; all 18 measured chains reached
manifest revision 2.

| Policy | Cold runtime mean (SD) | Warm runtime mean (SD) | Cold/warm peak mean | Cold/warm post-15s mean |
|---|---:|---:|---:|---:|
| `keep_loaded` | 170.89s (2.24) | 153.10s (0.12) | 13,449.95 / 13,467.30MiB | 8,083.22 / 7,987.22MiB |
| `clear_execution_cache` | 188.28s (0.80) | 185.08s (0.42) | 13,434.52 / 13,408.94MiB | 1,229.63 / 1,229.63MiB |
| `unload_all_models` | 189.08s (0.68) | 197.13s (3.07) | 13,421.52 / 13,384.03MiB | 1,229.63 / 1,229.63MiB |

For each seed, segment-0 video, segment-0 accepted AV-tail tensor payload, segment-1 video, and
final H.264/AAC file hashes were identical across all six policy/temperature conditions. Every
paired whole-device peak difference was below the existing 128MiB material-difference threshold;
the roughly 1GiB strong-unload peak advantage in the earlier single run did not repeat.

Relative to `clear_execution_cache`, `keep_loaded` was 17.39 seconds faster cold and 31.97 seconds
faster warm on average, but retained 6,853.59/6,757.59MiB more device memory after 15 seconds.
Global unload was only 0.80 seconds slower cold but 12.05 seconds slower warm than the default,
with no material peak advantage and broader side effects. The default therefore remains
`clear_execution_cache`; `keep_loaded` is an explicit throughput-for-residency tradeoff.

The first `keep_loaded` attempt had exposed that current ComfyUI mutates prompt nodes with runtime
`is_changed` fingerprints: replaying those values cached the whole second prompt and prevented the
manifest from advancing. Sanitizing the snapshot fixed the issue. Across the complete matrix,
keep-loaded cached only unchanged loader nodes 1-5; orchestration, sampling, saving, and terminal
acceptance reran. The other policies had no cached nodes. These results remain specific to the
local GPU/model/profile and do not establish cross-GPU or universal no-OOM behavior.

## 1.6.0 total-duration orchestration and resume checkpoint

Implemented locally on 2026-08-08 without changing the stable sampler:

- `MiniMaxH3LongVideoOrchestratorT8` registers after the prior 22 nodes and emits only
  plan/state values; it does not own a MODEL or retain historical IMAGE/AUDIO tensors;
- total duration is quantized once to an exact 24fps frame count, then split across a
  fixed legal `17n+5` render window. The default 60-second/124-window/22-context plan is
  14 segments with effective frame counts `124 + 12*102 + 92 = 1440`;
- the final short tail still renders through the same fixed 124-frame H3 window and is
  trimmed only as final output, preserving the bounded per-segment sequence contract;
- global and per-segment prompt/seed/note values are supported, with fixed, incrementing,
  and deterministic chain/segment hash seed policies;
- steps, video/audio shifts, sampler, and scheduler are single-source Orchestrator values.
  They drive the stable sampler and a machine-generated candidate `sampling_summary`; accepted
  chains reject a changed summary before another segment is sampled;
- the contiguous accepted-manifest length selects the first unaccepted segment. Accepted
  fps, frame count, absolute timeline, and final identity are validated against the plan;
  incompatible settings fail instead of silently resuming the wrong chain;
- after the final segment is accepted, the node returns full progress and a ComfyUI
  `block_execution` reason, preventing an accidental extra sampling pass;
- the frontend and API graphs are respectively
  `examples/workflows/H3_Long_Video_Auto_Resume_22F_EXP.json` and
  `examples/long_video_auto_resume_api.json`.

Validation evidence:

- 104 project unit/structure tests pass, including duration quantization, fixed-window
  planning, final-tail trimming, prompt/seed overrides, deterministic hash seeds,
  manifest resume, changed-plan rejection, and complete-chain execution blocking;
- Ruff and `git diff --check` pass;
- stable `sampling.py` remains SHA-256
  `111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`;
- ComfyUI `--quick-test-for-ci` imports the plugin successfully in isolation;
- a temporary isolated live server registers 23 T8 nodes, exposes the Orchestrator with
  13 required inputs and 22 outputs, and lists the installed auto-resume frontend
  workflow through `/userdata`.

A real execution probe then ran the new API graph in the user's normal DynamicVRAM environment:

- non-pruned FL2VA INT8, Standard Turbo LoRA, NVFP4 H3 CLIP, both H3 VAEs, 736x416,
  a 124-frame internal window, one sampling step, and a one-second target;
- the joint H3 sampling, video/audio decode, exact final trim, and candidate save completed;
- the candidate descriptor contains 24 frames, absolute audio samples `[0, 32000)`, final=true,
  and no continuation context. PyAV reports 24fps/24 frames and exactly 1.0-second video,
  audio, and container streams;
- accepting the candidate produced manifest revision 1. Re-queueing the complete graph returned
  success with no downstream outputs and did not create another candidate, proving the final
  execution block in live ComfyUI;
- accepted-file composition produced a 736x416, 24fps, 24-frame MP4 whose video, audio, and
  container durations are all exactly 1.0 seconds.

The first controlled attempt used `--novram` and completed H3 sampling but failed at audio VAE
decode with a CUDA-input/CPU-filter device mismatch. A separate core-only graph using
`EmptyMiniMaxH3LatentAV -> VAEDecodeAudio` reproduced the same traceback at
`comfy/ldm/minimax/audio_vae.py:102`, without any T8 node in the decode path. This isolates the
failure to current ComfyUI H3 Audio VAE dynamic buffer handling under `--novram`; the normal
DynamicVRAM route succeeds. No plugin workaround was added without an independent correctness
and memory proof.

During the one-step probe, the temporary API mutation changed the sampler step count but initially
left the example's manually entered description at its four-step default. This exposed a genuine
provenance weakness rather than a sampling failure. The post-probe hardening removes that duplicate
entry from the auto-resume workflows: Orchestrator now emits steps/shifts/sampler/scheduler to the
sampler and emits the corresponding machine-generated summary to Candidate Save. Targeted schema,
API, frontend-link, resume-conflict, and completion tests pass for this revised route. A second
real-model run changed only Orchestrator `steps` to 1; the actual sampler ran 1/1 step and the
candidate recorded `1-step dual_clock_euler/native_flow shift12/3`. Acceptance, completion
blocking, and exact one-second composition succeeded again, so the provenance fix is covered by
runtime evidence rather than structure tests alone.

A second real probe covered four-step multi-segment auto-resume with the same non-pruned FL2VA
INT8/Standard Turbo/NVFP4 CLIP/dual-VAE stack, 736x416, a fixed 124-frame window, 22-frame AV
context, and a six-second target. Segment 0 produced 124 frames; after acceptance, the same API
graph resumed at segment 1 with the accepted parent/context and produced a 20-frame final tail.
The revision-2 manifest covers exactly 144 frames. A further queue returned success with zero
output nodes and no extra candidate. Both unbridged and 5ms-bridge compositions report 24fps,
144 frames, and exact 6.000-second video/audio/container streams.

The bridge reduced the post-AAC single-sample boundary jump from 0.000178755 to 0.000035433,
about 80.2%, but the adjacent audio-window level changed by about 33.30dB. Video contact-sheet
inspection found no obvious identity/composition cut, while boundary MAD and SSIM discontinuity
both ranked at the 100th percentile among 16 nearby intra-segment transitions and flow ranked at
87.5%. Device peaks were about 15,461.4 and 16,181.5MiB; the second leaves only about 198MiB and
fails the 512MiB safety gate even with `--vram-headroom 0.5`. Background auto-queue,
pause/cancel, automatic retry/model release, the multi-material 8-16 segment quality matrix,
blind listening, and VRAM safety tiers remain open.

### Four-step 60-second / 14-segment runtime checkpoint

A third real probe used the same FL2VA INT8/Standard Turbo/NVFP4 CLIP/dual-VAE stack,
736x416 canvas, 124-frame render window, 22-frame AV context, four-step
`dual_clock_euler/native_flow shift12/3`, and DynamicVRAM. It ran all segments in one warm
ComfyUI process without an explicit `/free` call:

- all 14 candidates were generated and accepted in parent/context order; the manifest reached
  revision 14 with effective segment frames `124 + 12*102 + 92 = 1440`, timeline end frame
  1440, audio end sample 1,920,000 at 32kHz, and a final segment;
- unbridged and 5ms cosine-bridge compositions both report 24fps/1440 frames and exactly
  60.000-second video, audio, and container streams;
- the first uncached full re-queue after final acceptance returned the expected ComfyUI
  `ExecutionBlocked` terminal status with an empty traceback and execution stopped at the
  Orchestrator. A later cached re-queue reported success while executing only the review node.
  Neither path sampled or created a fifteenth candidate;
- measured device peaks ranged from 15,479.95 to 16,228.17MiB. The descriptive warm-run peak
  slope was +28.03MiB/segment, the baseline slope was negative rather than a monotonic staircase,
  and the peak range was 716.83MiB. This one run does not show a cumulative VRAM leak, but five
  segments left less than 512MiB and segment 12 left only 151.33MiB. It fails the proposed 16GB
  safety gate; 0.25-second polling can also miss shorter spikes;
- generation time summed to about 1,336.8 seconds for the 14 segments, excluding orchestration,
  acceptance, composition, and offline analysis.

All 13 video boundaries were decoded and compared. Median/max boundary MAD were
0.01618/0.01906; median/min SSIM were 0.96374/0.92868; median/max mean optical flow were
0.2603/0.4748. The worst values occurred at the 11-to-12 seam. Contact-sheet inspection did not
show a hard subject/background cut at the worst two seams, while the segment-middle timeline
showed gradual subject-appearance and exposure drift. Pixel and flow metrics are not identity
verification.

Audio evidence rejects a seamless or lossless claim. Adjacent half-second level change had a
median of -9.51dB and a maximum absolute value of 40.83dB at the 12-to-13 seam. The above-8kHz
energy ratio fell from -32.14dB in segment 0 to -68.44dB in segment 13, a -36.30dB change that is
consistent with substantial recursive high-frequency loss. Descriptive adjacent-window NCC had
a median of 0.206 and median absolute best lag of 65.16ms; these are not overlap-reconstruction
scores. The cosine bridge reduced the median post-AAC single-sample boundary jump by 97.23%,
but it cannot repair level, spectrum, speaker identity, speech semantics, or lip sync.

This checkpoint proves one exact, resumable 60-second execution and bounded peak behavior in one
warm process. It does not satisfy the multi-material/multi-seed quality matrix, cold/warm repeat
matrix, blind listening, ASR/speaker/lip-sync evaluation, or a publishable 16GB safety tier.

### 5-frame versus 22-frame context pilot

Two controlled 736x416 probes used the same prompt, base/incremented seeds, model, Standard
Turbo LoRA, four-step sampler, render window, duration, DynamicVRAM arguments, and no explicit
`/free` between segments. Only context length and isolation labels differed. Segment 0 was
bit-identical across the 5-frame, historical 22-frame, and control 22-frame chains. The historical
and control 22-frame segment-1 MP4s were also bit-identical. Nevertheless, the same 22-frame
segment-1 output had measured peaks of about 16,181.55 and 15,194.31MiB, a 987.24MiB range.
This directly demonstrates that a single polled peak cannot be attributed to context length.

In the matched no-`/free` pair, the 5-frame segment-1 peak was 15,330.47MiB and the 22-frame
control was 15,194.31MiB; normalizing each to its own context-free segment 0 still left 5 frames
about 66.94MiB higher. The 5-frame segment was 8.30 seconds (8.77%) faster. Video seam metrics
were mixed and visually both contacts remained continuous in this low-motion sample. The
5-frame audio level drop was 43.65dB versus 33.29dB for 22 frames, while its descriptive NCC/lag
were better; neither route supports a seamless-audio claim.

A second pair raised the canvas to 1056x608 (642,048 pixels) and used 0.10-second VRAM polling.
Both two-segment chains completed and all four bridge/unbridged assemblies were exactly
24fps/144 frames with 6.000-second video, audio, and container streams. Segment 0 was again
bit-identical. The 5/22-frame segment-1 peaks were 15,205.91/15,341.03MiB with free margins
1,173.59/1,038.47MiB. After normalizing against each pair's segment-0 peak, the context delta
differed by only 11.44MiB. The 5-frame segment was 30.28 seconds (12.86%) faster, but the
22-frame route had lower video MAD (0.01477 versus 0.01612), higher SSIM (0.95616 versus
0.95035), a smaller audio level change (-2.39 versus +7.01dB), and higher descriptive NCC
(0.406 versus 0.111) in this single sample.

The planned alternating-order repeat gate was then completed at both resolutions. Each matrix
used a fixed accepted segment-0 baseline per context, three paired seeds, three isolated cold
runs per context, a same-process primer plus three warm runs per context, and 0.10-second polling.
Within each resolution, segment-0 MP4s and AV tail tensors were bit-identical; all six matching
context+seed cold/warm outputs were also bit-identical.

At 736x416, cold absolute peak means were 15,279.5MiB for 5 frames and 15,224.0MiB for 22 frames.
Paired `22-5` differences were +96.6/-78.3/-184.9MiB, so the device-peak direction was not
repeatable. Sampler PyTorch-pool means were 3,189.9/3,495.3MiB, a repeatable local reduction of
about 305MiB for 5 frames. Cold runtime means were 86.53/93.08 seconds; warm means were
69.27/78.01 seconds. The warm minimum margin was 97.6MiB and 5/6 runs were below 512MiB.
The three-seed 22-frame mean video MAD/SSIM and audio level/NCC were better at this resolution.

At 1056x608, cold absolute peak means were 15,739.0/15,724.2MiB, but paired `22-5` differences
were -752.0/+10.8/+696.7MiB. Sampler pool means were 5,753.4/6,381.2MiB, a repeatable local
reduction of about 628MiB for 5 frames. Cold runtime means were 200.29/230.38 seconds; warm means
were 187.89/218.40 seconds. All six warm runs were below the 512MiB gate and the minimum margin
was only 33.6MiB. Five frames had lower mean MAD/higher SSIM, which may include reduced motion;
22 frames had materially better audio level/NCC, while one seed showed a clear front-to-profile
video boundary change.

The 39-frame gate then reused the 736x416 accepted baseline and the same three paired seeds. Three
isolated cold starts and three same-process post-primer warm trials all completed without OOM;
matching cold/warm candidate MP4s were bit-identical, and the 5/22/39 segment-0 MP4 plus video/audio
tail tensors were identical. The 39-frame cold/warm sampler-pool means were 3,799.32/3,798.63MiB,
repeatably about 303-304MiB above 22 frames. Cold/warm runtime means were 101.65/87.38 seconds.
The warm process reached only 77.35MiB free and 3/3 measured trials failed the 512MiB gate.

Quality did not improve monotonically. The three-seed 39-frame mean video MAD/SSIM was
0.08801/0.68411 versus 0.00839/0.95501 at 22 frames. Manual contact-sheet inspection found one
continuous low-motion boundary, one visible pose/framing jump, and one severe identity/shot change;
the latter two were visibly worse than their 5/22-frame controls. Audio NCC increased to 0.626 on
average, but mean absolute level change was 10.27dB, so the audio evidence also does not establish
a universal quality tier.

Decision: keep 5 frames as `fast_context_5_experimental`, not `memory_safe`. Its runtime and
sampler-pool savings are real, but the end-to-end device peak is dominated by DynamicVRAM model
residency/conditioning state and changes direction between paired runs. Twenty-two frames remains
the current balanced default candidate. Thirty-nine frames is downgraded to
`context_39_high_risk_experimental`, not a quality or safety tier. The 1056x608/39-frame treatment
was not forced because its 22-frame warm control already failed the 512MiB gate in all six trials
and reached only 33.6MiB free. This is a predefined safety-gate denial, not a claim that the
unexecuted treatment must OOM. The controlled Block Cache/Sage/DynamicVRAM headroom gate was
subsequently completed below; multi-material dialogue/fast-motion/rhythmic-audio quality tests
and the 1056x608/39-frame treatment remain open.

## 2026-08-09 local memory-policy gate

The controlled matrix fixed the model, four-step sampling, 124-frame render window, 22-frame AV
context, prompt, paired seeds, and DynamicVRAM headroom 2.0GiB. Stock and Sage each completed
three cold and three warm trials at both 736x416 and 1056x608; every trial retained more than
512MiB, and matching strategy+seed cold/warm outputs were bit-identical. Default Block Cache
hit 0 of 4 forwards, held about 117.7MiB of CPU cache, and cannot remove the mandatory first
full forward, so it was rejected as the default OOM treatment.

Sage reduced runtime, but at equal headroom its whole-device peak was higher than Stock by an
average of about 1833.92MiB at 736x416 and 1411.86MiB at 1056x608 in the warm trials. It also
changed the AV output. Manual inspection of the 1056x608 sheets found material camera distance,
framing, pose, or trajectory divergence in two of three seeds. Sage is therefore only a
high-risk approximate speed experiment, not the default memory profile.

The conservative Stock+headroom-2.0 policy then completed a real uninterrupted 60-second chain:
14/14 accepted segments, manifest revision 14, exactly 1440 timeline frames, 1,920,000 audio
samples, and two 736x416 24fps H.264/AAC assemblies whose video, audio, and container durations
are all 60.000 seconds. No restart or explicit `/free` occurred between segments. Device peaks
ranged from 12,829.44 to 13,640.09MiB with a 13,137.67MiB median and 2739.41MiB minimum free
margin. The warm first-to-last difference was +182.81MiB and descriptive OLS slope was
+26.31MiB/segment, but the sequence was non-monotonic rather than a staircase leak.

Against the prior same-prompt/same-seed Stock headroom-0.5 chain, all 14 segment MP4 SHA-256
values and all 13 continuation `video_tail`/`audio_tail` tensor payloads were identical. The
context container hashes differ because chain/model metadata differs. Median device peak fell
by about 2635MiB while total generation time increased by about 1.63%. This establishes a
**validated local conservative profile** for the exact RTX 4060 Ti 16GiB, FL2VA INT8,
Standard four-step LoRA, 736x416, 124-frame, 22-context and plugin contract. It is not a general
`memory_safe` tier or never-OOM guarantee; other GPUs, higher resolutions/reference counts,
desktop VRAM pressure, and background queue/release behavior require separate gates.

### Three-base-seed 60-second cold-start follow-up

The fixed Stock+DynamicVRAM-headroom-2.0 contract was repeated with base seeds `2608082000`,
`2608083101`, and `2608083202`. Each chain used a separate ComfyUI process while prompt, FL2VA
INT8 model, Standard four-step LoRA, 736x416 canvas, 124-frame render window, 22-frame AV context,
and sampling settings remained unchanged. The first run preserves its real schema-1 manifest as
read-only migration evidence; the two new runs created native schema-2 manifests.

All 42/42 segments completed once with no OOM, retry, or candidate reuse. Independent analysis
recomputed and matched every candidate/accepted MP4 and context SHA-256, verified every parent ID
and manifest revision, and confirmed exact 1440-frame/1,920,000-sample timelines, final completion
blocking, and six 60.000-second assemblies. The three per-chain maximum peaks were 13,640.09,
13,414.01, and 13,426.72MiB; the aggregate minimum free margin was 2739.41MiB, no segment fell
below 512MiB, and all three peak sequences were non-monotonic. Total generation time per chain
ranged from 1334.251 to 1358.531 seconds. This passes the fixed local profile's cross-base-seed
cold-start mechanical and memory gate. It does not establish same-seed whole-chain warm
repeatability or a general `memory_safe` tier.

Visual quality did not pass the long-term identity gate. Worst-seam contact sheets remained
locally continuous without an obvious hard cut, yet all three 14-segment middle-frame timelines
accumulated facial-age and identity drift; seed `2608083101` changed most severely. Across runs,
median seam MAD was 0.01525-0.01651 and median seam SSIM was 0.91555-0.96374, demonstrating why
local seam metrics alone cannot establish long-term identity preservation. Maximum adjacent
half-second audio level gaps were 23.59-48.06dB, descriptive NCC medians were 0.127-0.206,
median absolute lags were 64.97-81.00ms, and the final segment's above-8kHz energy ratio was
9.66-36.30dB below the first. The 5ms bridge reduced median post-AAC single-sample jumps by
94.93%-97.33%, but cannot repair level, timbre, semantics, speaker identity, lip sync, or
recursive high-frequency loss.

The aggregate report is
`artifacts/long-video-generation-check/stock-headroom2-60s-multiseed/analysis/REPORT.md`. Its
remaining gates are same-seed whole-chain warm repeats; different prompts/materials; dialogue
ASR/speaker/lip-sync and blind listening; fast motion and rhythmic music; other GPUs, higher
resolutions/reference counts, and desktop-load profiles. The 0.10-second `/system_stats` polling
may also miss shorter peaks, and adjacent-window NCC is descriptive rather than overlap
reconstruction correlation.

### Default-off persistent first-frame identity-reference checkpoint

Source audit of the failing three-seed timelines found that the recommended background workflow
has no reference image connected. Even when a user connects `first_frame`, legacy continuation
behavior deliberately ignores it after segment 0 because previous-tail motion keyframes own the
target head. The chain therefore has motion continuity but no direct observation of the original
appearance after the first segment.

Long Video Conditioning now appends the optional `first_frame_reuse` input without changing the
existing input/widget prefix. `segment0_only` remains the default and preserves the old path.
`persistent_identity_reference` requires a connected first frame, retains its exact segment-0
keyframe role, and on continuation segments prepends the same image as non-timeline `<Picture 1>`
before any user reference images. The local MiniMax H3 payload patch already supports motion
keyframes, image references, and the marked continuation-audio window together, so no global
ComfyUI patch or new model is introduced. Explicit FL2VA plus the reference still fails closed;
users must select `auto` or `Hybrid`. Persistent plus user reference images are capped at nine.

Automated coverage verifies legacy first-frame ignore behavior, segment-0 behavior, persistent
reference ordering/media mapping, motion+image+audio payload coexistence, missing-image rejection,
reference-limit rejection, explicit-FL2VA rejection, and the appended schema default. This is a
structurally isolated implementation. Each continuation adds an image-reference block and a VAE
encode, so VRAM/runtime can increase and motion may be overconstrained.

A first same-source/prompt/base-seed two-segment real A/B ran in independent isolated ComfyUI
processes with FL2VA INT8, Standard four-step LoRA, Qwen3-VL NVFP4, both H3 VAEs, 736x416,
124/22 frames, Stock plus native DynamicVRAM headroom 2 GiB, and `unload_all_models`. Both sides
completed 2/2 prompts without OOM, retry, or cache reuse. The accepted segment-0 MP4 SHA-256 and AV
context tensor hash matched exactly, while segment 1 changed only when the persistent reference was
enabled. Both final streams decode to exact 144-frame/6.000-second video and 6.000-second 32 kHz
audio. That first quality probe was invalid for identity because the subject turned fully away
before frame 123; its single paired +607.43 MiB whole-device delta is retained as raw evidence but
is not treated as a fixed persistent-reference cost.

The replacement face-visible matrix used the same source identity and low-motion prompt with base
seeds `2608096001/2/3`. Six independent cold chains completed 12/12 prompts without OOM, retry, or
cache reuse. Every pair retained byte-identical accepted segment-0 MP4/context hashes and changed
only segment 1; all six final files again decode to exact 144-frame/6.000-second video and
6.000-second 32 kHz audio. InsightFace buffalo_l selected the largest face above a 0.8% frame-area
floor and detected the primary face on all 60 continuation frames in both modes. Pooled
persistent-minus-legacy source-embedding cosine mean/median were +0.0424474/+0.0268420, and the
persistent side was higher on 45/60 frames. Seed means were -0.0020399/+0.0699808/+0.0594012:
two seeds improved clearly and one was approximately neutral/slightly negative. Descriptive median
absolute age-estimator error improved by 3.0/3.5/1.5 years, but estimated age is not ground truth.
Frames inside one generated continuation are correlated, so 45/60 is not an independent-sample
significance test.

Whole-device paired peak deltas were -336.46/+294.12/-43.20 MiB and changed sign. They are dominated
by model-load/poll timing and do not establish a fixed peak overhead, despite the structurally added
reference block and VAE encode. PyTorch peak deltas were +122.29/+120.61/+877.19 MiB, runtime deltas
were +5.03/+3.61/+21.17 seconds, and every pair returned to the same 1231.63 MiB whole-device usage
after 15 seconds. Contact sheets show no obvious hard cut or new face artifact. Because the prompt
intentionally allowed only subtle movement, this matrix alone cannot rule out motion suppression.

The follow-up motion-rich matrix used the same source and runtime controls, base seeds
`2608097001/2/3`, and a prompt requiring continuous football tosses and wide arm movement across the
boundary. Six independent cold chains again completed 12/12 prompts without OOM, retry, or cache
reuse. Every pair retained byte-identical accepted segment-0 MP4/context hashes and changed only
segment 1; all six final files remained exact 144-frame/6.000-second video and 6.000-second 32 kHz
audio. Legacy primary-face detection passed 59/60 continuation frames and persistent passed 60/60;
the missing legacy frame occurred during vigorous motion/occlusion and was not imputed.

Across the 59 paired detected frames, persistent-minus-legacy source-embedding cosine mean/median
were +0.0269638/+0.0196597 and 38/59 frames were higher. Seed means were
-0.0254073/+0.0292898/+0.0796427: one negative and two positive. Descriptive median absolute age
error improved by 1.5/3.0/-0.5 years. Persistent/legacy temporal-MAD ratios were
0.9658/1.1970/1.0030, flow-P90 ratios 0.8405/1.3747/1.0159, and normalized face-center path ratios
1.1084/1.2221/1.1541. Visual contact sheets show active ball and arm motion in both modes. This is
evidence against an obvious systematic freeze in this short probe, not proof of equivalent action
freedom, prompt adherence, or preference.

Persistent boundary MAD was slightly higher in all three seeds, although contact sheets show no
obvious hard cut. Whole-device peak deltas were +549.93/-378.29/-169.03 MiB and therefore do not
establish a fixed device-level cost. Torch peak deltas were consistently positive at
+122.29/+133.10/+120.60 MiB; runtime deltas were +15.08/-15.14/+8.89 seconds, and all pairs returned
to equal post-15-second device occupancy. The persistent path remains default-off and must not be
called identity locking, action preserving, or memory-safe. At that checkpoint the probes authorized
one controlled intermediate 8-16-segment A/B, while the three-seed 60-second matrix remained
conditional on its identity, motion, audio, and VRAM result.

The intermediate gate then ran as one matched 32-second/eight-segment A/B with base seed
`2608097101`, the same source/prompt/model/sampling controls, independent cold processes, and global
`unload_all_models` after each accepted segment. Both modes completed 8/8 prompts without OOM,
retry, or cache reuse. Both manifests reached revision 8; segment 0 MP4/context tensors remained
byte-identical, all seven continuation MP4 hashes differed, and both final files decode to exact
768-frame/32.000-second video plus 32.000-second 32 kHz audio.

Persistent-minus-legacy whole-device peak was +84.54 MiB, Torch peak +122.29 MiB, and runtime
+61.91 seconds. Persistent minimum free VRAM was 3321.63 MiB and the post-15-second occupancy delta
was 0 MiB. This fixed-case mechanical and 512 MiB margin screen passed; it is not a general memory
safety result.

Ten evenly spaced frames per accepted segment were screened with the same InsightFace model.
Across 60 paired detected continuation samples, persistent-minus-legacy cosine mean/median was
+0.1008925/+0.0844391 and 53/60 samples were higher. Persistent won 6/7 continuation-segment
medians. The relative result is positive, but the depth target failed: legacy medians fell from
0.5482 to 0.1092, while persistent fell from 0.6134 to 0.1336. The source, both eight-frame
timelines, metric-worst seams, and eight-frame strips for segments 5-7 were inspected; both modes
visibly drift away from the source face. Relative improvement therefore does not establish long-term
identity preservation.

Persistent/legacy temporal-MAD ratios by continuation were
1.002/1.031/1.180/1.105/0.841/0.746/0.834 and flow-P90 ratios
1.118/0.945/0.924/0.899/0.823/0.590/0.903. Late strips show active football and arm motion in both
modes, including the 0.590-ratio segment, so this is not a literal freeze; it remains a material
motion-amplitude/trajectory warning. Video seam MAD median/max was 0.04871/0.08773 legacy versus
0.06791/0.08125 persistent. No metric-worst contact shows an obvious scene cut.

Audio maximum absolute adjacent half-second level gap was 4.10/3.93 dB, descriptive NCC median
0.250/0.267, and first-to-last above-8 kHz energy change -11.23/-4.73 dB for legacy/persistent.
This relative screen shows no material persistent-side regression in the one case; it includes no
blind listening, ASR, speaker, or lip-sync test.

**Decision:** the eight-segment intermediate gate fails because identity still collapses with depth
and a late motion warning remains. The predeclared gate therefore denies the three-seed 60-second
persistent/legacy matrix. `persistent_identity_reference` stays Experimental and default-off; the
identity-conditioning strategy must be redesigned before more long-chain generation. Raw metrics,
contacts, and the final report are in
`artifacts/long-video-generation-check/identity-anchor-intermediate-8segment-ab-seed2608097101/analysis/`.

### Dedicated identity crop and scene-plus-identity redesign

The redesign appends `persistent_identity_image` and `persistent_identity_strategy` after the old
input schema. Segment 0 always retains the exact original `first_frame`; the dedicated image is
continuation-only. `single_reference` prefers the crop and falls back to the full first frame, while
`scene_plus_identity` emits both as separate image references and fails closed if the dedicated
image is absent. The implementation reports the exact reference sources/count, includes the real
count in the nine-image limit, and leaves the default `segment0_only` path and old widget/API prefix
unchanged. Automated coverage includes segment-0 invariance, crop preference, inactive-policy
ignore behavior, legacy fallback, dual-reference ordering, count reporting, and missing-crop
rejection.

The motion-rich crop-only matrix reused base seeds `2608097001/2/3` and all fixed model, sampling,
canvas, context, cold-process, and release controls. Three new crop chains completed 6/6 prompts
without OOM, retry, or cache reuse; accepted segment-0 MP4 and AV-tail hashes matched both existing
baselines. Crop-minus-legacy pooled InsightFace cosine mean/median was +0.0827160/+0.0984536, with
54/59 paired detections higher and all three seed medians positive. Crop-minus-full-scene was
+0.0557522/+0.0990135 on 47/59 frames, but seed `2608097003` regressed in mean/median. Crop-only is
therefore useful evidence, not a universally dominant strategy.

The scene-plus-crop strategy then completed another three independent two-segment chains with 6/6
successful prompts and matching segment-0 hashes. Relative to legacy, pooled mean/median was
+0.1127789/+0.1162798 and 56/59 paired frames were higher; seed medians were all positive. Relative
to the full-scene strategy, pooled mean/median was +0.0845382/+0.0945541 and 52/60 frames were
higher, again with a positive median in every seed. All per-seed absolute medians were within 0.02
of the better single-reference result. Manual three-way/four-way contacts showed active football
and arm motion without a new hard cut, freeze, or obvious face artifact. These seeds selected the
strategy, so this is a development gate rather than unbiased generalization evidence.

One independent scene-plus-crop 32-second/eight-segment chain used base seed `2608097101`. It
completed 8/8 distinct prompts once without OOM, retry, or cache reuse; manifest revision 8 contains
124/102/102/102/102/102/102/32 output frames and the final H.264/AAC streams decode to exactly
768 frames and 32.000 seconds. Segment 0 video and AV-tail hashes match both old baselines. Whole
device peak/minimum-free/Torch peak were 12473.43/3906.07/3747.61 MiB, runtime was 961.84 seconds,
and post-15-second occupancy returned to 1231.63 MiB. This is one fixed local profile, not a general
memory-safety or no-leak result.

Ten evenly spaced samples per segment gave scene-plus-crop continuation medians
0.6989496/0.6388768/0.6441070/0.6092270/0.7367208/0.6013633/0.5735967. Its last/first ratio is
0.8206553, versus 0.1991358 legacy and 0.2178784 full-scene. Pooled scene-plus-crop minus legacy
mean/median was +0.4294471/+0.4666961 on 58 paired detections (57 higher); minus full-scene was
+0.3377138/+0.3580239 on 63 (59 higher). It won all seven continuation medians versus both
baselines. Timeline review confirms materially stronger identity retention through the last segment.

The predeclared composite gate nevertheless did not pass. Scene-plus-crop/legacy temporal-MAD
ratios were 0.986/0.770/1.178/0.982/0.647/1.003/0.867 and flow-P90 ratios
1.073/0.546/0.875/0.897/0.538/0.839/0.828. The 0.70 floor therefore fails in two flow segments and
one MAD segment. Two-fps strips for the flagged segments show ongoing football, arm, and pose motion,
so there is no literal freeze, but the lower action amplitude/trajectory cannot be dismissed. Video
seam MAD median/max was 0.04459/0.07521; audio maximum half-second level gap was 3.59 dB and
descriptive NCC median 0.390. Timeline and metric-worst seam review found no obvious new hard cut.

**Decision:** the redesign clears the fixed-case mechanical, identity-depth, relative-audio, and
512 MiB VRAM screens, but fails the predeclared relative-motion floor. It remains Experimental and
default-off. Do not run or claim the three-seed 60-second matrix yet. The next gate is a genuinely
unseen multi-seed/multi-source 32-second replication plus a motion-regression investigation; only a
pass may authorize a bounded 60-second matrix. Raw evidence is in
`artifacts/long-video-generation-check/identity-anchor-crop-motion-rich-multiseed-analysis/`,
`artifacts/long-video-generation-check/identity-anchor-scene-plus-crop-motion-rich-multiseed-analysis/`,
and
`artifacts/long-video-generation-check/identity-anchor-scene-plus-crop-intermediate-8segment-seed2608097101/analysis/`.

The fixed-cadence motion-regression experiment then appended the optional
`persistent_identity_interval`, with default `1` preserving the existing every-continuation
behavior and old workflow/API prefix. Interval `2` injects on continuation segments 1/3/5/7 and
uses bounded motion/audio context alone on 2/4/6. One matched development chain completed all eight
prompts and exact 32-second media without OOM, retry, or cache use. Runtime was 825.92 seconds;
whole-device peak/minimum free/Torch peak were 12744.43/3635.07/3735.12 MiB, and post-15 occupancy
returned to 1231.63 MiB. Its identity continuation last/first ratio was 0.52509, below the
every-segment strategy's 0.82066. Relative flow-P90 fell to 0.648/0.457/0.652 in three later
continuations, including both injected and skipped segments. Fixed alternating injection therefore
does not establish a causal or repeatable release of motion constraint. Interval 1 remains the
workflow/default behavior; larger values are an Experimental research control, not a recommended
quality mode.

Two genuinely new source/prompt/base-seed chains then used interval 1, 736x416, 124/22 frames,
four-step Stock plus DynamicVRAM h2, and `unload_all_models`. Each completed 8/8 prompts once,
without OOM, retry, or cached nodes, and produced exact 768-frame/32-second A/V:

- The qipao fan-dance/drum case ran 872.95 seconds with peak/minimum-free/Torch values
  12170.76/4208.74/3726.08 MiB and post-15 1231.63 MiB. Manual timeline and motion-strip review
  retained the same woman, qipao, courtyard, and active fan choreography; the metric-worst seam did
  not show a hard cut. Face sampling was sparse during motion, but the continuation last/first
  median ratio was 0.93654. Audio maximum half-second level gap was 2.68 dB and final-minus-first
  above-8-kHz energy was -3.41 dB. The prompt requested a steady 120 BPM, while Librosa's
  half/double-aware descriptive estimate was about 104.17 BPM; strict rhythm adherence therefore
  did not pass, and no listening result is inferred from beat tracking.
- The two-woman dialogue case ran 862.41 seconds with peak/minimum-free/Torch values
  12227.57/4151.93/3738.38 MiB and the same 1231.63 MiB post-15 occupancy. Two-source InsightFace
  assignment found both people in every one of 80 sampled frames and both remain visible at the
  end. Manual review nevertheless rejects seamless framing: segment 6 to 7 jumps from full-body to
  a close two-shot. A checksum-verified `Systran/faster-whisper-small.en` CPU model recognized both
  requested phrases with best-window word error rate 0 throughout the 32 seconds, but the speech
  mostly repeats those phrases and is not natural long-form conversation. A cropped-face
  mouth-aperture/audio-envelope proxy reached only 0.042/-0.008 correlation with 85.4/27.1 percent
  track coverage. This is not SyncNet and cannot establish lip sync; human viewing/listening and a
  trained audio-visual metric remain open gates.

The current user target is typical approximately 30-second creation, so this cycle treats 32
seconds as the complete-chain gate. The arbitrary-duration and existing 60-second capability remain
implemented, but no additional 60-second rendering is required. These results support local
32-second mechanical/memory stability for the fixed profile; they do not close rhythm, framing seam,
lip-sync, blind-review, high-resolution, or cross-GPU gates.

The final regression for this checkpoint is 150 passed tests with four third-party Triton
deprecation warnings and no project failure. Ruff passes for the project and all local analysis
scripts, 21 non-artifact JSON files parse, `git diff --check` passes, isolated ComfyUI whitelist
import succeeds against `cbbc9dab1`, and stable `sampling.py` remains SHA-256
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`. Raw telemetry,
metrics, and contact sheets remain under the local excluded
`artifacts/long-video-generation-check/` tree.

## 1.5.0 accepted-state and file-composition checkpoint

Implemented locally on 2026-08-08, following the 1.4.0 P1 generation run:

- four additional Long Video Experimental nodes register after the original 18;
- candidate MP4/context/descriptor writes are atomic and do not mutate accepted state;
- acceptance verifies SHA-256, uses a same-chain manifest lock and atomic replacement,
  retains one valid prior manifest revision, and is idempotent for the same candidate;
- every continuation candidate records its accepted parent candidate ID. A stale parent,
  a gap, or a final segment followed by another segment is rejected before commit;
- intentional replacement of segment N retains invalidated history and removes N and all
  later dependent segments from the active accepted chain without deleting their files;
- composition verifies every accepted file and all absolute frame/sample boundaries, then
  re-encodes with memory bounded to one video frame plus one segment PCM buffer;
- the duration-preserving cosine bridge changes samples only at the start of the new segment,
  decays to zero over a configurable window, and never shortens the total timeline;
- seven synthetic MP4 tests cover candidate/review/accept, accepted-context loading, stale/gap/revision
  rejection, replacement invalidation, backup fallback, exact sample accounting, streaming
  composition, and the bridge invariant.

The prior real 124/102/102-frame H3 outputs were then ingested through an isolated accepted
manifest and composed with bridge disabled/enabled. Both outputs contain 328 frames with a
13.6667-second video stream and a 13.667-second AAC stream. Before the final AAC encode, the
two value jumps changed from about 0.04164/0.04124 to zero. After decoding the final AAC,
the boundary jumps changed from about 0.04226/0.03509 to 0.00434/0.00704, reductions of
about 89.7% and 79.9%. The local report is under
`artifacts/long-video-generation-check/delivery-real-check-20260808-174353/REPORT.json`.

This is evidence for lower zero-order amplitude discontinuity, not a listening test or proof
of phase/semantic continuity. H.264/AAC composition is a re-encode, not lossless concatenation.
Human-reviewed total-duration resume is now implemented in 1.6.0. Later checkpoints above add
the 14-segment run, background auto-queue/retry/release, one bound local memory profile, and its
fixed-prompt three-base-seed cold follow-up. The multi-material long-chain quality matrix,
same-seed whole-chain warm repeats, and general/cross-configuration VRAM tiers remain open.

## 1.4.0 experimental long-video continuation checkpoint

Validated on 2026-08-08 against ComfyUI `a464ac335`:

- four new nodes register after the unchanged original 14-node list, under
  `T8/MiniMax H3/Long Video/Experimental`;
- no global `PackedLayout` or `MiniMaxH3.extra_conds` monkey patch is installed;
  the long-video node clones MODEL and adds one local `extra_conds` object patch;
- state loading selects exactly segment N-1; saving keeps at most a 39-frame
  video/audio latent tail on CPU, validates tensor hashes and metadata, and uses
  same-directory atomic replacement;
- 5/22/39 frame math maps to 2/7/12 H3 video latent steps;
- direct video latent tails avoid a full previous-video IMAGE load and video-VAE
  decode/re-encode, but do not remove their current-segment Transformer rows;
- non-final segments preserve the sampled endpoint; a final exact trim disables
  the next checkpoint to prevent continuation from hidden frames;
- the long-video examples use core `CreateVideo -> SaveVideo`, not VHS
  `apad + -shortest`, so the tested MP4 audio streams match their video timelines;
- all 84 tests and Ruff passed at the original P1 checkpoint.

The initial structural probe used one-step sampling and no LoRA. A later real
four-step run used the non-pruned FL2VA INT8 model, Standard Turbo LoRA, NVFP4
H3 CLIP, both H3 VAEs, 736x416, 124-frame windows, 22-frame AV context, and
DynamicVRAM. Three direct-latent segments completed without OOM and produced
124/102/102 video frames with matching audio-stream durations. Their measured
device peaks were approximately 15,998/15,881/16,135MiB, all below the planned
512MiB free-margin gate. In a same-source A/B, one decoded last frame was clearly
worse than both 22-frame routes; video-VAE re-encoding showed no convincing
quality win over direct sampler latent and was slower. Audio sample-boundary jumps
remained near the top of local differences. At this 1.4.0 checkpoint there was no 8-16 segment
degradation run or controlled VRAM safety matrix, so no `memory_safe`, lossless,
arbitrary-length, seamless, or no-OOM claim is made.

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
  mismatch; it also resolves the Orchestrator's literal steps/shifts/sampler/scheduler output links
  in the auto-resume API instead of reporting a false `steps=None` mismatch;
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
