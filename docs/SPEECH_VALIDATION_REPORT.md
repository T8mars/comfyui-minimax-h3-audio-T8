# MiniMax H3 T8 Experimental Speech Validation Report

Validation date: 2026-08-10
Speech checkpoint: 1.10.0 reliability/long-form expansion
ComfyUI commit: `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`
Status: Experimental; this report does not establish a stable TTS, high-fidelity voice clone,
or 16GiB memory-safe tier.

## Scope

This validation covers the native ComfyUI graph implemented by this plugin:

`Voice Profile -> Speech Plan -> H3 Conditioning -> stock H3 sampling -> audio-only decode -> optional ASR/speaker QA -> Finalize`

The speech generator reuses the connected MiniMax H3 MODEL, Qwen3-VL CLIP, video VAE, and
audio VAE. No second H3 or text-encoder loader exists inside the speech nodes. The tests below
use the joint AV model on a 32x32 dark video canvas and retain only decoded audio.

## Controlled environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti 16GiB |
| OS / Python | Windows / Python 3.12.10 |
| PyTorch | 2.10.0+cu130 |
| ComfyUI | 0.31.0 at `cbbc9dab1` |
| H3 described/dialogue checkpoint | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| H3 reference checkpoint | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| VAEs | `minimax_h3_video_vae_fp16.safetensors`; `minimax_h3_audio_vae_fp32.safetensors` |
| Sampling | 20 steps, `res_multistep`, `simple`, video/audio shift 12/3 |
| Canvas / window | 32x32, 10.0s requested, 243 aligned H3 frames |
| Server | isolated `127.0.0.1:8195`, DynamicVRAM, cache-none, only this plugin whitelisted |

No Turbo LoRA was used. A video Turbo result must not be extrapolated to speech accuracy or
speaker identity without a separate controlled A/B.

Optional QA models are installed as physical directories under the standard ComfyUI model tree;
they remain optional plugin dependencies and are not distributable project assets:

- [faster-whisper small.en](https://huggingface.co/Systran/faster-whisper-small.en), revision
  `d1d751a5f8271d482d14ca55d9e2deeebbae577f`, runtime directory
  `F:\AI-T8-video-onekey\ComfyUI\models\TTS\faster-whisper-small.en-d1d751a5`;
- [faster-whisper multilingual small](https://huggingface.co/Systran/faster-whisper-small), revision
  `536b0662742c02347bc0e980a01041f333bce120`, runtime directory
  `F:\AI-T8-video-onekey\ComfyUI\models\TTS\faster-whisper-small-multilingual-536b0662`;
- [Microsoft WavLM Base Plus Speaker Verification](https://huggingface.co/microsoft/wavlm-base-plus-sv), revision
  `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, runtime directory
  `F:\AI-T8-video-onekey\ComfyUI\models\TTS\wavlm-base-plus-sv-feb593a6`.

Both runtime directories were copied with SHA-256 manifest comparison and verified not to be
reparse points. The original `G:\CodexModels` copies remain only as preserved backups; runtime
resolution no longer depends on them. No model file is inside this repository or Git.

## Results

### Described voice

- Target: `The lantern is still burning. We can begin when you're ready.`
- Output duration: 10.125s, 32kHz stereo.
- faster-whisper small.en returned the exact 11/11 normalized word sequence; WER 0.
- A same-seed repeat produced identical decoded PCM. The FLAC file bytes differed because of
  container metadata, so determinism is claimed only for the compared PCM under this exact
  environment.
- Representative pre-final-limiter file:
  `described_stock20_00002.flac`, SHA-256
  `AB166EBE62C2AE56D7803314DD8B4E0D05F99DC76458CB5FD6DDF85FFFE0F46F`.

This proves one exact English generation and same-environment repeat, not general pronunciation,
multilingual, acting, or cross-device determinism.

### Reference voice and exact-target alignment

- Reference: a locally generated licensed test anchor; 10.125s, 32kHz stereo.
- Target: `The door is open now. I'll wait beside the window.`
- Raw Ref2VA output duration: 10.125s.
- Raw ASR contained approximately 4.3s of unrelated lead-in, followed by the complete target.
  Therefore a bare Ref2VA completion cannot be assumed to contain only the requested line.
- `trim_exact_target` located the complete normalized target token sequence, applied fixed
  pre/post padding and edge fades, then re-transcribed the result.
- Aligned output duration: 4.465s; final normalized target sequence exact; similarity 1.0.
- Same-seed aligned PCM repeated identically.
- Representative aligned file: `reference_clone_stock20_00003.flac`, SHA-256
  `6EC0CAE433270D96C6FA05DA9998223FD53ACFD07C44552112D43B9436BF47BA`.

The node refuses fuzzy timing: if the complete ordered target is absent, no exact-target trim is
applied. ASR text success still does not prove speaker identity.

### Preliminary speaker signal and negative control

WavLMForXVector cosine was computed on CPU after ASR alignment:

| Pair | Cosine |
|---|---:|
| Reference anchor vs aligned reference-conditioned generation | 0.949587 |
| Same reference anchor vs deliberately different older male described-voice control | 0.484272 |
| Positive-minus-negative gap | 0.465315 |

The male control had the same target text and exact ASR similarity 1.0, so the comparison separates
text correctness from one preliminary identity signal. The control file is
`impostor_control_stock20_00001.flac`, SHA-256
`ACC1D347F3AECD1C6D720D575C09560F2D5A6C76E228DECD21243E79168099FA`.

One positive and one negative are insufficient to calibrate a decision threshold. No claim of
high-fidelity or exact voice cloning is allowed until a licensed multi-speaker set, impostor
distribution, and blind ABX test pass.

### Independent two-speaker dialogue

- Two described profiles were rendered independently with different seeds.
- Each turn used exact-target ASR alignment before deterministic sample-timeline assembly.
- Final duration: 9.81s, 32kHz stereo.
- Combined target: `you came back before the rain I promised I would find the road home`.
- Combined normalized transcript was exact.
- WavLM cosine between ASR-located turn spans: 0.247203, a useful separation signal for this pair.
- Output: `dialogue_two_speaker_stock20_00001.flac`, SHA-256
  `E82C5FA833D5AC5306A0782D8937DE28FF58249AC2DE67A176F1CB525AA2486C`.

This is evidence for the independent-turn mechanical path. It does not validate joint
multi-speaker generation, long-term character identity, overlapping-dialogue naturalness, or
speaker attribution across a population.

### Final attenuation-only peak protection

The first described output contained four samples at digital full scale. After the final limiter
was added, the exact saved FLAC was passed through the current `MiniMaxH3SpeechVerifyT8` node on a
fresh isolated ComfyUI runtime with ASR and speaker QA disabled:

| Signal | Peak | Samples >= 0.999 | Frames |
|---|---:|---:|---:|
| Input | 0.000000dBFS | 4 | 324000 |
| Output | -0.999854dBFS | 0 | 324000 |

The node completed in 0.625s and preserved sample count and sample rate. Unit tests also verify
that the limiter never boosts an already quiet signal.

## 1.10 reliability and creator-workflow expansion

### Upstream failure guard

`Speech Studio` now arms a prompt-lifecycle guard before Conditioning. Normal Finalize completion
disarms it. If the prompt ends first, the guard requests the configured release policy through the
same Comfy queue mechanism and writes a recovery event under
`output/minimax_h3_t8/speech_recovery`.

A real Stock20 prompt completed sampling and was then forced to fail in ASR before Finalize. Prompt
`f02cc595-8c46-4101-8df7-d5efff1c3c7c` produced an
`abnormal_prompt_end_before_finalize` recovery event with `unload_all_models` and
`release_requested=true`; after 15 seconds the isolated device reported about 15.9GiB free and a
32MiB torch pool. A successful Chinese prompt did not create an abnormal event. Current ComfyUI
also globally unloads on recognized CUDA OOM. These observations close the tested non-OOM/cancel
gap; they do not guarantee callback execution after process termination, driver reset, power loss,
or on untested ComfyUI releases.

### Multilingual text metrics

A pinned multilingual faster-whisper small CTranslate2 directory is installed at
`ComfyUI/models/TTS/faster-whisper-small-multilingual-536b0662`. The new validation tool reports WER
for space-delimited languages and raw Unicode CER for CJK. It deliberately does not normalize
simplified and traditional Chinese into the same character.

One Chinese Stock20 sentence completed. Expected `夜色很安静，我们现在可以开始了。`; transcript
`夜色很安靜,我们现在可以开始了`; raw CER 1/14 = 7.14%. This is a useful plumbing probe but fails the
pre-registered minimum of 30 H3 samples per language. No stable Chinese or multilingual claim is
allowed.

### Ten-speaker identity distribution and ABX package

Ten distinct speakers from the licensed LibriSpeech test-clean split each produced one Ref2VA
Stock20 target sentence. WavLM evaluation generated 10 genuine and 90 impostor scores:

- genuine range: 0.91315-0.98267;
- impostor 95th percentile: 0.885771;
- impostor maximum: 0.897372;
- 10/10 genuine scores exceeded the impostor 95th-percentile threshold.

A randomized 30-file blind ABX package, listener sheet, and separate answer key were created. No
human listener has completed it, and there is only one generated sentence per speaker. Therefore
this is population-level machine evidence, not a high-fidelity-clone claim.

### Prompt-level performance controls and deterministic ADR

A same-seed seven-case Stock20 matrix compared slow/natural/fast pace, low/natural/high pitch and
low/natural/high energy. Every generation succeeded, but all monotonic gates failed:

| Control | Ordered measurements | Result |
|---|---|---|
| pace words/s | 2.0258 / 2.0110 / 2.0110 | failed |
| median F0 Hz | 105.263 / 100.629 / 100.000 | failed |
| RMS dBFS | -17.498 / -17.577 / -17.833 | failed |

Emotion/intensity has no independent perceptual classifier or human rating matrix. Pace, pitch,
energy and intensity remain `uncalibrated_prompt_direction`; they are not numeric acoustic controls.

The separate ADR node provides deterministic operations: exact-sample refuse/pad-trim/bounded
phase-vocoder fit and optional pitch shift. Unit and real workflow probes reached zero output-sample
error and rejected rates outside the explicit range. Sample-exact duration is not phoneme alignment
or lip-sync evidence.

The explicit persistent voice path also completed a live isolated-server Save -> Load -> recoverable
Delete cycle. Save/Load are output nodes so standalone maintenance workflows execute; same-name
replacement remains opt-in, and Delete moved the entry to local trash instead of erasing it.

### Durable long-form state and completed-chunk preview

The long-form path atomically writes a CPU safetensors chunk and SHA-256 before advancing its
manifest, also writes a playable FLAC, and returns that FLAC to the node UI. The Start/Resume node
uses the durable manifest hash as its ComfyUI cache fingerprint, so requeueing the same workflow
advances rather than replaying a cached segment.

- A real 32-second, four-segment Stock20 chain composed to exactly 1,024,000 samples at 32kHz. Full
  ASR WER was 21.43%, so exact duration did not pass the strict long-text accuracy gate.
- A separate four-segment workflow was queued four times. Accepted count advanced 1/4, 2/4, 3/4,
  4/4 with no stale cache; every per-segment text similarity was 1.0. Final crossfaded output was
  513,280 samples (16.04s), matching its accepted timeline.
- Synthetic state matrices covered 32 seconds (8 chunks), 2 minutes (30 chunks), and 10 minutes
  (150 chunks). Every accepted chunk survived an in-memory session discard, SHA verification and
  restart-style resume; final sample counts were exact and cooperative cancel/clear worked.

The 2/10-minute matrix validates persistence mechanics only. Real H3 voice continuity, spectral
drift and identity over those durations remain untested. `request_cancel` is checked between
segments; use ComfyUI Stop for the currently running sampler. The UI preview is completed-chunk
preview, not token/frame realtime streaming.

### Joint multi-speaker denial

Two 2-speaker Joint Ref2VA Stock20 probes completed mechanically. Both generated the requested
sentences plus substantial unrequested speech. WER was 2.25 before prompt tightening and 2.375
after tightening. Speaker attribution was therefore not evaluated further. Joint remains an
explicitly denied EXP path; independent-turn generation and exact-sample assembly remain the
recommended dialogue design.

## VRAM and release observations

| Probe | Runtime | Whole-device peak |
|---|---:|---:|
| Described stock20 | 50.938s | 16262.2MiB |
| Described stock20 + global unload | 39.938s | 16282.1MiB |
| Reference stock20 raw | 49.844s | 16271.4MiB |
| Reference + ASR alignment | 65.515s | 16281.3MiB |
| Reference + ASR + speaker report | 74.109s | 16263.7MiB |
| Different-speaker control | 43.313s | 16289.6MiB |
| Two independent dialogue turns | 87.781s | 16315.7MiB |

The GPU reports 16379.5MiB total. Depending on the run, observed whole-device peak headroom was
only about 64-118MiB. Desktop processes contribute to whole-device occupancy, but that does not
make the user-visible OOM risk disappear. The pre-registered 512MiB safety gate fails, so this
configuration must not be labelled `memory_safe` or never-OOM.

With explicit `unload_all_models`, the isolated ComfyUI process's reported torch pool returned to
32-64MiB within 15s after successful completion. `clear_execution_cache` did not prove H3-only
weight unloading. `unload_all_models` is global and may evict every ComfyUI model.

Three separate cold processes succeeded with peaks 16284.6/16299.3/16284.2MiB and only
80.7-95.8MiB free. Three same-process `keep_loaded` runs also succeeded; peak range was 78.56MiB,
but minimum free headroom reached about 16.72MiB and the baseline residency increased by about
15143MiB from first to last. This is a material residency staircase even though the peak ceiling
did not stair-step. `keep_loaded` is therefore not a safe 16GiB default.

The lifecycle guard now covers the tested upstream non-OOM/cancel exit before Finalize. Recognized
OOM cleanup still relies on current ComfyUI's global OOM handler. Cross-GPU behavior remains
unvalidated.

## Mechanical verification

- Full project suite: `204 passed` (four upstream Triton deprecation warnings only).
- Ruff: all checks passed.
- ComfyUI CPU whitelist import: passed.
- Live isolated `/object_info`: all 22 speech nodes found; Studio category/eight outputs and new
  guard, preflight, persistence, ADR, long-form and Joint schemas loaded.
- Twelve speech ComfyUI 0.4 frontend workflows and their API prompts were generated; placeholder
  references remain intentionally unresolved until users supply licensed audio.

Raw WebSocket events, 0.1s telemetry samples, server logs, and probe prompts remain under the
gitignored `artifacts/speech-generation-check` directory. Generated audio remains under the normal
ComfyUI `output/MiniMaxH3_T8_Speech` directory and is not part of the repository.

## Denied claims and next gates

The following remain unvalidated or explicitly denied:

- 30 H3 samples per language for Chinese and other multilingual WER/CER;
- at least three human listeners completing ABX, with multiple generated sentences per speaker;
- calibrated emotion/intensity/rate/pitch controls (the first acoustic monotonic matrix failed);
- stable Joint multi-speaker generation and overlapping-dialogue attribution (current probe denied);
- real H3 2min/10min continuity, identity and spectrum drift;
- active-sampler background hard cancellation and token/frame realtime streaming;
- ADR phoneme alignment and visual lip synchronization;
- persistent-library cross-process/network-share/privacy-lifecycle stress;
- cross-GPU/high-resolution profiles and three-cold/three-warm repeats on another GPU;
- a 16GiB memory-safe tier.

These gates remain tracked in `roadmap.md`. Reference-voice use requires actual consent or another
lawful right; a UI checkbox cannot create that right.
