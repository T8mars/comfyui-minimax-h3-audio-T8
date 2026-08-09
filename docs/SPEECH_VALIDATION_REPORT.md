# MiniMax H3 T8 Experimental Speech Validation Report

Validation date: 2026-08-10
Speech checkpoint: 1.8.0; included in plugin release 1.9.0
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

The Finalize node is downstream. If sampling raises or OOMs before Finalize, ComfyUI aborts the
graph and the release request does not run. This version therefore has no finally-level global
release guarantee for upstream failures.

## Mechanical verification

- Full project suite: `173 passed`.
- Ruff: all checks passed.
- ComfyUI CPU whitelist import: passed.
- Live isolated `/object_info`: all 10 speech nodes found; Studio category and eight outputs
  correct; `peak_limit_dbfs` present.
- Three API examples and three ComfyUI 0.4 frontend workflows parse and are covered by structural
  tests; the frontend files also passed live isolated `/object_info` input/output type validation.

Raw WebSocket events, 0.1s telemetry samples, server logs, and probe prompts remain under the
gitignored `artifacts/speech-generation-check` directory. Generated audio remains under the normal
ComfyUI `output/MiniMaxH3_T8_Speech` directory and is not part of the repository.

## Denied claims and next gates

The following remain unvalidated or explicitly denied:

- Chinese and other multilingual WER/CER;
- at least 10 licensed speakers, impostor calibration, and blind ABX;
- calibrated emotion/intensity/rate/pitch controls;
- joint multi-speaker generation and stable overlapping dialogue;
- 32s/2min/10min continuity, accepted-state resume, cancellation, and crash recovery;
- ADR exact-duration fitting and real-time streaming;
- three cold plus three warm runs and a continuous three-job staircase test;
- cross-GPU/high-resolution profiles;
- a 16GiB memory-safe tier.

These gates remain tracked in `roadmap.md`. Reference-voice use requires actual consent or another
lawful right; a UI checkbox cannot create that right.
