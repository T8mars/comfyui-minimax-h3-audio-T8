# MiniMax H3 Audio T8

[简体中文](README.md) | English

A ComfyUI node pack for MiniMax H3 video and audio generation, with reference control, long-video workflows, face refinement, and acceleration tools.

Current version: **1.65.0** · GPL-3.0-or-later

## Features

- Text-to-video, image-to-video, first/last-frame, image/video/audio references, and mixed references
- Independent video/audio dual-clock sampling
- Source-audio preservation, references, and mixing; optional Audio Refine for Turbo4/8, two-pass, PDD, EAV, Prompt Relay, and 8-step long video
- Multi-keyframes, long-video continuation, one-queue serial generation, resume support, and latent upscaling; the optional v8 subject-safe RGB post-process takes T2 only inside a reviewed per-frame alpha and preserves D0 elsewhere, including D0 audio
- Single- and multi-person face refinement, SAM3.1 tracking, and Skin Finish
- Prompt Relay, SPEED, SLA, PDD, and Enhance-A-Video integrations
- Fully local MV Vocal Lock V3 with the official Ref2V Turbo4 recipe, isolated-vocal drive, per-scene visual contracts, serial rendering, resume, and final one-time original-song mux
- FastH3 Preview: T2VA four-step inference with optional real learned-gate 90% VSA execution
- NVIDIA H3 Super Acceleration: H3 4-step draft, full LTX VAE encode, and a 3-step LTX-2.5 refiner; TAEHV is final-decode only
- FlashVSR v1.1: 2x/4x decoded-video restoration with fixed LCSA, an opt-in dynamic budget, memory-safe tiling, and untouched audio
- RAFT motion audits, trajectory control, RealBasicVSR, FreeNoise, an AYS calibration contract, and CADS visual-reference annealing

Nodes marked `Advanced` or `EXP` are advanced or experimental. Start with their bundled workflows.

## Installation

Search for `MiniMax H3 Audio T8` in ComfyUI Manager, install it, and restart ComfyUI.

> **Update ComfyUI first:** this node pack uses the recent native MiniMax H3 core, `comfy_api.latest`, model-patching, and weight-adapter APIs. Updating only this node pack while keeping an older ComfyUI build may make every T8 node turn red or appear missing. Update the ComfyUI core, frontend, and Manager together, then fully exit and restart ComfyUI.

Manual installation:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

## Quick Start

1. Open [`examples/workflows`](examples/workflows).
2. New users should start with `01-basic-generation` or `02-audio-control`.
3. Drag a JSON workflow into ComfyUI, replace the models and media, then run it.
4. For advanced workflows, read the folder's `README.md` and the NOTE nodes on the canvas.

For one-queue serial long-video generation, use an `In_Node_Long_Video_Loop` workflow from `04-long-video`. Keep using the Background/Accepted workflows when you want to review each segment manually.

For long video with both Prompt Relay and Enhance-A-Video, use `In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced` from the same folder. It uses the native 20-step route and does not use a Turbo LoRA.

For tail subdivision or a separate low-sigma second pass, use the long-video workflow containing `Long_Video_Sampling_Plan`. Disconnecting that node restores the old route.

[Browse all workflow categories](examples/workflows/README.md)

## SLA Precision V2 quality-correction route

Use [`15-sla-attention/2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json`](examples/workflows/15-sla-attention/2026-09-02_H3_SLA_Precision_V2_FL2VA_FP8_8Step_Advanced_EXP.json) for the new SLA quality-correction path. It appends a dynamic model-only SLA LoRA loader, Precision V2 Attention, and a fail-closed post-sampler Runtime Audit. Every old SLA node and workflow remains unchanged.

The recorded problem was not a seed issue. The historical route used pooled-BF16 routing, quantized `spas-sage-attn` Sage2 Q/K, 128x64 tiles, model-invocation step counting, coarse prefix protection, and an all-sparse exact route. Precision V2 pins the PlagueKind v1.4.3 implementation at commit `066ada9`: FP32 routing, a direct Triton sparse kernel with FP32 online softmax, sigma-derived logical steps, exact language/audio protection, and dense first and last steps. The SLA LoRA is applied as a dynamic residual over the FP8 base, without merging and re-quantizing base weights.

The workflow fixes 736x416x124, eight NFE, shifts 12/3, 32x32 blocks, requested 90% sparsity, dense steps 0/7, and sparse steps 1-6. A real audit observed exactly 50 dense H3 calls on each boundary step and 50 sparse calls on each middle step (300 total), 20 protected blocks, and no kernel fallback. The decoded video and audio are byte-identical to the same-seed render made before per-step observability was added. Against the same-input, same-seed dense XFormers control, this one environment measured about 12.38% lower end-to-end time and 18.75% lower sampler time.

Both dialogue clips strictly decode. Local ASR recovered the intended Mandarin sentence with an equivalent 呀/啊 substitution; official SyncNet measured -1 frame at 25fps for both, while a fixed 400ms delayed-video control measured +9 frames. A 32-frame review found no old post-one-second face/motion collapse. The user then watched the anonymous pair at normal speed and judged that both were about the same and acceptable. After reveal, A was Dense and B was Precision V2, so this material passes the perceptual non-inferiority gate without establishing a general Precision V2 quality advantage. Minimum free VRAM was 236MiB on the latest Precision V2 run (211MiB previously) and 245MiB for dense, below this project's 512MiB gate. The route therefore remains Advanced EXP and is not advertised as universally safe on 16GB GPUs.

## Fully Local MV / Lip Scenes

Prefer the `VocalLock_V3_Official_Ref2V_Turbo4` workflow under [`24-mv-lipsync`](examples/workflows/24-mv-lipsync). Load a performer reference image, the complete `full_song`, and a timeline-aligned isolated vocal or clear-dialogue `vocal_lock_audio`. V3 reuses the local V2 scene planner; only the isolated track enters each local H3 `lock_source` window, while the complete song stays out of H3 and segment candidates and is muxed once after video assembly.

V3 emits the official six-section Ref2VA structure with `<Subject 1>`, `<Picture 1>`, and `<Audio 1>: fully_copy`, adds one-person/one-face and per-scene visual contracts, and forces a front or three-quarter medium close-up with an unobstructed mouth for vocal scenes. The old workflows remain for compatibility but do not replace an isolated-vocal lip-sync test. The route calls no remote H3, LLM, TTS, separation, or video API and never submits HTTP `/prompt` from inside the node.

An early 5.152-second clear-English-speech sample passed SyncNet and normal-speed user lip-sync review, but showed softness around the performer. A later same-image, same-audio, same-failing-seed comparison showed that the main fault was not the seed or H3 base model: a generic LarryVrh EMA Turbo LoRA and a non-official eight-step/shift-6:3 schedule had been applied to Ref2VA. The recommended route now pins the official Ref2V Turbo v0.1 LoRA at strength 1.0, four steps, Euler/simple, shifts 12/3, and 1024x768. The same seed no longer shows persistent double-face ghosting.

The official Ref2V configuration has now produced an accepted 32-second/five-shot film: 5/5 scenes, 768/768 frames, 1024x768 at 24fps, with the complete song muxed once. Strict video, audio, and combined decoding pass, and default multithreaded video decode reports zero anomalies in 20 repeats. Scene sampling found no duplicate face, background face, or persistent subject-edge ghosting. Official SyncNet measured `0/-1/0/-1/0` frames across the five isolated-vocal scenes; a 400ms delayed-video control measured nine frames. After complete viewing, the user reported that the 32-second result had no problem and was perfect, and explicitly removed the approximately 90-second requirement. The acceptance is bound to master SHA-256 `e833277844e6980fdeacf9bdfd5c61ffe48aefdb3e1eba6869c363777b7dd75f`.

## Model Folders

| Model | Folder |
| --- | --- |
| H3 diffusion model | `models/diffusion_models` |
| Text encoder | `models/text_encoders` |
| Video/audio VAE | `models/vae` |
| Turbo, SLA, PDD, and other LoRAs | `models/loras` |
| Latent upscaler | `models/latent_upscale_models` |
| TAEHV Wide for H3 Super | `models/taehv` |
| H3 Fun Control (new Model Patch / old ControlNet) | `models/model_patches` / `models/controlnet` |
| TAEH3 preview model | `models/vae_approx` |
| RAFT optical-flow weights | `models/optical_flow` |
| RealBasicVSR weights | `models/upscale_models` |
| Complete FlashVSR v1.1 folder | `models/FlashVSR-v1.1` |

Do not mix FL2VA, Ref2VA, pruned, and full base-model variants.

## Common Settings

- Common Turbo dual-clock shifts: video `12`, audio `3`
- Preserve source audio: set `audio_mode=lock_source` and connect `mux_audio` to the save node
- Media tags: `<Picture 1>`, `<Video 1>`, and `<Audio 1>` must match the connected inputs
- Use width and height values divisible by 32; reduce resolution, frame count, and reference count first when VRAM is insufficient
- `1920×1088` is a risk-reference area, not an execution limit; larger canvases warn but are not blocked, and the user accepts the OOM and quality risk
- Do not stack multiple nodes that each take ownership of the sampler, attention, or MODEL forward path

## PDD 8-Step

Workflows are under [`19-pdd-acceleration`](examples/workflows/19-pdd-acceleration). PDD requires its dedicated setup node and must not be loaded as a normal LoRA.

Download the converted FL2VA/Ref2VA PDD acceleration LoRAs from [t8star/MiniMax-H3-Acc-8Step-comfy](https://huggingface.co/t8star/MiniMax-H3-Acc-8Step-comfy), place them in `ComfyUI/models/loras`, and select the variant matching your base model.

Defaults: Euler/simple, 8 NFE, shifts `12 / 3`, CFG 1.

Recent ComfyUI builds automatically use the official PDD FinalLayer; older builds keep the project's compatibility fallback. Selection is capability-based and never blocked by a ComfyUI version, model hash, or file size.

The folder also includes learned-latent two-pass workflows for FL2VA and Ref2VA. They split the same PDD trajectory into LOW 4 steps and HIGH 4 steps, so total NFE remains 8. The validated Ref2VA preset uses 864×480×22 with 1.5× upscale; the FL2VA two-pass route remains experimental.

## FastH3 VSA 4-Step

Use the `FastH3_VSA_T2VA_4Step` workflow under [`10-speed`](examples/workflows/10-speed). Download the official
[VSA Data-Free adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/blob/main/vsa-datafree/adapter_model.safetensors) to
`models/loras/FastH3-VSA/vsa-datafree/adapter_model.safetensors`.

This route is limited to plain T2VA, four NFE, and shifts `12 / 3`. Real VSA additionally requires a Comfy Kitchen build exposing `topk_ratio`, `block_len`, and `coarse_gate`. If the kernel or all 50 learned gates are unavailable, the node reports the reason and falls back to valid dense four-step inference; it never labels dense attention as VSA. See [`10-speed/README.md`](examples/workflows/10-speed/README.md) for build details.

## Official Core Compatibility

[`20-core-compatibility`](examples/workflows/20-core-compatibility) provides optional AV-latent, H3 Attention Hook, and per-step host-sync compatibility nodes. H3 Audio VAE encoding now disables the legacy aligned-length tail crop, preserving the final latent step for non-aligned audio; recent cores and non-H3 VAEs are unchanged. The tiled-VAE global-coordinate proposal remains a bypass-by-default audit. Existing workflows do not need changes.

## NVIDIA H3 Super Acceleration

[`22-sol-engine-h3-super`](examples/workflows/22-sol-engine-h3-super) implements NVIDIA's two-stage route: an H3 4-step draft is encoded by the full LTX-2.5 Video VAE, enlarged with the official x2 LTX latent upscaler, refined by LTX-2.5 for three Euler updates, and finally decoded by TAEHV Wide. TAEHV Encode must not feed the refiner. H3 audio bypasses Stage 2 and is muxed back unchanged.

The same folder also includes an experimental low-sigma identity-preserving Stage 2 using `0.5 → 0.412 → 0.350 → 0` (three Euler updates). It defaults to Dense Attention so the sigma schedule is the only changed variable. Use it for A/B review when the official full-denoise route changes faces too much; it is not NVIDIA's parity schedule and is not a universal identity guarantee.

Download the complete model bundle from [t8star/Minimax-H3-Super-Acceleration-Comfy](https://huggingface.co/t8star/Minimax-H3-Super-Acceleration-Comfy). Preserve its folder structure and copy the folders into `ComfyUI/models`. Exact filenames and paths are listed in the [`22-sol-engine-h3-super`](examples/workflows/22-sol-engine-h3-super) README. Sol-Attn is optional; the route falls back to dense attention when it is unavailable.

## FlashVSR Video Restoration

Workflows are under [`23-flashvsr`](examples/workflows/23-flashvsr). Download the official [FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) folder into `ComfyUI/models/FlashVSR-v1.1`. The model repository does not include `posi_prompt.pth`; copy it from the [official FlashVSR repository](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor) into the same folder. Then install a `spas_sage_attn` build compatible with the active Torch/CUDA runtime from [SpargeAttn](https://github.com/thu-ml/SpargeAttn) or the [Windows wheel releases](https://github.com/woct0rdho/SpargeAttn/releases).

Start with `Quality Locked`, which keeps the fixed `2.0 / 3.0 / 11` LCSA budget. `Balanced Dynamic` changes low-motion chunk budgets and requires visual review. `Memory Safe` trades speed for spatial tiling and staged offload. Official FlashVSR primarily targets 4x restoration; the included 2x route remains conservative EXP use. The nodes impose no model-hash, file-size, or pixel ceiling, and return the original audio object unchanged.

## Community Creation Tools

[`21-community-advanced`](examples/workflows/21-community-advanced) contains Fun Control, long-video character/voice and sentence-boundary planning, seam-drift auditing, residency policies, Creator semantic-cache planning, native TAEH3 preview inspection, and read-only diagnostics. Fun Control supports both the recent official `MODEL_PATCH` contract and the old ControlNet contract: place recent weights in `models/model_patches` and old weights in `models/controlnet`. Runtime capability, not a version string or model hash, selects the route. Models are available from [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet). Download `taeh3.safetensors` from [madebyollin/taehv](https://github.com/madebyollin/taehv) into `models/vae_approx`.

For a Qwen reference-prefix cache plus the separately installed [T8 BlockCache](https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8), use the Ref2VA Stock20 template under [`12-system-memory`](examples/workflows/12-system-memory). It is a performance-first EXP route, not a bit-exact, VRAM-saving, or universal 16GB-safety claim.

## Paper-Inspired Experiments

- [`07-motion-detail`](examples/workflows/07-motion-detail): RAFT motion audit/mask propagation, trajectory control, RealBasicVSR temporal restoration, and the H3 dual-clock AYS calibration contract.
- [`04-long-video`](examples/workflows/04-long-video): FreeNoise video-initialization rescheduling for the standard or Prompt Relay/EAV in-node loop.
- [`03-image-video-edit`](examples/workflows/03-image-video-edit): CADS visual-reference annealing; audio conditions remain unchanged.

There is no official H3-optimized AYS schedule to copy, so the default remains native flow. FreeInit and PAG do not yet have a validated H3 joint-AV mathematical/attention contract and were not implemented under misleading names.

## Troubleshooting

- **After an August 22 or later update, every T8 node turns red or appears missing:** the extension failed during startup; this is not a model-file or workflow-parameter problem. Update the ComfyUI core, frontend, and Manager together, fully exit ComfyUI, and restart it.
- **Use the first startup error to identify the cause:** missing `comfy_api.latest`, `comfy.weight_adapter`, `comfy.patcher_extension`, or `comfy.ldm.minimax` means that the ComfyUI core is too old. Missing `torch`, `torchaudio`, `numpy`, `safetensors`, or `PIL` (package name: `Pillow`) means that the Python environment used by ComfyUI is incomplete.
- **Repairing dependencies:** an empty `requirements.txt` in this project is intentional; the base packages come from ComfyUI. Reinstall **ComfyUI core's** `requirements.txt` with the exact Python executable that starts ComfyUI; users of bundled distributions should prefer that bundle's updater. Do not install into the system Python, and do not blindly add optional SLA, Transformers, or OpenCV packages for the basic nodes, because that can replace the working Torch/CUDA stack. When reporting the issue, include the first complete `IMPORT FAILED` / `ModuleNotFoundError`, the ComfyUI version, and this node-pack version.
- **Shifted parameters or NaN widgets:** fully restart ComfyUI, then reload the workflow.
- **Media-tag error:** check the connected media and tag numbering.
- **Prompt Relay tokenizer error:** use the native `Load CLIP` node with `type=minimax`. Since 1.52.3, wrapped CLIP objects with hidden internal tokenizers are supported only when every token matches native H3 tokenization.
- **Source audio is missing:** use `lock_source` and connect `mux_audio`.
- **OOM:** disable concurrency, lower resolution/frame count, then disable advanced nodes one by one to isolate the cause.

## Documentation and Support

- [Workflow index](examples/workflows/README.md)
- [Full usage guide](docs/README_ComfyUI.md)
- [Validation results and limitations](docs/VERIFICATION_REPORT.md)
- [Issue tracker](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues)
- [Third-party projects and licenses](THIRD_PARTY_NOTICES.md)

## Links

- [Bilibili](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI Apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI Package](https://pan.quark.cn/s/264edb7e36bd)
- [Model Storage](https://pan.quark.cn/s/c9c267081fbf)
- [Hugging Face](https://huggingface.co/t8star)

Models and third-party components remain subject to their own licenses. Users are responsible for the rights to any people, voices, and reference media they use.
