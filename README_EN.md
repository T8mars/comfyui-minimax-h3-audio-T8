# MiniMax H3 Audio T8

[简体中文](README.md) | English

A ComfyUI node pack for MiniMax H3 video and audio generation, with reference control, long-video workflows, face refinement, and acceleration tools.

Current version: **1.53.0** · GPL-3.0-or-later

## Features

- Text-to-video, image-to-video, first/last-frame, image/video/audio references, and mixed references
- Independent video/audio dual-clock sampling
- Source-audio preservation, audio references, track mixing, and low-step Audio Refine
- Multi-keyframes, long-video continuation, one-queue serial generation, resume support, and latent upscaling
- Single- and multi-person face refinement, SAM3.1 tracking, and Skin Finish
- Prompt Relay, SPEED, SLA, PDD, and Enhance-A-Video integrations

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

[Browse all workflow categories](examples/workflows/README.md)

## Model Folders

| Model | Folder |
| --- | --- |
| H3 diffusion model | `models/diffusion_models` |
| Text encoder | `models/text_encoders` |
| Video/audio VAE | `models/vae` |
| Turbo, SLA, PDD, and other LoRAs | `models/loras` |
| Latent upscaler | `models/latent_upscale_models` |
| H3 Fun ControlNet | `models/controlnet` |

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

The folder also includes learned-latent two-pass workflows for FL2VA and Ref2VA. They split the same PDD trajectory into LOW 4 steps and HIGH 4 steps, so total NFE remains 8. The validated Ref2VA preset uses 864×480×22 with 1.5× upscale; the FL2VA two-pass route remains experimental.

## Official Core Compatibility

[`20-core-compatibility`](examples/workflows/20-core-compatibility) provides optional AV-latent, H3 Attention Hook, and per-step host-sync compatibility nodes. The tiled-VAE global-coordinate proposal produced stronger grid artifacts in the current fp16 VAE validation, so it remains a report-only, bypass-by-default audit and is not recommended as a fix. Existing workflows do not need changes.

## Community Creation Tools

[`21-community-advanced`](examples/workflows/21-community-advanced) contains Fun Control, long-video character/voice and sentence-boundary planning, seam-drift auditing, residency policies, Creator semantic-cache planning, and read-only diagnostics. Download the Fun Control model from [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/controlnet) and place it in `models/controlnet`. The other nodes are report/plan-only by default: they do not delete cache files, unload models, or enable the still-draft official Generic Loops backend.

## Troubleshooting

- **All T8 nodes turn red or appear missing after an update:** this usually means that the extension failed during startup, not that a workflow parameter or model file is wrong. Update the ComfyUI core, frontend, and Manager together, fully exit ComfyUI, and restart it. If the problem remains, find the first `IMPORT FAILED` / `ModuleNotFoundError` in the startup console. Missing `comfy_api.latest`, `comfy.weight_adapter`, `comfy.patcher_extension`, or `comfy.ldm.minimax` means that the ComfyUI core is too old; missing `torch`, `torchaudio`, `safetensors`, or `PIL` means that the active ComfyUI Python environment is incomplete. The basic nodes require no extra pip packages, so do not install optional packages blindly. Include the first complete import traceback, ComfyUI version, and node-pack version when reporting the problem.
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
