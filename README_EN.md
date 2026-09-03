# MiniMax H3 Audio T8

[简体中文](README.md) | English

MiniMax H3 Audio T8 is a ComfyUI node pack for joint video and audio generation. It includes practical workflows for text and image animation, first/last-frame control, image/video/audio references, long video, lip sync, acceleration, and final-video restoration.

Current version: **1.69.0** · 284 nodes · GPL-3.0-or-later

## Where to start

If this is your first time using the pack:

1. Run a basic H3 workflow from [`examples/workflows/01-basic-generation`](examples/workflows/01-basic-generation).
2. For image or audio control, use the workflows under [`02-audio-control`](examples/workflows/02-audio-control) and the matching reference folders.
3. For OpenVDN eight-step generation, use an `OpenVDN_DMD8_*_Advanced.json` workflow from [`10-speed`](examples/workflows/10-speed).
4. For long video or music video work, start with [`04-long-video`](examples/workflows/04-long-video) or [`24-mv-lipsync`](examples/workflows/24-mv-lipsync).

Advanced workflows include notes on the canvas. Replace the model and input media before running them. Avoid stacking several LoRAs, attention backends, or sampler owners unless the workflow explicitly asks for it.

## What it can do

### H3 generation and references

- Text-to-video with audio (T2VA)
- First-frame image-to-video (I2VA)
- Last-frame generation (L2VA)
- First-and-last-frame generation (FL2VA)
- Single-image and multi-image Ref2VA
- Video, audio, and mixed references
- Native joint video/audio decoding and saving

### Audio and lip sync

- Source-audio locking, voice references, dialogue, and final-track mixing
- Vocal Lock: drive H3 with isolated vocals and mix the full song only once at the end
- Audio Refine routes for Turbo, PDD, EAV, Prompt Relay, and long video
- Local ASR, speaker, and SyncNet verification tools

The project's 32-second Vocal Lock V3 sample completed five serial H3 shots, per-shot lip-sync checks, and a full human review. The reviewer accepted that exact video. This is evidence for the reviewed sample, not a guarantee for every face, song, or prompt.

### Long video

- Multi-keyframe planning and segmented continuation
- One-queue serial rendering
- Resume support and accepted manifests
- Native Masked Context Plan B
- Optional Color Match, enabled by default, to reduce seam color changes

### Acceleration and finishing

- OpenVDN DMD eight-step and Stage B 50-step execution
- PDD, SLA, SPEED, FastH3 VSA, and Enhance-A-Video integrations
- FlashVSR, RealBasicVSR, RAFT, and Skin Finish
- Experimental two-stage MiniMax H3 plus LTX-2.5 upscaling

Features labeled `Advanced` or `EXP` should be used through their bundled workflows. Acceleration methods are usually alternatives, not add-ons to be stacked together.

## Installation

### ComfyUI Manager

Search for `MiniMax H3 Audio T8`, install it, then fully restart ComfyUI.

### Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

This node pack follows recent native MiniMax H3 APIs in ComfyUI. If every node turns red or appears missing, update the ComfyUI core, frontend, and Manager together, then exit and restart the application.

## Model folders

| Model | ComfyUI folder |
| --- | --- |
| H3 diffusion model | `models/diffusion_models` |
| Qwen3-VL text encoder | `models/text_encoders` |
| Video and audio VAEs | `models/vae` |
| Turbo, PDD, SLA, and other LoRAs | `models/loras` |
| FlashVSR / RealBasicVSR | `models/upscale_models` or the workflow's named folder |
| Face, optical-flow, and segmentation models | The folder named by the workflow or its documentation |

Similar filenames do not guarantee compatible H3 structures. Use the model and LoRA combination named by the workflow.

## OpenVDN: the recommended eight-step route

Complete model bundle: [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)

The model repository already follows ComfyUI's `diffusion_models`, `text_encoders`, and `vae` directory layout. After your model-access request is approved, download it directly into `ComfyUI/models`:

```bash
hf auth login
hf download t8star/Vdn-Minimax-H3-Comfy --local-dir ComfyUI/models
```

Then load a formal OpenVDN workflow from [`examples/workflows/10-speed`](examples/workflows/10-speed). The set includes:

- T2VA
- first-frame I2VA
- last-frame L2VA
- first-and-last-frame FL2VA
- single-image Ref2VA
- multi-image Ref2VA
- reference video with its audio
- standalone reference audio
- first-frame plus reference-audio Hybrid

OpenVDN upstream documents T2VA. The other modes are T8 extensions implemented and real-tested through ComfyUI's native MiniMax H3 conditioning layout.

### Full and supported pruned bases both work

The formal workflows default to:

```text
models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
```

This is the full, non-pruned base with a 2688-column AdaLN input. After installing the updated model bundle, the same workflow can also use:

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

The Composer hashes the base checkpoint's `adaln_t_table` and automatically selects the matching curve-projected Turbo adapter; no extra LoRA node is required. The adapter keeps 208 directly compatible targets and converts the 51 AdaLN targets into eight-column LoRA factors plus 51 bias residuals, for 310 applied patches. Full-width bases continue to use the native 2688-column adapter.

Compatibility is bound to the curve-table content, not merely to a filename. The model bundle's FL2VA pruned INT8/ConvRot base is supported, and the FL2VA pruned FP8 base shares the same statically verified curve signature. An unknown pruned/curve-basis checkpoint still fails before sampling with a specific diagnostic instead of silently using the wrong adapter.

### Validation performed

Using the full base above, the project ran I2VA, L2VA, FL2VA, single-image, two-image, video-plus-audio, audio-only, and first-frame-plus-audio routes one at a time on the same RTX 4060 Ti 16 GB. It then repeated T2VA and the same eight multimodal routes with the pruned INT8 base at 320x192x39. Every pruned route completed:

- 800 OpenVDN branch tensors
- 104 default-adapter targets
- 259 logical turbo-adapter targets, with bias residuals for all 51 AdaLN targets and 310 applied patches in total
- eight Euler/native-flow steps with video/audio shifts 12/3
- strict H.264 video, AAC audio, and combined decoding
- zero `ERROR lora` events in the runtime log

This proves that the tested workflows compose correctly with both base types. It does not promise identical quality or memory use for every prompt, reference, GPU, or driver. The full-base matrix retained 535–890 MiB; the nine short pruned runs fell as low as 290 MiB, and only T2VA and I2VA cleared this project's 512 MiB margin. On 16 GB GPUs, run one H3 generation at a time and reduce the canvas or frame count when needed.

## Common problems

### All nodes are red or missing

Update the ComfyUI core, frontend, and Manager, then fully restart. Updating only this custom node may not be enough.

### Out of memory at startup or sampling

Lower the resolution, frame count, and number of references. Close other generation tasks. Do not run two H3 jobs concurrently on a 16 GB GPU.

### Noisy or unusually quiet audio

Check the sampler, scheduler, step count, video/audio shifts, and exact LoRA named by the workflow. Generic EMA, Ref2VA, OpenVDN turbo, and other acceleration LoRAs are not interchangeable.

### OpenVDN reports an AdaLN mismatch

Confirm that the updated model bundle contains `stage-dmd-step-250/adapters/turbo_pruned_curve_fl2va/adapter_model.safetensors`. If it does and the error remains, the selected pruned base has an unsupported curve signature. Use the bundle's FL2VA pruned INT8 base or fall back to the full `minimax_h3_fl2va_int8_convrot.safetensors`; do not rename files to bypass the signature check.

### Can OpenVDN be combined with SLA, VSA, Sol-Attn, or BlockCache?

OpenVDN already owns the model branch and adapters. Do not add another MODEL or attention owner. More acceleration nodes do not mean more speed when they compete for the same forward pass.

## Documentation

- [Workflow index](examples/workflows)
- [ComfyUI and model guide](docs/README_ComfyUI.md)
- [Verification record](docs/VERIFICATION_REPORT.md)
- [Feature inventory](features.json)

## Licenses

The custom-node source is GPL-3.0-or-later.

Model files have separate terms. MiniMax H3 and its derivatives use the MiniMax H3 Community License Agreement. Its defined Applicable Territory excludes the European Union, the United Kingdom, the Republic of Korea, and the United States of America. Read the complete agreement and Acceptable Use Policy before downloading, running, or redistributing the models.

This GitHub repository does not contain model weights. The separate Hugging Face bundle preserves the relevant license, NOTICE, source links, and modification notices.
