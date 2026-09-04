# MiniMax H3 Audio T8

[简体中文](README.md) | English

MiniMax H3 Audio T8 is a ComfyUI node pack for joint video and audio generation. It includes practical workflows for text and image animation, first/last-frame control, image/video/audio references, long video, lip sync, acceleration, and final-video restoration.

Current version: **1.71.1** · 292 nodes · GPL-3.0-or-later

## Where to start

If this is your first time using the pack:

1. Run a basic H3 workflow from [`examples/workflows/01-basic-generation`](examples/workflows/01-basic-generation).
2. For image or audio control, use the workflows under [`02-audio-control`](examples/workflows/02-audio-control) and the matching reference folders.
3. For OpenVDN eight-step generation, download [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy), then use an `OpenVDN_DMD8_*_Advanced.json` workflow from [`10-speed`](examples/workflows/10-speed).
4. For WASD character and camera control, download [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy), then use [`26-h3-world`](examples/workflows/26-h3-world); the first release is fixed to first-frame I2VA.
5. For long video or music video work, start with [`04-long-video`](examples/workflows/04-long-video) or [`24-mv-lipsync`](examples/workflows/24-mv-lipsync).
6. For image or finished-video upscaling, use [`25-dlss-nr`](examples/workflows/25-dlss-nr), an optional Windows RTX post-process.

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
- H3-World first-frame I2VA with WASD and IJKL/F character/camera timelines

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
- DLSS-NR image/short-frame/file-video upscaling, plus FlashVSR, RealBasicVSR, RAFT, and Skin Finish
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
| Complete OpenVDN model bundle | [`t8star/Vdn-Minimax-H3-Comfy`](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy); already arranged under the correct `models` folders |
| H3-World action LoRA | [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy), arranged for `models/loras/minimax/H3-World`; original source: [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World) |
| FlashVSR / RealBasicVSR | `models/upscale_models` or the workflow's named folder |
| DLSS-NR v1.3 external runtime (not a model) | `models/DLSS-NR/1.3` (user supplied; never downloaded or redistributed by the node) |
| Face, optical-flow, and segmentation models | The folder named by the workflow or its documentation |

Similar filenames do not guarantee compatible H3 structures. Use the model and LoRA combination named by the workflow.

## DLSS-NR: optional Windows RTX post-processing

[`examples/workflows/25-dlss-nr`](examples/workflows/25-dlss-nr) contains four independent workflows:
runtime audit, image, short-video frames, and file-backed video. They start from v1.3 `Standard + 2x`.

**DLSS-NR needs no new PyTorch or safetensors model, but it does need an external executable runtime.**
Download the full `video2dlssnr_release.zip` from the official
[`video2dlssnr` v1.3 release](https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3).
Do not use the light archive, and do not install the upstream ComfyUI node package. This project never
downloads, installs, or redistributes the EXE or proprietary NVIDIA DLLs.

Requirements:

- Windows 10/11, an NVIDIA RTX GPU, and NVIDIA driver **616.56 or newer**
- No extra pip package for the image or frame workflows; they use ComfyUI's existing Torch, NumPy, and PyAV environment
- The `Video File` workflow also needs `ffprobe` available on `PATH`, normally provided by an FFmpeg installation
- Read and accept the applicable external-runtime and NVIDIA terms; the workflow acknowledgement is off by default

Keep the full ZIP, copy the four files from its `out` folder into the `bin` folder below, and copy
[`examples/runtime-manifests/dlss-nr-v1.3.json`](examples/runtime-manifests/dlss-nr-v1.3.json)
as `t8-runtime-manifest.json`:

```text
ComfyUI/models/DLSS-NR/1.3/
├── t8-runtime-manifest.json
├── video2dlssnr_release.zip
└── bin/
    ├── video2dlssnr.exe
    ├── nvngx_dlss.dll
    ├── nvngx_dlssnr.dll
    └── nvngx.dll_dlssnr.dll
```

Run `Runtime_Audit` first. It verifies the full archive and extracted-file hashes, driver, GPU mapping,
and a real feature probe. Only continue after it reports `READY`. Runtime v1.2 is limited to 1x NR-only;
the default 2x workflows require v1.3.

All four methods passed the non-regression gate on the three fixed review clips. Human review found
different skin and texture character but no universal winner. Standard is therefore a practical
starting point, not a claim that DLSS-NR is always best. Upscaling cannot repair identity, lip-sync,
or real texture information already missing from the source.

## H3-World: WASD character and camera control

Upstream project: [`Danzer1xxxxChan/H3-World`](https://github.com/Danzer1xxxxChan/H3-World) ·
ComfyUI model bundle: [`t8star/Minimax-H3-World-Comfy`](https://huggingface.co/t8star/Minimax-H3-World-Comfy) ·
original model: [`DANNY621/H3-World`](https://huggingface.co/DANNY621/H3-World)

The first release has one deliberate contract: a first-frame I2VA at 832x480, 124 frames, and 24 fps.
Its 37 latent-time controls can move the character forward, backward, or sideways and tilt or pan the
camera. The `custom` preset accepts a tiled JSON timeline for mixed actions. Other input aspect ratios
are scaled to cover and center-cropped, never stretched.

Download the LoRA:

```bash
hf download t8star/Minimax-H3-World-Comfy --include "loras/**" --local-dir ComfyUI/models
```

The resulting path must be
`ComfyUI/models/loras/minimax/H3-World/step-10000.safetensors`. The bundled file is byte-identical to
the pinned upstream revision, with SHA-256
`DDD9187B920B1E52C2D090F4E264FD83D8D433EFC2A5B159E58883AEAF96E526`; T8star did not convert, merge,
or quantize it. It is already a directly loadable ComfyUI LoRA with 104 A/B pairs. The workflow also
uses the existing full `minimax_h3_fl2va_int8_convrot.safetensors`, Qwen3-VL
encoder, video VAE, and audio VAE. It adds no pip dependency, but its safe final MP4 writer requires
`ffmpeg` on `PATH`. Start from [`examples/workflows/26-h3-world`](examples/workflows/26-h3-world) and do
not stack OpenVDN, SLA, VSA, Sol-Attn, BlockCache, or another model/attention owner on this route.

The fixed parking-garage blind review passed: the reviewer correctly identified the continuously moving
H3-World candidate, accepted its stability and both audio tracks, and rated visual quality as a tie. This
promotes the fixed contract to a formal Advanced feature; it is not a universal quality or VRAM claim.

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
