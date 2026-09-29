# MiniMax H3 Audio T8

MiniMax H3 video/audio nodes for ComfyUI: reference control, two-pass and long-video workflows, speech and singing, camera editing, and optional enhancement.

[简体中文](README.md) | English · Current version: **1.86.0** · [Changelog](CHANGELOG.md)

## Install

Requires recent native ComfyUI H3 support. Manual installation:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-minimax-h3-audio-T8.git minimax-h3-audio-T8
```

Exit and restart ComfyUI after installing/updating, then refresh the page. Manager can search `MiniMax H3 Audio T8`; Registry activation is separate from GitHub publication. Use GitHub if the latest version is not available there.

## Choose a workflow

Replace placeholder models and media with your own before running.

|Goal|Start here|
|---|---|
|First H3 generation / image-to-video|[Basic generation](examples/workflows/01-basic-generation)|
|Audio references, talking or singing|[Audio control](examples/workflows/02-audio-control) · [Native voice/emotion and Avatar](examples/workflows/36-avatar-voice/README.md)|
|Dual-model 4+4 or segmented long video|[Long-video workflows](examples/workflows/04-long-video)|
|Visually arrange shots, media and audio|[Obsidian Director](examples/workflows/39-director-console/README.md) (or use the dedicated `T8 Obsidian Director` entry in the ComfyUI left sidebar)|
|OpenVDN, FastH3, memory-saving patches|[Acceleration](examples/workflows/10-speed) · [Memory nodes](docs/H3_MEMORY_NODES_EXP.md)|
|Meridian camera and source-time editing|[Four-node workflows](examples/workflows/37-meridian/README.md)|
|First-pass dynamic preview / scoped cancellation|[TAEH3 preview](docs/TAEH3_SAMPLING_PREVIEW_EXP.md)|
|Enhancement / frame interpolation|[Topaz](docs/TOPAZ_EXP.md) · [DLSS-NR](examples/workflows/25-dlss-nr) · [DLSS interpolation](examples/workflows/29-dlss-fi)|
|H16-3 chunked second pass / refined audio|[H16-3 workflow](examples/workflows/13-latent-upscale/2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json) · [Guide](docs/H16_3_CHUNKED_PASS2_EXP.md)|
|More features and advanced setup|[All workflows](examples/workflows) · [Detailed guide](docs/README_DETAILS_EN.md)|

## Download models

Read each repository's setup and licenses. Do not overwrite different architectures with identically named checkpoints.

|Model / resource|Download and ComfyUI location|
|---|---|
|H3 transformer, Qwen, video/audio VAEs|`models/diffusion_models`, `models/text_encoders`, `models/vae`; [base setup](docs/README_ComfyUI.md)|
|TAEH3 temporal / 2D previews|[Taeh3-Comfy](https://huggingface.co/t8star/Taeh3-Comfy) → `models/vae_approx/`; the 2D file has a separate name|
|Meridian ConvRot INT8 + Omega1B512|[Meridian-Comfy](https://huggingface.co/t8star/Meridian-Comfy) → `models/meridian/`; Omega in `vggt-omega/checkpoints/`; code/assets and H3 VAE are separate|
|Semantic Bridge|[Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy) → `models/semantic_bridge/t8_compat/`|
|T8 comic-combat Semantic Bridge|[Model](https://huggingface.co/t8star/semantic_bridge_T8-comic-combat) → `models/semantic_bridge/t8_compat/`; use the latest GitHub source and follow the [Bridge guide](docs/SEMANTIC_BRIDGE_EXP.md)|
|OpenVDN bundle|[Vdn-Minimax-H3-Comfy](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy); retain the repository layout|
|H3-World action LoRA|[Minimax-H3-World-Comfy](https://huggingface.co/t8star/Minimax-H3-World-Comfy); [setup](examples/workflows/26-h3-world)|

Meridian's DMD is already merged: **do not apply the same DMD LoRA again**. Omega is the original PT geometry checkpoint, not INT8 or ordinary VGGT. TAEH3 is approximate preview only, not a final VAE or audio model.

## Important limits

- Use the matching workflows for `Advanced` / `EXP`. Accepted examples are not universal quality, hardware, or long-window guarantees.
- Existing graphs need no forced migration. Unknown LoRA/Sage/Sol stacks receive warnings rather than blanket connection bans; see [patch-stack policy](docs/PATCH_STACK_POLICY.md).
- Exact voice identity, word timing, seamless joins and universal 16GB performance are not promised. Slight seam color changes remain a known limitation.
- Topaz and DLSS are independent postprocessing with their own runtimes and resources; H3 weights alone are insufficient.

For red nodes, audio errors and model compatibility, see the [detailed guide](docs/README_DETAILS_EN.md). [Report issues](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues) with the full error, workflow and Core/node versions; remove credentials and private media.

## T8 links

|Platform|Link|
|---|---|
|Bilibili / YouTube|[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/)|
|API / online apps|[API (affiliate)](https://api.seedance.nz/sign-up?aff=5f4w) · [RunningHub (invite)](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)|
|ComfyUI bundle / model drive|[ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Models](https://pan.quark.cn/s/c9c267081fbf)|
|Models / nodes|[Hugging Face](https://huggingface.co/t8star) · [Node GitHub](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)|

## Licenses and documentation

Node code: [GPL-3.0-or-later](LICENSE). No weights are included in this GitHub repository. Models/external assets retain their own licenses: Meridian weights use the MiniMax H3 Community License; Omega separately uses the FAIR Noncommercial Research License. Conversion grants no additional rights.

[Changelog](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Feature list](features.json) · [Homepage policy](docs/README_POLICY.md)
