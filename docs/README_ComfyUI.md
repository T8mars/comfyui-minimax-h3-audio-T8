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
Version 1.3.3 exposes internal sampler and scheduler dropdowns while preserving
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

Three complete frontend workflows are installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/`: stable 4/4, experimental 4/8,
and experimental 4/10. They share the same seed, prompt, EMA LoRA, loaders, and
MP4 settings for direct comparison. Drag a JSON file into the ComfyUI canvas or
open it from the Workflows menu.

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
