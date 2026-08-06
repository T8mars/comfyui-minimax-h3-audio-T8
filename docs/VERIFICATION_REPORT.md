# LoRA and sampler verification report

This report records the historical LoRA, stable sampler, and multi-rate sampler
verification checkpoint. For the current plugin version, node inventory, and
Ref2VA still-image status, also read the project-root `README.md` and
`features.json`.

Verified on 2026-08-06 against ComfyUI `0.30.0`, source commit
`563b98eefbe643a4cd510ee7f0b43e79880d5a3f`.

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
