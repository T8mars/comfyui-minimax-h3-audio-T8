# MiniMax H3 Hybrid Compatibility Audit (Advanced)

Version 1.17.0 adds an isolated, opt-in compatibility checkpoint for the
experimental FL2VA×Ref2VA Hybrid model path. It does not modify model weights,
Conditioning, sampler mathematics, caches, global H3 code, or the connected
MODEL object. The first output is the exact same Python MODEL object received
by the node.

## Placement

Place the node at the end of the MODEL patch chain and before `BasicGuider`:

```text
Hybrid Loader
  -> optional LoRA
  -> optional SageAttention
  -> optional Block Cache / Long Video / MultiKeyframe
  -> stable or EXP sampler setup
  -> Hybrid Compatibility Audit (Advanced)
  -> BasicGuider
```

Connect the final H3 `positive` Conditioning as well when using Long Video,
MultiKeyframe, or reference media. The Conditioning input is optional so a
model-only preflight remains possible.

The default `report_only` mode always passes the MODEL through. It can report
`compatible=false`, but it does not stop generation. The opt-in
`block_hard_conflicts` mode raises before the guider executes when a mechanical
contract failure is proven.

## What is audited

- Hybrid attachment schema, exact validated FL2VA/Ref2VA hashes, curve hashes,
  canonical recipe, identity fingerprint, operation count, and payload size.
- Every expected AdaLN `offset-set` entry, including offset and target shape.
- Hybrid-before-LoRA ordering. A patch before the Hybrid set is invalid;
  a later patch overlapping selected AdaLN rows is mechanically ordered but is
  rejected as uncalibrated. Attention/MLP-only patches outside selected AdaLN
  tensors remain mechanically compatible.
- MiniMax H3 Block Cache prototype, first/last `double_block` replacements, and
  both wrapper groups. Partial or foreign double-block replacement fails closed.
- MiniMax SageAttention replacement across every H3 DiT block. Partial coverage
  or an unknown attention-forward replacement fails closed.
- Long Video and MultiKeyframe scoped MODEL patches, matching Conditioning
  markers, patch version, and mutual exclusion.
- Stock/unpatched, stable dual-clock/native AV, experimental multi-rate, or
  unknown custom `model_sampling` routes.
- Loader-recorded VRAM-policy provenance, ModelPatcher DynamicVRAM state,
  current whole-device free VRAM, ComfyUI/AIMDO state, and host commit headroom.

The report uses stable issue codes. Examples include
`patch_precedes_hybrid_set`, `adaln_patch_overlaps_hybrid`,
`block_cache_contract_incomplete`, `sage_attention_contract_incomplete`,
`long_video_multikeyframe_conflict`,
`multikeyframe_conditioning_model_mismatch`,
`vram_policy_required_not_applied`,
`current_gpu_headroom_below_gate`, and
`host_commit_headroom_below_gate`.

## VRAM and VBAR boundary

When a T8 VRAM policy is connected to the Hybrid Loader, the Loader stores a
small immutable provenance attachment on the returned MODEL. It records whether
the policy was actually applied, the policy fingerprint and mode, the reserve
target, the ComfyUI/AIMDO setter routes, cleanup scope, and gate results. It
does not embed the large before/after telemetry tree.

`require_applied_vram_policy=true` rejects a missing or `report_only` policy.
The defaults of 512 MiB current whole-device headroom and 16 GiB host commit
headroom are preflight gates only. They are deliberately configurable because
hardware and workflows differ.

The audit always reports `memory_safe_claim=false`. VBAR pages model weights;
it does not provide an upper bound for activations, attention workspaces,
VAE/CLIP allocations, CUDA context, pinned host buffers, another process,
driver allocations, or host commit. A large page file prevents one class of
host-backing failure but does not prove that a future denoising peak cannot OOM.

## Quality boundary

Passing this node means that the observed patch stack satisfies the known
mechanical contracts. It does not prove that a Hybrid recipe is better than
stock FL2VA or Ref2VA, removes waxy/oily rendering, improves identity, preserves
audio, or is safe on every 16 GiB GPU.

Current H3 has three AdaLN tags: video, text, and audio. Reference and target
rows share their video/audio tags. Static AdaLN fusion therefore affects target
and reference rows together and is not reference-only routing. An AdaLN LoRA
that overlaps selected Hybrid rows remains blocked until that exact base,
artifact, LoRA, order, sampler, and quality matrix is calibrated.

## Examples

- API: `examples/hybrid_compatibility_audit_api.json`
- Frontend: `examples/workflows/H3_Hybrid_Compatibility_Audit_Stock20_EXP.json`

The frontend example keeps the validated 4 GiB reserve starting point for the
exact RTX 4060 Ti 16 GiB, 736×416, 124-frame Stock20 experiment, requires an
applied policy, connects final Conditioning, and uses `report_only`. It is not a
general memory-safe preset.
