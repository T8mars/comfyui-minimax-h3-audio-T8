from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

import comfy.controlnet
import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths
from comfy.ldm.minimax.model import DiTBlock


FUN_CONTROL_ATTACHMENT_KEY = "t8_minimax_h3_fun_control_v1"
FUN_CONTROL_ADDITIONAL_MODEL_KEY = "t8_minimax_h3_fun_control_models_v1"
CONTROL_TYPE = "H3_T8_FUN_CONTROL"
PATCH_SIZE = (1, 2, 2)
INJECTION_LAYERS = (0, 10, 20, 30, 40)


def _native_fun_control_available() -> bool:
    return bool(
        hasattr(comfy.controlnet, "MiniMaxH3ControlNet")
        and callable(getattr(comfy.controlnet, "load_controlnet_minimax_h3", None))
    )


def _convert_diffusers_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert the published VideoX-Fun names without using a file allowlist."""
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".attn.to_q.weight"):
            base = key[: -len("to_q.weight")]
            converted[base + "qkv_proj.weight"] = torch.cat(
                [
                    state_dict[base + "to_q.weight"],
                    state_dict[base + "to_k.weight"],
                    state_dict[base + "to_v.weight"],
                ],
                dim=0,
            )
        elif key.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        elif key.endswith(".ff.net.0.proj.weight"):
            half = value.shape[0] // 2
            converted[key.replace(".ff.net.0.proj.", ".mlp.fc1.")] = torch.cat(
                [value[half:], value[:half]], dim=0
            )
        else:
            converted[
                key.replace(".attn.norm_q.", ".attn.q_norm.")
                .replace(".attn.norm_k.", ".attn.k_norm.")
                .replace(".attn.to_out.0.", ".attn.out_proj.")
                .replace(".ff.net.2.", ".mlp.fc2.")
            ] = value
    return converted


class _ControlDiTBlock(DiTBlock):
    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        t_dim: int,
        eps: float,
        qk_eps: float,
        *,
        first_block: bool,
        apply_silu: bool,
        adaln_dtype,
        dtype,
        device,
        operations,
    ):
        super().__init__(
            hidden,
            heads,
            head_dim,
            ffn,
            t_dim,
            eps,
            qk_eps,
            apply_silu=apply_silu,
            adaln_dtype=adaln_dtype,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if first_block:
            self.before_proj = operations.Linear(
                hidden, hidden, bias=True, dtype=dtype, device=device
            )
        self.after_proj = operations.Linear(
            hidden, hidden, bias=True, dtype=dtype, device=device
        )


class _MiniMaxH3FunControl(nn.Module):
    def __init__(
        self,
        *,
        control_in_dim: int,
        injection_layers: tuple[int, ...],
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        time_embed_dim: int,
        use_adaln_curves: bool,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = PATCH_SIZE
        self.injection_layers = tuple(injection_layers)
        patch_dim = control_in_dim * 4
        self.control_proj_in = operations.Linear(
            patch_dim,
            hidden_size,
            bias=True,
            dtype=torch.float32,
            device=device,
        )
        self.control_blocks = nn.ModuleList(
            [
                _ControlDiTBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    1e-5,
                    1e-5,
                    first_block=index == 0,
                    apply_silu=not use_adaln_curves,
                    adaln_dtype=torch.float32 if use_adaln_curves else dtype,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for index in range(len(self.injection_layers))
            ]
        )


@dataclass(frozen=True)
class H3FunControlBundle:
    backend: str
    control: Any
    filename: str
    path: str
    report: Mapping[str, Any]


def _checkpoint_structure(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if "control_blocks.0.attn.to_q.weight" in state_dict:
        state_dict = _convert_diffusers_state_dict(state_dict)
    required = (
        "control_proj_in.weight",
        "control_blocks.0.adaln_proj.linear.weight",
        "control_blocks.0.attn.qkv_proj.weight",
        "control_blocks.0.attn.q_norm.weight",
        "control_blocks.0.mlp.fc1.weight",
        "control_blocks.0.after_proj.weight",
    )
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(
            "The selected checkpoint does not expose the MiniMax H3 Fun ControlNet "
            f"tensor contract; missing {missing[:4]}"
        )
    block_count = 0
    while f"control_blocks.{block_count}.after_proj.weight" in state_dict:
        block_count += 1
    if block_count <= 0:
        raise RuntimeError("MiniMax H3 Fun ControlNet contains no control blocks")
    proj = state_dict["control_proj_in.weight"]
    qkv = state_dict["control_blocks.0.attn.qkv_proj.weight"]
    q_norm = state_dict["control_blocks.0.attn.q_norm.weight"]
    adaln = state_dict["control_blocks.0.adaln_proj.linear.weight"]
    return {
        "state_dict": state_dict,
        "block_count": block_count,
        "control_in_dim": int(proj.shape[1] // 4),
        "hidden_size": int(proj.shape[0]),
        "head_dim": int(q_norm.shape[0]),
        "num_heads": int(qkv.shape[0] // (3 * q_norm.shape[0])),
        "ffn_hidden_size": int(
            state_dict["control_blocks.0.mlp.fc1.weight"].shape[0] // 2
        ),
        "time_embed_dim": int(adaln.shape[1]),
    }


def _load_compatibility_control(path: str):
    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    structure = _checkpoint_structure(state_dict)
    state_dict = structure.pop("state_dict")
    load_device = comfy.model_management.get_torch_device()
    quantization = comfy.utils.detect_layer_quantization(state_dict, "")
    if quantization is not None:
        compute_dtype = torch.bfloat16
        operations = comfy.ops.mixed_precision_ops(quantization, compute_dtype)
    else:
        compute_dtype = comfy.utils.weight_dtype(state_dict)
        manual_cast = comfy.model_management.unet_manual_cast(
            compute_dtype, load_device
        )
        operations = comfy.ops.pick_operations(compute_dtype, manual_cast)
    offload_device = comfy.model_management.unet_offload_device()
    injection_layers = tuple(range(0, structure["block_count"] * 10, 10))
    control = _MiniMaxH3FunControl(
        control_in_dim=structure["control_in_dim"],
        injection_layers=injection_layers,
        hidden_size=structure["hidden_size"],
        num_attention_heads=structure["num_heads"],
        attention_head_dim=structure["head_dim"],
        ffn_hidden_size=structure["ffn_hidden_size"],
        time_embed_dim=structure["time_embed_dim"],
        use_adaln_curves=structure["time_embed_dim"] <= 16,
        dtype=compute_dtype,
        device=offload_device,
        operations=operations,
    )
    missing, unexpected = control.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "MiniMax H3 Fun ControlNet framework load found incompatible tensors: "
            f"missing={sorted(missing)[:6]}, unexpected={sorted(unexpected)[:6]}"
        )
    control.eval().requires_grad_(False)
    patcher_class = getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )
    patcher = patcher_class(
        control, load_device=load_device, offload_device=offload_device
    )
    return control, patcher, {
        **structure,
        "injection_layers": injection_layers,
        "compute_dtype": str(compute_dtype),
        "quantization": type(quantization).__name__ if quantization is not None else None,
    }


def load_h3_fun_control(control_net_name: str) -> tuple[H3FunControlBundle, str]:
    path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    file_path = Path(path)
    if _native_fun_control_available():
        native = comfy.controlnet.load_controlnet(path)
        native_class = getattr(comfy.controlnet, "MiniMaxH3ControlNet")
        if not isinstance(native, native_class):
            raise RuntimeError(
                "The current ComfyUI native loader did not recognize this checkpoint as "
                "a MiniMax H3 Fun ControlNet"
            )
        details = {
            "schema": "t8_minimax_h3_fun_control_loader_v1",
            "status": "native_core",
            "backend": "native_core_controlnet",
            "filename": control_net_name,
            "file_bytes_diagnostic": file_path.stat().st_size,
            "fingerprint_policy": "diagnostic_only_framework_loader_is_authoritative",
            "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15860",
        }
        bundle = H3FunControlBundle(
            "native", native, control_net_name, str(file_path), details
        )
        return bundle, json.dumps(details, ensure_ascii=False, sort_keys=True)

    control, patcher, structure = _load_compatibility_control(path)
    holder = {"model": control, "patcher": patcher}
    details = {
        "schema": "t8_minimax_h3_fun_control_loader_v1",
        "status": "compatibility_fallback",
        "backend": "clone_scoped_dit_replacement",
        "filename": control_net_name,
        "file_bytes_diagnostic": file_path.stat().st_size,
        "structure": structure,
        "fingerprint_policy": "diagnostic_only_framework_loader_is_authoritative",
        "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15860",
        "community_audit": "https://github.com/wyzborrero/ComfyUI-H3-FunControl",
    }
    bundle = H3FunControlBundle(
        "compatibility", holder, control_net_name, str(file_path), details
    )
    return bundle, json.dumps(details, ensure_ascii=False, sort_keys=True)


def _fit_control_video(
    frames: torch.Tensor,
    *,
    width: int,
    height: int,
    length: int,
    fit_mode: str,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] < 3 or frames.shape[0] <= 0:
        raise ValueError("control_video must be an IMAGE batch [T,H,W,C]")
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("H3 Fun Control width and height must be positive multiples of 32")
    if length < 5 or length % 17 != 5:
        raise ValueError("H3 Fun Control length must follow the 17n+5 frame grid")
    if frames.shape[0] < length:
        frames = torch.cat(
            [frames, frames[-1:].expand(length - frames.shape[0], -1, -1, -1)],
            dim=0,
        )
    else:
        frames = frames[:length]
    current = (int(frames.shape[2]), int(frames.shape[1]))
    if fit_mode == "exact":
        if current != (width, height):
            raise ValueError(
                "control_video geometry must exactly match the target canvas in exact mode: "
                f"got {current[0]}x{current[1]}, expected {width}x{height}"
            )
        return frames[..., :3]
    crop = "center" if fit_mode == "center_crop" else "disabled"
    chw = frames[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(chw, width, height, "lanczos", crop).movedim(
        1, -1
    )


def _patchify_control(latent: torch.Tensor, projection_width: int) -> torch.Tensor:
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24:
        raise ValueError(
            "H3 video VAE must encode control_video to [1,24,T,H,W], got "
            f"{tuple(latent.shape)}"
        )
    batch, channels, time, height, width = latent.shape
    pt, ph, pw = PATCH_SIZE
    if time % pt or height % ph or width % pw:
        raise ValueError("H3 control latent is not divisible by the (1,2,2) patch size")
    rows = (
        latent.view(
            batch,
            channels,
            time // pt,
            pt,
            height // ph,
            ph,
            width // pw,
            pw,
        )
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .contiguous()
        .view((time // pt) * (height // ph) * (width // pw), -1)
    )
    if rows.shape[1] > projection_width:
        raise ValueError(
            f"control latent packs to {rows.shape[1]} columns but checkpoint expects {projection_width}"
        )
    if rows.shape[1] < projection_width:
        rows = torch.nn.functional.pad(rows, (0, projection_width - rows.shape[1]))
    return rows


def _video_span(mod_segments) -> tuple[int, int] | None:
    if not mod_segments:
        return None
    start, stop = int(mod_segments[-1][0]), int(mod_segments[-1][1])
    return (start, stop) if stop > start else None


def _adaln_input_dim(module: Any) -> int | None:
    """Read the executable AdaLN input width without using file identity."""
    table = getattr(module, "adaln_t_table", None)
    if table is not None and getattr(table, "ndim", 0) == 2:
        return int(table.shape[-1])
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        blocks = getattr(module, "control_blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    block = blocks[0]
    projection = getattr(getattr(block, "adaln_proj", None), "linear", None)
    weight = getattr(projection, "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return None
    return int(weight.shape[1])


def _assert_compatible_adaln_pair(model, bundle: H3FunControlBundle) -> None:
    """Reject an impossible base/control pair before encoding or sampling.

    Kijai's ``adaln_basis`` ControlNet consumes the same eight-dimensional
    curve coordinates as a pruned H3 base.  The original dense H3 and original
    Fun-Control release both consume the full 2688-dimensional time embedding.
    Mixing those representations cannot be repaired by padding or reshaping the
    weights, so compare the live module contracts instead of filenames, hashes,
    or byte sizes.
    """
    if bundle.backend != "compatibility":
        return
    base = getattr(getattr(model, "model", None), "diffusion_model", None)
    holder = bundle.control if isinstance(bundle.control, Mapping) else {}
    control = holder.get("model")
    base_dim = _adaln_input_dim(base) if base is not None else None
    control_dim = _adaln_input_dim(control) if control is not None else None
    if base_dim is None or control_dim is None or base_dim == control_dim:
        return
    raise RuntimeError(
        "MiniMax H3 Fun Control AdaLN representation mismatch: "
        f"the selected base model consumes {base_dim} values but the selected "
        f"ControlNet consumes {control_dim}. Use an adaln_basis/curve-form "
        "ControlNet with a *_pruned_* H3 base, or pair the original full-width "
        "ControlNet with the original full-width H3 base. This check uses live "
        "tensor shapes only; filenames, hashes, and file sizes are not restricted."
    )


def _dense_control_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(options or {})
    for key in tuple(cleaned):
        if key == "optimized_attention_override" or key.startswith(("sol_", "sla_")):
            cleaned.pop(key, None)
    return cleaned


def _enter_control_stream(control, rows, seed, start: int, stop: int):
    """Build one full H3 control stream in a caller-owned seed clone."""
    embedded = control.control_proj_in(
        rows.to(device=seed.device, dtype=torch.float32)
    )
    embedded = control.control_blocks[0].before_proj(embedded.to(seed.dtype))
    if embedded.shape[0] != stop - start:
        raise RuntimeError(
            "H3 Fun Control video-token mismatch after VAE encode: "
            f"control={embedded.shape[0]}, target={stop - start}. "
            "Use the same width, height and 17n+5 frame count as generation."
        )
    stream = seed
    stream[start:stop] += embedded.to(stream.dtype)
    return stream


def _apply_native(
    positive,
    bundle: H3FunControlBundle,
    vae,
    frames: torch.Tensor,
    strength: float,
    start_percent: float,
    end_percent: float,
):
    hint = frames.movedim(-1, 1)
    cache = {}
    output = []
    for conditioning, metadata in positive:
        copied = metadata.copy()
        previous = copied.get("control")
        if previous not in cache:
            control = bundle.control.copy().set_cond_hint(
                hint, strength, (start_percent, end_percent), vae=vae
            )
            control.set_previous_controlnet(previous)
            cache[previous] = control
        copied["control"] = cache[previous]
        copied["control_apply_to_uncond"] = True
        output.append([conditioning, copied])
    return output


def _apply_compatibility(
    model,
    bundle: H3FunControlBundle,
    latent: torch.Tensor,
    strength: float,
    start_percent: float,
    end_percent: float,
):
    patched = model.clone()
    transformer = patched.model_options.get("transformer_options", {})
    if transformer.get("sol_morton"):
        raise RuntimeError(
            "H3 Fun Control cannot use Sol-Attn Morton token reordering. Disable morton; "
            "raster-order control rows would otherwise target the wrong video tokens."
        )
    holder = bundle.control
    control = holder["model"]
    control_patcher = holder["patcher"]
    projection_width = int(control.control_proj_in.weight.shape[1])
    rows = _patchify_control(latent.to(torch.float32), projection_width)
    sampling = getattr(model.model, "model_sampling", None)
    start_sigma = sampling.percent_to_sigma(start_percent) if sampling else 1.0
    end_sigma = sampling.percent_to_sigma(end_percent) if sampling else 0.0
    state: dict[str, Any] = {"stream": None}

    existing = (
        transformer.get("patches_replace", {}).get("dit", {})
        if isinstance(transformer, Mapping)
        else {}
    )

    def active(options: Mapping[str, Any]) -> bool:
        sigmas = options.get("sigmas")
        if sigmas is None:
            return True
        sigma = float(sigmas[0])
        return float(end_sigma) <= sigma <= float(start_sigma)

    def make_hook(layer: int, inner):
        def hook(args, extra):
            options = args.get("transformer_options", {}) or {}
            span = _video_span(args.get("mod_segments")) if active(options) else None
            # The base DiT block mutates its input in place, so capture the
            # pre-block stream before dispatching either the original block or
            # an already-installed replacement.
            seed = (
                args["img"].clone()
                if span is not None and layer == control.injection_layers[0]
                else None
            )
            output = inner(args, extra) if inner is not None else extra["original_block"](args)
            image = output["img"]
            if span is None:
                return output
            start, stop = span
            if seed is not None:
                # VideoX-Fun enters the control tower exactly once as
                # before_proj(control_proj_in(control_rows)) + base_stream.  The
                # checkpoint input projection is intentionally FP32 even when
                # the transformer blocks are BF16/INT8, so never infer the
                # activation dtype from a quantized storage tensor.
                state["stream"] = _enter_control_stream(
                    control, rows, seed, start, stop
                )
            stream = state.get("stream")
            if stream is None:
                return output
            index = control.injection_layers.index(layer)
            block = control.control_blocks[index]
            stream = DiTBlock.forward(
                block,
                stream,
                args["t_emb"],
                args["mod_segments"],
                args["rope_freqs"],
                transformer_options=_dense_control_options(options),
            )
            state["stream"] = stream
            skip = block.after_proj(stream[start:stop]) * strength
            image[start:stop] += skip.to(image.dtype)
            return {"img": image}

        hook._t8_h3_fun_control = True
        hook._t8_h3_fun_control_layer = layer
        return hook

    for layer in control.injection_layers:
        inner = existing.get(("double_block", layer))
        patched.set_model_patch_replace(
            make_hook(layer, inner), "dit", "double_block", layer
        )
    additional = list(
        patched.get_additional_models_with_key(FUN_CONTROL_ADDITIONAL_MODEL_KEY)
    )
    additional.append(control_patcher)
    patched.set_additional_models(FUN_CONTROL_ADDITIONAL_MODEL_KEY, additional)
    return patched


def apply_h3_fun_control(
    model,
    positive,
    control_bundle: H3FunControlBundle,
    vae,
    control_video: torch.Tensor,
    width: int,
    height: int,
    length: int,
    control_kind: str,
    fit_mode: str,
    strength: float,
    start_percent: float,
    end_percent: float,
):
    if not isinstance(control_bundle, H3FunControlBundle):
        raise TypeError("control_bundle must come from the T8 H3 Fun Control loader")
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise ValueError("start_percent and end_percent must satisfy 0 <= start <= end <= 1")
    if strength == 0.0:
        report = {
            "schema": "t8_minimax_h3_fun_control_apply_v1",
            "status": "bypass",
            "reason": "strength_zero",
            "model_identity_preserved": True,
            "conditioning_identity_preserved": True,
        }
        return model, positive, json.dumps(report, ensure_ascii=False, sort_keys=True)
    transformer = getattr(model, "model_options", {}).get("transformer_options", {})
    if transformer.get("sol_morton"):
        raise RuntimeError(
            "H3 Fun Control requires raster-order video rows; disable Sol-Attn morton."
        )
    frames = _fit_control_video(
        control_video,
        width=width,
        height=height,
        length=length,
        fit_mode=fit_mode,
    )
    _assert_compatible_adaln_pair(model, control_bundle)
    prior = model.get_attachment(FUN_CONTROL_ATTACHMENT_KEY) if hasattr(model, "get_attachment") else None
    prior_controls = list(prior.get("controls", ())) if isinstance(prior, Mapping) else []
    entry = {
        "kind": control_kind,
        "strength": float(strength),
        "start_percent": float(start_percent),
        "end_percent": float(end_percent),
        "filename": control_bundle.filename,
    }
    if control_bundle.backend == "native":
        conditioned = _apply_native(
            positive,
            control_bundle,
            vae,
            frames,
            strength,
            start_percent,
            end_percent,
        )
        # Attach metadata to a clone so applying native ControlNet does not
        # mutate the user's upstream MODEL object. Conditioning already owns
        # the native ControlNet chain; the clone exists for isolation/reporting.
        patched = model.clone()
    else:
        latent = vae.encode(frames)
        patched = _apply_compatibility(
            model,
            control_bundle,
            latent,
            strength,
            start_percent,
            end_percent,
        )
        conditioned = positive
    controls = prior_controls + [entry]
    attachment = {
        "schema": "t8_minimax_h3_fun_control_attachment_v1",
        "backend": control_bundle.backend,
        "controls": controls,
        "combined_strength": sum(item["strength"] for item in controls),
        "audio_rows_policy": "never_receive_control_skip",
        "control_tower_attention": "dense_raster_order",
        "sol_morton": False,
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(FUN_CONTROL_ATTACHMENT_KEY, attachment)
    report = {
        "schema": "t8_minimax_h3_fun_control_apply_v1",
        "status": "applied",
        "backend": control_bundle.backend,
        "target": {"width": width, "height": height, "length": length},
        "fit_mode": fit_mode,
        "control": entry,
        "control_count": len(controls),
        "combined_strength": attachment["combined_strength"],
        "warnings": (
            ["combined control strength exceeds 1.0; saturation is possible"]
            if attachment["combined_strength"] > 1.0
            else []
        ),
        "scientific_boundary": (
            "Control type is descriptive; this node consumes preprocessed frames and does not "
            "certify the upstream depth/pose/edge estimator."
        ),
    }
    return patched, conditioned, json.dumps(report, ensure_ascii=False, sort_keys=True)
