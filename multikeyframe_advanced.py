from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import math
from pathlib import Path
import types

import node_helpers
import torch
from comfy.ldm.minimax import model as minimax_model

from .conditioning import (
    HYBRID_KEYFRAME_SENTINEL,
    HYBRID_LAYOUT_LEGACY_SENTINEL,
    NATIVE_GUIDE_PACKED_LAYOUT_SHA256,
    _encode_reference_audio,
    _resize_reference_image,
    assert_hybrid_layout_contract,
    build_conditioning,
    build_packed_layout,
    resolve_task_type,
)
from .core import (
    CANVAS_MULTIPLE,
    FPS,
    MAX_PIXELS,
    adapt_canvas,
    align_frame_count_down,
    empty_av_latent,
    encode_audio_once,
    fit_audio_latent,
    nested_av_parts,
    replace_audio_latent,
    resize_image,
    sorted_autogrow_items,
    sorted_autogrow_values,
    temporal_shape,
)
from .prompt_tags import media_map_json, prepare_prompt


KEYFRAME_PLAN_TYPE = "H3_T8_KEYFRAME_PLAN"
MULTIKEYFRAME_SCHEMA_KEY = "t8_multikeyframe_schema"
MULTIKEYFRAME_SCHEMA = 1
MULTIKEYFRAME_PATCH_VERSION = 1
ACTUAL_FRAME_INDEX = "t8_multikeyframe_frame_index"
KEYFRAME_NOISE_AUG = "t8_multikeyframe_visual_noise_aug"
VISUAL_NOISE_AUGS_KEY = "minimax_visual_cond_noise_augs"
PAYLOAD_VISUAL_NOISE_AUGS_KEY = "visual_cond_noise_augs"
NATIVE_LAYOUT_KEY = "t8_multikeyframe_native_layout"
FRAME_RESCALE = 5.0 / 3.0
MAX_MIDDLE_KEYFRAMES = 7
DEFAULT_VISUAL_NOISE_AUG = 0.999
VALIDATED_FORWARD_SHA256S = {
    "ec62dafa65d6eaf36c670b926a05b42503702cbd6e1e4bb9db279c0db2b4a3c5",
    "f40e52b23fb2f9c76ac4cac48c7a2f899e6e37d517cd801a939fff551ab89867",
    # ComfyUI 187eda8: adds row-wise video/audio denoise-mask timesteps.
    "14bdfccd6860f252005b8d43ab446aa9a938a13dc819061724b8f914218f5fd1",
}
RESIZE_MODES = {"center_crop", "stretch"}
POSITION_MODES = {"frame", "seconds", "percent"}


def append_keyframe_plan(
    previous_plan,
    image: torch.Tensor,
    position_mode: str,
    position: float,
    visual_noise_aug: float,
    resize_mode: str,
    enabled: bool,
):
    if previous_plan is None:
        entries: list[dict] = []
    elif isinstance(previous_plan, Sequence) and not isinstance(previous_plan, (str, bytes)):
        entries = list(previous_plan)
    else:
        raise ValueError("previous_plan must be an H3 T8 keyframe plan")
    if len(entries) >= MAX_MIDDLE_KEYFRAMES:
        raise ValueError(
            f"MiniMax H3 Advanced supports at most {MAX_MIDDLE_KEYFRAMES} middle keyframes"
        )
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("image must be a non-empty ComfyUI IMAGE batch [B,H,W,C]")
    mode = str(position_mode).lower()
    if mode not in POSITION_MODES:
        raise ValueError(f"Unknown keyframe position mode: {position_mode}")
    value = float(position)
    if not math.isfinite(value):
        raise ValueError("keyframe position must be finite")
    aug = float(visual_noise_aug)
    if not math.isfinite(aug) or not 0.0 <= aug <= 1.0:
        raise ValueError("visual_noise_aug must be finite and between 0.0 and 1.0")
    resize = str(resize_mode).lower()
    if resize not in RESIZE_MODES:
        raise ValueError(f"Unknown keyframe resize mode: {resize_mode}")

    entries.append(
        {
            "image": image[:1],
            "position_mode": mode,
            "position": value,
            "visual_noise_aug": aug,
            "resize_mode": resize,
            "enabled": bool(enabled),
            "source_ordinal": len(entries) + 1,
        }
    )
    return tuple(entries)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def resolve_keyframe_plan(plan, frame_count: int) -> list[dict]:
    if plan is None:
        return []
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise ValueError("keyframe_plan must be an H3 T8 keyframe plan")
    if len(plan) > MAX_MIDDLE_KEYFRAMES:
        raise ValueError(
            f"MiniMax H3 Advanced supports at most {MAX_MIDDLE_KEYFRAMES} middle keyframes"
        )

    resolved: list[dict] = []
    used: dict[int, int] = {}
    for fallback_ordinal, raw in enumerate(plan, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"keyframe plan entry {fallback_ordinal} is not a mapping")
        if not bool(raw.get("enabled", True)):
            continue
        mode = str(raw.get("position_mode", "")).lower()
        if mode not in POSITION_MODES:
            raise ValueError(f"keyframe plan entry {fallback_ordinal} has invalid position_mode")
        value = float(raw.get("position"))
        if not math.isfinite(value):
            raise ValueError(f"keyframe plan entry {fallback_ordinal} position must be finite")
        if mode == "frame":
            rounded = round(value)
            if not math.isclose(value, rounded, abs_tol=1e-9):
                raise ValueError(
                    f"keyframe plan entry {fallback_ordinal} frame position must be an integer"
                )
            frame_index = int(rounded)
        elif mode == "seconds":
            if value < 0.0:
                raise ValueError(
                    f"keyframe plan entry {fallback_ordinal} seconds must be non-negative"
                )
            frame_index = _round_half_up(value * FPS)
        else:
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"keyframe plan entry {fallback_ordinal} percent must be between 0 and 100"
                )
            frame_index = _round_half_up((value / 100.0) * (frame_count - 1))

        if frame_index <= 0 or frame_index >= frame_count - 1:
            raise ValueError(
                f"keyframe plan entry {fallback_ordinal} resolves to frame {frame_index}; "
                f"middle keyframes must be in 1..{frame_count - 2}"
            )
        if frame_index in used:
            raise ValueError(
                f"keyframe plan entries {used[frame_index]} and {fallback_ordinal} both resolve "
                f"to frame {frame_index}"
            )
        image = raw.get("image")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError(f"keyframe plan entry {fallback_ordinal} has no valid IMAGE")
        aug = float(raw.get("visual_noise_aug", DEFAULT_VISUAL_NOISE_AUG))
        if not math.isfinite(aug) or not 0.0 <= aug <= 1.0:
            raise ValueError(
                f"keyframe plan entry {fallback_ordinal} visual_noise_aug must be in 0..1"
            )
        resize_mode = str(raw.get("resize_mode", "center_crop")).lower()
        if resize_mode not in RESIZE_MODES:
            raise ValueError(
                f"keyframe plan entry {fallback_ordinal} has invalid resize_mode"
            )
        source_ordinal = int(raw.get("source_ordinal", fallback_ordinal))
        used[frame_index] = fallback_ordinal
        resolved.append(
            {
                "image": image[:1],
                "requested_mode": mode,
                "requested_value": value,
                "frame_index": frame_index,
                "visual_noise_aug": aug,
                "resize_mode": resize_mode,
                "source_ordinal": source_ordinal,
            }
        )
    resolved.sort(key=lambda item: (item["frame_index"], item["source_ordinal"]))
    return resolved


def _assert_unmodified_packed_layout() -> None:
    init = minimax_model.PackedLayout.__init__
    code = getattr(init, "__code__", None)
    code_file = Path(code.co_filename).resolve() if code is not None else None
    module_file = Path(minimax_model.__file__).resolve()
    if (
        getattr(init, "__module__", None) != minimax_model.__name__
        or getattr(init, "__qualname__", None) != "PackedLayout.__init__"
        or code_file != module_file
    ):
        raise RuntimeError(
            "An external process-global MiniMax H3 PackedLayout patch is active. "
            "Disable it before using the T8 Advanced multi-keyframe node."
        )


def native_middle_keyframe_support() -> bool:
    _assert_unmodified_packed_layout()
    keyframe = {
        "resolved_frame_index": 2,
        "latent": torch.zeros((1, 24, 1, 2, 2)),
    }
    parameters = inspect.signature(minimax_model.PackedLayout.__init__).parameters
    if "frame_count" not in parameters:
        packed_source = inspect.getsource(minimax_model.PackedLayout.__init__)
        packed_sha256 = hashlib.sha256(packed_source.encode("utf-8")).hexdigest()
        if packed_sha256 != NATIVE_GUIDE_PACKED_LAYOUT_SHA256:
            raise RuntimeError(
                "This ComfyUI build accepts a new MiniMax H3 Guide layout, but its exact "
                "contract has not been validated by this plugin version."
            )
        layout = build_packed_layout(1, 2, 2, 2, 1, keyframes=[keyframe])
        cond_segments = [
            (start, stop) for start, stop, kind in layout.segments if kind == "cond"
        ]
        expected_t = 1.0 + FRAME_RESCALE * 2
        if len(cond_segments) != 1 or not math.isclose(
            float(layout.position_ids[cond_segments[0][0], 0]), expected_t
        ):
            raise RuntimeError(
                "The native MiniMax H3 Guide position contract changed; Advanced "
                "multi-keyframe execution was refused."
            )
        return True
    try:
        build_packed_layout(1, 2, 2, 2, 1, keyframes=[keyframe], frame_count=5)
    except ValueError as exc:
        if "only first/last keyframe anchors are supported" in str(exc):
            return False
        raise RuntimeError(
            "MiniMax H3 PackedLayout rejected the Advanced capability probe with an "
            f"unknown contract: {exc}"
        ) from exc
    raise RuntimeError(
        "This ComfyUI build accepts middle MiniMax H3 keyframes through an unknown legacy "
        "constructor contract. Advanced multi-keyframe execution was refused."
    )


def _pixel_frames_from_latent_t(latent_t: int) -> int:
    if latent_t < 1:
        raise ValueError("MiniMax H3 video latent T must be positive")
    spans = getattr(minimax_model, "FRAME_PER_TOKEN", (1, 4, 4, 4, 4))
    return sum(spans[index % len(spans)] for index in range(latent_t))


def _ref_advance(ref: Mapping) -> float:
    kind = ref.get("kind")
    if kind == HYBRID_KEYFRAME_SENTINEL:
        return 0.0
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(ref.get("ref_audio_t", 0))
    if kind in {"video", "video_audio"}:
        return max(
            float(ref.get("ref_audio_t", 0)),
            FRAME_RESCALE * _pixel_frames_from_latent_t(int(ref.get("latent_t", 0))),
        )
    raise RuntimeError(f"Unsupported MiniMax H3 reference kind: {kind!r}")


def _repair_layout_positions(layout, keyframes: list[dict], refs: list[dict], frame_count: int) -> None:
    text_len, latent_t = int(layout.signature[0]), int(layout.signature[1])
    ref_offset = sum(_ref_advance(ref) for ref in refs)
    cond_segments = [(start, stop) for start, stop, kind in layout.segments if kind == "cond"]
    if len(cond_segments) != len(keyframes):
        raise RuntimeError("MiniMax H3 keyframe/layout count changed; Advanced patch refused")
    for (start, stop), keyframe in zip(cond_segments, keyframes):
        pixel_index = int(keyframe.get(ACTUAL_FRAME_INDEX, keyframe["resolved_frame_index"]))
        if pixel_index < 0 or pixel_index >= int(frame_count):
            raise RuntimeError(f"Advanced keyframe index {pixel_index} is outside the target")
        if pixel_index == 0:
            cond_t = float(text_len)
        elif pixel_index == int(frame_count) - 1:
            cond_t = (
                float(text_len)
                + FRAME_RESCALE * _pixel_frames_from_latent_t(latent_t)
                - FRAME_RESCALE
            )
        else:
            cond_t = float(text_len) + FRAME_RESCALE * pixel_index
        layout.position_ids[start:stop, 0] = cond_t + ref_offset


def _visual_condition_count(layout) -> int:
    return sum(kind in {"cond", "ref_img"} for _, _, kind in layout.segments)


def _keyframes_for_core(
    keyframes: list[dict], frame_count: int, native_layout: bool
) -> list[dict]:
    core_keyframes = []
    for keyframe in keyframes:
        item = dict(keyframe)
        actual = int(item.get(ACTUAL_FRAME_INDEX, item["resolved_frame_index"]))
        if actual < 0 or actual >= frame_count:
            raise RuntimeError(f"Advanced keyframe index {actual} is outside the target")
        if native_layout or actual in {0, frame_count - 1}:
            item["resolved_frame_index"] = actual
        else:
            item["resolved_frame_index"] = 0
        core_keyframes.append(item)
    return core_keyframes


def repair_multikeyframe_payload(out: dict, kwargs: dict) -> dict:
    if int(kwargs.get(MULTIKEYFRAME_SCHEMA_KEY, 0) or 0) != MULTIKEYFRAME_SCHEMA:
        return out
    cond = out.get("minimax_payload")
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError("Advanced model patch could not access the MiniMax H3 payload")

    frame_count_value = kwargs.get("minimax_frame_count")
    if frame_count_value is None:
        raise RuntimeError("Advanced MiniMax H3 payload is missing minimax_frame_count")
    frame_count = int(frame_count_value)
    native_layout = bool(kwargs.get(NATIVE_LAYOUT_KEY, False))
    keyframes = list(kwargs.get("minimax_keyframes") or [])
    core_keyframes = _keyframes_for_core(keyframes, frame_count, native_layout)
    refs = list(kwargs.get("minimax_refs") or [])
    payload["cond_video_latents"] = [
        keyframe["latent"] for keyframe in keyframes if "latent" in keyframe
    ] + [
        ref["latent"]
        for ref in refs
        if ref.get("kind") != HYBRID_KEYFRAME_SENTINEL and "latent" in ref
    ]
    payload["cond_audio_latents"] = [
        ref["audio_latent"] for ref in refs if ref.get("audio_latent") is not None
    ]
    payload["frame_count"] = frame_count
    payload["keyframes"] = core_keyframes
    payload["refs"] = refs

    augs = [float(value) for value in kwargs.get(VISUAL_NOISE_AUGS_KEY, [])]
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("Advanced MiniMax H3 payload has no PackedLayout")
    _repair_layout_positions(layout, keyframes, refs, frame_count)
    expected_visual = _visual_condition_count(layout)
    if len(payload["cond_video_latents"]) != expected_visual:
        raise RuntimeError(
            "Advanced video payload/layout count mismatch: "
            f"{len(payload['cond_video_latents'])} latents for {expected_visual} segments"
        )
    if len(augs) != expected_visual:
        raise RuntimeError(
            "Advanced visual noise-augmentation count mismatch: "
            f"{len(augs)} values for {expected_visual} visual segments"
        )
    expected_audio = sum(kind == "ref_audio" for _, _, kind in layout.segments)
    if len(payload["cond_audio_latents"]) != expected_audio:
        raise RuntimeError(
            "Advanced audio payload/layout count mismatch: "
            f"{len(payload['cond_audio_latents'])} latents for {expected_audio} segments"
        )
    payload[PAYLOAD_VISUAL_NOISE_AUGS_KEY] = augs
    payload[MULTIKEYFRAME_SCHEMA_KEY] = MULTIKEYFRAME_SCHEMA
    payload[NATIVE_LAYOUT_KEY] = native_layout
    if augs and all(math.isclose(value, augs[0], abs_tol=0.0) for value in augs[1:]):
        payload["visual_cond_noise_aug"] = augs[0]
    payload["t8_multikeyframe_patch_version"] = MULTIKEYFRAME_PATCH_VERSION
    return out


def _condition_rows_with_augs(self, payload: dict, device):
    latents = list(payload.get("cond_video_latents", []))
    augs = payload.get(PAYLOAD_VISUAL_NOISE_AUGS_KEY)
    if augs is None:
        aug = float(payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP))
        augs = [aug] * len(latents)
    if len(augs) != len(latents):
        raise RuntimeError(
            f"MiniMax H3 Advanced received {len(augs)} visual augs for {len(latents)} latents"
        )
    seed = int(payload.get("seed", 0))
    rows = []
    for latent, raw_aug in zip(latents, augs):
        aug = float(raw_aug)
        row = minimax_model.patchify_video(latent.to(torch.float32), self.patch_size)
        if aug < 1.0:
            generator = torch.Generator("cpu").manual_seed(seed)
            noise = torch.randn(row.shape, generator=generator, dtype=torch.float32)
            row = aug * row + (1.0 - aug) * noise.to(row.device)
        rows.append(row.to(device))
    return torch.cat(rows, dim=0) if rows else None


def _segment_timestep_plan(layout, t_video: float, t_audio: float, visual_augs, audio_aug: float):
    expected = _visual_condition_count(layout)
    if len(visual_augs) != expected:
        raise RuntimeError(
            f"MiniMax H3 Advanced layout has {expected} visual segments but {len(visual_augs)} augs"
        )
    visual_index = 0
    segment_times: list[float] = []
    for _start, _stop, kind in layout.segments:
        if kind in {"cond", "ref_img"}:
            segment_times.append(max(t_video, float(visual_augs[visual_index])))
            visual_index += 1
        elif kind in {"cond_audio", "ref_audio"}:
            segment_times.append(max(t_audio, float(audio_aug)))
        elif kind in {"text", "video"}:
            segment_times.append(t_video)
        elif kind == "audio":
            segment_times.append(t_audio)
        else:
            raise RuntimeError(f"Unknown MiniMax H3 packed segment kind: {kind}")
    return segment_times


def _target_mask_timestep_rows(
    denoise_mask,
    audio_denoise_mask,
    sigma_video,
    t_video: float,
    t_audio: float,
    latent_t: int,
    latent_h: int,
    latent_w: int,
):
    video_time = t_video
    audio_time = t_audio
    video_rows = None
    audio_rows = None
    if denoise_mask is not None:
        mask_row_values = getattr(minimax_model, "mask_row_values", None)
        if mask_row_values is None:
            raise RuntimeError(
                "This ComfyUI build supplied a video denoise mask without the validated "
                "MiniMax H3 row-mask contract"
            )
        mask = mask_row_values(
            denoise_mask[0, 0].to(torch.float32), latent_t, latent_h, latent_w
        )
        if mask is not None:
            pin = max(t_video, minimax_model.VISUAL_COND_TIMESTEP)
            rows = (1.0 - mask * sigma_video.to(mask.device)).clamp(max=pin)
            if rows.unique().numel() == 1:
                video_time = float(rows[0])
            else:
                video_rows = rows
    if audio_denoise_mask is not None:
        mask = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((mask >= 1.0 - 1e-3).all()):
            pin = max(t_audio, minimax_model.AUDIO_COND_TIMESTEP)
            rows = (1.0 - mask * (1.0 - t_audio)).clamp(max=pin)
            if rows.unique().numel() == 1:
                audio_time = float(rows[0])
            else:
                audio_rows = rows
    return video_time, audio_time, video_rows, audio_rows


def _multikeyframe_forward(
    self,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    denoise_mask=None,
    audio_denoise_mask=None,
    **kwargs,
):
    transformer_options = transformer_options or {}
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = minimax_model.comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = build_packed_layout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )
        _repair_layout_positions(
            layout,
            list(payload.get("keyframes") or []),
            list(payload.get("refs") or []),
            int(payload.get("frame_count")),
        )

    shift_v = float(
        transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)
    )
    shift_a = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio)
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    visual_augs = [float(value) for value in payload[PAYLOAD_VISUAL_NOISE_AUGS_KEY]]
    audio_aug = float(
        payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
    )
    segment_times = _segment_timestep_plan(layout, t_v, t_a, visual_augs, audio_aug)

    # ComfyUI 187eda8 added row-wise target denoise-mask timesteps. Preserve that
    # behavior while independently scheduling the visual conditioning segments.
    video_time, audio_time, video_rows_t, audio_rows_t = _target_mask_timestep_rows(
        denoise_mask,
        audio_denoise_mask,
        sigma_v,
        t_v,
        t_a,
        latent_t,
        lat_h,
        lat_w,
    )

    def set_segment_time(kind: str, value: float) -> None:
        matches = [
            index
            for index, (_start, _stop, segment_kind) in enumerate(layout.segments)
            if segment_kind == kind
        ]
        if len(matches) != 1:
            raise RuntimeError(f"MiniMax H3 layout must contain one {kind} segment")
        segment_times[matches[0]] = value

    if not math.isclose(video_time, t_v, abs_tol=0.0):
        set_segment_time("video", video_time)
    if not math.isclose(audio_time, t_a, abs_tol=0.0):
        set_segment_time("audio", audio_time)

    unique_t = sorted(
        set(segment_times)
        | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
        | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    segment_tags = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "cond_audio": 2,
        "ref_audio": 2,
    }

    def rows_to_mod_index(rows, tag: int):
        levels = rows.unique()
        base = torch.tensor(
            [t_row[value] * 3 + tag for value in levels.tolist()],
            dtype=torch.long,
            device=rows.device,
        )
        return base[torch.searchsorted(levels, rows)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for segment_index, (start, stop, kind) in enumerate(layout.segments):
        row_base = t_row[segment_times[segment_index]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, stop - start + 1):
                if index == stop - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (start + run_start, start + index, row_base + int(tags[run_start]))
                    )
                    run_start = index
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append(
                (start, stop, rows_to_mod_index(video_rows_t, segment_tags[kind]))
            )
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append(
                (start, stop, rows_to_mod_index(audio_rows_t, segment_tags[kind]))
            )
        else:
            mod_segments.append((start, stop, row_base + segment_tags[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = _condition_rows_with_augs(self, payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = audio_offset = 0
    for start, stop, kind in layout.segments:
        count = stop - start
        if kind == "text":
            hidden[start:stop] = text_states
        elif kind in {"cond", "ref_img", "video"}:
            hidden[start:stop] = video_embed[video_offset : video_offset + count]
            video_offset += count
        else:
            hidden[start:stop] = audio_embed[audio_offset : audio_offset + count]
            audio_offset += count

    t_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = minimax_model.comfy.model_management.cast_to(self.adaln_t_table, device=device)
        position = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(
            table[lower], table[lower + 1], (position - lower).unsqueeze(1)
        )
    else:
        t_emb = self.time_embedder(t_values).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = minimax_model.comfy.model_prefetch.make_prefetch_queue(
        list(self.blocks), device, transformer_options
    )
    for index, block in enumerate(self.blocks):
        minimax_model.comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", index) in blocks_replace:

            def block_wrap(args, current_block=block):
                return {
                    "img": current_block(
                        args["img"],
                        args["t_emb"],
                        args["mod_segments"],
                        args["rope_freqs"],
                        transformer_options=args["transformer_options"],
                    )
                }

            hidden = blocks_replace[("double_block", index)](
                {
                    "img": hidden,
                    "t_emb": t_emb,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )["img"]
        else:
            hidden = block(
                hidden,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
    if prefetch_queue is not None:
        minimax_model.comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    video_index, (video_start, video_stop, _kind) = next(
        (index, segment)
        for index, segment in enumerate(layout.segments)
        if segment[2] == "video"
    )
    audio_index, (audio_start, audio_stop, _kind) = next(
        (index, segment)
        for index, segment in enumerate(layout.segments)
        if segment[2] == "audio"
    )
    if video_rows_t is not None:
        video_segment = (
            video_start,
            video_stop,
            rows_to_mod_index(video_rows_t, 0) // 3,
        )
    else:
        video_segment = (
            video_start,
            video_stop,
            t_row[segment_times[video_index]],
        )
    if audio_rows_t is not None:
        audio_segment = (
            audio_start,
            audio_stop,
            rows_to_mod_index(audio_rows_t, 0) // 3,
        )
    else:
        audio_segment = (
            audio_start,
            audio_stop,
            t_row[segment_times[audio_index]],
        )
    video_out, audio_out = self.final_layer(
        hidden, t_emb, video_segment, audio_segment
    )
    video_out = minimax_model.unpatchify_video(
        video_out,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio_out)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def patch_multikeyframe_model(model, require_per_condition_forward: bool):
    if not hasattr(model, "clone") or not hasattr(model, "add_object_patch"):
        raise ValueError("model is not a ComfyUI MODEL patcher")
    _assert_unmodified_packed_layout()
    patched = model.clone()
    base_model = patched.model
    if type(base_model).__name__ != "MiniMaxH3":
        diffusion_name = type(getattr(base_model, "diffusion_model", None)).__name__
        if diffusion_name != "MiniMaxH3Model":
            raise ValueError("MiniMax H3 Multi-Keyframe Advanced requires a MiniMax H3 MODEL")

    original_extra_conds = patched.get_model_object("extra_conds")
    if getattr(original_extra_conds, "_t8_long_video_patch_version", None) is not None:
        raise ValueError(
            "MiniMax H3 Multi-Keyframe Advanced and Long Video Conditioning cannot be "
            "stacked until their patch order has been validated"
        )
    existing_extra_version = getattr(
        original_extra_conds, "_t8_multikeyframe_patch_version", None
    )
    if existing_extra_version not in {None, MULTIKEYFRAME_PATCH_VERSION}:
        raise RuntimeError(
            "A different MiniMax H3 Multi-Keyframe Advanced extra_conds patch version "
            f"is active ({existing_extra_version}); stacking was refused"
        )
    extra_function = getattr(original_extra_conds, "__func__", original_extra_conds)
    if (
        getattr(original_extra_conds, "_t8_multikeyframe_patch_version", None) is None
        and getattr(extra_function, "__module__", None) != "comfy.model_base"
    ):
        raise RuntimeError(
            "An unrecognized MiniMax H3 extra_conds object patch is already active; "
            "Advanced multi-keyframe patching refused to avoid an unsafe patch-order conflict"
        )
    if getattr(original_extra_conds, "_t8_multikeyframe_patch_version", None) is None:

        def _patched_extra_conds(_self, **kwargs):
            core_kwargs = kwargs
            if int(kwargs.get(MULTIKEYFRAME_SCHEMA_KEY, 0) or 0) == MULTIKEYFRAME_SCHEMA:
                frame_count_value = kwargs.get("minimax_frame_count")
                if frame_count_value is None:
                    raise RuntimeError(
                        "Advanced MiniMax H3 conditioning is missing minimax_frame_count"
                    )
                core_kwargs = dict(kwargs)
                core_kwargs["minimax_keyframes"] = _keyframes_for_core(
                    list(kwargs.get("minimax_keyframes") or []),
                    int(frame_count_value),
                    bool(kwargs.get(NATIVE_LAYOUT_KEY, False)),
                )
            out = original_extra_conds(**core_kwargs)
            return repair_multikeyframe_payload(out, kwargs)

        _patched_extra_conds._t8_multikeyframe_patch_version = MULTIKEYFRAME_PATCH_VERSION
        patched.add_object_patch(
            "extra_conds", types.MethodType(_patched_extra_conds, base_model)
        )

    if require_per_condition_forward:
        diffusion_model = base_model.diffusion_model
        original_forward = patched.get_model_object("diffusion_model._forward")
        existing_forward_version = getattr(
            original_forward, "_t8_multikeyframe_patch_version", None
        )
        if existing_forward_version not in {None, MULTIKEYFRAME_PATCH_VERSION}:
            raise RuntimeError(
                "A different MiniMax H3 Multi-Keyframe Advanced _forward patch version "
                f"is active ({existing_forward_version}); stacking was refused"
            )
        if getattr(original_forward, "_t8_multikeyframe_patch_version", None) is None:
            forward_function = getattr(original_forward, "__func__", original_forward)
            if getattr(forward_function, "__module__", None) != minimax_model.__name__:
                raise RuntimeError(
                    "An unrecognized MiniMax H3 _forward object patch is already active; "
                    "per-keyframe strength patching refused"
                )
            forward_source = inspect.getsource(forward_function)
            forward_sha256 = hashlib.sha256(forward_source.encode("utf-8")).hexdigest()
            if forward_sha256 not in VALIDATED_FORWARD_SHA256S:
                raise RuntimeError(
                    "This ComfyUI build does not match the validated MiniMax H3 _forward "
                    "contract required for independent per-keyframe augmentation. "
                    f"Expected one of {sorted(VALIDATED_FORWARD_SHA256S)}, got "
                    f"{forward_sha256}."
                )
            required_forward_contract = (
                "layout = payload.get(\"layout\")",
                "cond_video_rows = self._cond_video_rows(payload, device)",
                "mod_segments = []",
                "rope_rotation_table",
                "self.final_layer",
            )
            missing = [
                snippet for snippet in required_forward_contract if snippet not in forward_source
            ]
            if missing:
                raise RuntimeError(
                    "This ComfyUI build changed the MiniMax H3 forward contract; "
                    "independent per-keyframe strength is disabled until revalidated. "
                    f"Missing contract marker(s): {missing}"
                )

            def _patched_forward(_self, *args, **kwargs):
                payload = kwargs.get("minimax_payload") or {}
                if int(payload.get(MULTIKEYFRAME_SCHEMA_KEY, 0) or 0) != MULTIKEYFRAME_SCHEMA:
                    return original_forward(*args, **kwargs)
                return _multikeyframe_forward(_self, *args, **kwargs)

            _patched_forward._t8_multikeyframe_patch_version = MULTIKEYFRAME_PATCH_VERSION
            patched.add_object_patch(
                "diffusion_model._forward",
                types.MethodType(_patched_forward, diffusion_model),
            )
    return patched


def _encode_keyframe(
    video_vae,
    image: torch.Tensor,
    width: int,
    height: int,
    frame_index: int,
    noise_aug: float,
    resize_mode: str,
):
    crop = "center" if resize_mode == "center_crop" else "disabled"
    resized = resize_image(image[:1], width, height, crop)
    latent = video_vae.encode(resized)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[2] != 1:
        raise ValueError(
            "MiniMax H3 keyframe VAE encode must produce a one-frame 5D video latent"
        )
    return resized, {
        "resolved_frame_index": frame_index,
        ACTUAL_FRAME_INDEX: frame_index,
        KEYFRAME_NOISE_AUG: float(noise_aug),
        "latent": latent,
    }


def _build_advanced_conditioning(
    clip,
    video_vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_type: str,
    audio_mode: str,
    audio_denoise_strength: float,
    add_source_as_reference: bool,
    prompt_primary_audio_ordinal: int,
    strict_prompt_tags: bool,
    ref_image_size: str,
    reference_video_policy: str,
    first_frame,
    last_frame,
    resolved_middle: list[dict],
    first_frame_noise_aug: float,
    last_frame_noise_aug: float,
    reference_visual_noise_aug: float,
    drive_audio=None,
    final_audio=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
):
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Requested canvas has {width * height:,} pixels and exceeds the configured "
            f"MiniMax H3 2.0MP cap of {MAX_PIXELS:,} pixels (1920x1088); reduce width/height"
        )
    for label, value in {
        "audio_denoise_strength": audio_denoise_strength,
        "first_frame_noise_aug": first_frame_noise_aug,
        "last_frame_noise_aug": last_frame_noise_aug,
        "reference_visual_noise_aug": reference_visual_noise_aug,
    }.items():
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be finite and between 0 and 1")
    if resolved_middle and (first_frame is None or last_frame is None):
        raise ValueError(
            "Advanced middle keyframes currently require both first_frame and last_frame; "
            "use the stable node for other task layouts"
        )

    ref_image_values = sorted_autogrow_values(ref_images)
    ref_video_entries = sorted_autogrow_items(ref_videos)
    ref_video_values = [value for _, value in ref_video_entries]
    ref_audio_values = sorted_autogrow_values(ref_audios)
    ref_video_audio_by_ordinal = dict(sorted_autogrow_items(ref_video_audios))
    if len(ref_image_values) > 9 or len(ref_video_values) > 3 or len(ref_audio_values) > 3:
        raise ValueError("MiniMax H3 reference limits are 9 pictures, 3 videos, and 3 standalone audios")
    video_ordinals = {ordinal for ordinal, _ in ref_video_entries}
    orphan_soundtracks = sorted(set(ref_video_audio_by_ordinal) - video_ordinals)
    if orphan_soundtracks:
        raise ValueError(
            "Reference-video soundtrack(s) have no same-numbered video: "
            + ", ".join(map(str, orphan_soundtracks))
        )

    mode = audio_mode.lower()
    if mode not in {"native", "reference_only", "lock_source", "remix_source"}:
        raise ValueError(f"Unknown audio mode: {audio_mode}")
    if mode != "native" and drive_audio is None:
        raise ValueError(f"Audio mode {audio_mode} requires drive_audio")
    if drive_audio is None and add_source_as_reference:
        add_source_as_reference = False

    latent, frame_count = empty_av_latent(width, height, length)
    _, template_audio = nested_av_parts(latent)
    native_layout = native_middle_keyframe_support()

    keyframe_specs = []
    if first_frame is not None:
        keyframe_specs.append(
            {
                "source": "first_frame",
                "image": first_frame,
                "frame_index": 0,
                "visual_noise_aug": float(first_frame_noise_aug),
                "resize_mode": "stretch",
                "requested_mode": "frame",
                "requested_value": 0,
            }
        )
    for item in resolved_middle:
        keyframe_specs.append({"source": f"middle_frame_{item['source_ordinal']}", **item})
    if last_frame is not None:
        keyframe_specs.append(
            {
                "source": "last_frame",
                "image": last_frame,
                "frame_index": frame_count - 1,
                "visual_noise_aug": float(last_frame_noise_aug),
                "resize_mode": "center_crop",
                "requested_mode": "frame",
                "requested_value": frame_count - 1,
            }
        )
    keyframe_specs.sort(key=lambda item: item["frame_index"])
    if len(keyframe_specs) > 9:
        raise ValueError("MiniMax H3 Advanced supports at most 9 total timeline keyframes")

    keyframes = []
    keyframe_images = []
    picture_labels: list[str] = []
    for spec in keyframe_specs:
        image, keyframe = _encode_keyframe(
            video_vae,
            spec["image"],
            width,
            height,
            int(spec["frame_index"]),
            float(spec["visual_noise_aug"]),
            str(spec["resize_mode"]),
        )
        keyframe_images.append(image)
        keyframes.append(keyframe)
        picture_labels.append(
            f"{spec['source']} (exact frame {spec['frame_index']}, "
            f"raw noise_aug {float(spec['visual_noise_aug']):.6f})"
        )

    real_ref_items: list[dict] = []
    real_ref_blocks: list[dict] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    for index, image in enumerate(ref_image_values, 1):
        resized, ref_width, ref_height = _resize_reference_image(
            image, width, height, ref_image_size
        )
        encoded = video_vae.encode(resized)
        real_ref_items.append({"type": "image", "data": resized})
        real_ref_blocks.append(
            {
                "kind": "image",
                "latent_h": ref_height // 16,
                "latent_w": ref_width // 16,
                "latent": encoded,
            }
        )
        picture_labels.append(f"ref_image_{index}")

    for index, (video_ordinal, frames) in enumerate(ref_video_entries, 1):
        if frames.ndim != 4 or frames.shape[0] < 5:
            raise ValueError(f"ref_video_{index} must contain at least 5 IMAGE frames")
        input_frame_count = int(frames.shape[0])
        if reference_video_policy == "official_2_to_15s" and not (
            2 * FPS <= input_frame_count <= 15 * FPS
        ):
            raise ValueError(
                f"ref_video_{index} has {input_frame_count} frames; official guidance is "
                "48-360 frames at 24fps"
            )
        source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
        canvas_width, canvas_height = adapt_canvas(source_width, source_height)
        if source_width * source_height < canvas_width * canvas_height:
            canvas_width = max(
                CANVAS_MULTIPLE,
                round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            canvas_height = max(
                CANVAS_MULTIPLE,
                round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
        frames = resize_image(frames, canvas_width, canvas_height)
        frames = frames[:frame_count]
        aligned_count = align_frame_count_down(int(frames.shape[0]))
        if aligned_count < 5:
            raise ValueError(f"ref_video_{index} is too short after 17n+5 alignment")
        frames = frames[:aligned_count]
        encoded_video = video_vae.encode(frames)

        soundtrack = ref_video_audio_by_ordinal.get(video_ordinal)
        encoded_soundtrack, soundtrack_t = None, 0
        if soundtrack is not None:
            encoded_soundtrack, soundtrack_t = _encode_reference_audio(audio_vae, soundtrack)
            real_ref_items.append({"type": "audio"})
            audio_labels.append(f"ref_video_audio_{video_ordinal}")
        sample_indices = list(range(0, frames.shape[0], FPS // 2))
        real_ref_items.append(
            {
                "type": "video",
                "data": frames[sample_indices],
                "timestamps": [sample_index / FPS for sample_index in sample_indices],
            }
        )
        real_ref_blocks.append(
            {
                "kind": "video_audio" if soundtrack_t else "video",
                "latent_t": int(encoded_video.shape[2]),
                "latent_h": canvas_height // 16,
                "latent_w": canvas_width // 16,
                "ref_audio_t": soundtrack_t,
                "latent": encoded_video,
                "audio_latent": encoded_soundtrack,
            }
        )
        video_labels.append(f"ref_video_{video_ordinal}")

    encoded_source = None
    source_audio_ordinal = 0
    if drive_audio is not None:
        encoded_source = fit_audio_latent(
            encode_audio_once(audio_vae, drive_audio), template_audio
        )
        if add_source_as_reference:
            real_ref_items.append({"type": "audio"})
            real_ref_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": int(encoded_source.shape[-1]),
                    "audio_latent": encoded_source,
                }
            )
            audio_labels.append("drive_audio (primary source)")
            source_audio_ordinal = len(audio_labels)

    for index, audio in enumerate(ref_audio_values, 1):
        encoded_audio, audio_t = _encode_reference_audio(audio_vae, audio)
        real_ref_items.append({"type": "audio"})
        real_ref_blocks.append(
            {"kind": "audio", "ref_audio_t": audio_t, "audio_latent": encoded_audio}
        )
        audio_labels.append(f"ref_audio_{index}")

    has_refs = bool(real_ref_blocks)
    resolved_task = resolve_task_type(task_type, first_frame, last_frame, has_refs)
    if resolved_task not in {"fl2va", "hybrid"}:
        raise ValueError(
            "Multi-Keyframe Advanced only supports FL2VA with both endpoints or Hybrid "
            "with both endpoints plus references; use the stable Conditioning and optional "
            "Visual Reference Strength node for T2VA/I2VA/L2VA/Ref2VA"
        )
    counts = {
        "pictures": len(picture_labels),
        "videos": len(video_labels),
        "audios": len(audio_labels),
    }
    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        counts,
        source_audio_ordinal=source_audio_ordinal,
        prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
        strict=strict_prompt_tags,
    )

    if keyframes and real_ref_blocks:
        hybrid_route = assert_hybrid_layout_contract()
        ref_items = [
            {"type": "image", "data": image} for image in keyframe_images
        ] + real_ref_items
        if hybrid_route == HYBRID_LAYOUT_LEGACY_SENTINEL:
            refs = [
                {"kind": HYBRID_KEYFRAME_SENTINEL, "latent": keyframe["latent"]}
                for keyframe in keyframes
            ] + real_ref_blocks
        else:
            refs = real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=ref_items)
    elif real_ref_blocks:
        refs = real_ref_blocks
        tokens = clip.tokenize(conditioned_prompt, minimax_ref_items=real_ref_items)
    else:
        refs = []
        tokens = clip.tokenize(conditioned_prompt, images=keyframe_images)

    conditioning = clip.encode_from_tokens_scheduled(tokens)
    keyframe_augs = [float(keyframe[KEYFRAME_NOISE_AUG]) for keyframe in keyframes]
    ref_visual_count = sum(
        ref.get("kind") in {"image", "video", "video_audio"} for ref in real_ref_blocks
    )
    visual_augs = keyframe_augs + [float(reference_visual_noise_aug)] * ref_visual_count
    values = {
        MULTIKEYFRAME_SCHEMA_KEY: MULTIKEYFRAME_SCHEMA,
        NATIVE_LAYOUT_KEY: native_layout,
        VISUAL_NOISE_AUGS_KEY: visual_augs,
        "minimax_visual_cond_noise_aug": float(reference_visual_noise_aug),
    }
    if keyframes:
        values.update({"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
    if refs:
        values["minimax_refs"] = refs
    conditioning = node_helpers.conditioning_set_values(conditioning, values)

    if mode == "lock_source":
        latent = replace_audio_latent(latent, encoded_source, 0.0)
    elif mode == "remix_source":
        latent = replace_audio_latent(latent, encoded_source, audio_denoise_strength)

    media_map = media_map_json(
        picture_labels, video_labels, audio_labels, source_audio_ordinal
    )
    report_lines = [
        f"task={resolved_task}",
        f"audio_mode={mode}",
        f"frames={frame_count} ({frame_count / FPS:.3f}s at 24fps)",
        f"pictures={len(picture_labels)}, videos={len(video_labels)}, audios={len(audio_labels)}",
        f"source_audio_tag={'<Audio ' + str(source_audio_ordinal) + '>' if source_audio_ordinal else 'none'}",
    ]
    report_lines.extend(f"warning: {warning}" for warning in prompt_warnings)
    output_audio = final_audio if final_audio is not None else drive_audio
    return (
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        "\n".join(report_lines),
        keyframe_specs,
        visual_augs,
        native_layout,
    )


def build_multikeyframe_conditioning(
    model,
    clip,
    video_vae,
    audio_vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_type: str = "auto",
    audio_mode: str = "lock_source",
    audio_denoise_strength: float = 0.35,
    add_source_as_reference: bool = True,
    prompt_primary_audio_ordinal: int = 1,
    strict_prompt_tags: bool = True,
    ref_image_size: str = "match",
    reference_video_policy: str = "official_2_to_15s",
    first_frame_noise_aug: float = DEFAULT_VISUAL_NOISE_AUG,
    last_frame_noise_aug: float = DEFAULT_VISUAL_NOISE_AUG,
    reference_visual_noise_aug: float = DEFAULT_VISUAL_NOISE_AUG,
    drive_audio=None,
    final_audio=None,
    first_frame=None,
    last_frame=None,
    keyframe_plan=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
):
    frame_count = temporal_shape(length)[0]
    resolved_middle = resolve_keyframe_plan(keyframe_plan, frame_count)
    default_strengths = all(
        math.isclose(float(value), DEFAULT_VISUAL_NOISE_AUG, abs_tol=0.0)
        for value in (
            first_frame_noise_aug,
            last_frame_noise_aug,
            reference_visual_noise_aug,
        )
    )
    if not resolved_middle and default_strengths:
        stable = build_conditioning(
            clip,
            video_vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            task_type,
            audio_mode,
            audio_denoise_strength,
            add_source_as_reference,
            prompt_primary_audio_ordinal,
            strict_prompt_tags,
            ref_image_size,
            reference_video_policy,
            drive_audio,
            final_audio,
            first_frame,
            last_frame,
            ref_images,
            ref_videos,
            ref_video_audios,
            ref_audios,
        )
        keyframes = []
        if first_frame is not None:
            keyframes.append(
                {
                    "ordinal": len(keyframes) + 1,
                    "source": "first_frame",
                    "frame_index": 0,
                    "seconds": 0.0,
                    "percent": 0.0,
                    "visual_noise_aug": DEFAULT_VISUAL_NOISE_AUG,
                }
            )
        if last_frame is not None:
            keyframes.append(
                {
                    "ordinal": len(keyframes) + 1,
                    "source": "last_frame",
                    "frame_index": frame_count - 1,
                    "seconds": (frame_count - 1) / FPS,
                    "percent": 100.0,
                    "visual_noise_aug": DEFAULT_VISUAL_NOISE_AUG,
                }
            )
        report = {
            "status": "stable_fast_path",
            "advanced_patch_applied": False,
            "model_cloned": False,
            "middle_keyframe_count": 0,
            "frame_count": frame_count,
            "stable_conditioning_report": stable[5],
            "warnings": [],
        }
        return (
            model,
            *stable[:5],
            json.dumps(keyframes, ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    (
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        stable_report,
        keyframe_specs,
        visual_augs,
        native_layout,
    ) = _build_advanced_conditioning(
        clip,
        video_vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        task_type,
        audio_mode,
        audio_denoise_strength,
        add_source_as_reference,
        prompt_primary_audio_ordinal,
        strict_prompt_tags,
        ref_image_size,
        reference_video_policy,
        first_frame,
        last_frame,
        resolved_middle,
        first_frame_noise_aug,
        last_frame_noise_aug,
        reference_visual_noise_aug,
        drive_audio,
        final_audio,
        ref_images,
        ref_videos,
        ref_video_audios,
        ref_audios,
    )
    require_per_condition_forward = len({float(value) for value in visual_augs}) > 1
    patched_model = patch_multikeyframe_model(model, require_per_condition_forward)

    keyframe_map = []
    for ordinal, spec in enumerate(keyframe_specs, 1):
        frame_index = int(spec["frame_index"])
        keyframe_map.append(
            {
                "ordinal": ordinal,
                "source": spec["source"],
                "requested_mode": spec["requested_mode"],
                "requested_value": spec["requested_value"],
                "frame_index": frame_index,
                "seconds": frame_index / FPS,
                "percent": frame_index / max(1, frame_count - 1) * 100.0,
                "visual_noise_aug": float(spec["visual_noise_aug"]),
                "resize_mode": spec["resize_mode"],
            }
        )
    _, latent_t, _ = temporal_shape(frame_count)
    latent_h, latent_w = height // 16, width // 16
    frame_rows = ((latent_h + 1) // 2) * ((latent_w + 1) // 2)
    added_rows = len(resolved_middle) * frame_rows
    warnings = [
        "Advanced/Experimental: middle-keyframe quality and 16GiB memory safety are not guaranteed.",
        "visual_noise_aug is a raw H3 conditioning value, not a calibrated linear strength.",
        (
            "added_rows_vs_target_video_rows_percent is a packed DiT sequence-row ratio, "
            "not a VRAM percentage; it excludes CLIP image work, refs, VAE peaks, allocator/"
            "offload behavior, and nonlinear attention interactions."
        ),
        "Do not stack this node with Long Video Conditioning or an external global H3 layout patch.",
    ]
    if require_per_condition_forward:
        warnings.append(
            "Independent per-condition noise/timestep routing is active and requires quality monotonicity validation."
        )
    if any(float(value) <= 0.95 for value in visual_augs):
        warnings.append(
            "A visual_noise_aug value at or below 0.950 is aggressive and may weaken identity, motion, or composition."
        )
    report = {
        "status": "experimental_advanced",
        "advanced_patch_applied": True,
        "model_cloned": True,
        "scoped_model_patch_version": MULTIKEYFRAME_PATCH_VERSION,
        "native_middle_layout_supported": native_layout,
        "scoped_position_repair": not native_layout,
        "per_condition_forward_patch": require_per_condition_forward,
        "frame_count": frame_count,
        "middle_keyframe_count": len(resolved_middle),
        "total_keyframe_count": len(keyframe_specs),
        "visual_condition_count": len(visual_augs),
        "frame_rows_per_keyframe": frame_rows,
        "added_middle_rows": added_rows,
        "added_rows_vs_target_video_rows_percent": (
            added_rows / max(1, latent_t * frame_rows) * 100.0
        ),
        "added_rows_estimate_scope": (
            "DiT packed visual-condition rows only; excludes CLIP image work, ordinary refs, "
            "VAE peaks, allocator/offload behavior, and nonlinear attention interactions. "
            "This value is not a VRAM percentage."
        ),
        "memory_safety_tier": "unproven_experimental",
        "recommended_default_visual_noise_aug": DEFAULT_VISUAL_NOISE_AUG,
        "stable_conditioning_report": stable_report,
        "warnings": warnings,
    }
    return (
        patched_model,
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        json.dumps(keyframe_map, ensure_ascii=False, indent=2),
        json.dumps(report, ensure_ascii=False, indent=2),
    )
