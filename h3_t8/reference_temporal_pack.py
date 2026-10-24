"""Explicit reference-only temporal sampling on the native H3 physical clock.

Independently implemented against Core PackedLayout. The phase recipe is
documented by VPack ee13c64 (Apache-2.0); no upstream implementation is vendored.
This changes neither generated latent, Qwen presentation nor reference audio.
"""
from __future__ import annotations

from copy import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import torch
from comfy import model_management
from comfy.ldm.minimax import model as native

from .nfe_run_contract_advanced import _ConditioningDigester, _feed


SCHEMA = "t8.h3.reference_temporal_pack.v1"
KEY = "t8_reference_temporal_pack"
PAIR_KEY = "t8_reference_temporal_pack_binding"
PROFILES = {"0/5": (0, 1, 2, 3, 4), "1/5": (0, 1, 2, 3),
            "2/5": (0, 1, 3), "3/5": (0, 3)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tensor_record(value):
    if (type(value) is not torch.Tensor or not value.is_floating_point()
            or value.is_meta or value.is_sparse or value.is_quantized):
        raise ValueError("Reference packing requires an actual dense floating tensor")
    raw = value.detach().contiguous().view(torch.uint8).reshape(-1)
    result = hashlib.sha256()
    for start in range(0, raw.numel(), 8 * 1024**2):
        model_management.throw_exception_if_processing_interrupted()
        block = raw[start:start + 8 * 1024**2].cpu()
        if not bool(torch.isfinite(block.view(value.dtype)).all()):
            raise ValueError("Reference packing received non-finite tensor content")
        result.update(block.numpy().tobytes())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": result.hexdigest()}


class _ExecutionConditionDigester(_ConditioningDigester):
    """Hash known content even beside opaque owners, without certifying them.

    The public persistent digester deliberately uses a nonce for unknown
    objects. This request-local inspector uses only their process identity;
    neither their repr/equality/call nor their internals are executed or trusted.
    """
    def _digest(self, value, hasher, path):
        kind = type(value)
        constant = (kind.__module__ == "comfy.conds" and kind.__qualname__ == "CONDConstant"
                    and hasattr(value, "cond"))
        known = value is None or isinstance(value, (torch.Tensor, bool, int, float, str, bytes,
                                                   Mapping, Sequence)) or constant
        if known:
            return super()._digest(value, hasher, path)
        self.opaque_paths.append(path)
        _feed(hasher, "opaque-process-selection-not-content-proof", str(id(value)).encode("ascii"))


def condition_identity(positive):
    inspector = _ExecutionConditionDigester(8 * 1024**2)
    result = inspector.digest(positive)
    return {"sha256": result, "opaque_paths": list(inspector.opaque_paths),
            "scope": "known_content_and_process_local_opaque_selection",
            "opaque_internal_state_verified": False, "portable_cache_identity": False}


def retained_indices(count, ratio="2/5"):
    if type(count) is not int or count < 1:
        raise ValueError("Reference latent time dimension must be a positive integer")
    if ratio not in PROFILES:
        raise ValueError("Unknown reference compression ratio; select 0/5, 1/5, 2/5 or 3/5")
    return [index for index in range(count) if index % 5 in PROFILES[ratio]]


def _video(ref):
    latent = ref.get("latent")
    if (type(latent) is not torch.Tensor or latent.ndim != 5 or latent.shape[0] != 1
            or min(latent.shape) < 1 or type(ref.get("latent_t")) is not int
            or type(ref.get("latent_h")) is not int or type(ref.get("latent_w")) is not int
            or ref["latent_t"] != latent.shape[2]
            or ref.get("latent_h") != latent.shape[3] or ref.get("latent_w") != latent.shape[4]
            or latent.shape[3] % 2 or latent.shape[4] % 2):
        raise ValueError("Reference video latent must match [1,C,T,H,W] and even spatial patch geometry")
    tensor_record(latent)
    return latent


def verify_ref(ref):
    record = ref.get(KEY)
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("Reference temporal-pack descriptor is missing or has a bad version")
    unsigned = dict(record)
    seal = unsigned.pop("sha256", None)
    if seal != digest(unsigned):
        raise ValueError("Reference temporal-pack descriptor SHA changed")
    indices = retained_indices(record.get("original_t"), record.get("ratio"))
    if record.get("indices") != indices or len(indices) != ref.get("latent_t"):
        raise ValueError("Reference temporal-pack indices do not match the actual time dimension")
    if tensor_record(_video(ref)) != record.get("packed_tensor"):
        raise ValueError("Reference temporal-pack latent content SHA changed")
    return record


def pack_conditioning(positive, ratio="2/5", *, phase="SINGLE"):
    if ratio not in PROFILES:
        retained_indices(1, ratio)
    if ratio == "0/5":
        return positive, {"schema": SCHEMA, "status": "disabled_identity", "ratio": ratio,
                          "portable_cache_identity": False}
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("Reference temporal-pack phase must be a nonempty label")
    source = condition_identity(positive)
    result, observations = [], []
    for row_number, (embedding, metadata) in enumerate(positive):
        if KEY in metadata:
            raise ValueError("Conditioning is already packed; branch from the original positive for a new stage")
        refs = metadata.get("minimax_refs")
        if refs is None:
            result.append([embedding, metadata])
            continue
        if type(refs) is not list or any(type(ref) is not dict for ref in refs):
            raise ValueError("Reference metadata must contain an ordered native reference list")
        updated_refs = []
        for ordinal, ref in enumerate(refs):
            if ref.get("kind") not in ("video", "video_audio"):
                updated_refs.append(ref)
                continue
            if KEY in ref:
                raise ValueError("Reference is already packed; use the original reference latent")
            original = _video(ref)
            indices = retained_indices(ref["latent_t"], ratio)
            packed = original.index_select(2, torch.tensor(indices, device=original.device))
            cursor, all_starts = 0, []
            for index in range(ref["latent_t"]):
                all_starts.append(cursor)
                cursor += native.FRAME_PER_TOKEN[index % 5]
            starts = [all_starts[index] for index in indices]
            record = {"schema": SCHEMA, "ratio": ratio, "original_t": ref["latent_t"],
                      "indices": indices, "original_pixel_frame_starts": starts,
                      "original_seconds_24fps": [index / 24. for index in starts],
                      "physical_rope_offsets": native._video_t_grid(ref["latent_t"], 0)[indices].tolist(),
                      "source_tensor": tensor_record(original), "packed_tensor": tensor_record(packed)}
            record["sha256"] = digest(record)
            updated_refs.append({**ref, "latent": packed, "latent_t": len(indices), KEY: record})
            observations.append({"conditioning_row": row_number, "reference_ordinal": ordinal,
                                 "descriptor": record})
        updated = {**metadata, "minimax_refs": updated_refs}
        result.append([embedding, updated])
    if not observations:
        return positive, {"schema": SCHEMA, "status": "no_video_references_original_retained",
                          "ratio": ratio, "portable_cache_identity": False}
    contract = {"schema": SCHEMA, "ratio": ratio, "phase": phase,
                "source_condition": source, "references": observations,
                "provider_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "portable_cache_identity": False}
    contract["sha256"] = digest(contract)
    result = [[embedding, {**metadata, KEY: contract}] for embedding, metadata in result]
    return result, contract


def physical_layout(layout, refs):
    """Select original full-layout rows; never renumber physical time.

    Existing compatible positional repairs are preserved as additive deltas.
    No payload, layout, tensor, global constructor or source reference is edited.
    """
    if not hasattr(layout, "signature") or len(layout.signature) != 5:
        raise ValueError("Reference temporal pack needs an actual native layout signature")
    dense_refs, records = [], []
    for ref in refs:
        if KEY in ref:
            record = verify_ref(ref)
            dense_refs.append({**ref, "latent_t": record["original_t"]})
            records.append(record)
        else:
            dense_refs.append(ref)
            records.append(None)
    # Keyframe geometry is supplied by the caller, not inferred from segment
    # counts or arbitrary labels. See repair_payload for the payload contract.
    return dense_refs, records


def repair_payload(payload, signature):
    refs = payload.get("refs") or []
    if not any(KEY in ref for ref in refs):
        return payload, {"applied": False, "reason": "no_packed_video"}
    compressed = payload.get("layout")
    if compressed is None or tuple(compressed.signature) != tuple(signature):
        compressed = native.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=refs)
    dense_refs, records = physical_layout(compressed, refs)
    dense = native.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=dense_refs)
    baseline = native.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=refs)
    if (compressed.segments != baseline.segments or compressed.seq_len != baseline.seq_len
            or compressed.position_ids.shape != baseline.position_ids.shape
            or not bool(torch.isfinite(compressed.position_ids).all())):
        raise ValueError("Reference temporal-pack actual layout has incompatible segment/position geometry")
    for name in ("img_pos", "img_update", "audio_pos", "audio_update"):
        actual, expected = getattr(compressed, name, None), getattr(baseline, name)
        if (type(actual) is not torch.Tensor or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected)):
            raise ValueError("Reference temporal-pack actual layout row map differs: " + name)
    segment_index = 1
    while segment_index < len(dense.segments) and dense.segments[segment_index][2] in ("cond", "cond_audio"):
        segment_index += 1
    selected = torch.ones(dense.seq_len, dtype=torch.bool)
    for ref, record in zip(dense_refs, records, strict=True):
        kind = ref.get("kind")
        if kind == "audio":
            segment_index += int(ref.get("ref_audio_t", 0) > 0)
            continue
        if kind in ("video", "video_audio") and ref.get("ref_audio_t", 0) > 0:
            segment_index += 1
        if kind not in ("image", "video", "video_audio"):
            raise ValueError("Actual reference kind has no native spatial segment API")
        start, stop, segment_kind = dense.segments[segment_index]
        if segment_kind != "ref_img":
            raise ValueError("Reference temporal-pack reference/layout order mismatch")
        if record is not None:
            rows = (ref["latent_h"] // 2) * (ref["latent_w"] // 2)
            if stop - start != ref["latent_t"] * rows:
                raise ValueError("Reference temporal-pack rows are not frame aligned")
            keep = torch.zeros(ref["latent_t"], dtype=torch.bool)
            keep[record["indices"]] = True
            selected[start:stop] = keep.repeat_interleave(rows)
        segment_index += 1
    original_positions = dense.position_ids[selected]
    if original_positions.shape != baseline.position_ids.shape:
        raise ValueError("Reference temporal-pack selected rows do not match the actual compressed layout")
    repaired = copy(compressed)
    # Preserve compatible upstream keyframe/position repairs and spatial coords.
    repaired.position_ids = original_positions + (compressed.position_ids - baseline.position_ids)
    return {**payload, "layout": repaired}, {
        "applied": True, "original_sequence_rows": dense.seq_len,
        "packed_sequence_rows": repaired.seq_len, "signature": list(signature),
        "position_sha256": tensor_record(repaired.position_ids)["sha256"],
        "generated_and_audio_rows_removed": 0,
        "original_latent_indices": [None if row is None else row["indices"] for row in records]}
