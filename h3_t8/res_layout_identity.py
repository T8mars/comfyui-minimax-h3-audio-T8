"""Bind actual processed H3 position rows to their source condition geometry.

Reconstruction uses an authenticated original Core constructor on a private
instance. It does not run Sol's constructor hook or mutate its global lookups.
"""
import hashlib
import math
from pathlib import Path
from types import FunctionType

import torch
from comfy import conds
from comfy.ldm.minimax import model as core_h3

from .nfe_run_contract_advanced import _ConditioningDigester
from .patch_stack_policy import UnverifiedModelStack
from .res_kitchen_identity import _SourceAudit, _digest, _require
from .vdn_attention_compat import _factory_closure


_ROWS_LIMIT = 1048576
_FIELDS = ("position_ids", "img_pos", "img_update", "audio_pos", "audio_update")


def _constructor(audit):
    constructor = core_h3.PackedLayout.__init__
    if type(constructor) is not FunctionType:
        raise UnverifiedModelStack("RES position constructor is opaque")
    if constructor.__globals__ is not vars(core_h3):
        from . import sol_attn_minimax_v2 as sol
        from .res_sol_identity import _SOL_SOURCE_SHA256
        _require(hashlib.sha256(Path(sol.__file__).read_bytes()).hexdigest() == _SOL_SOURCE_SHA256,
                 "position constructor's Sol source changed")
        closure = _factory_closure(constructor, sol._patch_packed_layout, "__init__")
        _require(closure is not None and set(closure) == {"original_init"},
                 "position constructor has an unknown wrapper")
        audit.function(constructor, sol, dependencies=False)
        constructor = closure["original_init"]
    _require(constructor.__qualname__ == "PackedLayout.__init__",
             "position constructor delegates to another computation")
    audit.function(constructor, core_h3, dependencies=False)
    _require(core_h3.torch is torch and core_h3.math is math, "position constructor dependencies changed")
    for name in ("_frame_grid", "_axis_from_sqrt_area", "_ref_t_span", "_audio_grid",
                 "_video_grid", "_video_t_grid", "_video_t_spans"):
        audit.function(getattr(core_h3, name), core_h3, dependencies=False)
    return constructor


def _count_rows(signature, payload):
    text, vt, h, w, at = signature
    frame_rows = (h // 2) * (w // 2)
    count = text + vt * frame_rows + at * 2
    for name in ("refs", "keyframes"):
        values = payload.get(name) or []
        _require(type(values) in (list, tuple) and len(values) <= 64,
                 "position reference/keyframe collection exceeds this bounded inspector")
        for value in values:
            _require(type(value) is dict, "position source entry is opaque")
            if name == "keyframes":
                index = value["resolved_frame_index"]
                _require(type(index) in (int, float) and math.isfinite(index), "position keyframe index is not finite")
                visual, audio = value.get("latent"), value.get("audio_latent")
                if visual is not None:
                    _require(type(visual) is torch.Tensor and visual.ndim == 5,
                             "position keyframe latent is opaque")
                    count += visual.shape[2] * frame_rows
                if audio is not None:
                    _require(type(audio) is torch.Tensor and audio.ndim >= 1,
                             "position audio keyframe latent is opaque")
                    count += audio.shape[-1] * 2
            else:
                kind = value["kind"]
                _require(kind in ("image", "audio", "video", "video_audio"), "position reference kind changed")
                if kind != "audio":
                    rh, rw = value["latent_h"], value["latent_w"]
                    _require(type(rh) is int and type(rw) is int and rh >= 2 and rw >= 2,
                             "position reference spatial geometry is malformed")
                    rvt = 1 if kind == "image" else value["latent_t"]
                    _require(type(rvt) is int and rvt > 0, "position reference time geometry is malformed")
                    count += rvt * (rh // 2) * (rw // 2)
                if kind != "image":
                    rat = value["ref_audio_t"]
                    _require(type(rat) is int and rat >= 0, "position reference audio geometry is malformed")
                    count += rat * 2
    _require(count <= _ROWS_LIMIT, "position reconstruction exceeds its bounded adapter; ordinary sampling remains available")
    return count


def _inspect(guider, chunk_bytes):
    rows = getattr(guider, "conds", None)
    _require(type(rows) is dict and bool(rows), "processed position conditions are unavailable")
    audit = _SourceAudit()
    constructor = _constructor(audit)
    for name in ("process_cond", "concat"):
        audit.function(getattr(conds.CONDConstant, name), conds, dependencies=False)
    contracts = {}
    for branch, values in sorted(rows.items()):
        _require(type(branch) is str and type(values) is list and bool(values), "processed condition rows changed")
        contracts[branch] = []
        for row in values:
            _require(type(row) is dict and type(row.get("model_conds")) is dict, "processed condition entry changed")
            selected = row["model_conds"]
            payload_cond = selected.get("minimax_payload")
            shapes_cond = selected.get("latent_shapes")
            context_cond = selected.get("c_crossattn")
            _require(type(payload_cond) is conds.CONDConstant and type(shapes_cond) is conds.CONDConstant
                     and type(context_cond) is conds.CONDRegular, "processed position condition owner changed")
            payload, shapes, context = payload_cond.cond, shapes_cond.cond, context_cond.cond
            _require(type(payload) is dict and type(shapes) in (list, tuple) and len(shapes) == 2
                     and all(type(shape) in (list, tuple, torch.Size) for shape in shapes)
                     and len(shapes[0]) == 5 and len(shapes[1]) >= 1
                     and type(context) is torch.Tensor and context.ndim == 3 and context.shape[0] == 1,
                     "processed position payload geometry changed")
            _require(all(type(number) is int and number > 0 for shape in shapes for number in shape),
                     "processed latent shape values changed")
            vs = shapes[0]
            signature = (context.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2, shapes[1][-1])
            expected_rows = _count_rows(signature, payload)
            actual = payload.get("layout")
            _require(type(actual) is core_h3.PackedLayout and actual.signature == signature,
                     "actual processed layout signature differs from its condition")
            expected = object.__new__(core_h3.PackedLayout)
            constructor(expected, *signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            _require(actual.seq_len == expected_rows == expected.seq_len and actual.segments == expected.segments,
                     "actual position segment table differs from native condition reconstruction")
            for field in _FIELDS:
                tensor, reference = getattr(actual, field), getattr(expected, field)
                _require(type(tensor) is torch.Tensor and tensor.dtype == reference.dtype
                         and tensor.shape == reference.shape and torch.equal(tensor.detach().cpu(), reference),
                         "actual position content differs from native condition reconstruction: " + field)
            semantic = {key: value for key, value in payload.items() if key != "layout"}
            semantic.update(actual_position_fields={name: getattr(actual, name) for name in _FIELDS},
                            actual_context=context, latent_shapes=shapes, signature=signature, segments=actual.segments)
            digester, hasher = _ConditioningDigester(chunk_bytes), hashlib.sha256()
            digester._digest(semantic, hasher, "$.processed_position_condition")
            _require(not digester.opaque_paths, "processed position payload has an opaque semantic owner")
            contracts[branch].append({"sha256": hasher.hexdigest(), "signature": list(signature),
                                      "sequence_rows": actual.seq_len})
    result = {"schema": "t8.minimax_h3.RES_processed_position.v1", "branches": contracts,
              "sources": audit.sources, "executables": audit.functions,
              "clock_constants": {"FRAME_RESCALE": core_h3.FRAME_RESCALE, "FRAME_PER_TOKEN": core_h3.FRAME_PER_TOKEN},
              "all_position_fields_and_segments_verified": True, "portable_condition_layout": True,
              "original_layout_Sol_cache_or_conditions_changed": False, "reconstruction_row_limit": _ROWS_LIMIT,
              "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["sha256"] = _digest(result)
    return result


def inspect_processed_layouts(guider, chunk_bytes=8 * 1024**2):
    try:
        return _inspect(guider, chunk_bytes)
    except UnverifiedModelStack:
        raise
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise UnverifiedModelStack("RES processed position identity unavailable: " + str(error)) from error
