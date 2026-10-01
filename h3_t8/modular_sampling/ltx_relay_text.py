"""Process-local LTX Relay text segments, encoded by the selected CLIP.

Token boundaries are measured from actual selected-CLIP embeddings, not
guessed from Unicode character offsets or H3 packed-token conventions.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import torch

from .ltx_relay_plan import validate_ltx_relay_plan


BINDING_TYPE = "T8_LTX_RELAY_TEXT_BINDING"


def tensor_digest(value: torch.Tensor) -> str:
    """Exact logical tensor identity in bounded 4 MiB host slices."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("LTX Relay conditioning must be a floating tensor")
    hasher = hashlib.sha256()
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(str(value.dtype).encode("ascii"))
    raw = value.detach().contiguous().view(torch.uint8).reshape(-1)
    for offset in range(0, raw.numel(), 4 * 1024 * 1024):
        host = raw[offset:offset + 4 * 1024 * 1024].to(device="cpu")
        hasher.update(memoryview(host.numpy()).cast("B"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class LTXRelayTextBinding:
    plan: dict
    token_spans: tuple[tuple[int, int], ...]  # global, then each local event
    token_count: int
    embedding_sha256s: tuple[str, ...]
    schedule_windows: tuple[tuple[float | None, float | None], ...]
    unprocessed_ltxav_embeds: bool
    metadata_unverified: tuple[str, ...]


def _entry(value, *, segment, schedule):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not isinstance(value[1], Mapping)):
        raise ValueError(f"LTX Relay {segment} schedule {schedule} is not CLIP conditioning")
    embedding, metadata = value
    if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 3
            or embedding.shape[0] != 1 or min(embedding.shape) < 1
            or not embedding.is_floating_point() or not torch.isfinite(embedding).all()):
        raise ValueError(f"LTX Relay {segment} schedule {schedule} has invalid text embeddings")
    return embedding, dict(metadata)


def _window(metadata):
    start, end = metadata.get("clip_start_percent"), metadata.get("clip_end_percent")
    return (float(start) if start is not None else None,
            float(end) if end is not None else None)


def _same_metadata(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if a is b:
        return True
    try:
        result = a == b
        return isinstance(result, bool) and result
    except (TypeError, ValueError, RuntimeError):
        return False


def _mask(metadata, embedding):
    mask = metadata.get("attention_mask")
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (1, 2):
        raise ValueError("LTX Relay CLIP attention_mask must be [tokens] or [1,tokens]")
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != embedding.shape[:2]:
        raise ValueError("LTX Relay CLIP attention_mask does not match text tokens")
    if not torch.isfinite(mask).all():
        raise ValueError("LTX Relay CLIP attention_mask is non-finite")
    return mask


def encode_ltx_relay_conditioning(clip, ltx_relay_plan, max_text_tokens=2048):
    plan = validate_ltx_relay_plan(ltx_relay_plan)
    if not callable(getattr(clip, "tokenize", None)) or not callable(
            getattr(clip, "encode_from_tokens_scheduled", None)):
        raise TypeError("LTX Relay requires the selected CLIP tokenizer and scheduled encoder")
    if type(max_text_tokens) is not int or not 1 <= max_text_tokens <= 8192:
        raise ValueError("LTX Relay max_text_tokens must be within 1..8192")
    texts = ["Global scene: " + plan["global_prompt"]] + [
        f"Event {event['event_index']}: {event['local_prompt']}" for event in plan["events"]
    ]
    encoded = []
    for index, text in enumerate(texts):
        result = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        if not isinstance(result, (list, tuple)) or not result:
            raise ValueError(f"LTX Relay segment {index} returned no CLIP conditioning")
        encoded.append([_entry(item, segment=index, schedule=i)
                        for i, item in enumerate(result)])
    schedule_count = len(encoded[0])
    if any(len(part) != schedule_count for part in encoded):
        raise ValueError("LTX Relay CLIP schedules differ between text segments")

    conditions = []
    spans = None
    unverified = set()
    windows = []
    for schedule in range(schedule_count):
        parts = [segment[schedule] for segment in encoded]
        tensors = [part[0] for part in parts]
        metas = [part[1] for part in parts]
        reference = tensors[0]
        if any(t.shape[2] != reference.shape[2] or t.device != reference.device
               or t.dtype != reference.dtype for t in tensors):
            raise ValueError("LTX Relay selected CLIP returned incompatible segment embeddings")
        first_window = _window(metas[0])
        if any(_window(meta) != first_window for meta in metas):
            raise ValueError("LTX Relay CLIP schedule windows differ between segments")
        windows.append(first_window)
        raw_flag = metas[0].get("unprocessed_ltxav_embeds", False)
        if any(meta.get("unprocessed_ltxav_embeds", False) != raw_flag for meta in metas):
            raise ValueError("LTX Relay segments disagree on LTXAV embedding projection")
        if schedule and bool(raw_flag) != bool(conditions[0][1].get("unprocessed_ltxav_embeds", False)):
            raise ValueError("LTX Relay schedules disagree on LTXAV embedding projection")
        lengths = [int(t.shape[1]) for t in tensors]
        total = sum(lengths)
        if total > max_text_tokens:
            raise ValueError(f"LTX Relay has {total} tokens, exceeding max_text_tokens={max_text_tokens}")
        cursor = 0
        current_spans = []
        for length in lengths:
            current_spans.append((cursor, cursor + length))
            cursor += length
        if spans is None:
            spans = tuple(current_spans)
        elif spans != tuple(current_spans):
            raise ValueError("LTX Relay scheduled text lengths differ; one binding cannot cover them")

        metadata = dict(metas[0])
        for segment_meta in metas[1:]:
            for key in set(metadata) | set(segment_meta):
                if key not in {"pooled_output", "attention_mask"} and not _same_metadata(
                        metadata.get(key), segment_meta.get(key)):
                    unverified.add(key)
        masks = [_mask(meta, tensor) for tensor, meta in parts]
        if any(mask is not None for mask in masks):
            dtype = next(mask.dtype for mask in masks if mask is not None)
            device = reference.device
            metadata["attention_mask"] = torch.cat([
                mask.to(device=device, dtype=dtype) if mask is not None else
                torch.ones((1, tensor.shape[1]), device=device, dtype=dtype)
                for mask, tensor in zip(masks, tensors)
            ], dim=1)
        merged = torch.cat(tensors, dim=1)
        conditions.append([merged, metadata])

    binding = LTXRelayTextBinding(
        plan=plan, token_spans=spans, token_count=spans[-1][1],
        embedding_sha256s=tuple(tensor_digest(entry[0]) for entry in conditions),
        schedule_windows=tuple(windows),
        unprocessed_ltxav_embeds=bool(conditions[0][1].get("unprocessed_ltxav_embeds", False)),
        metadata_unverified=tuple(sorted(unverified)),
    )
    report = {"status": "encoded_bound_not_applied", "plan_hash": plan["plan_hash"],
              "token_spans": binding.token_spans, "token_count": binding.token_count,
              "schedule_windows": windows, "selected_clip_used": True,
              "metadata_unverified": binding.metadata_unverified,
              "local_pooled_outputs_not_merged": any("pooled_output" in meta
                                                      for part in encoded[1:] for _, meta in part),
              "attention_applied": False, "portable_identity": False,
              "note": "Separate selected-CLIP encodings; later LTX connector registers may append keys."}
    return conditions, binding, json.dumps(report, ensure_ascii=False)


def validate_ltx_relay_binding(binding, positive, ltx_latent=None):
    if type(binding) is not LTXRelayTextBinding:
        raise TypeError("LTX Relay needs its live text binding")
    validate_ltx_relay_plan(binding.plan, ltx_latent)
    if not isinstance(positive, (list, tuple)) or len(positive) != len(binding.embedding_sha256s):
        raise ValueError("LTX Relay positive conditioning schedule differs from binding")
    for index, item in enumerate(positive):
        embedding, metadata = _entry(item, segment="positive", schedule=index)
        if embedding.shape[1] != binding.token_count or tensor_digest(embedding) != binding.embedding_sha256s[index]:
            raise ValueError("LTX Relay positive conditioning is not the encoded binding")
        if _window(metadata) != binding.schedule_windows[index] or bool(
                metadata.get("unprocessed_ltxav_embeds", False)) != binding.unprocessed_ltxav_embeds:
            raise ValueError("LTX Relay positive conditioning metadata changed its text contract")
    return binding
