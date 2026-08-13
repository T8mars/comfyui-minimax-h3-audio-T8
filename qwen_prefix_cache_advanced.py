from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import threading
import time
from typing import Any

import torch


QWEN_PREFIX_CACHE_SCHEMA = "t8.minimax_h3.qwen_reference_prefix_cache.v1"
SUPPORTED_CORE_CONTRACTS = {
    (
        "bed1e84fba459099df310aed267fb2c51bbd8a768b9727767300225afe28d361",
        "441453d9e244ac56caa2e697ac6b37afed6463c929eebc19959a54e28423ef20",
        "be8eb446e143c6ed0c50b653390f2286db5c65ddeabfc8d991b0bd91727ce846",
        "19bae67ffee1e8a6cf3f2b2bbd96dae540bc29da6b412d683f51dcde10e23d0e",
        "5af2c49496af674f99dbfda4d1e2bd7873ca058e22d5765f6bc5d423f261ff40",
        "490c7189de876492ccef7be6520a8dc0f9c7225225110ad6d2de5186adb76f4c",
    )
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _source_hash(value: Any) -> str | None:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def core_contract() -> dict[str, Any]:
    try:
        from comfy.text_encoders.llama import Attention, Llama2_
        from comfy.text_encoders.minimax import (
            MiniMaxH3ClipModel,
            MiniMaxH3Tokenizer,
            MiniMaxQwen3VL,
        )
        from comfy.text_encoders.qwen3vl import Qwen3VL
    except Exception as error:
        return {
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
            "hashes": None,
        }
    values = (
        _source_hash(MiniMaxH3Tokenizer.tokenize_with_weights),
        _source_hash(MiniMaxH3ClipModel.encode_token_weights),
        _source_hash(MiniMaxQwen3VL.forward),
        _source_hash(Qwen3VL.build_image_inputs),
        _source_hash(Llama2_.forward),
        _source_hash(Attention.forward),
    )
    names = (
        "minimax_tokenizer",
        "minimax_clip_encode",
        "minimax_qwen_forward",
        "qwen_image_inputs",
        "llama_forward",
        "attention_forward",
    )
    return {
        "supported": None not in values and values in SUPPORTED_CORE_CONTRACTS,
        "hashes": dict(zip(names, values, strict=True)),
        "contract": "H3 reference presentation is a strict causal prefix before prompt text",
    }


def _hash_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(value)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))


def _contains_visual(tokens: Mapping[str, Any]) -> bool:
    return any(
        isinstance(pair[0], Mapping) and pair[0].get("type") == "image"
        for row in _token_pairs(tokens)
        for pair in row
    )


def fingerprint_tokens(tokens: Any, model_identity: str) -> str:
    digest = hashlib.sha256()
    digest.update(QWEN_PREFIX_CACHE_SCHEMA.encode("ascii"))
    digest.update(model_identity.encode("utf-8"))
    _hash_value(digest, tokens)
    return digest.hexdigest()


def _token_pairs(tokens: Mapping[str, Any]) -> list[list[tuple[Any, ...]]]:
    if set(tokens) != {"qwen3vl_32b"}:
        raise ValueError("Qwen prefix cache accepts only native MiniMax H3 qwen3vl_32b tokens")
    batches = tokens["qwen3vl_32b"]
    if not isinstance(batches, list) or len(batches) != 1:
        raise ValueError("Qwen prefix cache supports exactly one token batch")
    for pair in batches[0]:
        if len(pair) < 2 or float(pair[1]) != 1.0:
            raise ValueError("Qwen prefix cache requires unweighted native H3 tokens")
    return batches


def prefix_split_matches_full(full_tokens, prefix_tokens, suffix_tokens) -> bool:
    full_rows = _token_pairs(full_tokens)
    prefix_rows = _token_pairs(prefix_tokens)
    suffix_rows = _token_pairs(suffix_tokens)
    if not (len(full_rows) == len(prefix_rows) == len(suffix_rows) == 1):
        return False
    full_digest = hashlib.sha256()
    split_digest = hashlib.sha256()
    _hash_value(full_digest, full_rows[0])
    _hash_value(split_digest, prefix_rows[0] + suffix_rows[0])
    return full_digest.digest() == split_digest.digest()


def _plain_token_rows(tokens: Mapping[str, Any]) -> list[list[Any]]:
    return [[pair[0] for pair in row] for row in _token_pairs(tokens)]


class H3PrefixTokens(dict):
    def __init__(
        self,
        full_tokens: Mapping[str, Any],
        prefix_tokens: Mapping[str, Any] | None,
        suffix_tokens: Mapping[str, Any],
        prefix_fingerprint: str | None,
    ):
        super().__init__(full_tokens)
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.prefix_fingerprint = prefix_fingerprint


@dataclass
class PrefixCacheEntry:
    key: str
    hidden: torch.Tensor
    key_values: list[tuple[torch.Tensor, torch.Tensor, int]]
    next_position: torch.Tensor
    token_tags: torch.Tensor
    bytes: int
    prefix_tokens: int
    created_unix: float


def _entry_bytes(hidden, key_values, next_position, tags) -> int:
    tensors = [hidden, next_position, tags]
    for key, value, _ in key_values:
        tensors.extend((key, value))
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors)


def _cpu_detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def build_prefix_entry(
    clip_model: Any,
    prefix_tokens: Mapping[str, Any],
    key: str,
) -> PrefixCacheEntry:
    from comfy.text_encoders.minimax import token_tags_from_embeds_info

    token_rows = _plain_token_rows(prefix_tokens)
    device = clip_model.execution_device
    if device is None:
        device = clip_model.transformer.get_input_embeddings().weight.device
    embeds, attention_mask, num_tokens, embeds_info = clip_model.process_tokens(token_rows, device)
    if embeds.shape[0] != 1 or not bool(torch.all(attention_mask == 1)):
        raise ValueError("Qwen prefix cache requires one unpadded native H3 prefix")
    qwen = clip_model.transformer
    llama = qwen.model
    position_ids, visual_pos_masks, deepstack = qwen.build_image_inputs(embeds, embeds_info)
    if position_ids is None or visual_pos_masks is None or deepstack is None:
        raise ValueError("Qwen prefix cache requires at least one visual H3 reference")
    prefix_len = int(embeds.shape[1])
    config = llama.config
    kv_dtype = embeds.dtype
    past = [
        (
            torch.empty(
                (1, config.num_key_value_heads, prefix_len, config.head_dim),
                device=device,
                dtype=kv_dtype,
            ),
            torch.empty(
                (1, config.num_key_value_heads, prefix_len, config.head_dim),
                device=device,
                dtype=kv_dtype,
            ),
            0,
        )
        for _ in llama.layers
    ]
    output = llama.forward(
        None,
        embeds=embeds,
        attention_mask=None,
        past_key_values=past,
        position_ids=position_ids,
        deepstack_embeds=deepstack,
        visual_pos_masks=visual_pos_masks,
        embeds_info=embeds_info,
        intermediate_output=None,
        final_layer_norm_intermediate=False,
        dtype=torch.float32,
        num_tokens=num_tokens,
    )
    if len(output) < 3 or len(output[2]) != len(llama.layers):
        raise RuntimeError("Qwen prefix build did not return one KV entry per language layer")
    hidden = _cpu_detached(output[0])
    key_values = [
        (_cpu_detached(k[:, :, :prefix_len]), _cpu_detached(v[:, :, :prefix_len]), prefix_len)
        for k, v, index in output[2]
        if int(index) == prefix_len
    ]
    if len(key_values) != len(llama.layers):
        raise RuntimeError("Qwen prefix KV lengths do not match the prefix length")
    next_position = _cpu_detached(position_ids[:, -1] + 1)
    tags = _cpu_detached(token_tags_from_embeds_info(prefix_len, embeds_info))
    size = _entry_bytes(hidden, key_values, next_position, tags)
    return PrefixCacheEntry(
        key=key,
        hidden=hidden,
        key_values=key_values,
        next_position=next_position,
        token_tags=tags,
        bytes=size,
        prefix_tokens=prefix_len,
        created_unix=time.time(),
    )


def encode_suffix_from_entry(
    clip_model: Any,
    suffix_tokens: Mapping[str, Any],
    entry: PrefixCacheEntry,
) -> tuple[torch.Tensor, torch.Tensor]:
    from comfy.ldm.modules.attention import optimized_attention_for_device

    token_rows = _plain_token_rows(suffix_tokens)
    device = clip_model.execution_device
    if device is None:
        device = clip_model.transformer.get_input_embeddings().weight.device
    embeds, attention_mask, _num_tokens, embeds_info = clip_model.process_tokens(token_rows, device)
    if embeds_info or embeds.shape[0] != 1 or not bool(torch.all(attention_mask == 1)):
        raise ValueError("Qwen prefix-cache suffix must be one unpadded text-only prompt")
    qwen = clip_model.transformer
    llama = qwen.model
    suffix_len = int(embeds.shape[1])
    prefix_len = int(entry.prefix_tokens)
    next_position = entry.next_position.to(device=device)
    position_ids = next_position[:, None] + torch.arange(
        suffix_len,
        device=device,
        dtype=next_position.dtype,
    )[None, :]
    freqs_cis = llama.compute_freqs_cis(position_ids, device)
    minimum = torch.finfo(embeds.dtype).min / 4
    mask = torch.zeros(
        (suffix_len, prefix_len + suffix_len),
        dtype=embeds.dtype,
        device=device,
    )
    mask[:, prefix_len:] = torch.full(
        (suffix_len, suffix_len),
        minimum,
        dtype=embeds.dtype,
        device=device,
    ).triu_(1)
    optimized_attention = optimized_attention_for_device(device, mask=True, small_input=True)

    hidden = embeds
    for layer_index, layer in enumerate(llama.layers):
        prefix_key, prefix_value, index = entry.key_values[layer_index]
        hidden, _ = layer(
            x=hidden,
            attention_mask=mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=(
                prefix_key.to(device=device),
                prefix_value.to(device=device),
                int(index),
            ),
        )
    if llama.norm is not None:
        hidden = llama.norm(hidden)
    output = torch.cat((entry.hidden.to(device=device), hidden), dim=1).float()
    tags = torch.cat(
        (
            entry.token_tags.to(device=device),
            torch.ones(suffix_len, dtype=torch.long, device=device),
        )
    )
    return output, tags


class QwenReferencePrefixCache:
    def __init__(self, max_entries: int, maximum_cache_mib: float):
        if max_entries < 1 or max_entries > 16:
            raise ValueError("max_entries must be between 1 and 16")
        if not math.isfinite(maximum_cache_mib) or maximum_cache_mib < 64 or maximum_cache_mib > 65536:
            raise ValueError("maximum_cache_mib must be between 64 and 65536")
        self.max_entries = int(max_entries)
        self.maximum_bytes = int(maximum_cache_mib * 1024**2)
        self.entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.rejected_oversize = 0
        self.lock = threading.RLock()

    def get(self, key: str) -> PrefixCacheEntry | None:
        with self.lock:
            value = self.entries.pop(key, None)
            if value is None:
                self.misses += 1
                return None
            self.entries[key] = value
            self.hits += 1
            return value

    def put(self, value: PrefixCacheEntry) -> bool:
        with self.lock:
            if value.bytes > self.maximum_bytes:
                self.rejected_oversize += 1
                return False
            self.entries.pop(value.key, None)
            self.entries[value.key] = value
            while len(self.entries) > self.max_entries or self.total_bytes() > self.maximum_bytes:
                self.entries.popitem(last=False)
            return True

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()

    def total_bytes(self) -> int:
        return sum(value.bytes for value in self.entries.values())

    def report(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema": QWEN_PREFIX_CACHE_SCHEMA,
                "entries": len(self.entries),
                "entry_keys": list(self.entries),
                "total_mib": self.total_bytes() / 1024**2,
                "maximum_cache_mib": self.maximum_bytes / 1024**2,
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "bypasses": self.bypasses,
                "rejected_oversize": self.rejected_oversize,
                "storage": "bounded_cpu_memory_lru",
                "disk_writes": False,
                "memory_safe_claim": False,
                "bit_exact_claim": False,
            }


def _model_identity(clip: Any, contract: Mapping[str, Any]) -> str:
    patcher = getattr(clip, "patcher", None)
    model = getattr(patcher, "model", None)
    patch_uuid = getattr(patcher, "patches_uuid", None)
    payload = {
        "model_object_id": id(model),
        "patch_uuid": None if patch_uuid is None else str(patch_uuid),
        "core_hashes": contract.get("hashes"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _unwrap_h3_clip_model(clip: Any) -> Any:
    cond_stage = getattr(clip, "cond_stage_model", None)
    clip_model = getattr(cond_stage, "qwen3vl_32b", None)
    transformer = getattr(clip_model, "transformer", None)
    if clip_model is None or transformer is None or transformer.__class__.__name__ != "MiniMaxQwen3VL":
        raise ValueError("CLIP is not the native MiniMax H3 Qwen3-VL-32B conditioning encoder")
    return clip_model


class MiniMaxH3CachedClip:
    def __init__(self, clip: Any, cache: QwenReferencePrefixCache, contract: Mapping[str, Any]):
        self._clip = clip
        self.cache = cache
        self.contract = dict(contract)
        self.model_identity = _model_identity(clip, contract)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._clip, name)

    def clone(self, disable_dynamic: bool = False):
        return MiniMaxH3CachedClip(
            self._clip.clone(disable_dynamic=disable_dynamic),
            self.cache,
            self.contract,
        )

    def tokenize(self, text, return_word_ids=False, **kwargs):
        full = self._clip.tokenize(text, return_word_ids=return_word_ids, **kwargs)
        has_refs = bool(kwargs.get("minimax_ref_items")) or bool(kwargs.get("images"))
        if return_word_ids or not has_refs:
            return H3PrefixTokens(full, None, full, None)
        prefix = self._clip.tokenize("", return_word_ids=False, **kwargs)
        if not _contains_visual(prefix):
            return H3PrefixTokens(full, None, full, None)
        suffix_kwargs = dict(kwargs)
        suffix_kwargs.pop("minimax_ref_items", None)
        suffix_kwargs.pop("images", None)
        suffix = self._clip.tokenize(text, return_word_ids=False, **suffix_kwargs)
        if not prefix_split_matches_full(full, prefix, suffix):
            return H3PrefixTokens(full, None, full, None)
        key = fingerprint_tokens(prefix, self.model_identity)
        return H3PrefixTokens(full, prefix, suffix, key)

    def encode_from_tokens_scheduled(
        self,
        tokens,
        unprojected=False,
        add_dict: dict[str, Any] | None = None,
        show_pbar=True,
    ):
        add_dict = {} if add_dict is None else add_dict
        if (
            not isinstance(tokens, H3PrefixTokens)
            or tokens.prefix_tokens is None
            or tokens.prefix_fingerprint is None
            or unprojected
            or getattr(self._clip, "use_clip_schedule", False)
            or getattr(getattr(self._clip, "patcher", None), "forced_hooks", None) is not None
        ):
            self.cache.bypasses += 1
            return self._clip.encode_from_tokens_scheduled(
                dict(tokens),
                unprojected=unprojected,
                add_dict=add_dict,
                show_pbar=show_pbar,
            )

        clip_model = _unwrap_h3_clip_model(self._clip)
        self._clip.load_model(dict(tokens))
        device = self._clip.patcher.load_device
        self._clip.cond_stage_model.reset_clip_options()
        self._clip.cond_stage_model.set_clip_options({"execution_device": device})
        import comfy.model_management as model_management

        with self.cache.lock, model_management.cuda_device_context(device):
            entry = self.cache.get(tokens.prefix_fingerprint)
            if entry is None:
                entry = build_prefix_entry(
                    clip_model,
                    tokens.prefix_tokens,
                    tokens.prefix_fingerprint,
                )
                self.cache.put(entry)
            output, tags = encode_suffix_from_entry(clip_model, tokens.suffix_tokens, entry)
        output = output.to(device=model_management.intermediate_device())
        tags = tags.to(device=model_management.intermediate_device())
        pooled = {"pooled_output": None, "minimax_token_tags": tags}
        pooled.update(add_dict)
        self._clip.add_hooks_to_dict(pooled)
        return [[output, pooled]]


def build_prefix_cache_clip(
    clip: Any,
    mode: str,
    max_entries: int,
    maximum_cache_mib: float,
    cache_epoch: int,
) -> tuple[Any, QwenReferencePrefixCache, dict[str, Any]]:
    if mode not in {"report_only", "memory_lru_exp"}:
        raise ValueError(f"unsupported Qwen prefix cache mode: {mode!r}")
    if cache_epoch < 0 or cache_epoch > 0x7FFFFFFF:
        raise ValueError("cache_epoch must be between 0 and 2147483647")
    contract = core_contract()
    cache = QwenReferencePrefixCache(max_entries, maximum_cache_mib)
    report = {
        "schema": QWEN_PREFIX_CACHE_SCHEMA,
        "mode": mode,
        "cache_epoch": int(cache_epoch),
        "core_contract": contract,
        "applied": False,
        "cache": cache.report(),
        "scientific_boundaries": [
            "Only an exact visual-reference prefix is cached; prompt text is recomputed.",
            "Audio-only references add labels but no Qwen visual compute and do not enable this cache by themselves.",
            "The cache is bounded CPU memory only, is not persisted, and is released with the node output or cache clear/restart.",
            "Unknown core source, token weights, CLIP schedules/hooks and non-H3 CLIP inputs bypass or fail closed.",
            "The causal formula is equivalent, but different attention shapes may change floating-point rounding; no bit-exact claim is made before real-model comparison.",
        ],
    }
    if not contract.get("supported"):
        report["status"] = "unsupported_core"
        if mode == "memory_lru_exp":
            raise RuntimeError("Qwen Reference Prefix Cache refuses this unknown ComfyUI source contract")
        return clip, cache, report
    _unwrap_h3_clip_model(clip)
    if mode == "report_only":
        report["status"] = "report_only"
        return clip, cache, report
    report["status"] = "applied_exp"
    report["applied"] = True
    return MiniMaxH3CachedClip(clip, cache, contract), cache, report
