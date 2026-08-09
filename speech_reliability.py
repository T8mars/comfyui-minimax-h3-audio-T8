from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
import uuid
from typing import Callable

import folder_paths
import torch

from .long_video_background import ComfyQueueRuntime, RELEASE_POLICIES


LOGGER = logging.getLogger(__name__)
SPEECH_GUARD_TYPE = "H3_T8_SPEECH_GUARD"
SPEECH_GUARD_SCHEMA = "minimax_h3_t8_speech_guard_v1"


@dataclass
class _GuardEntry:
    prompt_id: str
    token: str
    release_policy: str
    armed_unix: float
    runtime: object


class SpeechGuardRegistry:
    """Prompt-lifecycle guard without patching ComfyUI's executor.

    A normal graph calls ``complete`` from Finalize. If execution ends before that,
    ComfyUI's public cache-provider lifecycle callback calls ``on_prompt_end`` and the
    requested error policy is applied. Recognised CUDA OOMs are already globally
    unloaded by current ComfyUI; a second release request is intentionally idempotent.
    """

    def __init__(self, event_writer: Callable[[dict], None] | None = None):
        self._entries: dict[str, _GuardEntry] = {}
        self._lock = threading.Lock()
        self._event_writer = event_writer or _write_guard_event

    def arm(self, prompt_id: str, release_policy: str, runtime) -> dict:
        if release_policy not in RELEASE_POLICIES:
            raise ValueError(f"unknown release policy: {release_policy}")
        prompt_id = str(prompt_id or "").strip()
        if not prompt_id:
            raise ValueError("speech guard requires a ComfyUI prompt id")
        token = uuid.uuid4().hex
        entry = _GuardEntry(
            prompt_id=prompt_id,
            token=token,
            release_policy=release_policy,
            armed_unix=time.time(),
            runtime=runtime,
        )
        with self._lock:
            self._entries[prompt_id] = entry
        return {
            "schema": SPEECH_GUARD_SCHEMA,
            "prompt_id": prompt_id,
            "token": token,
            "error_release_policy": release_policy,
            "armed": True,
        }

    def complete(self, guard: dict) -> dict:
        prompt_id = str((guard or {}).get("prompt_id", ""))
        token = str((guard or {}).get("token", ""))
        with self._lock:
            entry = self._entries.get(prompt_id)
            if entry is None:
                return {"completed": False, "reason": "guard_not_active"}
            if entry.token != token:
                raise ValueError("speech guard token does not match the active prompt")
            self._entries.pop(prompt_id, None)
        return {"completed": True, "prompt_id": prompt_id, "armed_seconds": time.time() - entry.armed_unix}

    def on_prompt_end(self, prompt_id: str) -> dict | None:
        with self._lock:
            entry = self._entries.pop(str(prompt_id), None)
        if entry is None:
            return None
        event = {
            "schema": SPEECH_GUARD_SCHEMA,
            "prompt_id": entry.prompt_id,
            "event": "abnormal_prompt_end_before_finalize",
            "release_policy": entry.release_policy,
            "scope": (
                "none"
                if entry.release_policy == "keep_loaded"
                else (
                    "global_comfyui_models"
                    if entry.release_policy == "unload_all_models"
                    else "execution_cache_and_soft_memory"
                )
            ),
            "armed_seconds": time.time() - entry.armed_unix,
            "release_requested": False,
            "timestamp_unix": time.time(),
        }
        try:
            if entry.release_policy != "keep_loaded":
                entry.runtime.request_release(entry.release_policy)
                event["release_requested"] = True
        except Exception as error:  # lifecycle callbacks must never hide the original error
            event["release_error"] = f"{type(error).__name__}: {error}"
            LOGGER.exception("MiniMax H3 speech abnormal-exit release failed")
        try:
            self._event_writer(event)
        except Exception:
            LOGGER.exception("MiniMax H3 speech guard event could not be persisted")
        LOGGER.warning(
            "MiniMax H3 speech prompt %s ended before Finalize; release=%s requested=%s",
            entry.prompt_id,
            entry.release_policy,
            event["release_requested"],
        )
        return event

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)


def _guard_event_root() -> Path:
    output = Path(folder_paths.get_output_directory()).resolve()
    root = (output / "minimax_h3_t8" / "speech_recovery").resolve()
    if output != root and output not in root.parents:
        raise ValueError("speech recovery path escaped the ComfyUI output directory")
    return root


def _write_guard_event(event: dict) -> None:
    root = _guard_event_root()
    root.mkdir(parents=True, exist_ok=True)
    prompt_id = "".join(c for c in str(event.get("prompt_id", "unknown")) if c.isalnum() or c in "-_")
    path = root / f"{prompt_id or 'unknown'}.json"
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


_REGISTRY = SpeechGuardRegistry()
_PROVIDER = None
_PROVIDER_LOCK = threading.Lock()


def _ensure_lifecycle_provider() -> None:
    global _PROVIDER
    if _PROVIDER is not None:
        return
    with _PROVIDER_LOCK:
        if _PROVIDER is not None:
            return
        from comfy_api.latest._caching import CacheProvider
        from comfy_execution.cache_provider import register_cache_provider

        class _SpeechLifecycleProvider(CacheProvider):
            async def on_lookup(self, context):
                return None

            async def on_store(self, context, value):
                return None

            def should_cache(self, context, value=None):
                return False

            def on_prompt_end(self, prompt_id: str) -> None:
                _REGISTRY.on_prompt_end(prompt_id)

        _PROVIDER = _SpeechLifecycleProvider()
        register_cache_provider(_PROVIDER)


def arm_speech_guard(release_policy: str, runtime=None) -> dict:
    _ensure_lifecycle_provider()
    runtime = runtime or ComfyQueueRuntime()
    return _REGISTRY.arm(runtime.current_prompt_id(), release_policy, runtime)


def complete_speech_guard(guard: dict | None) -> dict:
    if not guard:
        return {"completed": False, "reason": "no_guard_connected"}
    if guard.get("schema") != SPEECH_GUARD_SCHEMA:
        raise ValueError("expected an H3 T8 speech guard token")
    return _REGISTRY.complete(guard)


def vram_preflight(minimum_headroom_mib: float = 512.0) -> dict:
    """Report present headroom; never infer peak-generation safety from it."""
    result = {
        "schema": "minimax_h3_t8_speech_vram_preflight_v1",
        "minimum_headroom_mib": float(minimum_headroom_mib),
        "current_gate_only": True,
        "memory_safe_claim": False,
        "warning": (
            "A passing preflight is not a peak-VRAM guarantee. The memory_safe label still "
            "requires three cold, three warm, and consecutive-task measurements."
        ),
    }
    try:
        import comfy.model_management as model_management
        from comfy.cli_args import args

        device = model_management.get_torch_device()
        result.update(
            {
                "device": str(device),
                "dynamic_vram_enabled": bool(getattr(model_management, "aimdo_enabled", False)),
                "configured_vram_headroom_gib": float(getattr(args, "vram_headroom", 0.0) or 0.0),
                "reserve_vram_gib": getattr(args, "reserve_vram", None),
            }
        )
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_mib = free_bytes / (1024**2)
            result.update(
                {
                    "whole_device_free_mib": free_mib,
                    "whole_device_total_mib": total_bytes / (1024**2),
                    "torch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
                    "torch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2),
                    "current_gate_pass": free_mib >= float(minimum_headroom_mib),
                }
            )
        else:
            result.update({"current_gate_pass": False, "reason": "CUDA device unavailable"})
    except Exception as error:
        result.update(
            {
                "current_gate_pass": False,
                "inspection_error": f"{type(error).__name__}: {error}",
            }
        )
    return result
