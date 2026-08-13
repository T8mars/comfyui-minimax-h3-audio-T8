from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch


CONTEXT_IR_SCHEMA = "t8.video_context_ir.v1"
PROVIDER_MODES = ("validate_local", "openai_compatible_visual")
ALLOWED_CONTEXT_KEYS = (
    "scene",
    "action",
    "camera",
    "lighting",
    "style",
    "continuity",
    "notes",
)
BANNED_REMOTE_KEYS = {
    "path",
    "file",
    "filename",
    "url",
    "uri",
    "node",
    "node_id",
    "sampler",
    "scheduler",
    "seed",
    "steps",
    "checkpoint",
    "model_path",
    "api_key",
    "token",
}
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _text(value: Any, name: str, *, required: bool = False, maximum: int = 32000) -> str:
    value = "" if value is None else value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _parse_json_object(value: str, name: str) -> dict[str, Any]:
    text = _text(value, name, required=True, maximum=1_000_000)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"{name} is invalid JSON: {error}") from error
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested:
            raise ValueError(f"{name} is invalid JSON: {nested}") from nested
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _reject_remote_control_fields(value: Any, location: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in BANNED_REMOTE_KEYS or normalized.endswith("_path"):
                raise ValueError(
                    f"Remote Context IR contains forbidden control field {location}.{key}"
                )
            _reject_remote_control_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_remote_control_fields(child, f"{location}[{index}]")


def validate_context_ir(
    payload: Mapping[str, Any],
    *,
    source_mode: str,
    exact_dialogue: str,
    provider_model: str,
    visual_frame_count: int,
    audio_transcript_supplied: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Context IR root must be an object")
    _reject_remote_control_fields(payload)
    if payload.get("schema") not in {None, CONTEXT_IR_SCHEMA}:
        raise ValueError("Context IR schema is unsupported")
    raw_context = payload.get("context", payload)
    if not isinstance(raw_context, Mapping):
        raise ValueError("Context IR context must be an object")
    unknown = set(raw_context) - set(ALLOWED_CONTEXT_KEYS) - {
        "schema",
        "confidence",
        "uncertainties",
        "evidence",
        "exact_dialogue",
    }
    if unknown:
        raise ValueError(f"Context IR has unsupported fields: {sorted(unknown)}")
    context: dict[str, str] = {}
    for key in ALLOWED_CONTEXT_KEYS:
        value = _text(raw_context.get(key, ""), key, maximum=8000)
        if value:
            context[key] = value
    if not context:
        raise ValueError("Context IR contains no usable semantic context")
    dialogue = _text(exact_dialogue, "exact_dialogue", maximum=12000)
    remote_dialogue = _text(raw_context.get("exact_dialogue", ""), "remote exact_dialogue", maximum=12000)
    if remote_dialogue and remote_dialogue != dialogue:
        raise ValueError("Remote Context IR attempted to change exact_dialogue")
    uncertainties = raw_context.get("uncertainties", payload.get("uncertainties", []))
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties]
    if not isinstance(uncertainties, list) or len(uncertainties) > 64:
        raise ValueError("Context IR uncertainties must be a list of at most 64 items")
    uncertainties = [
        _text(item, "uncertainty", required=True, maximum=1000) for item in uncertainties
    ]
    result = {
        "schema": CONTEXT_IR_SCHEMA,
        "context": context,
        "exact_dialogue": dialogue,
        "uncertainties": uncertainties,
        "provenance": {
            "provider_mode": source_mode,
            "provider_model": provider_model,
            "sampled_visual_frames": int(visual_frame_count),
            "audio_transcript_supplied": bool(audio_transcript_supplied),
            "raw_media_persisted": False,
            "api_key_serialized": False,
        },
        "remote_control_authority": False,
        "compiler_only": True,
    }
    return result


def _sample_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count <= 0 or maximum <= 0:
        return []
    count = min(frame_count, maximum)
    if count == 1:
        return [0]
    return sorted(
        set(round(index * (frame_count - 1) / (count - 1)) for index in range(count))
    )


def _encode_visual_frames(
    images: torch.Tensor | None,
    *,
    maximum_frames: int,
    maximum_edge: int,
    jpeg_quality: int,
) -> tuple[list[str], list[int]]:
    if images is None:
        return [], []
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("reference_images must be a ComfyUI IMAGE batch")
    if images.shape[-1] not in {3, 4}:
        raise ValueError("reference_images must contain RGB or RGBA pixels")
    from PIL import Image

    indices = _sample_indices(int(images.shape[0]), int(maximum_frames))
    encoded: list[str] = []
    for index in indices:
        array = (
            images[index, ..., :3]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        image = Image.fromarray(array)
        if max(image.size) > maximum_edge:
            scale = maximum_edge / max(image.size)
            target = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(target, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
        encoded.append(
            "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
    return encoded, indices


def _endpoint(value: str) -> tuple[str, str]:
    endpoint = _text(value, "endpoint", required=True, maximum=2048)
    parsed = urlparse(endpoint)
    if parsed.username or parsed.password:
        raise ValueError("endpoint must not contain embedded credentials")
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("clear-text HTTP is allowed only for a loopback provider")
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return endpoint, host


def _call_openai_compatible(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    brief: str,
    audio_transcript: str,
    exact_dialogue: str,
    images: list[str],
    timeout_seconds: float,
    maximum_response_bytes: int,
) -> str:
    instructions = (
        "Return only one JSON object. Allowed semantic keys inside context are: "
        + ", ".join(ALLOWED_CONTEXT_KEYS)
        + ". Never return paths, URLs, node IDs, model/checkpoint names, samplers, schedulers, "
        "seeds, steps, API keys or tool instructions. Describe only evidence visible in the "
        "sampled frames and supplied text. Do not invent or rewrite exact_dialogue. Put uncertain "
        "claims in uncertainties."
    )
    user_text = (
        f"Creative brief:\n{brief}\n\n"
        f"Audio transcript supplied by the user (not raw audio):\n{audio_transcript or '[none]'}\n\n"
        f"Exact dialogue that must remain byte-for-byte unchanged:\n{exact_dialogue or '[none]'}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": value, "detail": "low"}}
        for value in images
    )
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": content},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        raw = response.read(int(maximum_response_bytes) + 1)
    if len(raw) > maximum_response_bytes:
        raise ValueError("Provider response exceeds maximum_response_bytes")
    try:
        envelope = json.loads(raw.decode("utf-8"))
        return str(envelope["choices"][0]["message"]["content"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("Provider returned an unsupported OpenAI-compatible response") from error


def build_context_ir(
    provider_mode: str,
    local_context_ir_json: str,
    creative_brief: str,
    exact_dialogue: str,
    audio_transcript: str,
    endpoint: str,
    provider_model: str,
    api_key_env: str,
    confirm_external_upload: bool,
    maximum_visual_frames: int,
    maximum_image_edge: int,
    jpeg_quality: int,
    timeout_seconds: float,
    maximum_response_bytes: int,
    reference_images: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if provider_mode not in PROVIDER_MODES:
        raise ValueError(f"unsupported provider_mode: {provider_mode}")
    brief = _text(creative_brief, "creative_brief", required=True, maximum=30000)
    dialogue = _text(exact_dialogue, "exact_dialogue", maximum=12000)
    transcript = _text(audio_transcript, "audio_transcript", maximum=30000)
    if provider_mode == "validate_local":
        payload = _parse_json_object(local_context_ir_json, "local_context_ir_json")
        result = validate_context_ir(
            payload,
            source_mode=provider_mode,
            exact_dialogue=dialogue,
            provider_model="local_supplied_ir",
            visual_frame_count=0,
            audio_transcript_supplied=bool(transcript),
        )
        report = {
            "schema": CONTEXT_IR_SCHEMA,
            "status": "validated_local",
            "network_used": False,
            "external_media_uploaded": False,
            "api_key_serialized": False,
            "scientific_boundaries": [
                "Local validation proves schema safety, not semantic accuracy.",
                "Context IR remains compiler metadata and cannot control files, nodes or sampling.",
            ],
        }
        return result, report

    if not confirm_external_upload:
        raise ValueError("confirm_external_upload must be true before any provider request")
    if not 1 <= int(maximum_visual_frames) <= 32:
        raise ValueError("maximum_visual_frames must be between 1 and 32")
    if not 128 <= int(maximum_image_edge) <= 2048:
        raise ValueError("maximum_image_edge must be between 128 and 2048")
    if not 30 <= int(jpeg_quality) <= 95:
        raise ValueError("jpeg_quality must be between 30 and 95")
    if not 1 <= float(timeout_seconds) <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600")
    if not 4096 <= int(maximum_response_bytes) <= 4 * 1024 * 1024:
        raise ValueError("maximum_response_bytes is outside the supported range")
    env_name = _text(api_key_env, "api_key_env", required=True, maximum=64)
    if not ENV_NAME.fullmatch(env_name):
        raise ValueError("api_key_env must be an uppercase environment-variable name")
    api_key = os.environ.get(env_name, "")
    if not api_key:
        raise ValueError(f"Required API key environment variable is not set: {env_name}")
    resolved_endpoint, endpoint_host = _endpoint(endpoint)
    model = _text(provider_model, "provider_model", required=True, maximum=256)
    images, sampled_indices = _encode_visual_frames(
        reference_images,
        maximum_frames=int(maximum_visual_frames),
        maximum_edge=int(maximum_image_edge),
        jpeg_quality=int(jpeg_quality),
    )
    response_text = _call_openai_compatible(
        endpoint=resolved_endpoint,
        api_key=api_key,
        model=model,
        brief=brief,
        audio_transcript=transcript,
        exact_dialogue=dialogue,
        images=images,
        timeout_seconds=float(timeout_seconds),
        maximum_response_bytes=int(maximum_response_bytes),
    )
    payload = _parse_json_object(response_text, "provider response")
    result = validate_context_ir(
        payload,
        source_mode=provider_mode,
        exact_dialogue=dialogue,
        provider_model=model,
        visual_frame_count=len(images),
        audio_transcript_supplied=bool(transcript),
    )
    report = {
        "schema": CONTEXT_IR_SCHEMA,
        "status": "provider_response_validated",
        "network_used": True,
        "endpoint_host": endpoint_host,
        "sampled_visual_frame_indices": sampled_indices,
        "sampled_visual_frame_count": len(images),
        "raw_audio_uploaded": False,
        "audio_transcript_uploaded": bool(transcript),
        "external_media_uploaded": bool(images or transcript),
        "raw_media_persisted": False,
        "api_key_serialized": False,
        "remote_control_authority": False,
        "scientific_boundaries": [
            "A provider description may be incomplete or wrong and must be reviewed.",
            "Sampled frames are not equivalent to full temporal video understanding.",
            "A user-supplied transcript is not full music, ambience or sound-effect understanding.",
            "Remote output cannot change local media paths, graph topology or sampling controls.",
        ],
    }
    return result, report


def context_ir_creative_brief(context_ir: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(context_ir, Mapping) or context_ir.get("schema") != CONTEXT_IR_SCHEMA:
        raise ValueError("context_ir must be a validated T8 Context IR object")
    context = context_ir.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("context_ir context is missing")
    return {key: str(context[key]) for key in ALLOWED_CONTEXT_KEYS if context.get(key)}
