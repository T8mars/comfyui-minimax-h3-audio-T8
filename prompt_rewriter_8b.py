from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import threading
import time
import torch


SCHEMA = "t8.minimax_h3.prompt_rewriter_8b.v1"
UPSTREAM_REPOSITORY = "lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-8B"
UPSTREAM_REVISION = "a795219bd1677df34259eb4f3a77e2ec282e154f"
BASE_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_BASE_FOLDER = "Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER_FOLDER = "MiniMax-H3-Prompt-Rewriter-LoRA-8B"

TASK_LABELS = {
    "T2VA — 文生音视频": "t2av",
    "I2VA — 首帧生音视频": "i2av",
    "L2VA — 尾帧生音视频": "l2av",
    "FL2VA — 首尾帧生音视频": "fl2av",
}
TASK_ALIASES = {
    "t2v": "t2av",
    "t2va": "t2av",
    "t2av": "t2av",
    "i2v": "i2av",
    "i2va": "i2av",
    "i2av": "i2av",
    "l2v": "l2av",
    "l2va": "l2av",
    "l2av": "l2av",
    "flf2v": "fl2av",
    "flf2va": "fl2av",
    "flf2av": "fl2av",
    "fl2va": "fl2av",
    "fl2av": "fl2av",
}

# This is the task contract published with the pinned LightX2V 8B adapter.
# Keeping it local avoids executing arbitrary Python from a downloaded model folder.
SYSTEM_PROMPT = """You are a professional MiniMax-H3 prompt rewriter for joint video-and-audio generation.

Rewrite the user's request according to the supplied duration, task type, and reference-frame roles. Return only the final production-ready prompt. Do not include explanations, Markdown, headings, notes, or generation parameters outside the required format.

Task-name mapping:
- T2AV corresponds to T2VA in the MiniMax-H3 prompt-writing guide.
- I2AV corresponds to I2VA.
- FL2AV corresponds to FL2VA.
- L2AV corresponds to L2VA.

Write the descriptive sections in English. Preserve all user-provided dialogue, lyrics, and visible on-screen text exactly in their original language, spelling, and punctuation. Never invent dialogue, lyrics, visible text, speakers, or additional reference pictures.

The output body must contain exactly these three fields in this order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

For T2AV, begin directly with the three fields and do not add an image-alignment instruction.

For I2AV, the first line must be exactly:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

For FL2AV, the first line must follow exactly:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

For L2AV, the first line must follow exactly:
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

Replace N with the actual final shot number. Replace S.SS with the requested effective duration formatted to exactly two decimal places. Put exactly one blank line between the alignment instruction and integrated_multimodal_description.

Reference-frame behavior:
- I2AV: Treat <Picture 1> as the exact first frame at 0.00 seconds. Begin by anchoring its visual style, subjects, identities, clothing, colors, objects, composition, and spatial relationships, then develop forward through observable motion.
- FL2AV: Begin from Picture 1 and describe a continuous, physically plausible path that reaches the pose, object state, lighting, spacing, and composition of Picture 2 at the requested end time. Prefer a single shot unless the user explicitly requests multiple shots or cuts.
- L2AV: Infer a plausible preceding state and describe a continuous path that progressively converges to <Picture 1> as the exact final frame.
- Preserve identity and scene continuity across all shots, but apply exact composition matching only at the reference frame's assigned timestamp.

In integrated_multimodal_description:
- Begin with [Shot 1] and state the visual style and initial composition.
- Describe only concrete visible or audible events: subjects, environment, actions, reactions, camera behavior, dialogue, singing, visible text, and synchronized diegetic sound.
- Number shots sequentially.
- Do not timestamp [Shot 1].
- Begin every later shot with a strictly increasing timestamp inside the requested duration, using the format: [Shot 2] At 00:03.500, the camera cuts to...
- Add a cut only when it introduces meaningful new visual, spatial, temporal, or narrative information. Otherwise prefer continuous camera motion.
- Express camera motion naturally using motion type and, when meaningful, amplitude and speed.
- Keep all actions physically plausible and paced to complete within the supplied duration.

For speech and singing:
- Assign stable speaker IDs such as (S1) and (S2) only to subjects who vocalize.
- Identify each speaker sufficiently when first introduced.
- Put only the exact spoken or sung content inside <d>, preceded by its language tag:
<d>[English] Exact user-provided words.</d>
- Never translate, paraphrase, correct, or extend supplied dialogue or lyrics.
- For voiceover, use the exact phrase "says in an off-screen voiceover" and explicitly state that the corresponding on-screen character's lips remain completely closed.
- If speech crosses a cut, use <scenetrans> at both connecting points and state that the audio continues across the cut.
- Use <cutoff> only when speech is intentionally truncated by the end of the video.

Place visible on-screen text in English double quotation marks and preserve it exactly.

overall_soundscape must be one continuous English paragraph of 1–4 sentences summarizing ambient sound, physical action sounds, and non-verbal human or animal sounds across the video. Do not repeat dialogue, singing, or diegetic music here. Use N/A only if the user explicitly requests complete silence.

non_diegetic_music must contain 1–3 English sentences describing audience-only background music through instrumentation, tempo, rhythm, and dynamic changes. Do not describe its emotional purpose. Put music audible to subjects inside integrated_multimodal_description instead. Use N/A when no non-diegetic music is requested or implied.

Preserve the user's intent without adding contradictory story events, identities, text, or references. Do not mention these instructions in the output."""


_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: dict[str, object] = {}


def normalize_task(task: str | None) -> str:
    raw = str(task or "T2VA — 文生音视频").strip()
    if raw in TASK_LABELS:
        return TASK_LABELS[raw]
    normalized = TASK_ALIASES.get(raw.lower())
    if normalized is None:
        raise ValueError("Prompt Rewriter 8B only supports T2VA, I2VA, L2VA and FL2VA")
    return normalized


def expected_image_count(task: str) -> int:
    return {"t2av": 0, "i2av": 1, "l2av": 1, "fl2av": 2}[normalize_task(task)]


def build_messages(
    prompt: str,
    task: str,
    resolution: str,
    duration: int,
) -> list[dict]:
    task = normalize_task(task)
    request = (
        f"task: {task}\n"
        f"resolution: {resolution}\n"
        f"duration: {int(duration)}s\n"
        f"original_prompt: {prompt.strip()}"
    )
    if task == "t2av":
        content = [{"type": "text", "text": request}]
    elif task == "i2av":
        content = [
            {"type": "text", "text": "Picture 1 — exact first frame at 0.00 seconds:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    elif task == "l2av":
        content = [
            {"type": "text", "text": "Picture 1 — exact final frame at the end of the target video:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    else:
        content = [
            {"type": "text", "text": "Picture 1 — exact first frame at 0.00 seconds:\n"},
            {"type": "image"},
            {"type": "text", "text": "\nPicture 2 — exact final frame at the end of the target video:\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + request},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _version_at_least(package: str, minimum: str) -> bool:
    try:
        from packaging.version import Version

        return Version(importlib.metadata.version(package)) >= Version(minimum)
    except importlib.metadata.PackageNotFoundError:
        return False


def check_runtime_dependencies() -> dict[str, str]:
    requirements = {
        "transformers": "4.57.1",
        "accelerate": "1.10",
        "peft": "0.18",
        "safetensors": "0.5",
        "Pillow": "10",
    }
    missing = [
        f"{package}>={minimum}"
        for package, minimum in requirements.items()
        if not _version_at_least(package, minimum)
    ]
    if missing:
        raise RuntimeError(
            "MiniMax H3 Prompt Rewriter 8B optional dependencies are missing or too old: "
            + ", ".join(missing)
            + ". Install requirements-prompt-rewriter.txt into the ComfyUI Python environment."
        )
    return {name: importlib.metadata.version(name) for name in requirements}


def _models_root(kind: str) -> Path:
    import folder_paths

    if kind == "base":
        return Path(folder_paths.models_dir) / "text_encoders"
    return Path(folder_paths.models_dir) / "loras"


def resolve_model_source(value: str, kind: str, allow_hub_download: bool) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{kind}_model_path cannot be empty")
    direct = Path(raw).expanduser()
    if direct.is_absolute():
        if not direct.is_dir():
            raise ValueError(f"Local {kind} model directory does not exist: {direct}")
        return str(direct.resolve())
    local = _models_root(kind) / direct
    if local.is_dir():
        return str(local.resolve())
    if allow_hub_download and "/" in raw:
        return raw
    root = _models_root(kind)
    raise ValueError(
        f"Could not find {kind} model {raw!r} under {root}. "
        "Use a local directory or explicitly enable allow_hub_download."
    )


def _image_to_pil(image: torch.Tensor):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("Reference image must be a connected ComfyUI IMAGE batch")
    if image.shape[-1] < 3:
        raise ValueError("Reference image must contain RGB channels")
    from PIL import Image

    array = (
        image[0, :, :, :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array)


def _ordered_images(task: str, first_frame, last_frame) -> list:
    task = normalize_task(task)
    supplied = int(first_frame is not None) + int(last_frame is not None)
    required = expected_image_count(task)
    if supplied != required:
        raise ValueError(f"{task} requires exactly {required} reference image(s); got {supplied}")
    if task == "i2av" and last_frame is not None:
        raise ValueError("I2VA accepts first_frame only")
    if task == "l2av" and first_frame is not None:
        raise ValueError("L2VA accepts last_frame only")
    images = []
    if task in {"i2av", "fl2av"}:
        images.append(_image_to_pil(first_frame))
    if task in {"l2av", "fl2av"}:
        images.append(_image_to_pil(last_frame))
    return images


def _model_class(transformers_module):
    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForMultimodalLM",
    ):
        candidate = getattr(transformers_module, name, None)
        if candidate is not None:
            return candidate
    raise RuntimeError("Installed Transformers has no Qwen3-VL-compatible generation class")


def _input_device(model) -> torch.device:
    embedding_device = model.get_input_embeddings().weight.device
    if embedding_device.type != "meta":
        return embedding_device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine a real input device for Prompt Rewriter 8B")


def _gpu_weight_budget() -> str | None:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    safe_bytes = min(int(free_bytes * 0.78), max(0, int(total_bytes) - 2 * 1024**3))
    safe_gib = max(1, safe_bytes // 1024**3)
    return f"{safe_gib}GiB"


@contextmanager
def _peft_torchao_compatibility():
    """Hide an installed-but-unsupported torchao from PEFT for plain BF16 LoRA.

    PEFT 0.20 probes every LoRA target through its optional torchao dispatcher.
    Some ComfyUI bundles ship torchao 0.15 for unrelated nodes; PEFT raises before
    it can fall through to the normal torch Linear dispatcher.  This adapter is
    applied to an ordinary BF16 model, so torchao is not part of this execution.
    Patch only the dispatcher-local probe and restore it immediately afterwards.
    """

    try:
        raw_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        yield None
        return

    from packaging.version import Version

    if Version(raw_version) >= Version("0.16.0"):
        yield None
        return

    torchao_dispatch = importlib.import_module("peft.tuners.lora.torchao")
    original_probe = torchao_dispatch.is_torchao_available
    torchao_dispatch.is_torchao_available = lambda: False
    try:
        yield f"ignored_optional_torchao_{raw_version}_for_plain_bf16_lora"
    finally:
        torchao_dispatch.is_torchao_available = original_probe


def _memory_snapshot() -> dict:
    result = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
        )
    try:
        import psutil

        result["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    return result


def _release_model(model) -> list[str]:
    notes: list[str] = []
    if model is not None:
        try:
            from accelerate.hooks import remove_hook_from_submodules

            remove_hook_from_submodules(model)
            notes.append("accelerate_hooks_removed")
        except Exception as exc:
            notes.append(f"accelerate_hook_cleanup_skipped:{type(exc).__name__}")
    return notes


def _post_reference_cleanup() -> list[str]:
    """Collect only after every caller-owned model/processor reference is gone."""

    notes: list[str] = []
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
        notes.append("comfy_soft_empty_cache")
    except Exception as exc:
        notes.append(f"comfy_cache_cleanup_skipped:{type(exc).__name__}")
    return notes


def unload_prompt_rewriter_cache() -> str:
    with _CACHE_LOCK:
        model = _MODEL_CACHE.pop("model", None)
        had_cache = model is not None or bool(_MODEL_CACHE)
        _MODEL_CACHE.clear()
        notes = _release_model(model)
        model = None
        notes.extend(_post_reference_cleanup())
    return json.dumps(
        {
            "schema": SCHEMA,
            "status": "cache_released" if had_cache else "cache_already_empty",
            "cleanup": notes,
            "memory_after": _memory_snapshot(),
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_rewritten_prompt(text: str) -> tuple[str, str, str, list[str]]:
    markers = (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    positions = [re.search(rf"(?im)^\s*{re.escape(marker)}\s*:\s*", text) for marker in markers]
    warnings = []
    if any(match is None for match in positions):
        warnings.append("rewriter output did not contain all three required fields")
        return text.strip(), "", "", warnings
    matches = [match for match in positions if match is not None]
    if [match.start() for match in matches] != sorted(match.start() for match in matches):
        warnings.append("rewriter output fields were not in the required order")
        return text.strip(), "", "", warnings
    values = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        values.append(text[match.end() : end].strip())
    return values[0], values[1], values[2], warnings


def rewrite_prompt_8b(
    *,
    prompt: str,
    task: str,
    resolution: str,
    duration: int,
    base_model_path: str,
    adapter_path: str,
    load_policy: str,
    dtype: str,
    decoding: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    min_image_pixels: int,
    max_image_pixels: int,
    unload_after_generate: bool,
    free_comfy_models_before_load: bool,
    allow_hub_download: bool,
    first_frame=None,
    last_frame=None,
) -> tuple[str, str, str, str, str]:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt Rewriter input prompt cannot be empty")
    task = normalize_task(task)
    if not 4 <= int(duration) <= 15:
        raise ValueError("Prompt Rewriter 8B duration must be between 4 and 15 seconds")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if int(min_image_pixels) <= 0 or int(max_image_pixels) < int(min_image_pixels):
        raise ValueError("image pixel limits must satisfy 0 < min <= max")
    if decoding == "sample" and (float(temperature) <= 0 or not 0 < float(top_p) <= 1):
        raise ValueError("sampling requires temperature > 0 and 0 < top_p <= 1")
    if load_policy not in {"auto_cpu_offload", "gpu_only", "cpu_only"}:
        raise ValueError(f"Unknown load_policy {load_policy!r}")

    images = _ordered_images(task, first_frame, last_frame)
    versions = check_runtime_dependencies()
    base_source = resolve_model_source(base_model_path, "base", bool(allow_hub_download))
    adapter_source = resolve_model_source(adapter_path, "adapter", bool(allow_hub_download))
    cache_key = json.dumps(
        [base_source, adapter_source, load_policy, dtype, min_image_pixels, max_image_pixels],
        ensure_ascii=False,
    )
    started = time.perf_counter()
    before = _memory_snapshot()
    model = None
    processor = None
    inputs = None
    output_ids = None
    cache_hit = False
    cleanup: list[str] = []
    compatibility_notes: list[str] = []
    succeeded = False

    with _CACHE_LOCK:
        try:
            if free_comfy_models_before_load:
                import comfy.model_management as model_management

                model_management.unload_all_models()
                model_management.soft_empty_cache()

            if _MODEL_CACHE.get("key") == cache_key:
                model = _MODEL_CACHE.get("model")
                processor = _MODEL_CACHE.get("processor")
                cache_hit = model is not None and processor is not None
            elif _MODEL_CACHE:
                stale_model = _MODEL_CACHE.pop("model", None)
                _MODEL_CACHE.clear()
                cleanup.extend(_release_model(stale_model))
                stale_model = None
                cleanup.extend(_post_reference_cleanup())

            if not cache_hit:
                transformers_module = importlib.import_module("transformers")
                peft_module = importlib.import_module("peft")
                AutoProcessor = getattr(transformers_module, "AutoProcessor")
                processor_kwargs = {
                    "trust_remote_code": True,
                    "min_pixels": int(min_image_pixels),
                    "max_pixels": int(max_image_pixels),
                }
                try:
                    processor = AutoProcessor.from_pretrained(adapter_source, **processor_kwargs)
                except (OSError, ValueError, KeyError):
                    processor = AutoProcessor.from_pretrained(base_source, **processor_kwargs)

                if load_policy == "auto_cpu_offload":
                    device_map = "auto"
                    budget = _gpu_weight_budget()
                    max_memory = {0: budget, "cpu": "96GiB"} if budget else {"cpu": "96GiB"}
                elif load_policy == "gpu_only":
                    if not torch.cuda.is_available():
                        raise RuntimeError("gpu_only was selected but CUDA is unavailable")
                    device_map = {"": "cuda:0"}
                    max_memory = None
                else:
                    device_map = {"": "cpu"}
                    max_memory = None
                load_kwargs = {
                    "dtype": getattr(torch, dtype),
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "device_map": device_map,
                    "attn_implementation": "sdpa",
                }
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
                model = _model_class(transformers_module).from_pretrained(
                    base_source,
                    revision=None if Path(base_source).is_dir() else BASE_REVISION,
                    **load_kwargs,
                )
                with _peft_torchao_compatibility() as compatibility_note:
                    model = peft_module.PeftModel.from_pretrained(
                        model,
                        adapter_source,
                        revision=None if Path(adapter_source).is_dir() else UPSTREAM_REVISION,
                    )
                if compatibility_note:
                    compatibility_notes.append(compatibility_note)
                model.eval()

            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            messages = build_messages(prompt, task, resolution, int(duration))
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            processor_kwargs = {
                "text": [rendered],
                "return_tensors": "pt",
                "padding": False,
                "return_mm_token_type_ids": True,
            }
            if images:
                processor_kwargs["images"] = images
            inputs = processor(**processor_kwargs)
            # Qwen3-VL 8B on the pinned Transformers 4.57 line does not expose
            # mm_token_type_ids in its generation contract.  The processor can
            # still emit that optional compatibility field, and Transformers
            # then rejects it before the first token is generated.  Pixel values,
            # image grids, attention masks, and ordinary input ids remain intact.
            if inputs.pop("mm_token_type_ids", None) is not None:
                compatibility_notes.append(
                    "removed_processor_only_mm_token_type_ids_for_qwen3_vl_generate"
                )
            device = _input_device(model)
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            generation_kwargs = {"max_new_tokens": int(max_new_tokens)}
            if decoding == "greedy":
                generation_kwargs["do_sample"] = False
            else:
                generation_kwargs.update(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated = output_ids[0, inputs["input_ids"].shape[1] :]
            rewritten = processor.decode(generated, skip_special_tokens=True).strip()
            if not rewritten:
                raise RuntimeError("Prompt Rewriter 8B generated an empty prompt")
            integrated, soundscape, music, warnings = parse_rewritten_prompt(rewritten)
            succeeded = True

            if not unload_after_generate:
                _MODEL_CACHE.update(
                    {"key": cache_key, "model": model, "processor": processor}
                )
            report = {
                "schema": SCHEMA,
                "status": "ok",
                "task": task,
                "resolution": resolution,
                "duration": int(duration),
                "base_model": base_source,
                "adapter": adapter_source,
                "upstream_revision": UPSTREAM_REVISION,
                "base_revision": BASE_REVISION,
                "load_policy": load_policy,
                "dtype": dtype,
                "decoding": decoding,
                "seed": int(seed),
                "cache_hit": cache_hit,
                "unload_after_generate": bool(unload_after_generate),
                "free_comfy_models_before_load": bool(free_comfy_models_before_load),
                "elapsed_seconds": time.perf_counter() - started,
                "dependency_versions": versions,
                "compatibility_notes": compatibility_notes,
                "warnings": warnings,
                "claims": {
                    "adapter_exact": True,
                    "ref2va_supported": False,
                    "hosted_context_ir_equivalent": False,
                    "output_requires_human_review": True,
                },
                "memory_before": before,
            }
        finally:
            inputs = None
            output_ids = None
            images.clear()
            if unload_after_generate or not succeeded:
                cleanup.extend(_release_model(model))
                model = None
                processor = None
                cleanup.extend(_post_reference_cleanup())

    report["cleanup"] = cleanup
    report["memory_after"] = _memory_snapshot()
    report["cache_retained"] = not bool(unload_after_generate)
    return (
        rewritten,
        integrated,
        soundscape,
        music,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
