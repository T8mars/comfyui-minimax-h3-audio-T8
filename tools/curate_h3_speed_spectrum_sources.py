#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "minimax_h3_speed_source_curation_v2"
VIDEO_EXTENSIONS = {
    ".avi",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".webm",
}
SUSPECT_NAME_TOKENS = (
    "baseline",
    "blind",
    "cached",
    "candidate",
    "cold",
    "composite",
    "compare",
    "comparison",
    "control",
    "contact_sheet",
    "contactsheet",
    "detail",
    "dynamic_guidance",
    "exp_",
    "face_refine",
    "6way",
    "grid",
    "hybrid",
    "labeled",
    "long_video",
    "matrix",
    "montage",
    "multikeyframe",
    "multiface",
    "ordinary",
    "passthrough",
    "preview",
    "probe",
    "quality_gate",
    "quality_",
    "refine",
    "reel_delivery",
    "repeat",
    "restart",
    "route",
    "side_by_side",
    "sidebyside",
    "six_way",
    "speed",
    "sigma",
    "stg",
    "stitched",
    "tail",
    "three_way",
    "two_way",
    "turbo",
    "upstream",
    "validation",
    "visual_strength",
    "warm",
)


class CurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodeSignature:
    raw_sha256: str
    frame_hashes: tuple[int, ...]
    decoded_frame_count: int


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(command), capture_output=True, check=False)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def parse_rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _probe_payload(path: Path, ffprobe: str) -> Mapping[str, Any]:
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
                "nb_frames,duration:format=duration:format_tags=prompt,workflow"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise CurationError(f"ffprobe failed: {detail}")
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationError("ffprobe returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CurationError("ffprobe did not return an object")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _sorted_unique_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def extract_embedded_h3_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a privacy-minimized H3 execution contract from MP4 metadata.

    The complete prompt/workflow text can contain private user material. This function
    hashes those tags but emits only model/task/sampler contract fields required for
    calibration grouping.
    """

    format_value = payload.get("format")
    tags = format_value.get("tags") if isinstance(format_value, Mapping) else None
    if not isinstance(tags, Mapping):
        tags = {}
    prompt_text = tags.get("prompt")
    workflow_text = tags.get("workflow")
    base = {
        "prompt_tag_present": isinstance(prompt_text, str) and bool(prompt_text),
        "workflow_tag_present": isinstance(workflow_text, str) and bool(workflow_text),
        "prompt_text_emitted": False,
        "workflow_text_emitted": False,
    }
    if isinstance(prompt_text, str) and prompt_text:
        base["prompt_tag_sha256"] = hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest().upper()
    if isinstance(workflow_text, str) and workflow_text:
        base["workflow_tag_sha256"] = hashlib.sha256(
            workflow_text.encode("utf-8")
        ).hexdigest().upper()
    if not isinstance(prompt_text, str) or not prompt_text:
        return {**base, "status": "missing_prompt_tag"}
    try:
        prompt = json.loads(prompt_text)
    except json.JSONDecodeError as exc:
        return {
            **base,
            "status": "malformed_prompt_tag",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(prompt, Mapping):
        return {**base, "status": "malformed_prompt_tag", "error": "not_an_object"}

    nodes = [node for node in prompt.values() if isinstance(node, Mapping)]
    class_types = _sorted_unique_strings(node.get("class_type", "") for node in nodes)
    h3_signal = any("minimaxh3" in value.lower().replace("_", "") for value in class_types)
    checkpoint_names: list[str] = []
    vae_names: list[str] = []
    lora_names: list[str] = []
    task_families: list[str] = []
    seeds: list[int] = []
    sampler_contracts: list[dict[str, Any]] = []
    conditioning_text_hashes: list[str] = []
    source_asset_identifier_hashes: list[str] = []
    mutator_tokens = (
        "lora",
        "patch",
        "blockcache",
        "sage",
        "activationchunk",
        "modeltime",
        "restart",
        "spatiotemporal",
        "speed",
    )
    generation_modifier_tokens = (
        "sampler",
        "scheduler",
        "sigma",
        "guidance",
        "dynamiccfg",
        "detail",
        "tail",
        "restart",
        "spatiotemporal",
        "modeltime",
        "speed",
        "multikeyframe",
        "keyframe",
        "hybrid",
        "visualreference",
        "face",
        "motion",
        "audioinjection",
        "multirate",
        "blockcache",
        "activationchunk",
        "qwenprefix",
    )
    model_mutator_classes: list[str] = []
    generation_modifier_classes: list[str] = []
    for node in nodes:
        class_type = str(node.get("class_type") or "")
        normalized_class = class_type.lower().replace("_", "")
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for key in ("unet_name", "model_name", "ckpt_name", "diffusion_model"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                checkpoint_names.append(value)
                h3_signal = h3_signal or "minimax_h3" in value.lower()
        value = inputs.get("vae_name")
        if isinstance(value, str) and value:
            vae_names.append(value)
            h3_signal = h3_signal or "minimax_h3" in value.lower()
        value = inputs.get("lora_name")
        if isinstance(value, str) and value:
            lora_names.append(value)
        task = inputs.get("task_type")
        if isinstance(task, str) and task.strip():
            task_families.append(task.strip().split(" ", 1)[0].upper())
        if "minimaxh3" in normalized_class:
            for key in ("prompt", "text", "positive_prompt"):
                value = inputs.get(key)
                if isinstance(value, str) and value:
                    conditioning_text_hashes.append(
                        hashlib.sha256(value.encode("utf-8")).hexdigest().upper()
                    )
        if normalized_class in {"loadimage", "loadvideo", "loadaudio"}:
            for key in ("image", "file", "video", "audio"):
                value = inputs.get(key)
                if isinstance(value, str) and value:
                    normalized_identifier = value.replace("\\", "/").casefold()
                    source_asset_identifier_hashes.append(
                        hashlib.sha256(normalized_identifier.encode("utf-8"))
                        .hexdigest()
                        .upper()
                    )
        for key in ("noise_seed", "seed"):
            value = inputs.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                seeds.append(value)
        if "sampler" in normalized_class or "scheduler" in normalized_class:
            contract = {"class_type": class_type}
            for key in (
                "steps",
                "shift_video",
                "shift_audio",
                "sampler_name",
                "scheduler",
                "denoise",
            ):
                value = inputs.get(key)
                if isinstance(value, (str, int, float, bool)):
                    contract[key] = value
            sampler_contracts.append(contract)
        if any(token in normalized_class for token in mutator_tokens):
            model_mutator_classes.append(class_type)
        if any(token in normalized_class for token in generation_modifier_tokens):
            generation_modifier_classes.append(class_type)

    checkpoint_names = _sorted_unique_strings(checkpoint_names)
    vae_names = _sorted_unique_strings(vae_names)
    video_vae_names = [name for name in vae_names if "video" in name.lower()]
    audio_vae_names = [name for name in vae_names if "audio" in name.lower()]
    lora_names = _sorted_unique_strings(lora_names)
    task_families = _sorted_unique_strings(task_families)
    model_mutator_classes = _sorted_unique_strings(model_mutator_classes)
    generation_modifier_classes = _sorted_unique_strings(
        generation_modifier_classes
    )
    conditioning_text_hashes = _sorted_unique_strings(conditioning_text_hashes)
    source_asset_identifier_hashes = _sorted_unique_strings(
        source_asset_identifier_hashes
    )
    sampler_contracts = sorted(
        sampler_contracts,
        key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False),
    )
    if not h3_signal:
        return {
            **base,
            "status": "non_h3_prompt",
            "class_types_sha256": _canonical_sha256(class_types),
        }
    contract = {
        "task_families": task_families,
        "checkpoint_names": checkpoint_names,
        "video_vae_names": video_vae_names,
        "audio_vae_names": audio_vae_names,
        "lora_names": lora_names,
        "model_mutator_classes": model_mutator_classes,
        "generation_modifier_classes": generation_modifier_classes,
        "sampler_contracts": sampler_contracts,
    }
    content_signature = {
        "task_families": task_families,
        "conditioning_text_sha256": conditioning_text_hashes,
        "source_asset_identifier_sha256": source_asset_identifier_hashes,
    }
    complete = bool(task_families and checkpoint_names and video_vae_names)
    return {
        **base,
        "status": "parsed_h3_contract" if complete else "partial_h3_contract",
        "contract_id": _canonical_sha256(contract),
        "contract": contract,
        "seed_values": sorted(set(seeds)),
        "conditioning_text_sha256": conditioning_text_hashes,
        "source_asset_identifier_sha256": source_asset_identifier_hashes,
        "content_signature_id": _canonical_sha256(content_signature),
        "class_types_sha256": _canonical_sha256(class_types),
    }


def provenance_contract_groups(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in items:
        provenance = row.get("embedded_provenance")
        if not isinstance(provenance, Mapping):
            continue
        if provenance.get("status") != "parsed_h3_contract":
            continue
        contract_id = str(provenance["contract_id"])
        group = groups.setdefault(
            contract_id,
            {
                "contract_id": contract_id,
                "contract": provenance["contract"],
                "total_files": 0,
                "provisional_candidate": 0,
                "manual_review_required": 0,
                "rejected": 0,
                "content_signature_ids": set(),
                "seed_values": set(),
                "paths": [],
            },
        )
        status = str(row.get("status"))
        group["total_files"] += 1
        if status in {"provisional_candidate", "manual_review_required", "rejected"}:
            group[status] += 1
        group["content_signature_ids"].add(str(provenance["content_signature_id"]))
        group["seed_values"].update(str(value) for value in provenance["seed_values"])
        group["paths"].append(str(row.get("path")))
    public_groups = []
    for group in groups.values():
        group["unique_content_signatures"] = len(group.pop("content_signature_ids"))
        group["unique_seed_values"] = len(group.pop("seed_values"))
        public_groups.append(group)
    return sorted(
        public_groups,
        key=lambda value: (-int(value["provisional_candidate"]), -int(value["total_files"]), value["contract_id"]),
    )


def probe_record_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise CurationError("No video stream was reported")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise CurationError("Video dimensions are missing or invalid")
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(
        stream.get("r_frame_rate")
    )
    if fps is None:
        raise CurationError("Video frame rate is missing or invalid")
    duration = _finite_positive(stream.get("duration"))
    if duration is None:
        format_value = payload.get("format")
        if isinstance(format_value, dict):
            duration = _finite_positive(format_value.get("duration"))
    frames = None
    raw_frames = stream.get("nb_frames")
    if raw_frames not in (None, "", "N/A"):
        try:
            parsed_frames = int(raw_frames)
        except (TypeError, ValueError):
            parsed_frames = 0
        if parsed_frames > 0:
            frames = parsed_frames
    if duration is None and frames is not None:
        duration = frames / fps
    if duration is None:
        raise CurationError("Video duration is missing or invalid")
    return {
        "codec": str(stream.get("codec_name") or "unknown"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "reported_frame_count": frames,
        "estimated_frame_count": int(math.floor(duration * fps + 1e-6)),
    }


def center_cover_geometry(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive")
    scale = max(target_width / source_width, target_height / source_height)
    scaled_width = source_width * scale
    scaled_height = source_height * scale
    retained_fraction = (target_width * target_height) / (
        scaled_width * scaled_height
    )
    return {
        "requires_upscale": scale > 1.0 + 1e-9,
        "scale": scale,
        "retained_source_fraction": min(1.0, retained_fraction),
        "anisotropic_stretch": False,
    }


def suspect_name_flags(path: Path, *, roots: Sequence[Path] = ()) -> list[str]:
    candidate = path
    resolved_path = path.resolve()
    relative_candidates: list[Path] = []
    for raw_root in roots:
        resolved_root = raw_root.expanduser().resolve()
        base = resolved_root.parent if resolved_root.is_file() else resolved_root
        try:
            relative_candidates.append(resolved_path.relative_to(base))
        except ValueError:
            continue
    if relative_candidates:
        candidate = min(relative_candidates, key=lambda value: len(value.parts))
    normalized = str(candidate).lower().replace("\\", "/")
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return [token for token in SUSPECT_NAME_TOKENS if token in normalized]


def strict_decode_window(
    path: Path,
    *,
    ffmpeg: str,
    window_seconds: float,
    attempts: int,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    trials = []
    for attempt in range(1, attempts + 1):
        completed = _run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-i",
                str(path),
                "-t",
                f"{window_seconds:.9f}",
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
        trials.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-1000:],
            }
        )
    return {"passed": all(row["returncode"] == 0 for row in trials), "trials": trials}


def _average_hash(frame: np.ndarray) -> int:
    threshold = float(frame.mean())
    bits = np.asarray(frame >= threshold, dtype=np.uint8).reshape(-1)
    result = 0
    for value in bits:
        result = (result << 1) | int(value)
    return result


def decode_signature(
    path: Path,
    *,
    ffmpeg: str,
    window_seconds: float,
    sample_fps: float = 3.0,
    side: int = 16,
) -> DecodeSignature:
    completed = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-t",
            f"{window_seconds:.9f}",
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (
                f"fps={sample_fps:.9f},"
                f"scale={side}:{side}:force_original_aspect_ratio=increase,"
                f"crop={side}:{side},format=gray"
            ),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise CurationError(f"Signature decode failed: {detail}")
    frame_bytes = side * side
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise CurationError("Signature decode returned empty or malformed raw frames")
    frames = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, side, side)
    hashes = tuple(_average_hash(frame) for frame in frames)
    return DecodeSignature(
        raw_sha256=hashlib.sha256(completed.stdout).hexdigest().upper(),
        frame_hashes=hashes,
        decoded_frame_count=len(hashes),
    )


def temporal_hash_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    bit_count = 256 * len(left)
    differences = sum((int(a) ^ int(b)).bit_count() for a, b in zip(left, right))
    return 1.0 - differences / bit_count


def discover_videos(roots: Iterable[Path]) -> list[Path]:
    paths: dict[str, Path] = {}
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file():
            candidates = [resolved]
        elif resolved.is_dir():
            candidates = resolved.rglob("*")
        else:
            continue
        for path in candidates:
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                paths[str(path).casefold()] = path
    return [paths[key] for key in sorted(paths)]


def _union_find_groups(pairs: Sequence[tuple[int, int]], count: int) -> list[list[int]]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs:
        union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def curate_sources(
    *,
    roots: Sequence[Path],
    mode: str,
    ffprobe: str,
    ffmpeg: str,
    target_width: int,
    target_height: int,
    required_frames: int,
    target_fps: float,
    decode_attempts: int,
    near_duplicate_threshold: float,
    reject_upscale: bool,
    require_embedded_provenance: bool = False,
) -> dict[str, Any]:
    if mode not in {"metadata", "strict", "signature"}:
        raise ValueError("mode must be metadata, strict or signature")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if required_frames < 2 or target_fps <= 0.0:
        raise ValueError("required_frames and target_fps are invalid")
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")

    window_seconds = (required_frames - 1) / target_fps + 1.0 / target_fps
    minimum_last_frame_time = (required_frames - 1) / target_fps
    items: list[dict[str, Any]] = []
    signatures: list[DecodeSignature | None] = []
    exact_files: dict[str, str] = {}
    exact_decoded: dict[str, str] = {}

    for path in discover_videos(roots):
        row: dict[str, Any] = {
            "path": str(path),
            "status": "pending",
            "rejection_reasons": [],
            "manual_review_flags": [],
        }
        signatures.append(None)
        try:
            row["bytes"] = path.stat().st_size
            row["file_sha256"] = sha256_file(path)
            duplicate = exact_files.get(row["file_sha256"])
            if duplicate is not None:
                row["exact_file_duplicate_of"] = duplicate
                row["rejection_reasons"].append("exact_file_duplicate")
            else:
                exact_files[row["file_sha256"]] = str(path)

            probe_payload = _probe_payload(path, ffprobe)
            metadata = probe_record_from_payload(probe_payload)
            row["video"] = metadata
            row["embedded_provenance"] = extract_embedded_h3_provenance(probe_payload)
            if (
                require_embedded_provenance
                and row["embedded_provenance"]["status"] != "parsed_h3_contract"
            ):
                row["manual_review_flags"].append(
                    {
                        "kind": "missing_or_incomplete_embedded_h3_provenance",
                        "status": row["embedded_provenance"]["status"],
                    }
                )
            geometry = center_cover_geometry(
                metadata["width"],
                metadata["height"],
                target_width,
                target_height,
            )
            row["center_cover"] = geometry
            if metadata["duration_seconds"] + 1e-6 < minimum_last_frame_time:
                row["rejection_reasons"].append("too_short_for_strict_h3_window")
            if reject_upscale and geometry["requires_upscale"]:
                row["rejection_reasons"].append("source_requires_upscale")
            flags = suspect_name_flags(path, roots=roots)
            if flags:
                row["manual_review_flags"].append(
                    {"kind": "suspected_derived_or_comparison_name", "tokens": flags}
                )

            if mode in {"strict", "signature"} and not row["rejection_reasons"]:
                strict = strict_decode_window(
                    path,
                    ffmpeg=ffmpeg,
                    window_seconds=window_seconds,
                    attempts=decode_attempts,
                )
                row["strict_decode"] = strict
                if not strict["passed"]:
                    row["rejection_reasons"].append("strict_decode_failed")

            if mode == "signature" and not row["rejection_reasons"]:
                signature = decode_signature(
                    path,
                    ffmpeg=ffmpeg,
                    window_seconds=window_seconds,
                )
                signatures[-1] = signature
                row["decoded_signature"] = {
                    "raw_sha256": signature.raw_sha256,
                    "sampled_frame_count": signature.decoded_frame_count,
                    "sample_fps": 3.0,
                    "side": 16,
                }
                decoded_duplicate = exact_decoded.get(signature.raw_sha256)
                if decoded_duplicate is not None:
                    row["exact_decoded_duplicate_of"] = decoded_duplicate
                    row["rejection_reasons"].append("exact_decoded_window_duplicate")
                else:
                    exact_decoded[signature.raw_sha256] = str(path)
        except (OSError, CurationError, ValueError) as exc:
            row["rejection_reasons"].append("inspection_failed")
            row["inspection_error"] = str(exc)

        if row["rejection_reasons"]:
            row["status"] = "rejected"
        elif row["manual_review_flags"]:
            row["status"] = "manual_review_required"
        else:
            row["status"] = "provisional_candidate"
        items.append(row)

    near_pairs: list[tuple[int, int]] = []
    if mode == "signature":
        for left in range(len(items)):
            left_signature = signatures[left]
            if left_signature is None or items[left]["status"] == "rejected":
                continue
            for right in range(left + 1, len(items)):
                right_signature = signatures[right]
                if right_signature is None or items[right]["status"] == "rejected":
                    continue
                similarity = temporal_hash_similarity(
                    left_signature.frame_hashes, right_signature.frame_hashes
                )
                if similarity >= near_duplicate_threshold:
                    near_pairs.append((left, right))
        for group_number, indices in enumerate(
            _union_find_groups(near_pairs, len(items)), start=1
        ):
            group_id = f"near-{group_number:04d}"
            paths = [items[index]["path"] for index in indices]
            for index in indices:
                items[index]["near_duplicate_group"] = group_id
                items[index]["manual_review_flags"].append(
                    {
                        "kind": "low_resolution_temporal_hash_near_duplicate",
                        "group": group_id,
                        "members": paths,
                        "threshold": near_duplicate_threshold,
                    }
                )
                if items[index]["status"] != "rejected":
                    items[index]["status"] = "manual_review_required"

    counts = {
        name: sum(row["status"] == name for row in items)
        for name in ("provisional_candidate", "manual_review_required", "rejected")
    }
    return {
        "schema": SCHEMA,
        "read_only": True,
        "mode": mode,
        "roots": [str(root.expanduser().resolve()) for root in roots],
        "policy": {
            "target_width": target_width,
            "target_height": target_height,
            "required_frames": required_frames,
            "target_fps": target_fps,
            "minimum_last_frame_time_seconds": minimum_last_frame_time,
            "strict_decode_attempts": decode_attempts if mode != "metadata" else 0,
            "reject_upscale": reject_upscale,
            "near_duplicate_threshold": near_duplicate_threshold,
            "near_duplicate_is_manual_only": True,
            "require_embedded_provenance": require_embedded_provenance,
        },
        "counts": {"discovered": len(items), **counts},
        "items": items,
        "provenance": {
            "raw_prompt_or_workflow_text_emitted": False,
            "status_counts": {
                status: sum(
                    row.get("embedded_provenance", {}).get("status") == status
                    for row in items
                )
                for status in (
                    "parsed_h3_contract",
                    "partial_h3_contract",
                    "missing_prompt_tag",
                    "malformed_prompt_tag",
                    "non_h3_prompt",
                )
            },
            "contract_groups": provenance_contract_groups(items),
        },
        "decision": {
            "formal_dataset_authorized": False,
            "independence_validated": False,
            "content_diversity_validated": False,
            "reason": (
                "This tool rejects mechanical corruption and exact duplicates, and only flags "
                "heuristic near-duplicates. A human must still verify that provisional clips are "
                "independent, representative and not derived from one another before they count "
                "toward the 100-clip SPEED calibration gate."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only inventory and mechanical curation for MiniMax H3 SPEED spectrum "
            "calibration sources. It never moves or deletes media."
        )
    )
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("metadata", "strict", "signature"), default="metadata"
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--target-width", type=int, default=736)
    parser.add_argument("--target-height", type=int, default=416)
    parser.add_argument("--required-frames", type=int, default=124)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--decode-attempts", type=int, default=1)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.98)
    parser.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Do not reject sources that are smaller than the target center-cover canvas.",
    )
    parser.add_argument(
        "--require-embedded-provenance",
        action="store_true",
        help=(
            "Require a parsed embedded H3 task/model/video-VAE contract for a provisional "
            "candidate; missing metadata becomes manual review rather than formal evidence."
        ),
    )
    args = parser.parse_args()
    result = curate_sources(
        roots=args.root,
        mode=args.mode,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        target_width=args.target_width,
        target_height=args.target_height,
        required_frames=args.required_frames,
        target_fps=args.target_fps,
        decode_attempts=args.decode_attempts,
        near_duplicate_threshold=args.near_duplicate_threshold,
        reject_upscale=not args.allow_upscale,
        require_embedded_provenance=args.require_embedded_provenance,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    print(result["decision"]["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
