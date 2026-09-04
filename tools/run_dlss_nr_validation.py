#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import av
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
DEFAULT_MODELS_DIR = COMFY_ROOT / "models"
REPORT_SCHEMA = "t8.dlss_nr.real_validation.v1"
P3_REVIEW_SCHEMA = "t8.dlss_nr.p3_human_review.v1"
SPEECH_PHRASE = "你在哪里"
INPUT_MANIFEST_SCHEMA = "t8.dlss_nr.validation_inputs.v1"


def _load_dlss_module():
    try:
        return importlib.import_module("h3_audio_t8_pkg.dlss_nr_advanced")
    except ModuleNotFoundError:
        package_name = "h3_audio_t8_dlss_validation"
        if str(COMFY_ROOT) not in sys.path:
            sys.path.insert(0, str(COMFY_ROOT))
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_name,
                ROOT / "__init__.py",
                submodule_search_locations=[str(ROOT)],
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            assert spec.loader is not None
            spec.loader.exec_module(package)
        return importlib.import_module(f"{package_name}.dlss_nr_advanced")


dlss = _load_dlss_module()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_output_directory(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing DLSS-NR validation evidence: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _bind_prepared_inputs(
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    image: Path | None,
    speech_video: Path | None,
    hard_cut_video: Path | None,
    fine_texture_video: Path | None,
) -> dict[str, Any]:
    manifest = manifest_path.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"validation input manifest does not exist: {manifest}")
    expected_sha = expected_manifest_sha256.strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError(
            "validation input manifest SHA-256 must contain exactly 64 hex characters"
        )
    actual_sha = _sha256_file(manifest)
    if actual_sha != expected_sha:
        raise ValueError(
            f"validation input manifest SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("validation input manifest is not valid JSON") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema") != INPUT_MANIFEST_SCHEMA
    ):
        raise ValueError(
            f"validation input manifest must use schema {INPUT_MANIFEST_SCHEMA!r}"
        )
    if document.get("status") != "PREPARED_NOT_DLSS_TESTED":
        raise ValueError("validation input manifest has an unsupported status")
    gate_effect = document.get("gate_effect")
    if not isinstance(gate_effect, dict) or any(
        gate_effect.get(key) is not False
        for key in (
            "p2_complete",
            "p3_complete",
            "p4_complete",
            "automatic_promotion",
        )
    ):
        raise ValueError(
            "validation input manifest must not claim a completed DLSS gate"
        )
    confirmations = document.get("operator_confirmations")
    if (
        not isinstance(confirmations, dict)
        or confirmations.get("speech_phrase") != SPEECH_PHRASE
    ):
        raise ValueError(
            "validation input manifest does not bind the required speech phrase"
        )
    if any(
        confirmations.get(key) is not True
        for key in (
            "speech_phrase_clearly_audible",
            "hard_cut_is_intentional",
            "fine_texture_overlay_is_intentional",
        )
    ):
        raise ValueError(
            "validation input manifest is missing an operator/source confirmation"
        )
    prepared = document.get("prepared_inputs")
    if not isinstance(prepared, dict):
        raise ValueError("validation input manifest has no prepared_inputs object")
    requested = {
        "p2_image": image,
        "speech": speech_video,
        "hard_cut": hard_cut_video,
        "fine_texture": fine_texture_video,
    }
    bound = {}
    root = manifest.parent
    for name, supplied in requested.items():
        if supplied is None:
            continue
        record = prepared.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"validation input manifest has no {name!r} record")
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative.strip()
            or Path(relative).is_absolute()
        ):
            raise ValueError(
                f"validation input manifest {name!r} path must be relative"
            )
        recorded_path = (root / relative).resolve()
        if not recorded_path.is_relative_to(root):
            raise ValueError(
                f"validation input manifest {name!r} path escapes its bundle"
            )
        if recorded_path != supplied.resolve():
            raise ValueError(
                f"supplied {name!r} path differs from the hash-bound manifest"
            )
        if not recorded_path.is_file():
            raise FileNotFoundError(
                f"prepared validation input is missing: {recorded_path}"
            )
        try:
            expected_bytes = int(record["bytes"])
            expected_file_sha = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"validation input manifest {name!r} file record is malformed"
            ) from exc
        actual_bytes = recorded_path.stat().st_size
        actual_file_sha = _sha256_file(recorded_path)
        if expected_bytes != actual_bytes or expected_file_sha != actual_file_sha:
            raise ValueError(
                f"prepared validation input {name!r} no longer matches its manifest"
            )
        bound[name] = {
            "path": str(recorded_path),
            "bytes": actual_bytes,
            "sha256": actual_file_sha,
        }
    return {
        "schema": INPUT_MANIFEST_SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": actual_sha,
        "bound_inputs": bound,
        "p2_p4_claimed_complete": False,
    }


def _image_tensor(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        original_mode = opened.mode
        rgb = np.array(opened.convert("RGB"), dtype=np.uint8, copy=True)
    height, width = rgb.shape[:2]
    megapixels = width * height / 1_000_000.0
    if not 0.40 <= megapixels <= 0.65:
        raise ValueError(
            "P2 representative image must be approximately 0.5 MP "
            f"(0.40..0.65 MP), got {width}x{height} ({megapixels:.4f} MP)"
        )
    tensor = torch.from_numpy(rgb).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
    return tensor, {
        "path": str(source),
        "sha256": _sha256_file(source),
        "original_mode": original_mode,
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 6),
        "bridge": "PIL RGB -> uint8 -> float32 BHWC 0..1",
    }


def _save_tensor_png(tensor: torch.Tensor, path: Path) -> dict[str, Any]:
    if tuple(tensor.shape[:1]) != (1,) or tensor.ndim != 4 or tensor.shape[-1] < 3:
        raise ValueError("validation PNG writer requires one BHWC RGB image")
    rgb = (
        tensor[0, ..., :3]
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(rgb, mode="RGB").save(path)
    with Image.open(path) as opened:
        width, height = opened.size
        mode = opened.mode
    quantized = bool(
        torch.equal(
            tensor[0, ..., :3].detach().float().cpu().mul(255.0).round(),
            tensor[0, ..., :3].detach().float().cpu().mul(255.0),
        )
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "bits_per_channel": 8,
        "rgb_quantized_to_uint8_grid": quantized,
    }


def _save_lanczos(source: torch.Tensor, path: Path) -> dict[str, Any]:
    rgb = source[0, ..., :3].mul(255.0).round().to(torch.uint8).cpu().numpy()
    height, width = rgb.shape[:2]
    target_width, target_height = dlss.target_dimensions(width, height, 2.0)
    image = Image.fromarray(rgb, mode="RGB").resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    image.save(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "width": target_width,
        "height": target_height,
        "mode": "RGB",
        "bits_per_channel": 8,
        "method": "Pillow Lanczos from the same rounded RGB8 source bridge",
    }


def _nvidia_memory_sample(gpu_index: int) -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=10,
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError(
            "nvidia-smi memory sampling failed: " + completed.stderr[-1000:]
        )
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4 and int(fields[0]) == int(gpu_index):
            return {
                "total_mib": float(fields[1]),
                "used_mib": float(fields[2]),
                "free_mib": float(fields[3]),
            }
    raise RuntimeError(f"nvidia-smi did not return GPU index {gpu_index}")


def _measure_vram(
    gpu_index: int, operation: Callable[[], Any]
) -> tuple[Any, dict[str, Any]]:
    stop = threading.Event()
    samples: list[dict[str, float]] = []
    errors: list[str] = []

    def sample() -> None:
        while not stop.is_set():
            try:
                samples.append(_nvidia_memory_sample(gpu_index))
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                errors.append(str(error))
                return
            stop.wait(0.1)

    started = time.perf_counter()
    worker = threading.Thread(target=sample, name="t8-dlss-vram-sampler", daemon=True)
    worker.start()
    try:
        result = operation()
    finally:
        stop.set()
        worker.join(timeout=12)
        if worker.is_alive():
            raise RuntimeError("VRAM sampler did not stop after the serial DLSS-NR run")
        try:
            samples.append(_nvidia_memory_sample(gpu_index))
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(str(error))
    if not samples:
        raise RuntimeError("no NVIDIA VRAM sample was captured")
    return result, {
        "sampler": "nvidia-smi at 100 ms while one DLSS-NR job ran",
        "sample_count": len(samples),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "total_mib": samples[0]["total_mib"],
        "start_used_mib": samples[0]["used_mib"],
        "peak_used_mib": max(item["used_mib"] for item in samples),
        "minimum_free_mib": min(item["free_mib"] for item in samples),
        "sampling_errors": errors,
    }


def _nvidia_index(revalidation: dict[str, Any]) -> int:
    selected = revalidation["device_mapping"]["nvidia"]
    return int(selected["index"])


def _run_p2(
    runtime: dict[str, Any], source_path: Path, output_root: Path
) -> dict[str, Any]:
    p2_root = output_root / "p2-image"
    p2_root.mkdir()
    source, source_report = _image_tensor(source_path)
    source_output = p2_root / "source_rgb8.png"
    source_saved = _save_tensor_png(source, source_output)
    lanczos = _save_lanczos(source, p2_root / "lanczos_2x.png")
    runs = []
    for name, mode, scale in (
        ("dlss_nr_1x", "nr_only", 1.0),
        ("dlss_sr_2x", "sr_only", 2.0),
        ("dlss_sr_nr_2x", "sr_nr", 2.0),
    ):
        revalidation = dlss.revalidate_runtime_handle(runtime)
        gpu_index = _nvidia_index(revalidation)

        def execute():
            return dlss.process_image_batch(runtime, source, mode=mode, scale=scale)

        (candidate, returned_source, process_report), memory = _measure_vram(
            gpu_index, execute
        )
        if returned_source is not source:
            raise RuntimeError(
                "P2 image execution did not return the exact source tensor object"
            )
        saved = _save_tensor_png(candidate, p2_root / f"{name}.png")
        if saved["rgb_quantized_to_uint8_grid"] is not True:
            raise RuntimeError(
                "DLSS-NR RGB output did not stay on its declared uint8 bridge grid"
            )
        runs.append(
            {
                "name": name,
                "mode": mode,
                "scale": scale,
                "runtime_revalidation": revalidation,
                "memory": memory,
                "output": saved,
                "process": process_report,
            }
        )
    return {
        "status": "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED",
        "source": source_report | {"saved_bridge": source_saved},
        "baseline": lanczos,
        "runs": runs,
        "fairness": {
            "same_rgb8_source_bridge": True,
            "same_exact_2x_dimensions": True,
            "automatic_quality_winner": None,
        },
        "human_review_complete": False,
    }


class FileBackedVideo:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with av.open(str(self.path), mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError(
                    "validation VIDEO must contain exactly one video stream"
                )
            stream = container.streams.video[0]
            if stream.average_rate is None:
                raise ValueError("validation VIDEO has no declared frame rate")
            self.width = int(stream.width)
            self.height = int(stream.height)
            self.rate = float(stream.average_rate)
            self.frame_count = sum(1 for _frame in container.decode(stream))

    def get_stream_source(self):
        return str(self.path)

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_frame_count(self):
        return self.frame_count

    def get_dimensions(self):
        return self.width, self.height

    def get_bit_depth(self):
        return 8

    def get_frame_rate(self):
        return self.rate


def _thumbnail_rgb(
    frame: av.VideoFrame, width: int = 96, height: int = 54
) -> np.ndarray:
    return (
        frame.reformat(width=width, height=height, format="rgb24")
        .to_ndarray()
        .astype(np.float32)
        / 255.0
    )


def _video_screen(
    source_path: Path, candidate_path: Path, *, hard_cut: bool
) -> dict[str, Any]:
    source_means = []
    candidate_means = []
    source_motion = []
    candidate_motion = []
    source_frames = []
    candidate_frames = []
    with (
        av.open(str(source_path), mode="r") as source_container,
        av.open(str(candidate_path), mode="r") as candidate_container,
    ):
        source_stream = source_container.streams.video[0]
        candidate_stream = candidate_container.streams.video[0]
        pairs = zip(
            source_container.decode(source_stream),
            candidate_container.decode(candidate_stream),
            strict=True,
        )
        previous_source = None
        previous_candidate = None
        for source_frame, candidate_frame in pairs:
            source = _thumbnail_rgb(source_frame)
            candidate = _thumbnail_rgb(candidate_frame)
            source_frames.append(source)
            candidate_frames.append(candidate)
            source_means.append(source.mean(axis=(0, 1)))
            candidate_means.append(candidate.mean(axis=(0, 1)))
            if previous_source is not None:
                source_motion.append(float(np.mean(np.abs(source - previous_source))))
                candidate_motion.append(
                    float(np.mean(np.abs(candidate - previous_candidate)))
                )
            previous_source = source
            previous_candidate = candidate
    if not source_frames or len(source_frames) != len(candidate_frames):
        raise RuntimeError(
            "P3 screening did not decode equal non-empty frame sequences"
        )
    black_regressions = []
    for index, (source, candidate) in enumerate(
        zip(source_frames, candidate_frames, strict=True)
    ):
        if (
            candidate.mean() < 0.01
            and candidate.std() < 0.01
            and (source.mean() >= 0.03 or source.std() >= 0.02)
        ):
            black_regressions.append(index)
    freeze_regressions = [
        index + 1
        for index, (source_delta, candidate_delta) in enumerate(
            zip(source_motion, candidate_motion, strict=True)
        )
        if source_delta > 4.0 / 255.0 and candidate_delta < 0.25 / 255.0
    ]
    cut_report = None
    if hard_cut:
        cut_offset = int(np.argmax(source_motion))
        cut_index = cut_offset + 1
        source_cut_delta = source_motion[cut_offset]
        candidate_cut_delta = candidate_motion[cut_offset]
        current_error = float(
            np.mean(np.abs(candidate_frames[cut_index] - source_frames[cut_index]))
        )
        previous_error = float(
            np.mean(np.abs(candidate_frames[cut_index] - source_frames[cut_index - 1]))
        )
        cut_report = {
            "cut_frame_index": cut_index,
            "source_cut_delta": source_cut_delta,
            "candidate_cut_delta": candidate_cut_delta,
            "source_has_mechanical_hard_cut": source_cut_delta >= 0.08,
            "candidate_preserves_cut_transition": candidate_cut_delta >= 0.02,
            "post_cut_closer_to_current_source_than_previous_source": current_error
            < previous_error,
            "current_source_mae": current_error,
            "previous_source_mae": previous_error,
        }
    source_color = np.mean(np.stack(source_means), axis=0)
    candidate_color = np.mean(np.stack(candidate_means), axis=0)
    return {
        "thumbnail_geometry": [96, 54],
        "decoded_frame_count": len(source_frames),
        "black_regression_frames": black_regressions,
        "freeze_regression_frames": freeze_regressions,
        "mean_rgb_source": source_color.tolist(),
        "mean_rgb_candidate": candidate_color.tolist(),
        "mean_rgb_absolute_delta": np.abs(candidate_color - source_color).tolist(),
        "hard_cut": cut_report,
        "quality_ranking": None,
        "limitations": "Mechanical screen only; normal-speed human review remains mandatory.",
    }


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        if _sha256_file(temporary) != _sha256_file(source):
            raise RuntimeError("review source changed while copying")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _p3_review_document(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return rf"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DLSS-NR P3 正常速度人工验收</title><style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#0e1117;color:#eef2f7}}header{{position:sticky;top:0;background:#171c26;padding:16px;z-index:2}}main{{padding:16px;display:grid;gap:16px}}article{{border:1px solid #394252;border-radius:10px;padding:14px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}video{{width:100%;background:#000}}.fields{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}}label{{display:grid;gap:4px}}select,textarea,button{{background:#242c39;color:white;border:1px solid #536077;border-radius:6px;padding:7px}}textarea{{min-height:60px}}@media(max-width:900px){{.pair,.fields{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>DLSS-NR P3 正常速度人工验收</h1><p>三组均须完整、正常速度观看并试听。机械 PASS 不代表口型、身份、皮肤、文字、颜色或时序质量通过。</p><button id="export">导出人工评审 JSON</button></header><main id="cards"></main><script>
const rows={payload};const metrics=[['overall','总体'],['mouth_lipsync','嘴部/口型'],['face_identity_skin','人脸/身份/皮肤'],['text_fine_texture','文字/细纹理'],['color','颜色'],['temporal_stability','时序稳定'],['cut_history','切镜历史拖带'],['audio','音频']];const choices=[['pending','未检查'],['pass','通过'],['fail','失败'],['not_applicable','不适用'],['unsure','不确定']];let saved={{}};try{{saved=JSON.parse(localStorage.getItem('t8-dlss-p3-review-v1')||'{{}}')}}catch(_){{}}function persist(e){{const x=e.currentTarget;saved[x.dataset.id]||={{}};saved[x.dataset.id][x.dataset.field]=x.value;localStorage.setItem('t8-dlss-p3-review-v1',JSON.stringify(saved))}}const cards=document.getElementById('cards');for(const row of rows){{const article=document.createElement('article');article.innerHTML=`<h2>${{row.label}}</h2><div class="pair"><section><h3>原片</h3><video controls preload="metadata" src="${{row.source}}"></video></section><section><h3>DLSS-NR</h3><video controls preload="metadata" src="${{row.candidate}}"></video></section></div>`;const fields=document.createElement('div');fields.className='fields';for(const [field,label] of metrics){{const box=document.createElement('label');box.textContent=label;const select=document.createElement('select');select.dataset.id=row.id;select.dataset.field=field;for(const [value,text] of choices){{const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option)}}select.value=saved[row.id]?.[field]||'pending';select.onchange=persist;box.append(select);fields.append(box)}}const notes=document.createElement('label');notes.textContent='问题与时间点';const area=document.createElement('textarea');area.dataset.id=row.id;area.dataset.field='notes';area.value=saved[row.id]?.notes||'';area.oninput=persist;notes.append(area);fields.append(notes);article.append(fields);cards.append(article)}}document.getElementById('export').onclick=()=>{{const reviews=rows.map(row=>({{clip_id:row.id,...(saved[row.id]||{{}})}}));const out={{schema:'{P3_REVIEW_SCHEMA}',exported_at:new Date().toISOString(),reviews}};const blob=new Blob([JSON.stringify(out,null,2)+'\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='dlss_nr_p3_human_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}};
</script></body></html>"""


def _run_p3(
    runtime: dict[str, Any],
    clips: dict[str, Path],
    output_root: Path,
    *,
    speech_phrase_confirmation: str,
    hard_cut_source_confirmed: bool,
    fine_texture_source_confirmed: bool,
    motion_engine: str,
    crf: float,
) -> dict[str, Any]:
    if speech_phrase_confirmation.strip() != SPEECH_PHRASE:
        raise ValueError(
            f"P3 requires an operator confirmation that the speech source clearly says {SPEECH_PHRASE!r}"
        )
    if not hard_cut_source_confirmed:
        raise ValueError(
            "P3 requires confirmation that the hard-cut source contains a real hard cut"
        )
    if not fine_texture_source_confirmed:
        raise ValueError(
            "P3 requires confirmation that the fine-texture source contains readable text or fine texture"
        )
    p3_root = output_root / "p3-video"
    p3_root.mkdir()
    review_media = p3_root / "review_media"
    review_media.mkdir()
    reports = []
    review_rows = []
    for clip_id, label in (
        ("speech", f"人物对白（{SPEECH_PHRASE}）"),
        ("hard_cut", "硬切镜头"),
        ("fine_texture", "字幕与细纹理"),
    ):
        source = FileBackedVideo(clips[clip_id])
        if clip_id == "speech":
            megapixels = source.width * source.height / 1_000_000.0
            if source.frame_count != 124 or not 0.40 <= megapixels <= 0.65:
                raise ValueError(
                    "P3 speech source must be 124 frames and near 0.5 MP; got "
                    f"{source.width}x{source.height}x{source.frame_count}"
                )
            source_contract = dlss._file_source_contract(source)[1]
            if not source_contract["audio_packets"]:
                raise ValueError(
                    "P3 speech source must contain a decodable audio stream"
                )
        revalidation = dlss.revalidate_runtime_handle(runtime)
        gpu_index = _nvidia_index(revalidation)
        candidate_path = p3_root / f"{clip_id}_dlss_sr_nr_2x.mp4"

        def execute():
            return dlss.process_video_file(
                runtime,
                source,
                output_path=candidate_path,
                mode="sr_nr",
                scale=2.0,
                motion_engine=motion_engine,
                crf=crf,
            )

        (published, returned_source, process_report), memory = _measure_vram(
            gpu_index, execute
        )
        if returned_source is not source or published != candidate_path.resolve():
            raise RuntimeError(
                "P3 file execution returned a different source or output"
            )
        screen = _video_screen(source.path, published, hard_cut=clip_id == "hard_cut")
        hard_cut_pass = True
        if clip_id == "hard_cut":
            cut = screen["hard_cut"]
            hard_cut_pass = all(
                (
                    cut["source_has_mechanical_hard_cut"],
                    cut["candidate_preserves_cut_transition"],
                    cut["post_cut_closer_to_current_source_than_previous_source"],
                )
            )
        mechanical_pass = all(
            (
                not screen["black_regression_frames"],
                not screen["freeze_regression_frames"],
                hard_cut_pass,
                process_report["audio"]["packet_payload_exact"],
                process_report["audio"]["packet_timeline_exact"],
                process_report["audio"]["decoded_pcm_exact"],
                process_report["audio"]["decoded_timeline_exact"],
                process_report["motion_engine_resolved"] in {"nvof", "lk"},
                process_report["video"]["input_frame_count"]
                == process_report["video"]["output_frame_count"],
            )
        )
        if not mechanical_pass:
            raise RuntimeError(f"P3 mechanical screen failed for {clip_id}")
        source_copy = review_media / f"{clip_id}_source{source.path.suffix.lower()}"
        _copy_or_link(source.path, source_copy)
        reports.append(
            {
                "clip_id": clip_id,
                "label": label,
                "source": {
                    "path": str(source.path),
                    "sha256": _sha256_file(source.path),
                    "width": source.width,
                    "height": source.height,
                    "frame_count": source.frame_count,
                    "fps": source.rate,
                },
                "candidate": {
                    "path": str(published),
                    "sha256": _sha256_file(published),
                },
                "runtime_revalidation": revalidation,
                "memory": memory,
                "process": process_report,
                "screen": screen,
                "mechanical_pass": True,
                "human_review_complete": False,
            }
        )
        review_rows.append(
            {
                "id": clip_id,
                "label": label,
                "source": f"review_media/{source_copy.name}",
                "candidate": candidate_path.name,
            }
        )
    (p3_root / "p3_review.html").write_text(
        _p3_review_document(review_rows), encoding="utf-8", newline="\n"
    )
    return {
        "status": "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED",
        "speech_phrase_operator_confirmation": speech_phrase_confirmation,
        "speech_phrase_machine_verified": False,
        "hard_cut_source_operator_confirmation": True,
        "fine_texture_source_operator_confirmation": True,
        "runs_are_strictly_serial": True,
        "stress_or_parallel_generation": False,
        "runs": reports,
        "review_page": str((p3_root / "p3_review.html").resolve()),
        "human_review_complete": False,
    }


def run_validation(
    *,
    models_dir: Path,
    runtime_version: str,
    output_dir: Path,
    stage: str,
    accept_external_runtime_license: bool,
    dxgi_adapter_index: int,
    cuda_device_index: int,
    image: Path | None = None,
    speech_video: Path | None = None,
    hard_cut_video: Path | None = None,
    fine_texture_video: Path | None = None,
    speech_phrase_confirmation: str = "",
    hard_cut_source_confirmed: bool = False,
    fine_texture_source_confirmed: bool = False,
    motion_engine: str = "auto",
    crf: float = 18.0,
    input_manifest: Path | None = None,
    input_manifest_sha256: str = "",
) -> dict[str, Any]:
    if stage not in {"p2", "p3", "all"}:
        raise ValueError("stage must be p2, p3 or all")
    if stage in {"p2", "all"} and image is None:
        raise ValueError("P2 requires --image")
    if stage in {"p3", "all"} and None in (
        speech_video,
        hard_cut_video,
        fine_texture_video,
    ):
        raise ValueError("P3 requires speech, hard-cut and fine-texture videos")
    output = _new_output_directory(output_dir)
    ready, audit = dlss.audit_dlss_nr_runtime(
        dlss.runtime_root(models_dir, runtime_version),
        runtime_version,
        accept_external_runtime_license=accept_external_runtime_license,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=dxgi_adapter_index,
        cuda_device_index=cuda_device_index,
    )
    _write_json_atomic(output / "runtime_audit.json", audit)
    if not ready:
        raise RuntimeError(
            "DLSS-NR runtime did not pass the real feature gate; see runtime_audit.json"
        )
    if input_manifest is None:
        raise ValueError(
            "a hash-bound --input-manifest is required after the runtime is ready"
        )
    input_binding = _bind_prepared_inputs(
        input_manifest,
        input_manifest_sha256,
        image=image if stage in {"p2", "all"} else None,
        speech_video=speech_video if stage in {"p3", "all"} else None,
        hard_cut_video=hard_cut_video if stage in {"p3", "all"} else None,
        fine_texture_video=fine_texture_video if stage in {"p3", "all"} else None,
    )
    runtime = dlss.runtime_handle_from_report(audit)
    if runtime is None:
        raise RuntimeError("ready DLSS-NR audit did not produce a runtime handle")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "INCOMPLETE",
        "requested_stage": stage,
        "runtime_audit": audit,
        "input_binding": input_binding,
        "serial_execution": True,
        "automatic_quality_winner": None,
        "p2": None,
        "p3": None,
    }
    if stage in {"p2", "all"}:
        assert image is not None
        report["p2"] = _run_p2(runtime, image, output)
    if stage in {"p3", "all"}:
        assert speech_video is not None
        assert hard_cut_video is not None
        assert fine_texture_video is not None
        report["p3"] = _run_p3(
            runtime,
            {
                "speech": speech_video,
                "hard_cut": hard_cut_video,
                "fine_texture": fine_texture_video,
            },
            output,
            speech_phrase_confirmation=speech_phrase_confirmation,
            hard_cut_source_confirmed=hard_cut_source_confirmed,
            fine_texture_source_confirmed=fine_texture_source_confirmed,
            motion_engine=motion_engine,
            crf=crf,
        )
    report["status"] = "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED"
    report["completed_at_unix"] = time.time()
    _write_json_atomic(output / "validation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated P2/P3 DLSS-NR real validation strictly serially."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--runtime-version", default="1.3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("p2", "p3", "all"), default="all")
    parser.add_argument("--accept-external-runtime-license", action="store_true")
    parser.add_argument("--dxgi-adapter-index", type=int, default=0)
    parser.add_argument("--cuda-device-index", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--speech-video", type=Path)
    parser.add_argument("--hard-cut-video", type=Path)
    parser.add_argument("--fine-texture-video", type=Path)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--confirm-speech-phrase", default="")
    parser.add_argument("--confirm-hard-cut-source", action="store_true")
    parser.add_argument("--confirm-fine-texture-source", action="store_true")
    parser.add_argument("--motion-engine", choices=dlss.MOTION_ENGINES, default="auto")
    parser.add_argument("--crf", type=float, default=18.0)
    args = parser.parse_args()
    report = run_validation(
        models_dir=args.models_dir,
        runtime_version=args.runtime_version,
        output_dir=args.output_dir,
        stage=args.stage,
        accept_external_runtime_license=args.accept_external_runtime_license,
        dxgi_adapter_index=args.dxgi_adapter_index,
        cuda_device_index=args.cuda_device_index,
        image=args.image,
        speech_video=args.speech_video,
        hard_cut_video=args.hard_cut_video,
        fine_texture_video=args.fine_texture_video,
        speech_phrase_confirmation=args.confirm_speech_phrase,
        hard_cut_source_confirmed=args.confirm_hard_cut_source,
        fine_texture_source_confirmed=args.confirm_fine_texture_source,
        motion_engine=args.motion_engine,
        crf=args.crf,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
    )
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output_dir.resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
