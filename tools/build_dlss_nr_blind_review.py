#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

try:
    from . import run_dlss_nr_validation as validation_tool
except ImportError:  # pragma: no cover - direct script execution
    import run_dlss_nr_validation as validation_tool


dlss = validation_tool.dlss
MANIFEST_SCHEMA = "t8.dlss_nr.blind_manifest.v1"
PACKAGE_SCHEMA = "t8.dlss_nr.blind_package.v1"
SCREENING_SCHEMA = "t8.dlss_nr.mechanical_screening.v1"
REVIEW_SCHEMA = "t8.dlss_nr.blind_review.v1"
METHODS = ("lanczos", "realbasicvsr", "flashvsr", "dlss_nr")
METHOD_PROFILES = {
    "lanczos": "lanczos_2x",
    "realbasicvsr": "conservative",
    "flashvsr": "quality_locked",
    "dlss_nr": "standard",
}
CLIP_TYPES = {"speech", "hard_cut", "fine_texture"}
CODES = ("A", "B", "C", "D")
PUBLIC_CLIP_LABELS = {
    "speech": "人物对白",
    "hard_cut": "硬切镜头",
    "fine_texture": "字幕与细纹理",
}


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} must contain exactly {sorted(keys)}, got {actual}")
    return value


def _resolve_file(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _media_contract(path: Path) -> dict[str, Any]:
    video = validation_tool.FileBackedVideo(path)
    resolved, contract = dlss._file_source_contract(video)
    if resolved != path.resolve():
        raise RuntimeError("strict VIDEO contract resolved a different source")
    return contract


def _validate_method_contract(
    *,
    clip_id: str,
    source_contract: dict[str, Any],
    candidate_contract: dict[str, Any],
) -> None:
    source_width = int(source_contract["width"])
    source_height = int(source_contract["height"])
    expected_width, expected_height = dlss.target_dimensions(
        source_width, source_height, 2.0
    )
    if (int(candidate_contract["width"]), int(candidate_contract["height"])) != (
        expected_width,
        expected_height,
    ):
        raise ValueError(
            f"P4 clip {clip_id} candidate must be exact 2x "
            f"{expected_width}x{expected_height}"
        )
    if int(candidate_contract["frame_count"]) != int(source_contract["frame_count"]):
        raise ValueError(f"P4 clip {clip_id} candidate frame count differs from source")
    if Fraction(candidate_contract["rate"]) != Fraction(source_contract["rate"]):
        raise ValueError(f"P4 clip {clip_id} candidate fps differs from source")
    source_duration = Fraction(int(source_contract["frame_count"]), 1) / Fraction(
        source_contract["rate"]
    )
    candidate_duration = Fraction(int(candidate_contract["frame_count"]), 1) / Fraction(
        candidate_contract["rate"]
    )
    if source_duration != candidate_duration:
        raise ValueError(f"P4 clip {clip_id} candidate duration differs from source")


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg is required to normalize the P4 blind candidates")
    return path


def _encoding_contract(bitrate_kbps: int) -> dict[str, Any]:
    return {
        "decoder": "ffmpeg default software decode, threads=1",
        "video_encoder": "libx264",
        "preset": "medium",
        "pixel_format": "yuv420p",
        "color_metadata": "bt709 primaries/transfer/matrix, limited range",
        "rate_control": "CBR with nal-hrd=cbr and force-cfr=1",
        "decoder_stability": "single-thread x264, one reference frame, no B-frames",
        "bitrate_kbps": int(bitrate_kbps),
        "minrate_kbps": int(bitrate_kbps),
        "maxrate_kbps": int(bitrate_kbps),
        "bufsize_kbps": int(bitrate_kbps) * 2,
        "encoder_threads": 1,
        "audio": "packet-copy from the authoritative source, candidate audio ignored",
    }


def _normalized_stream_probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,bit_rate,"
            "color_range,color_space,color_transfer,color_primaries",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError("FFprobe failed on normalized P4 media: " + completed.stderr[-2000:])
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("normalized P4 media has no unique video stream")
    stream = streams[0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError("normalized P4 media did not use the fixed H.264/yuv420p contract")
    color = {
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }
    if any(stream.get(field) != value for field, value in color.items()):
        raise RuntimeError("normalized P4 media did not use the fixed Rec.709 limited metadata")
    return stream


def _normalize_candidate(
    *,
    source_path: Path,
    source_contract: dict[str, Any],
    candidate_path: Path,
    target_path: Path,
    bitrate_kbps: int,
) -> dict[str, Any]:
    ffmpeg = _ffmpeg_path()
    rate = Fraction(source_contract["rate"])
    frame_count = int(source_contract["frame_count"])
    width = int(source_contract["width"]) * 2
    height = int(source_contract["height"]) * 2
    video_only = target_path.with_name(
        f".{target_path.stem}.video-only-{os.getpid()}-{time.time_ns()}.mp4"
    )
    combined = target_path.with_name(
        f".{target_path.stem}.partial-{os.getpid()}-{time.time_ns()}.mp4"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(candidate_path),
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        str(frame_count),
        "-r",
        str(rate),
        "-fps_mode",
        "cfr",
        "-vf",
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-threads:v",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-b:v",
        f"{bitrate_kbps}k",
        "-minrate",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{bitrate_kbps * 2}k",
        "-x264-params",
        "nal-hrd=cbr:force-cfr=1:ref=1:bframes=0:threads=1:"
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-movflags",
        "+faststart",
        str(video_only),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=3600,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            raise RuntimeError(
                f"P4 normalization failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
        dlss._packet_copy_video_and_audio(video_only, source_path, combined)
        final_validation = dlss._validate_final_file(
            combined,
            frame_count=frame_count,
            width=width,
            height=height,
            rate=rate,
            source_audio_packets=source_contract["audio_packets"],
            source_audio_pcm=source_contract["audio_pcm"],
        )
        stream = _normalized_stream_probe(combined)
        os.replace(combined, target_path)
    except BaseException:
        combined.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        raise
    finally:
        video_only.unlink(missing_ok=True)
    return {
        "path": str(target_path.resolve()),
        "sha256": validation_tool._sha256_file(target_path),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "encoding_contract": _encoding_contract(bitrate_kbps),
        "stream": stream,
        "validation": final_validation,
        "atomic_publish": True,
        "source_audio_packet_exact": True,
        "source_audio_pcm_exact": True,
    }


def _pair_seed(seed: int, review_id: str) -> int:
    digest = hashlib.sha256(review_id.encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _latin_orders(seed: int, review_id: str, count: int) -> list[list[str]]:
    base = list(METHODS)
    random.Random(_pair_seed(seed, review_id)).shuffle(base)
    return [base[index % len(base) :] + base[: index % len(base)] for index in range(count)]


def _review_document(review_id: str, clips: list[dict[str, Any]]) -> str:
    rows = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    safe_review_id = json.dumps(review_id)
    return rf"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DLSS-NR 四路匿名盲测</title><style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#0d1016;color:#eef2f7}}header{{position:sticky;top:0;z-index:3;background:#171c27f2;padding:16px;border-bottom:1px solid #394252}}main{{padding:16px;display:grid;gap:18px}}article{{border:1px solid #394252;border-radius:10px;padding:14px;background:#151a23}}.source video{{max-width:50%}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}video{{width:100%;background:#000}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}label{{display:grid;gap:4px}}select,textarea,button{{background:#242c39;color:#fff;border:1px solid #536077;border-radius:6px;padding:7px}}textarea{{min-height:70px}}.checks{{display:grid;grid-template-columns:repeat(2,1fr);gap:5px}}.checks label{{display:flex;gap:5px}}@media(max-width:900px){{.grid,.metrics{{grid-template-columns:1fr}}.source video{{max-width:100%}}}}
</style></head><body><header><h1>DLSS-NR 四路匿名盲测</h1><p>每组先完整观看原片，再逐条完整、正常速度观看 A–D；文件名和页面不包含方法名。不要打开 <code>blind_key.json</code> 或 <code>mechanical_screening.json</code>。</p><p>自动指标只筛查坏帧，不参与方法排名。人物、皮肤、嘴部、身份、文字或时序任一明显退化都必须勾选。</p><button id="export">导出评审 JSON</button><button id="clear">清空记录</button><span id="status"></span></header><main id="cards"></main><script>
const clips={rows};const reviewId={safe_review_id};const storageKey='t8-dlss-p4-'+reviewId;let saved={{}};try{{saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}')}}catch(_){{}}const metrics=[['overall','总体最好'],['face_identity_skin','人脸/身份/皮肤最好'],['mouth_lipsync','嘴部/口型最好'],['text_fine_texture','文字/细纹理最好'],['color','颜色最好'],['temporal_stability','时序稳定最好']];const choices=[['pending','未判断'],['tie','无明显差异/平'],['A','A'],['B','B'],['C','C'],['D','D'],['unsure','不确定']];const regressions=[['face_identity_skin','人脸/身份/皮肤退化'],['mouth_lipsync','嘴部/口型退化'],['text_fine_texture','文字/细纹理退化'],['color','颜色退化'],['temporal_stability','时序退化'],['blocking_failure','黑屏/花屏/冻结/坏帧']];function state(id){{return saved[id]||(saved[id]={{metrics:{{}},regressions:{{}},watched:{{}},notes:'',assessability:'pending'}})}}function persist(){{localStorage.setItem(storageKey,JSON.stringify(saved));document.getElementById('status').textContent='已保存'}}const cards=document.getElementById('cards');for(const row of clips){{const s=state(row.clip_id);const article=document.createElement('article');const title=document.createElement('h2');title.textContent=row.label;article.append(title);const source=document.createElement('section');source.className='source';source.innerHTML='<h3>原片</h3><video controls preload="metadata" src="'+row.source+'"></video>';article.append(source);const grid=document.createElement('div');grid.className='grid';for(const side of row.sides){{const box=document.createElement('section');const h=document.createElement('h3');h.textContent=side.code;const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=side.media;const watched=document.createElement('label');watched.textContent='已完整观看 '+side.code;const w=document.createElement('input');w.type='checkbox';w.checked=!!s.watched[side.code];w.onchange=()=>{{s.watched[side.code]=w.checked;persist()}};watched.prepend(w);const checks=document.createElement('div');checks.className='checks';for(const [field,text] of regressions){{const label=document.createElement('label');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=!!s.regressions[side.code]?.[field];checkbox.onchange=()=>{{s.regressions[side.code]||={{}};s.regressions[side.code][field]=checkbox.checked;persist()}};label.append(checkbox,document.createTextNode(text));checks.append(label)}}box.append(h,video,watched,checks);grid.append(box)}}article.append(grid);const fields=document.createElement('div');fields.className='metrics';const assess=document.createElement('label');assess.textContent='本组是否可判断';const aq=document.createElement('select');for(const [value,text] of [['pending','未判断'],['assessable','可判断'],['source_insufficient','原片不足'],['playback_problem','播放问题'],['unsure','不确定']]){{const option=document.createElement('option');option.value=value;option.textContent=text;aq.append(option)}}aq.value=s.assessability;aq.onchange=()=>{{s.assessability=aq.value;persist()}};assess.append(aq);fields.append(assess);for(const [field,text] of metrics){{const label=document.createElement('label');label.textContent=text;const select=document.createElement('select');for(const [value,caption] of choices){{const option=document.createElement('option');option.value=value;option.textContent=caption;select.append(option)}}select.value=s.metrics[field]||'pending';select.onchange=()=>{{s.metrics[field]=select.value;persist()}};label.append(select);fields.append(label)}}const notes=document.createElement('label');notes.textContent='问题与时间点';const area=document.createElement('textarea');area.value=s.notes;area.oninput=()=>{{s.notes=area.value;persist()}};notes.append(area);fields.append(notes);article.append(fields);cards.append(article)}}document.getElementById('export').onclick=()=>{{const reviews=clips.map(row=>({{clip_id:row.clip_id,...state(row.clip_id)}}));const value={{schema:'{REVIEW_SCHEMA}',review_id:reviewId,exported_at:new Date().toISOString(),reviews}};const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='dlss_nr_p4_blind_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}};document.getElementById('clear').onclick=()=>{{if(confirm('确认清空？')){{localStorage.removeItem(storageKey);location.reload()}}}};
</script></body></html>"""


def _prepare_manifest(manifest: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {"schema", "review_id", "bitrate_kbps", "clips"},
        "P4 manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    review_id = manifest["review_id"]
    if not isinstance(review_id, str) or not review_id.strip() or len(review_id) > 100:
        raise ValueError("review_id must be a non-empty string no longer than 100 characters")
    bitrate = manifest["bitrate_kbps"]
    if not isinstance(bitrate, int) or isinstance(bitrate, bool) or not 1_000 <= bitrate <= 100_000:
        raise ValueError("bitrate_kbps must be an integer within 1000..100000")
    clips = manifest["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("P4 manifest clips must be a non-empty list")
    prepared = []
    seen_ids = set()
    seen_types = set()
    for index, raw in enumerate(clips):
        row = _exact_keys(
            raw,
            {"clip_id", "label", "clip_type", "source", "source_sha256", "methods"},
            f"clip[{index}]",
        )
        clip_id = row["clip_id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen_ids:
            raise ValueError("clip_id must be a unique non-empty string")
        seen_ids.add(clip_id)
        clip_type = row["clip_type"]
        if clip_type not in CLIP_TYPES:
            raise ValueError(f"clip_type must be one of {sorted(CLIP_TYPES)}")
        seen_types.add(clip_type)
        source = _resolve_file(manifest_dir, row["source"], f"{clip_id}.source")
        if source.suffix.lower() != ".mp4":
            raise ValueError("P4 authoritative sources must be MP4 for browser review")
        source_sha = validation_tool._sha256_file(source)
        if row["source_sha256"] != source_sha:
            raise ValueError(f"P4 source hash mismatch for {clip_id}")
        source_contract = _media_contract(source)
        methods = _exact_keys(row["methods"], set(METHODS), f"{clip_id}.methods")
        prepared_methods = {}
        for method in METHODS:
            entry = _exact_keys(
                methods[method],
                {"path", "source_sha256", "candidate_sha256", "profile"},
                f"{clip_id}.{method}",
            )
            if entry["source_sha256"] != source_sha:
                raise ValueError(f"P4 {clip_id}/{method} was not bound to the same source hash")
            if entry["profile"] != METHOD_PROFILES[method]:
                raise ValueError(
                    f"P4 {clip_id}/{method} profile must be {METHOD_PROFILES[method]!r}"
                )
            candidate = _resolve_file(
                manifest_dir, entry["path"], f"{clip_id}.{method}.path"
            )
            candidate_sha = validation_tool._sha256_file(candidate)
            if entry["candidate_sha256"] != candidate_sha:
                raise ValueError(f"P4 candidate hash mismatch for {clip_id}/{method}")
            candidate_contract = _media_contract(candidate)
            _validate_method_contract(
                clip_id=clip_id,
                source_contract=source_contract,
                candidate_contract=candidate_contract,
            )
            prepared_methods[method] = {
                "path": candidate,
                "sha256": candidate_sha,
                "profile": entry["profile"],
                "contract": candidate_contract,
            }
        prepared.append(
            {
                "clip_id": clip_id,
                "label": str(row["label"]),
                "clip_type": clip_type,
                "source": source,
                "source_sha256": source_sha,
                "source_contract": source_contract,
                "methods": prepared_methods,
            }
        )
    missing = CLIP_TYPES - seen_types
    if missing:
        raise ValueError(f"P4 manifest is missing required clip types: {sorted(missing)}")
    return {"review_id": review_id.strip(), "bitrate_kbps": bitrate, "clips": prepared}


def build_package(
    manifest: dict[str, Any], manifest_dir: Path, output_dir: Path, blind_seed: int
) -> dict[str, Any]:
    prepared = _prepare_manifest(manifest, manifest_dir.resolve())
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing P4 blind-review evidence: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    media.mkdir()
    orders = _latin_orders(blind_seed, prepared["review_id"], len(prepared["clips"]))
    public_clips = []
    private_clips = []
    mechanical_clips = []
    for index, (clip, order) in enumerate(
        zip(prepared["clips"], orders, strict=True), 1
    ):
        source_target = media / f"clip-{index:02d}-source.mp4"
        validation_tool._copy_or_link(clip["source"], source_target)
        public_sides = []
        private_sides = []
        mechanical_sides = []
        for code, method in zip(CODES, order, strict=True):
            target = media / f"clip-{index:02d}-{code}.mp4"
            normalized = _normalize_candidate(
                source_path=clip["source"],
                source_contract=clip["source_contract"],
                candidate_path=clip["methods"][method]["path"],
                target_path=target,
                bitrate_kbps=prepared["bitrate_kbps"],
            )
            screen = validation_tool._video_screen(
                clip["source"], target, hard_cut=clip["clip_type"] == "hard_cut"
            )
            public_sides.append({"code": code, "media": f"media/{target.name}"})
            private_sides.append(
                {
                    "code": code,
                    "method": method,
                    "profile": clip["methods"][method]["profile"],
                    "source_candidate_path": str(clip["methods"][method]["path"]),
                    "source_candidate_sha256": clip["methods"][method]["sha256"],
                    "normalized_path": str(target),
                    "normalized_sha256": normalized["sha256"],
                }
            )
            mechanical_sides.append(
                {
                    "code": code,
                    "method": method,
                    "normalized": normalized,
                    "screen": screen,
                }
            )
        public_clips.append(
            {
                "clip_id": f"clip-{index:02d}",
                "label": f"第 {index} 组 · {PUBLIC_CLIP_LABELS[clip['clip_type']]}",
                "clip_type": clip["clip_type"],
                "source": f"media/{source_target.name}",
                "sides": public_sides,
            }
        )
        private_clips.append(
            {
                "clip_id": clip["clip_id"],
                "public_clip_id": f"clip-{index:02d}",
                "private_label": clip["label"],
                "clip_type": clip["clip_type"],
                "source_path": str(clip["source"]),
                "source_sha256": clip["source_sha256"],
                "source_contract": {
                    "width": clip["source_contract"]["width"],
                    "height": clip["source_contract"]["height"],
                    "frame_count": clip["source_contract"]["frame_count"],
                    "fps": float(clip["source_contract"]["rate"]),
                },
                "sides": private_sides,
            }
        )
        mechanical_clips.append(
            {
                "clip_id": clip["clip_id"],
                "clip_type": clip["clip_type"],
                "sides": mechanical_sides,
            }
        )
    key = {
        "schema": PACKAGE_SCHEMA,
        "review_id": prepared["review_id"],
        "blind_seed": int(blind_seed),
        "randomization": "seeded base permutation plus cyclic Latin rotation",
        "encoding_contract": _encoding_contract(prepared["bitrate_kbps"]),
        "limitations": [
            "The page contains no method mapping; keep this key private until review export.",
            "Mechanical screening never ranks methods and cannot replace complete human review.",
            "A result applies only to these hash-bound sources, candidates and profiles.",
        ],
        "clips": private_clips,
    }
    screening = {
        "schema": SCREENING_SCHEMA,
        "review_id": prepared["review_id"],
        "quality_ranking": None,
        "clips": mechanical_clips,
    }
    validation_tool._write_json_atomic(output / "blind_key.json", key)
    validation_tool._write_json_atomic(output / "mechanical_screening.json", screening)
    (output / "blind_review.html").write_text(
        _review_document(prepared["review_id"], public_clips),
        encoding="utf-8",
        newline="\n",
    )
    return key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize and package the four-way P4 DLSS-NR comparison blindly."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=2609032201)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = build_package(
        manifest, manifest_path.parent, args.output_dir, args.blind_seed
    )
    print(
        json.dumps(
            {
                "review_id": key["review_id"],
                "clips": len(key["clips"]),
                "output": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
