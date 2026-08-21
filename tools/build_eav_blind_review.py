#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any


MANIFEST_SCHEMA = "t8.eav_blind_manifest.v1"
PACKAGE_SCHEMA = "t8.eav_blind_package.v1"
REVIEW_SCHEMA = "t8.eav_blind_review.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def _media_contract(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_type,width,height,avg_frame_rate,nb_frames,"
                "sample_rate,channels,duration"
            ),
            "-of",
            "json",
            "--",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    streams = json.loads(completed.stdout).get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError(f"Expected exactly one video and one audio stream in {path}")
    video = videos[0]
    audio = audios[0]
    numerator, denominator = str(video["avg_frame_rate"]).split("/", 1)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(numerator) / float(denominator),
        "frame_count": int(video["nb_frames"]),
        "video_duration_seconds": float(video["duration"]),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "audio_duration_seconds": float(audio["duration"]),
    }


def _sync_media(source: Path, target: Path, expected_sha256: str) -> None:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        if _sha256_file(target) == expected_sha256:
            return
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        if _sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"Blind media hash changed while copying {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_media(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _validate_pair_contract(left: dict[str, Any], right: dict[str, Any], pair_id: str) -> None:
    for key in ("width", "height", "fps", "frame_count", "sample_rate", "channels"):
        if left[key] != right[key]:
            raise ValueError(
                f"EAV blind pair {pair_id} differs at {key}: {left[key]} != {right[key]}"
            )
    frame_seconds = 1.0 / float(left["fps"])
    for key in ("video_duration_seconds", "audio_duration_seconds"):
        if abs(float(left[key]) - float(right[key])) > frame_seconds:
            raise ValueError(
                f"EAV blind pair {pair_id} differs by more than one frame at {key}"
            )


def _document(public_pairs: list[dict[str, Any]]) -> str:
    rows = json.dumps(public_pairs, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniMax H3 EAV / FETA 匿名 A/B 评审</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#0d1015;color:#eef1f6}}
header{{position:sticky;top:0;z-index:2;padding:16px 22px;background:#151922ee;border-bottom:1px solid #343b49}}
header h1{{font-size:20px;margin:0 0 8px}}header p{{margin:4px 0;color:#cbd1dc}}button{{padding:8px 12px;border:1px solid #596579;border-radius:7px;background:#263247;color:white}}
main{{padding:18px;display:grid;gap:18px}}article{{border:1px solid #343b49;border-radius:12px;background:#171b24;padding:14px}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.side{{background:#0e1117;border-radius:8px;padding:9px}}
video{{width:100%;max-height:520px;background:#000;border-radius:6px}}.votes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:12px}}
label{{display:grid;gap:4px;color:#cbd1dc;font-size:13px}}select,textarea{{background:#222936;color:white;border:1px solid #4a5568;border-radius:5px;padding:7px}}
textarea{{min-height:64px;resize:vertical}}#status{{margin-left:10px;color:#8bd889}}@media(max-width:900px){{.pair,.votes{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>MiniMax H3 EAV / FETA 匿名 A/B 评审</h1>
<p>每组完整观看并听完 A、B。漏填按“平”导出；不要从文件名猜映射。</p>
<p>分别判断整段画面、运动连续性、提示词/参考遵循和完整音轨；锐度、单帧截图或自动指标不能代替整段审核。</p>
<button id="export">导出评审 JSON</button><span id="status"></span></header><main id="cards"></main>
<script>
const pairs={rows};const storageKey='t8-eav-blind-review-v1';let saved={{}};try{{saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}')}}catch(_){{saved={{}}}}
const choices=[['tie','平'],['A','A更好'],['B','B更好']];
function select(pair,field){{const e=document.createElement('select');e.dataset.pair=pair;e.dataset.field=field;for(const [v,t] of choices){{const o=document.createElement('option');o.value=v;o.textContent=t;e.append(o)}}e.value=saved[pair]?.[field]||'tie';e.onchange=persist;return e}}
function persist(ev){{const e=ev.currentTarget;saved[e.dataset.pair]||={{}};saved[e.dataset.pair][e.dataset.field]=e.value;localStorage.setItem(storageKey,JSON.stringify(saved));document.getElementById('status').textContent='已保存'}}
const cards=document.getElementById('cards');for(const row of pairs){{const a=document.createElement('article');const h=document.createElement('h2');h.textContent=row.label;a.append(h);const p=document.createElement('div');p.className='pair';for(const side of row.sides){{const s=document.createElement('section');s.className='side';const sh=document.createElement('h3');sh.textContent=side.code;const v=document.createElement('video');v.src=side.media;v.preload='metadata';v.controls=true;s.append(sh,v);p.append(s)}}a.append(p);const votes=document.createElement('div');votes.className='votes';const metrics=[['overall','整段画面总体'],['motion','运动连续与自然'],['prompt_reference','提示词/参考遵循'],['audio','完整音轨'],['stability','闪烁/崩坏稳定性']];for(const [field,label] of metrics){{if(field==='prompt_reference'&&!row.reference_metric)continue;const l=document.createElement('label');l.textContent=label;l.append(select(row.pair_id,field));votes.append(l)}}const failure=document.createElement('label');failure.textContent='硬失败';const f=document.createElement('select');f.dataset.pair=row.pair_id;f.dataset.field='blocking_failure';for(const [v,t] of [['none','无'],['A','A有黑屏/花屏/坏帧/明显音频故障'],['B','B有黑屏/花屏/坏帧/明显音频故障'],['both','两边都有']]){{const o=document.createElement('option');o.value=v;o.textContent=t;f.append(o)}}f.value=saved[row.pair_id]?.blocking_failure||'none';f.onchange=persist;failure.append(f);votes.append(failure);a.append(votes);const nl=document.createElement('label');nl.textContent='具体问题与时间点';const n=document.createElement('textarea');n.dataset.pair=row.pair_id;n.dataset.field='notes';n.value=saved[row.pair_id]?.notes||'';n.oninput=persist;nl.append(n);a.append(nl);cards.append(a)}}
document.getElementById('export').onclick=()=>{{const reviews=pairs.map(row=>{{const value=saved[row.pair_id]||{{}};return{{pair_id:row.pair_id,overall:value.overall||'tie',motion:value.motion||'tie',prompt_reference:row.reference_metric?(value.prompt_reference||'tie'):'not_applicable',audio:value.audio||'tie',stability:value.stability||'tie',blocking_failure:value.blocking_failure||'none',notes:value.notes||''}}}});const payload={{schema:'{REVIEW_SCHEMA}',exported_at:new Date().toISOString(),reviews}};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='eav_extended_blind_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}};
</script></body></html>"""


def build_package(
    manifest: dict[str, Any], manifest_dir: Path, output_dir: Path, blind_seed: int
) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest pairs must be a non-empty list")
    seen: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    public_pairs = []
    private_pairs = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("every EAV blind pair must be an object")
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in seen:
            raise ValueError(f"pair_id must be unique and non-empty: {pair_id!r}")
        seen.add(pair_id)
        baseline = _resolve_media(manifest_dir, row.get("baseline"), "baseline")
        candidate = _resolve_media(manifest_dir, row.get("apply_exp"), "apply_exp")
        baseline_contract = _media_contract(baseline)
        candidate_contract = _media_contract(candidate)
        _validate_pair_contract(baseline_contract, candidate_contract, pair_id)
        order = ["baseline", "apply_exp"]
        random.Random(int(blind_seed) ^ index).shuffle(order)
        public_sides = []
        private_sides = []
        for code, arm in zip(("A", "B"), order, strict=True):
            source = baseline if arm == "baseline" else candidate
            digest = _sha256_file(source)
            target_name = f"pair-{index:02d}-{code}{source.suffix.lower()}"
            _sync_media(source, media_dir / target_name, digest)
            public_sides.append({"code": code, "media": f"media/{target_name}"})
            private_sides.append(
                {"code": code, "arm": arm, "source": str(source), "sha256": digest}
            )
        public_pairs.append(
            {
                "pair_id": pair_id,
                "label": str(row.get("label") or f"第 {index} 组"),
                "reference_metric": bool(row.get("reference_metric", False)),
                "sides": public_sides,
            }
        )
        private_pairs.append(
            {
                "pair_id": pair_id,
                "contract": baseline_contract,
                "reference_metric": bool(row.get("reference_metric", False)),
                "sides": private_sides,
            }
        )
    key = {
        "schema": PACKAGE_SCHEMA,
        "blind_seed": int(blind_seed),
        "source_manifest_schema": MANIFEST_SCHEMA,
        "limitations": [
            "One material and one seed per route cannot establish general quality superiority.",
            "Audio must be reviewed because H3 joint AV layers can change it indirectly.",
            "The private key must remain hidden until the review export is complete.",
        ],
        "pairs": private_pairs,
    }
    _write_json_atomic(output_dir / "blind_key.json", key)
    (output_dir / "blind_review.html").write_text(
        _document(public_pairs), encoding="utf-8", newline="\n"
    )
    return key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, hash-traceable EAV/FETA multi-pair blind review."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=2608217301)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = build_package(
        manifest, manifest_path.parent, args.output_dir.resolve(), args.blind_seed
    )
    print(
        json.dumps(
            {"pairs": len(key["pairs"]), "output": str(args.output_dir.resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
