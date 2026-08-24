from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Mapping


MANIFEST_SCHEMA = "minimax_h3_t8_voice_clone_abx_manifest_v1"
KEY_SCHEMA = "minimax_h3_t8_voice_clone_abx_blind_key_v1"
REVIEW_SCHEMA = "minimax_h3_t8_voice_clone_abx_review_v1"
TARGET_POSITION_POLICIES = {"independent_random", "balanced_by_target_and_global"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be non-empty text")
    return value.strip()


def _resolve_regular_file(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    raw = Path(value)
    path = raw if raw.is_absolute() else base / raw
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{field} must resolve to a regular non-symlink file")
    return path


def _audio_contract(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,sample_rate,channels:format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise ValueError(
            f"ffprobe could not inspect {path.name}: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    audio_streams = [row for row in streams if row.get("codec_type") == "audio"]
    video_streams = [row for row in streams if row.get("codec_type") == "video"]
    if len(audio_streams) != 1:
        raise ValueError(f"{path.name} must contain exactly one audio stream")
    if video_streams:
        raise ValueError(
            f"{path.name} contains video; extract an audio-only review file to avoid visual cues"
        )
    stream = audio_streams[0]
    format_row = payload.get("format", {})
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(format_row["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has incomplete audio metadata") from exc
    if sample_rate <= 0 or channels <= 0 or duration <= 0:
        raise ValueError(f"{path.name} has invalid audio metadata")
    return {
        "codec_name": str(stream.get("codec_name", "")),
        "sample_rate": sample_rate,
        "channels": channels,
        "format_name": str(format_row.get("format_name", "")),
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
    }


def _fairness_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: contract[key]
        for key in ("codec_name", "sample_rate", "channels", "format_name")
    }


def _html(cases: list[dict[str, Any]], review_id: str) -> str:
    public_cases = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    review_id_json = json.dumps(review_id, ensure_ascii=False)
    schema_json = json.dumps(REVIEW_SCHEMA)
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>MiniMax H3 音色克隆 ABX 盲评</title>
<style>
body{{margin:0;background:#101217;color:#eef2f7;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1060px;margin:auto;padding:24px}} h1{{margin-bottom:8px}}
.notice,.case{{background:#181c24;border:1px solid #343b49;border-radius:14px;padding:18px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.audio{{background:#0f1218;padding:12px;border-radius:10px}} audio{{width:100%}}
label{{display:block;margin:8px 0}} select,input,textarea,button{{font:inherit}}
textarea{{width:100%;min-height:56px;box-sizing:border-box}}
.choices{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}}
button{{padding:10px 18px;border:0;border-radius:8px;background:#4b7bec;color:white;cursor:pointer}}
.muted{{color:#aeb7c6}} .warn{{color:#ffd27a}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>MiniMax H3 音色克隆 ABX 盲评</h1>
<section class=\"notice\">
<p>每组先听 A、B 两份真人参考，再听 X 生成音频。只判断 X 的说话人身份更接近 A 还是 B；不清楚就选“无法判断”，音频损坏才选“无效”。</p>
<p class=\"warn\">不要根据响度、编码或台词内容猜。自然度和吐字是独立评分；即使身份选择正确，也不等于高保真克隆。</p>
<label>评审代号（不要填真实姓名） <input id=\"reviewer\" maxlength=\"80\" placeholder=\"例如 reviewer-01\"></label>
</section>
<div id=\"cases\"></div>
<button id=\"export\">导出 JSON</button>
<p id=\"status\" class=\"muted\"></p>
<script>
const CASES={public_cases}; const REVIEW_ID={review_id_json}; const SCHEMA={schema_json};
const root=document.getElementById('cases');
for(const c of CASES){{
 const section=document.createElement('section'); section.className='case'; section.dataset.id=c.case_id;
 section.innerHTML=`<h2>${{c.label}}</h2><div class=\"grid\">
 <div class=\"audio\"><b>A · 真人参考</b><audio controls preload=\"metadata\" src=\"${{c.a}}\"></audio></div>
 <div class=\"audio\"><b>B · 真人参考</b><audio controls preload=\"metadata\" src=\"${{c.b}}\"></audio></div>
 <div class=\"audio\"><b>X · 生成音频</b><audio controls preload=\"metadata\" src=\"${{c.x}}\"></audio></div></div>
 <div class=\"choices\"><b>X 的身份更接近：</b>
 ${{['A','B','unclear','invalid'].map(v=>`<label><input type=\"radio\" name=\"choice-${{c.case_id}}\" value=\"${{v}}\">${{v==='unclear'?'无法判断':v==='invalid'?'音频无效':v}}</label>`).join('')}}</div>
 <label>判断信心 <select class=\"confidence\"><option value=\"0\">未评分</option>${{[1,2,3,4,5].map(v=>`<option>${{v}}</option>`).join('')}}</select></label>
 <label>X 自然度 <select class=\"naturalness\"><option value=\"0\">未评分</option>${{[1,2,3,4,5].map(v=>`<option>${{v}}</option>`).join('')}}</select></label>
 <label>X 吐字清晰度 <select class=\"articulation\"><option value=\"0\">未评分</option>${{[1,2,3,4,5].map(v=>`<option>${{v}}</option>`).join('')}}</select></label>
 <label>备注 <textarea class=\"notes\"></textarea></label>`; root.appendChild(section);
}}
document.getElementById('export').onclick=()=>{{
 const reviewer=document.getElementById('reviewer').value.trim();
 if(!reviewer){{document.getElementById('status').textContent='请先填写评审代号。';return;}}
 const reviews=[...document.querySelectorAll('.case')].map(section=>({{
   case_id:section.dataset.id,
   identity_choice:section.querySelector('input[type=radio]:checked')?.value||'',
   confidence:Number(section.querySelector('.confidence').value),
   candidate_naturalness:Number(section.querySelector('.naturalness').value),
   candidate_articulation:Number(section.querySelector('.articulation').value),
   notes:section.querySelector('.notes').value
 }}));
 const payload={{schema:SCHEMA,review_id:REVIEW_ID,reviewer_id:reviewer,exported_at:new Date().toISOString(),reviews}};
 const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});
 const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
 link.download='voice_clone_abx_review.json'; link.click(); URL.revokeObjectURL(link.href);
 const missing=reviews.filter(r=>!r.identity_choice).length;
 document.getElementById('status').textContent=missing?`已导出；仍有 ${{missing}} 组未回答，分析器会保留为未完成。`:'已导出完整评审。';
}};
</script></main></body></html>"""


def build_package(
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    output_dir: Path,
    random_seed: int,
) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA}")
    review_id = _required_text(manifest, "review_id", context="manifest")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest.cases must be a non-empty list")
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    target_position_policy = str(
        manifest.get("target_position_policy") or "independent_random"
    ).strip()
    if target_position_policy not in TARGET_POSITION_POLICIES:
        raise ValueError(
            "manifest.target_position_policy must be independent_random or "
            "balanced_by_target_and_global"
        )

    prepared: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    candidate_hashes: set[str] = set()
    rng = random.Random(random_seed)
    for index, raw in enumerate(raw_cases, start=1):
        context = f"manifest.cases[{index - 1}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be an object")
        case_id = _required_text(raw, "case_id", context=context)
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        target_speaker = _required_text(raw, "target_speaker_id", context=context)
        impostor_speaker = _required_text(raw, "impostor_speaker_id", context=context)
        if target_speaker == impostor_speaker:
            raise ValueError(f"{case_id} target and impostor speaker must differ")
        condition_id = _required_text(raw, "condition_id", context=context)
        utterance_id = _required_text(raw, "utterance_id", context=context)
        language_code = _required_text(raw, "language_code", context=context)
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{context}.seed must be an integer")
        seed_known = raw.get("seed_known", True)
        if not isinstance(seed_known, bool):
            raise ValueError(f"{context}.seed_known must be boolean when provided")

        paths = {
            role: _resolve_regular_file(
                manifest_dir,
                raw.get(role),
                field=f"{context}.{role}",
            )
            for role in ("target_reference", "impostor_reference", "candidate")
        }
        contracts = {role: _audio_contract(path) for role, path in paths.items()}
        fair = {role: _fairness_contract(value) for role, value in contracts.items()}
        if len({json.dumps(value, sort_keys=True) for value in fair.values()}) != 1:
            raise ValueError(
                f"{case_id} A/B/X codec, sample-rate, channel and container contracts differ"
            )
        suffixes = {path.suffix.lower() for path in paths.values()}
        if len(suffixes) != 1:
            raise ValueError(f"{case_id} A/B/X filename extensions must match")
        hashes = {role: _sha256_file(path) for role, path in paths.items()}
        if len(set(hashes.values())) != 3:
            raise ValueError(f"{case_id} A/B/X audio content must be distinct")
        if hashes["candidate"] in candidate_hashes:
            raise ValueError("candidate audio content must not be reused across cases")
        candidate_hashes.add(hashes["candidate"])

        prepared.append(
            {
                "case_id": case_id,
                "label": f"第 {index} 组",
                "target_speaker_id": target_speaker,
                "impostor_speaker_id": impostor_speaker,
                "condition_id": condition_id,
                "utterance_id": utterance_id,
                "language_code": language_code,
                "seed": seed,
                "seed_known": seed_known,
                "paths": paths,
                "hashes": hashes,
                "contracts": contracts,
                "suffix": next(iter(suffixes)),
            }
        )

    if target_position_policy == "independent_random":
        for row in prepared:
            row["target_code"] = rng.choice(("A", "B"))
            row["impostor_code"] = "B" if row["target_code"] == "A" else "A"
    else:
        target_groups: dict[str, list[dict[str, Any]]] = {}
        for row in prepared:
            target_groups.setdefault(str(row["target_speaker_id"]), []).append(row)
        for group_index, rows in enumerate(target_groups.values()):
            a_count = (len(rows) + 1) // 2 if group_index % 2 == 0 else len(rows) // 2
            codes = ["A"] * a_count + ["B"] * (len(rows) - a_count)
            rng.shuffle(codes)
            for row, target_code in zip(rows, codes, strict=True):
                row["target_code"] = target_code
                row["impostor_code"] = "B" if target_code == "A" else "A"

    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / "blind_key.json"
    if any(output_dir.iterdir()) and not key_path.exists():
        raise ValueError("Output directory is non-empty but has no blind_key.json")
    public_cases = []
    key_cases = []
    for index, row in enumerate(prepared, start=1):
        blind_case_id = f"case-{index:03d}"
        copied: dict[str, str] = {}
        for code, role in (
            (row["target_code"], "target_reference"),
            (row["impostor_code"], "impostor_reference"),
            ("X", "candidate"),
        ):
            destination = output_dir / "media" / f"case-{index:03d}-{code}{row['suffix']}"
            copied[code] = destination.relative_to(output_dir).as_posix()
        public_cases.append(
            {
                "case_id": blind_case_id,
                "label": row["label"],
                "a": copied["A"],
                "b": copied["B"],
                "x": copied["X"],
            }
        )
        key_cases.append(
            {
                "blind_case_id": blind_case_id,
                "case_id": row["case_id"],
                "target_speaker_id": row["target_speaker_id"],
                "impostor_speaker_id": row["impostor_speaker_id"],
                "condition_id": row["condition_id"],
                "utterance_id": row["utterance_id"],
                "language_code": row["language_code"],
                "seed": row["seed"],
                "seed_known": row["seed_known"],
                "target_code": row["target_code"],
                "impostor_code": row["impostor_code"],
                "media": {
                    code: {
                        "role": role,
                        "source": str(row["paths"][role]),
                        "sha256": row["hashes"][role],
                        "contract": row["contracts"][role],
                        "blind_path": copied[code],
                    }
                    for code, role in (
                        (row["target_code"], "target_reference"),
                        (row["impostor_code"], "impostor_reference"),
                        ("X", "candidate"),
                    )
                },
            }
        )
    key = {
        "schema": KEY_SCHEMA,
        "review_schema": REVIEW_SCHEMA,
        "review_id": review_id,
        "random_seed": random_seed,
        "target_position_policy": target_position_policy,
        "cases": key_cases,
        "scientific_boundary": (
            "This key supports blinded speaker-identity discrimination only. Even a passing "
            "panel does not establish high-fidelity cloning, naturalness, acting control, "
            "safety, consent or generalization."
        ),
    }
    if key_path.exists():
        existing = json.loads(key_path.read_text(encoding="utf-8"))
        if existing != key:
            raise ValueError("Output directory already contains a different immutable key")

    for index, row in enumerate(prepared, start=1):
        for code, role in (
            (row["target_code"], "target_reference"),
            (row["impostor_code"], "impostor_reference"),
            ("X", "candidate"),
        ):
            destination = output_dir / "media" / f"case-{index:03d}-{code}{row['suffix']}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and _sha256_file(destination) != row["hashes"][role]:
                raise ValueError(f"Existing blind media differs: {destination.name}")
            if not destination.exists():
                shutil.copy2(row["paths"][role], destination)
            if _sha256_file(destination) != row["hashes"][role]:
                raise ValueError(f"Copied blind media hash mismatch: {destination.name}")
    _write_json_atomic(key_path, key)
    html_path = output_dir / "blind_review.html"
    html_path.write_text(_html(public_cases, review_id), encoding="utf-8")
    return key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an immutable blinded A/B/X voice-clone identity review package."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = build_package(
        manifest,
        manifest_path.parent,
        args.output.resolve(),
        args.random_seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "review_id": key["review_id"],
                "cases": len(key["cases"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
