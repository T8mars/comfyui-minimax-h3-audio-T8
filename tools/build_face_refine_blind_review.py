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


SCHEMA = "t8.face_refine_blind_package.v1"


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


def _video_contract(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
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
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frame_count": int(stream["nb_frames"]),
        "duration_seconds": float(stream["duration"]),
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


def _document(public_pairs: list[dict[str, Any]]) -> str:
    rows = json.dumps(public_pairs, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniMax H3 Face Refine 匿名 A/B 评审</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #101114; color: #f2f3f5; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 16px 22px; background: #17191eee; border-bottom: 1px solid #343842; }}
    header h1 {{ margin: 0 0 8px; font-size: 20px; }}
    header p {{ margin: 4px 0; color: #c8ccd4; }}
    button {{ border: 1px solid #596171; border-radius: 7px; padding: 8px 12px; color: #fff; background: #2c3442; cursor: pointer; }}
    #status {{ margin-left: 10px; color: #89d185; }}
    main {{ padding: 18px; display: grid; gap: 18px; }}
    article {{ border: 1px solid #343842; border-radius: 12px; background: #181b21; padding: 16px; }}
    article h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .side {{ background: #111318; border-radius: 9px; padding: 10px; }}
    video {{ width: 100%; max-height: 460px; background: #000; border-radius: 6px; }}
    .ratings {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    label {{ display: grid; gap: 4px; color: #cbd0d9; font-size: 13px; }}
    select, textarea {{ color: #fff; background: #242833; border: 1px solid #4b5261; border-radius: 5px; padding: 6px; }}
    .verdicts {{ margin-top: 12px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 9px; }}
    textarea {{ min-height: 64px; resize: vertical; }}
    @media (max-width: 850px) {{ .pair, .ratings, .verdicts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>MiniMax H3 Face Refine 匿名 A/B 评审</h1>
  <p>完整观看并听完 A、B；不要只看暂停帧。重点检查身份、表情/嘴型、时序、接缝、自然度和动作保留。</p>
  <p>本包比较“保留原片”和“局部二次生成候选”的实际创作效用，不是同 NFE 算法优越性证明。评审保存在本浏览器。</p>
  <button id="export" type="button">导出评审 JSON</button><span id="status"></span>
</header>
<main id="cards"></main>
<script>
const pairs = {rows};
const storageKey = "minimax-h3-face-refine-blind-review-v1";
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }} catch (_) {{ saved = {{}}; }}
const dimensions = [
  ["identity", "身份稳定"], ["expression_mouth", "表情/嘴型"],
  ["temporal", "时序稳定"], ["seam", "回贴接缝"],
  ["naturalness", "脸部自然度"], ["motion", "动作保留"]
];
const scoreOptions = [["", "未评分"], ["1", "1 严重失败"], ["2", "2 较差"], ["3", "3 可接受"], ["4", "4 良好"], ["5", "5 优秀"]];
const preferOptions = [["", "未选择"], ["A", "A"], ["B", "B"], ["tie", "平"]];
function persist(event) {{
  const element = event.currentTarget;
  saved[element.dataset.pair] ||= {{}};
  saved[element.dataset.pair][element.dataset.field] = element.value;
  localStorage.setItem(storageKey, JSON.stringify(saved));
  document.getElementById("status").textContent = "已保存";
}}
function select(pairId, field, options) {{
  const element = document.createElement("select");
  element.dataset.pair = pairId; element.dataset.field = field;
  for (const [value, label] of options) {{ const option = document.createElement("option"); option.value = value; option.textContent = label; element.append(option); }}
  element.value = saved[pairId]?.[field] || ""; element.addEventListener("change", persist); return element;
}}
const cards = document.getElementById("cards");
for (const pairRow of pairs) {{
  const article = document.createElement("article");
  const title = document.createElement("h2"); title.textContent = pairRow.pair_id; article.append(title);
  const pair = document.createElement("div"); pair.className = "pair";
  for (const sideRow of pairRow.sides) {{
    const side = document.createElement("section"); side.className = "side";
    const heading = document.createElement("h3"); heading.textContent = sideRow.code;
    const video = document.createElement("video"); video.src = sideRow.media; video.preload = "metadata"; video.controls = true;
    side.append(heading, video);
    const ratings = document.createElement("div"); ratings.className = "ratings";
    for (const [field, labelText] of dimensions) {{ const label = document.createElement("label"); label.textContent = labelText; label.append(select(pairRow.pair_id, `${{sideRow.code}}_${{field}}_1_to_5`, scoreOptions)); ratings.append(label); }}
    side.append(ratings); pair.append(side);
  }}
  article.append(pair);
  const verdicts = document.createElement("div"); verdicts.className = "verdicts";
  for (const [field, labelText] of [["overall_preference", "总体偏好"], ["identity_preference", "身份偏好"], ["motion_preference", "动作偏好"]]) {{ const label = document.createElement("label"); label.textContent = labelText; label.append(select(pairRow.pair_id, field, preferOptions)); verdicts.append(label); }}
  article.append(verdicts);
  const notesLabel = document.createElement("label"); notesLabel.textContent = "错人、鬼脸、嘴型、闪烁、接缝、声音或具体失败时间";
  const notes = document.createElement("textarea"); notes.dataset.pair = pairRow.pair_id; notes.dataset.field = "failure_notes"; notes.value = saved[pairRow.pair_id]?.failure_notes || ""; notes.addEventListener("input", persist); notesLabel.append(notes); article.append(notesLabel);
  cards.append(article);
}}
document.getElementById("export").addEventListener("click", () => {{
  const reviews = pairs.map((row) => ({{pair_id: row.pair_id, ...(saved[row.pair_id] || {{}})}}));
  const payload = {{schema: "t8.face_refine_blind_review.v1", exported_at: new Date().toISOString(), review_completed: reviews.every((item) => item.overall_preference && item.identity_preference && item.motion_preference), reviews}};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}}); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "face_refine_blind_review.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}});
</script>
</body>
</html>
"""


def build_package(
    source: Path, candidates: list[Path], output_dir: Path, blind_seed: int
) -> dict[str, Any]:
    source = source.resolve()
    candidates = [path.resolve() for path in candidates]
    if not source.is_file() or any(not path.is_file() for path in candidates):
        raise FileNotFoundError("Source and every candidate must exist")
    if not candidates:
        raise ValueError("At least one candidate is required")
    contract = _video_contract(source)
    for candidate in candidates:
        candidate_contract = _video_contract(candidate)
        for key in ("width", "height", "fps", "frame_count"):
            if candidate_contract[key] != contract[key]:
                raise ValueError(
                    f"Candidate {candidate} differs from source contract at {key}: "
                    f"{candidate_contract[key]} != {contract[key]}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source)
    public_pairs = []
    private_pairs = []
    for index, candidate in enumerate(candidates, 1):
        pair_id = f"pair-{index:02d}"
        order = ["source", "candidate"]
        random.Random(blind_seed ^ index).shuffle(order)
        sides = []
        private_sides = []
        for code, arm in zip(("A", "B"), order, strict=True):
            media_source = source if arm == "source" else candidate
            digest = source_sha256 if arm == "source" else _sha256_file(candidate)
            name = f"{pair_id}-{code}{media_source.suffix.lower()}"
            _sync_media(media_source, media_dir / name, digest)
            sides.append({"code": code, "media": f"media/{name}"})
            private_sides.append(
                {
                    "code": code,
                    "arm": arm,
                    "source": str(media_source),
                    "sha256": digest,
                }
            )
        public_pairs.append({"pair_id": pair_id, "sides": sides})
        private_pairs.append({"pair_id": pair_id, "sides": private_sides})

    private_key = {
        "schema": SCHEMA,
        "blind_seed": blind_seed,
        "contract": contract,
        "limitations": [
            "The same source appears in every pair, so repeated reviewers may infer the control.",
            "This package tests practical candidate utility, not equal-NFE causal superiority.",
            "At least five independent reviewers are still required for the preregistered promotion gate.",
        ],
        "pairs": private_pairs,
    }
    _write_json_atomic(output_dir / "blind_key.json", private_key)
    html = _document(public_pairs)
    destination = output_dir / "blind_review.html"
    destination.write_text(html, encoding="utf-8", newline="\n")
    return private_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a randomized local Face Refine source/candidate A/B review package."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=260816)
    args = parser.parse_args()
    key = build_package(args.source, args.candidate, args.output_dir, args.blind_seed)
    print(json.dumps({"pairs": len(key["pairs"]), "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
