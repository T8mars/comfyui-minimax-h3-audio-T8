#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
from typing import Any


REVEAL_SCHEMA = "minimax_h3_speed_single_blind_reveal_v1"
REVIEW_SCHEMA = "minimax_h3_speed_single_blind_review_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_single_blind_review(
    *,
    baseline: Path,
    speed: Path,
    output_dir: Path,
    seed: int,
    title: str,
) -> dict[str, Any]:
    if not baseline.is_file() or not speed.is_file():
        raise ValueError("baseline and speed must both be existing files")
    output_dir.mkdir(parents=True, exist_ok=True)
    treatments = ["baseline", "speed"]
    random.Random(int(seed)).shuffle(treatments)
    mapping = {"A": treatments[0], "B": treatments[1]}
    source = {"baseline": baseline, "speed": speed}
    copied: dict[str, str] = {}
    for label in ("A", "B"):
        destination = output_dir / f"calibrated_t2va_{label}.mp4"
        shutil.copy2(source[mapping[label]], destination)
        copied[label] = destination.name

    reveal = {
        "schema": REVEAL_SCHEMA,
        "seed": int(seed),
        "mapping": mapping,
        "sources": {
            treatment: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for treatment, path in source.items()
        },
    }
    (output_dir / "reveal.json").write_text(
        json.dumps(reveal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    escaped_title = html.escape(title or "MiniMax H3 calibrated SPEED blind review")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{escaped_title}</title>
<style>
body{{font-family:system-ui;background:#11151b;color:#edf2f7;margin:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} video{{width:100%;background:#000}}
.card{{background:#1d2430;padding:14px;border-radius:12px}} label{{margin-right:16px}}
button{{padding:10px 18px;margin-top:18px}} .note{{color:#aeb9c8}}
</style></head><body><h1>{escaped_title}</h1>
<p class="note">只比较本页 A/B，不要打开 reveal.json。请完整观看并试听；缺失选择按平局处理。</p>
<div class="grid"><section class="card"><h2>A</h2><video controls src="{copied['A']}"></video></section>
<section class="card"><h2>B</h2><video controls src="{copied['B']}"></video></section></div>
<div id="form"></div><button onclick="saveReview()">导出评分 JSON</button>
<script>
const metrics=[['overall','总体画质'],['motion_detail','运动与细节'],['audio','声音']];
const root=document.getElementById('form');
for(const [key,label] of metrics){{const row=document.createElement('p');row.innerHTML='<b>'+label+'</b> ';
for(const value of ['A','B','tie']){{const item=document.createElement('label');item.innerHTML='<input type="radio" name="'+key+'" value="'+value+'">'+value;row.appendChild(item)}}root.appendChild(row)}}
function saveReview(){{const review={{schema:'{REVIEW_SCHEMA}',title:{json.dumps(title, ensure_ascii=False)},votes:{{}}}};
for(const [key] of metrics){{const chosen=document.querySelector('input[name="'+key+'"]:checked');review.votes[key]=chosen?chosen.value:'tie'}}
const blob=new Blob([JSON.stringify(review,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='h3_speed_calibrated_t2va_blind_review.json';a.click();URL.revokeObjectURL(a.href)}}
</script></body></html>"""
    (output_dir / "blind_review.html").write_text(page, encoding="utf-8")
    return reveal


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one randomized baseline-versus-calibrated-SPEED review page."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2608199501)
    parser.add_argument("--title", default="MiniMax H3 T2VA：基线 vs 数据集标定 SPEED")
    args = parser.parse_args()
    reveal = build_single_blind_review(
        baseline=args.baseline,
        speed=args.speed,
        output_dir=args.output_dir,
        seed=args.seed,
        title=args.title,
    )
    print(json.dumps({"mapping": reveal["mapping"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
