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


REVEAL_SCHEMA = "t8.minimax_h3.audio_refine.blind_reveal.v1"
REVIEW_SCHEMA = "t8.minimax_h3.blind_av_review.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_blind_review(
    *,
    original: Path,
    candidate: Path,
    output_dir: Path,
    private_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if not original.is_file() or not candidate.is_file():
        raise ValueError("original and candidate must both be existing files")
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    methods = ["original", "audio_refine"]
    random.Random(int(seed)).shuffle(methods)
    mapping = {"A": methods[0], "B": methods[1]}
    sources = {"original": original, "audio_refine": candidate}
    for label in ("A", "B"):
        shutil.copy2(sources[mapping[label]], output_dir / f"candidate_{label}.mp4")

    reveal = {
        "schema": REVEAL_SCHEMA,
        "seed": int(seed),
        "mapping": mapping,
        "sources": {
            method: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for method, path in sources.items()
        },
        "fixed_contract": {
            "resolution": "1056x608",
            "frames": 124,
            "fps": 24.0,
            "first_pass_nfe": 4,
            "refine_nfe": 4,
            "audio_denoise": 0.5,
            "prompt": "你在干嘛呢，我在这里呀，看看效果如何。",
        },
    }
    (private_dir / "reveal.json").write_text(
        json.dumps(reveal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    title = html.escape("MiniMax H3 音频候选匿名试听")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#101318;color:#eef2f7;margin:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:#1b222d;padding:14px;border-radius:12px}}
video{{width:100%;background:#000}}label{{display:inline-block;margin:5px 18px 5px 0}}button{{padding:10px 18px;margin:12px 8px 12px 0}}
.note{{color:#b8c2cf;line-height:1.6}}.metric{{background:#171d26;padding:8px 14px;border-radius:8px;margin:10px 0}}
textarea{{width:100%;min-height:90px;background:#0f141b;color:#eef2f7;border:1px solid #455064}}
</style></head><body><h1>{title}</h1>
<p class="note">请戴耳机完整观看并试听A/B。重点听台词是否仍为“你在干嘛呢，我在这里呀，看看效果如何”，同时看口型。不要寻找方法名称；本页不包含揭盲信息。</p>
<button onclick="syncPlay()">同步播放</button><button onclick="syncPause()">同步暂停</button><button onclick="syncReset()">同步归零</button>
<div class="grid"><section class="card"><h2>A</h2><video id="videoA" controls preload="metadata" src="candidate_A.mp4"></video></section>
<section class="card"><h2>B</h2><video id="videoB" controls preload="metadata" src="candidate_B.mp4"></video></section></div>
<div class="metric"><b>本组是否可判断</b><br>
<label><input type="radio" name="evaluability" value="can_judge">可判断</label>
<label><input type="radio" name="evaluability" value="source_insufficient">原素材不足</label>
<label><input type="radio" name="evaluability" value="playback_issue">播放问题</label>
<label><input type="radio" name="evaluability" value="uncertain">不确定</label></div>
<div id="metrics"></div><p><b>备注</b></p><textarea id="comment" placeholder="可记录改字、漏字、远近声跳变、金属感、音色、音乐/环境、口型等"></textarea><br>
<button onclick="saveReview()">导出评分JSON</button>
<script>
const metrics=[['dialogue_accuracy','台词准确/无增删字'],['voice_consistency','音色与远近稳定'],['naturalness','声音自然/少闷感金属感'],['ambience','环境声与瞬态连续'],['lip_sync','说话起止与口型同步'],['overall','总体更适合交付']];
const root=document.getElementById('metrics');
for(const [key,label] of metrics){{const row=document.createElement('div');row.className='metric';row.innerHTML='<b>'+label+'</b><br>';
for(const value of ['A','B','tie']){{const item=document.createElement('label');item.innerHTML='<input type="radio" name="'+key+'" value="'+value+'">'+(value==='tie'?'平':value);row.appendChild(item)}}root.appendChild(row)}}
const a=document.getElementById('videoA'),b=document.getElementById('videoB');
function syncPlay(){{b.currentTime=a.currentTime;Promise.allSettled([a.play(),b.play()])}}function syncPause(){{a.pause();b.pause()}}function syncReset(){{syncPause();a.currentTime=0;b.currentTime=0}}
function saveReview(){{const chosen=document.querySelector('input[name="evaluability"]:checked');const review={{schema:'{REVIEW_SCHEMA}',evaluability:chosen?chosen.value:'uncertain',votes:{{}},comment:document.getElementById('comment').value}};
for(const [key] of metrics){{const vote=document.querySelector('input[name="'+key+'"]:checked');review.votes[key]=vote?vote.value:'tie'}}
const blob=new Blob([JSON.stringify(review,null,2)+'\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='h3_audio_candidate_review.json';link.click();URL.revokeObjectURL(link.href)}}
</script></body></html>"""
    (output_dir / "blind_review.html").write_text(page, encoding="utf-8")
    return reveal


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one randomized H3 audio A/B review page.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2608260501)
    args = parser.parse_args()
    result = build_blind_review(
        original=args.original,
        candidate=args.candidate,
        output_dir=args.output_dir,
        private_dir=args.private_dir,
        seed=args.seed,
    )
    print(json.dumps({"mapping_sha256": hashlib.sha256(json.dumps(result["mapping"], sort_keys=True).encode()).hexdigest().upper()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
