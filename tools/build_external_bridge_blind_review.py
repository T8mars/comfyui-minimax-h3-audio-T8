#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import re
from typing import Any

try:
    from .build_eav_blind_review import (
        _media_contract,
        _resolve_media,
        _sha256_file,
        _sync_media,
        _validate_pair_contract,
        _write_json_atomic,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_eav_blind_review import (
        _media_contract,
        _resolve_media,
        _sha256_file,
        _sync_media,
        _validate_pair_contract,
        _write_json_atomic,
    )


MANIFEST_SCHEMA = "t8.external_bridge_blind_manifest.v1"
PACKAGE_SCHEMA = "t8.external_bridge_blind_package.v1"
REVIEW_SCHEMA = "t8.external_bridge_blind_review.v1"
ALLOWED_REFERENCE_METRICS = {"first_frame", "last_frame", "identity"}
DEFAULT_PAGE_TITLE = "MiniMax H3 外部桥接匿名 A/B 评审"
DEFAULT_PAGE_INTRO = (
    "每组先用“同步静音播放”比较画面，再分别完整试听 A、B。"
    "先选择本组是否可判断；只有“可判断”时 A/B/平才计入结论。"
)
DEFAULT_EXPORT_FILENAME = "external_bridge_blind_review.json"
EXPORT_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,78}\.json$")


def _pair_seed(blind_seed: int, pair_id: str) -> int:
    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    return int(blind_seed) ^ int.from_bytes(digest[:8], "big")


def _document(
    *,
    review_id: str,
    public_pairs: list[dict[str, Any]],
    page_title: str = DEFAULT_PAGE_TITLE,
    page_intro: str = DEFAULT_PAGE_INTRO,
    export_filename: str = DEFAULT_EXPORT_FILENAME,
) -> str:
    rows = json.dumps(public_pairs, ensure_ascii=False).replace("</", "<\\/")
    storage = json.dumps(f"t8-external-bridge-{review_id}")
    safe_title = html.escape(page_title)
    safe_intro = html.escape(page_intro)
    export_name = json.dumps(export_filename)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#0d1015;color:#eef1f6}}
header{{position:sticky;top:0;z-index:3;padding:16px 22px;background:#151922f2;border-bottom:1px solid #343b49}}
header h1{{font-size:21px;margin:0 0 8px}}header p{{margin:4px 0;color:#cbd1dc}}button{{padding:8px 12px;border:1px solid #596579;border-radius:7px;background:#263247;color:white;cursor:pointer}}
main{{padding:18px;display:grid;gap:18px}}article{{border:1px solid #343b49;border-radius:12px;background:#171b24;padding:14px}}
.prompt{{white-space:pre-wrap;background:#0e1117;border-radius:7px;padding:9px;color:#cbd1dc}}.refs{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}}.ref img{{max-width:220px;max-height:180px;object-fit:contain;background:#000;border-radius:6px}}.ref p{{margin:3px 0;color:#cbd1dc}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.side{{background:#0e1117;border-radius:8px;padding:9px}}video{{width:100%;max-height:540px;background:#000;border-radius:6px}}
.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}.votes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:12px}}label{{display:grid;gap:4px;color:#cbd1dc;font-size:13px}}
select,textarea{{background:#222936;color:white;border:1px solid #4a5568;border-radius:5px;padding:7px}}textarea{{min-height:64px;resize:vertical}}#status{{margin-left:10px;color:#8bd889}}@media(max-width:900px){{.pair,.votes{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>{safe_title}</h1>
<p>{safe_intro}</p>
<p>不要打开同目录的 <code>blind_key.json</code>，不要从文件名猜方法。单组结果只能关闭当前素材的人工门，不能外推普遍优势。</p>
<button id="export">导出评审 JSON</button><button id="clear">清空本页记录</button><span id="status"></span></header><main id="cards"></main>
<script>
const pairs={rows};const storageKey={storage};let saved={{}};try{{saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}')}}catch(_){{saved={{}}}}
const choices=[['tie','平'],['A','A更好'],['B','B更好']];
const assessabilityChoices=[['unsure','不确定'],['assessable','可判断（再选 A/B/平）'],['source_material_insufficient','原素材/参考本身不足'],['playback_problem','播放或音频加载问题']];
function persist(ev){{const e=ev.currentTarget;saved[e.dataset.pair]||={{}};saved[e.dataset.pair][e.dataset.field]=e.value;localStorage.setItem(storageKey,JSON.stringify(saved));document.getElementById('status').textContent='已保存'}}
function chooser(pair,field){{const e=document.createElement('select');e.dataset.pair=pair;e.dataset.field=field;for(const [v,t] of choices){{const o=document.createElement('option');o.value=v;o.textContent=t;e.append(o)}}e.value=saved[pair]?.[field]||'tie';e.onchange=persist;return e}}
const cards=document.getElementById('cards');for(const row of pairs){{const a=document.createElement('article');a.dataset.pair=row.pair_id;const h=document.createElement('h2');h.textContent=row.label+' · '+row.task_type;a.append(h);if(row.prompt){{const p=document.createElement('div');p.className='prompt';p.textContent='共同提示词：'+row.prompt;a.append(p)}}if(row.references.length){{const refs=document.createElement('div');refs.className='refs';for(const item of row.references){{const box=document.createElement('div');box.className='ref';const img=document.createElement('img');img.src=item.media;img.alt=item.label;const cap=document.createElement('p');cap.textContent=item.label;box.append(img,cap);refs.append(box)}}a.append(refs)}}const actions=document.createElement('div');actions.className='actions';const sync=document.createElement('button');sync.textContent='同步静音播放本组';sync.onclick=async()=>{{const vids=[...a.querySelectorAll('video')];for(const v of vids){{v.pause();v.currentTime=0;v.muted=true}}await Promise.allSettled(vids.map(v=>v.play()))}};const stop=document.createElement('button');stop.textContent='暂停本组';stop.onclick=()=>a.querySelectorAll('video').forEach(v=>v.pause());actions.append(sync,stop);a.append(actions);const p=document.createElement('div');p.className='pair';for(const side of row.sides){{const s=document.createElement('section');s.className='side';const sh=document.createElement('h3');sh.textContent=side.code;const v=document.createElement('video');v.src=side.media;v.preload='metadata';v.controls=true;s.append(sh,v);p.append(s)}}a.append(p);const assessment=document.createElement('label');assessment.textContent='本组是否足以判断';const q=document.createElement('select');q.dataset.pair=row.pair_id;q.dataset.field='assessability';for(const [v,t] of assessabilityChoices){{const o=document.createElement('option');o.value=v;o.textContent=t;q.append(o)}}q.value=saved[row.pair_id]?.assessability||'unsure';q.onchange=persist;assessment.append(q);a.append(assessment);const votes=document.createElement('div');votes.className='votes';const metrics=[['overall','整段总体'],['motion','运动连续与自然'],['audio','完整音轨'],['prompt_adherence','提示词遵循'],['stability','闪烁/崩坏稳定性']];for(const [field,title] of metrics){{const l=document.createElement('label');l.textContent=title;l.append(chooser(row.pair_id,field));votes.append(l)}}for(const metric of row.reference_metrics){{const titles={{first_frame:'首帧保持',last_frame:'尾帧保持',identity:'参考身份保持'}};const l=document.createElement('label');l.textContent=titles[metric];l.append(chooser(row.pair_id,'reference_'+metric));votes.append(l)}}const failure=document.createElement('label');failure.textContent='硬失败';const f=document.createElement('select');f.dataset.pair=row.pair_id;f.dataset.field='blocking_failure';for(const [v,t] of [['none','无'],['A','A有黑屏/花屏/坏帧/明显音频故障'],['B','B有黑屏/花屏/坏帧/明显音频故障'],['both','两边都有']]){{const o=document.createElement('option');o.value=v;o.textContent=t;f.append(o)}}f.value=saved[row.pair_id]?.blocking_failure||'none';f.onchange=persist;failure.append(f);votes.append(failure);a.append(votes);const nl=document.createElement('label');nl.textContent='具体问题与时间点';const n=document.createElement('textarea');n.dataset.pair=row.pair_id;n.dataset.field='notes';n.value=saved[row.pair_id]?.notes||'';n.oninput=persist;nl.append(n);a.append(nl);cards.append(a)}}
document.getElementById('export').onclick=()=>{{const reviews=pairs.map(row=>{{const value=saved[row.pair_id]||{{}};const out={{pair_id:row.pair_id,assessability:value.assessability||'unsure',overall:value.overall||'tie',motion:value.motion||'tie',audio:value.audio||'tie',prompt_adherence:value.prompt_adherence||'tie',stability:value.stability||'tie',blocking_failure:value.blocking_failure||'none',notes:value.notes||'',reference_metrics:{{}}}};for(const metric of row.reference_metrics)out.reference_metrics[metric]=value['reference_'+metric]||'tie';return out}});const payload={{schema:'{REVIEW_SCHEMA}',review_id:{json.dumps(review_id)},exported_at:new Date().toISOString(),reviews}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download={export_name};link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}};
document.getElementById('clear').onclick=()=>{{if(confirm('确认清空本页已保存选择？')){{localStorage.removeItem(storageKey);location.reload()}}}};
</script></body></html>"""


def build_package(
    manifest: dict[str, Any], manifest_dir: Path, output_dir: Path, blind_seed: int
) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    review_id = str(manifest.get("review_id", "")).strip()
    if not review_id:
        raise ValueError("manifest review_id must be non-empty")
    page_title = manifest.get("page_title", DEFAULT_PAGE_TITLE)
    page_intro = manifest.get("page_intro", DEFAULT_PAGE_INTRO)
    export_filename = manifest.get("export_filename", DEFAULT_EXPORT_FILENAME)
    if not isinstance(page_title, str) or not page_title.strip() or len(page_title) > 120:
        raise ValueError("manifest page_title must be 1..120 characters")
    if not isinstance(page_intro, str) or not page_intro.strip() or len(page_intro) > 500:
        raise ValueError("manifest page_intro must be 1..500 characters")
    if not isinstance(export_filename, str) or not EXPORT_FILENAME_RE.fullmatch(
        export_filename
    ):
        raise ValueError("manifest export_filename must be a safe ASCII .json basename")
    analysis_generalization = manifest.get("analysis_generalization")
    if analysis_generalization is not None and (
        not isinstance(analysis_generalization, str)
        or not analysis_generalization.strip()
        or len(analysis_generalization) > 800
    ):
        raise ValueError("manifest analysis_generalization must be 1..800 characters")
    page_title = page_title.strip()
    page_intro = page_intro.strip()
    if isinstance(analysis_generalization, str):
        analysis_generalization = analysis_generalization.strip()
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest pairs must be a non-empty list")
    public_pairs: list[dict[str, Any]] = []
    private_pairs: list[dict[str, Any]] = []
    media_jobs: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("every blind pair must be an object")
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in seen:
            raise ValueError(f"pair_id must be unique and non-empty: {pair_id!r}")
        seen.add(pair_id)
        control = _resolve_media(manifest_dir, row.get("control"), "control")
        candidate = _resolve_media(manifest_dir, row.get("candidate"), "candidate")
        control_contract = _media_contract(control)
        candidate_contract = _media_contract(candidate)
        _validate_pair_contract(control_contract, candidate_contract, pair_id)
        reference_metrics = [str(value) for value in row.get("reference_metrics", [])]
        unknown = set(reference_metrics) - ALLOWED_REFERENCE_METRICS
        if unknown:
            raise ValueError(f"unsupported reference_metrics for {pair_id}: {sorted(unknown)}")
        order = ["control", "candidate"]
        random.Random(_pair_seed(blind_seed, pair_id)).shuffle(order)
        public_sides = []
        private_sides = []
        for code, arm in zip(("A", "B"), order, strict=True):
            source = control if arm == "control" else candidate
            digest = _sha256_file(source)
            target_name = f"pair-{index:02d}-{code}{source.suffix.lower()}"
            media_jobs.append((source, Path("media") / target_name, digest))
            public_sides.append({"code": code, "media": f"media/{target_name}"})
            private_sides.append(
                {
                    "code": code,
                    "arm": arm,
                    "method": str(row.get(f"{arm}_method", arm)),
                    "source": str(source),
                    "sha256": digest,
                }
            )
        public_refs = []
        private_refs = []
        for ref_index, value in enumerate(row.get("reference_images", []), 1):
            source = _resolve_media(
                manifest_dir, value, f"reference_images[{ref_index - 1}]"
            )
            digest = _sha256_file(source)
            target_name = f"pair-{index:02d}-ref-{ref_index:02d}{source.suffix.lower()}"
            media_jobs.append((source, Path("media") / target_name, digest))
            label = f"参考图 {ref_index}"
            public_refs.append({"label": label, "media": f"media/{target_name}"})
            private_refs.append(
                {"label": label, "source": str(source), "sha256": digest}
            )
        public_pairs.append(
            {
                "pair_id": pair_id,
                "label": str(row.get("label") or f"第 {index} 组"),
                "task_type": str(row.get("task_type") or "H3"),
                "prompt": str(row.get("prompt") or ""),
                "reference_metrics": reference_metrics,
                "references": public_refs,
                "sides": public_sides,
            }
        )
        private_pairs.append(
            {
                "pair_id": pair_id,
                "contract": control_contract,
                "reference_metrics": reference_metrics,
                "references": private_refs,
                "sides": private_sides,
            }
        )
    key = {
        "schema": PACKAGE_SCHEMA,
        "review_id": review_id,
        "blind_seed": int(blind_seed),
        "source_manifest_schema": MANIFEST_SCHEMA,
        "limitations": [
            "One material and one seed per route cannot establish general quality superiority.",
            "Joint H3 video and audio must both be reviewed.",
            "Keep this key private until the review export is complete.",
        ],
        "pairs": private_pairs,
    }
    if any(
        field in manifest
        for field in ("page_title", "page_intro", "export_filename")
    ):
        key["display_contract"] = {
            "page_title": page_title,
            "page_intro": page_intro,
            "export_filename": export_filename,
        }
    if analysis_generalization is not None:
        key["analysis_contract"] = {
            "generalization": analysis_generalization,
        }
    key_path = output_dir / "blind_key.json"
    if output_dir.exists():
        existing_entries = list(output_dir.iterdir())
        if key_path.exists():
            try:
                existing_key = json.loads(key_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Existing blind_key.json is unreadable; use a new output directory "
                    "instead of replacing review evidence"
                ) from error
            if existing_key != key:
                raise ValueError(
                    "Existing blind review package has a different immutable key; use a "
                    "new output directory so earlier review exports remain revealable"
                )
        elif existing_entries:
            raise ValueError(
                "Output directory is non-empty but has no blind_key.json; use a new output "
                "directory instead of overwriting partial or unrelated evidence"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    for source, relative_target, digest in media_jobs:
        _sync_media(source, output_dir / relative_target, digest)
    _write_json_atomic(key_path, key)
    page_path = output_dir / "blind_review.html"
    # A completed blind package is evidence. Rebuilding an identical private key
    # must not silently replace the page that produced earlier review exports.
    if not page_path.exists():
        page_path.write_text(
            _document(
                review_id=review_id,
                public_pairs=public_pairs,
                page_title=page_title,
                page_intro=page_intro,
                export_filename=export_filename,
            ),
            encoding="utf-8",
            newline="\n",
        )
    return key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic MiniMax H3 external-bridge A/B review."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=2608230101)
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
