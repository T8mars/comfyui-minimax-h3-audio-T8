#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
from typing import Any
import uuid

import av
import run_skin_finish_live_sam31_validation as base


SCHEMA = "h3_t8_skin_finish_human_review/v1"
CRITERIA = (
    ("overall", "总体更自然"),
    ("skin_naturalness", "肤质自然、无蜡像感"),
    ("shine_highlight", "油光与高光控制"),
    ("tone_evenness", "肤色与亮度均匀"),
    ("texture_retention", "皮肤纹理保留"),
    ("eyes_lips_features", "眼、眉、鼻、嘴唇等五官保护"),
    ("temporal_flicker", "连续帧稳定、无闪烁泵动"),
    ("halo_edges", "脸部边缘无光晕或蒙版边界"),
    ("cross_person_spill", "多人之间无误涂或串色"),
    ("identity_mouth", "身份与口型保持（可判断时）"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest().upper()


def _video_contract(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if not isinstance(video, dict):
        raise ValueError("review media has no video stream")
    return {
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "frame_count": int(video.get("nb_frames", 0)),
        "frame_rate": str(video.get("r_frame_rate", "")),
        "duration": str(video.get("duration", "")),
        "audio_present": isinstance(audio, dict),
        "audio_codec": str((audio or {}).get("codec_name", "")),
        "sample_rate": int((audio or {}).get("sample_rate", 0) or 0),
        "channels": int((audio or {}).get("channels", 0) or 0),
    }


def _validate_pair(
    source_probe: dict[str, Any], candidate_probe: dict[str, Any]
) -> dict[str, Any]:
    source = _video_contract(source_probe)
    candidate = _video_contract(candidate_probe)
    required_equal = ("width", "height", "frame_count", "frame_rate")
    mismatch = {
        key: {"a": source[key], "b": candidate[key]}
        for key in required_equal
        if source[key] != candidate[key]
    }
    if mismatch:
        raise ValueError(f"review media contract mismatch: {mismatch}")
    return source


def _blind_mapping() -> dict[str, str]:
    return (
        {"A": "source", "B": "candidate"}
        if secrets.randbits(1) == 0
        else {"A": "candidate", "B": "source"}
    )


def _write_first_frame(path: Path, output: Path) -> None:
    with av.open(str(path), mode="r") as container:
        frame = next(container.decode(video=0), None)
        if frame is None:
            raise ValueError(f"review media has no decodable frame: {path}")
        frame.to_image().save(output, format="PNG", optimize=True)


def _render_html(public: dict[str, Any]) -> str:
    manifest = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    criteria = "\n".join(
        f"""
        <fieldset class="criterion" data-id="{key}">
          <legend>{label}</legend>
          <label><input type="radio" name="{key}" value="A">A更好</label>
          <label><input type="radio" name="{key}" value="B">B更好</label>
          <label><input type="radio" name="{key}" value="tie">平局</label>
          <label><input type="radio" name="{key}" value="abstain">无法判断</label>
        </fieldset>"""
        for key, label in CRITERIA
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skin Finish 匿名人工审片</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui,"Microsoft YaHei",sans-serif; }}
body {{ margin:0; background:#0d1015; color:#eef2f7; }}
main {{ max-width:1560px; margin:auto; padding:22px; }}
h1 {{ margin:0 0 6px; }} .muted {{ color:#aeb8c6; }}
.warning {{ background:#2b2210; border:1px solid #8e6a20; padding:12px; border-radius:10px; }}
.videos {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px; }}
.panel {{ background:#151a22; border:1px solid #313a48; border-radius:12px; padding:10px; }}
.panel h2 {{ margin:0 0 8px; }}
.viewport {{ aspect-ratio:960/704; overflow:hidden; background:#000; cursor:crosshair; border-radius:8px; }}
video {{ width:100%; height:100%; object-fit:contain; transform-origin:50% 50%; }}
.controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:14px 0; padding:12px; background:#151a22; border-radius:10px; }}
button,select,input,textarea {{ font:inherit; }} button,select {{ padding:7px 10px; }}
#seek {{ flex:1; min-width:260px; }}
.form {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
fieldset {{ border:1px solid #394557; border-radius:9px; padding:10px; }}
legend {{ font-weight:650; }} label {{ margin-right:14px; display:inline-block; padding:3px 0; }}
textarea {{ width:100%; min-height:88px; box-sizing:border-box; background:#0e1218; color:#fff; border:1px solid #394557; border-radius:8px; padding:8px; }}
.wide {{ grid-column:1/-1; }} .export {{ background:#1f7045; color:#fff; border:0; border-radius:8px; font-weight:700; }}
@media (max-width:900px) {{ .videos,.form {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Skin Finish 匿名人工审片</h1>
<p class="muted">Review ID：<code id="rid"></code>。A/B已匿名；先看完整运动，再慢放检查侧脸、嘴唇、眼睛、边缘与闪烁。点击画面可改变放大中心。</p>
<p class="warning">机械门通过不等于更美。若两边素材本身不足、无法看清皮肤或播放异常，请选择“无法判断”，不要强行投票。</p>
<section class="videos">
  <article class="panel"><h2>A</h2><div class="viewport"><video id="a" src="media/A.mp4" poster="poster/A.png" preload="metadata" muted></video></div></article>
  <article class="panel"><h2>B</h2><div class="viewport"><video id="b" src="media/B.mp4" poster="poster/B.png" preload="metadata" muted></video></div></article>
</section>
<div class="controls">
  <button id="play">播放/暂停</button><button id="prev">上一帧</button><button id="next">下一帧</button>
  <label>速度 <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option></select></label>
  <label>放大 <select id="zoom"><option selected>1</option><option>1.5</option><option>2</option><option>3</option></select></label>
  <label><input id="loop" type="checkbox" checked>循环</label>
  <input id="seek" type="range" min="0" max="1" step="0.001" value="0"><output id="time">0.000s</output>
</div>
<section class="form">
<fieldset class="wide"><legend>本组是否可判断</legend>
  <label><input type="radio" name="assessability" value="assessable">可判断</label>
  <label><input type="radio" name="assessability" value="source_insufficient">素材本身不足</label>
  <label><input type="radio" name="assessability" value="playback_problem">播放问题</label>
  <label><input type="radio" name="assessability" value="unsure">不确定</label>
</fieldset>
{criteria}
<fieldset><legend>A的硬失败（可多选）</legend>
  <label><input type="checkbox" name="failure_a" value="identity">身份/五官明显破坏</label>
  <label><input type="checkbox" name="failure_a" value="mouth_eye">嘴唇或眼睛被抹糊</label>
  <label><input type="checkbox" name="failure_a" value="flicker">持续闪烁/泵动</label>
  <label><input type="checkbox" name="failure_a" value="halo">明显光晕/边界</label>
  <label><input type="checkbox" name="failure_a" value="cross_person">跨人物误涂/串色</label>
  <label><input type="checkbox" name="failure_a" value="av_sync">音画不同步</label>
</fieldset>
<fieldset><legend>B的硬失败（可多选）</legend>
  <label><input type="checkbox" name="failure_b" value="identity">身份/五官明显破坏</label>
  <label><input type="checkbox" name="failure_b" value="mouth_eye">嘴唇或眼睛被抹糊</label>
  <label><input type="checkbox" name="failure_b" value="flicker">持续闪烁/泵动</label>
  <label><input type="checkbox" name="failure_b" value="halo">明显光晕/边界</label>
  <label><input type="checkbox" name="failure_b" value="cross_person">跨人物误涂/串色</label>
  <label><input type="checkbox" name="failure_b" value="av_sync">音画不同步</label>
</fieldset>
<label>左侧人物/轨迹观察<textarea id="left_notes"></textarea></label>
<label>右侧人物/轨迹观察<textarea id="right_notes"></textarea></label>
<label class="wide">总体备注<textarea id="notes"></textarea></label>
<div class="wide"><button class="export" id="export">导出评审JSON</button></div>
</section>
</main>
<script>
const manifest={manifest};
const A=document.querySelector('#a'),B=document.querySelector('#b'),seek=document.querySelector('#seek'),time=document.querySelector('#time');
document.querySelector('#rid').textContent=manifest.review_id;
let duration=0;
function sync(force=false){{ if(force||Math.abs(B.currentTime-A.currentTime)>0.035) B.currentTime=A.currentTime; }}
function update(){{ duration=Math.min(A.duration||Infinity,B.duration||Infinity); if(Number.isFinite(duration)) seek.max=duration; seek.value=A.currentTime; time.value=A.currentTime.toFixed(3)+'s'; sync(); }}
[A,B].forEach(v=>{{v.addEventListener('loadedmetadata',update);v.addEventListener('ended',()=>{{if(document.querySelector('#loop').checked){{A.currentTime=B.currentTime=0;A.play();B.play();}}}});}});
A.addEventListener('timeupdate',update);
document.querySelector('#play').onclick=()=>{{if(A.paused){{sync(true);A.play();B.play();}}else{{A.pause();B.pause();}}}};
seek.oninput=()=>{{A.currentTime=B.currentTime=Number(seek.value);update();}};
document.querySelector('#prev').onclick=()=>{{A.pause();B.pause();A.currentTime=B.currentTime=Math.max(0,A.currentTime-1/24);update();}};
document.querySelector('#next').onclick=()=>{{A.pause();B.pause();A.currentTime=B.currentTime=Math.min(duration||1,A.currentTime+1/24);update();}};
document.querySelector('#speed').onchange=e=>{{A.playbackRate=B.playbackRate=Number(e.target.value);}};
document.querySelector('#zoom').onchange=e=>{{const z=Number(e.target.value);A.style.transform=B.style.transform=`scale(${{z}})`;}};
document.querySelectorAll('.viewport').forEach(v=>v.onclick=e=>{{const r=v.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*100,y=(e.clientY-r.top)/r.height*100;A.style.transformOrigin=B.style.transformOrigin=`${{x}}% ${{y}}%`;}});
function radio(name){{return document.querySelector(`input[name="${{name}}"]:checked`)?.value||null;}}
document.querySelector('#export').onclick=()=>{{
 const criteria={{}}; document.querySelectorAll('.criterion').forEach(x=>criteria[x.dataset.id]=radio(x.dataset.id));
 const assessability=radio('assessability');
 if(!assessability){{alert('请先选择本组是否可判断。');return;}}
 const missing=Object.entries(criteria).filter(([,value])=>!value).map(([key])=>key);
 if(assessability==='assessable'&&missing.length){{alert('本组选择了“可判断”，请完成全部逐项选择。');return;}}
 if(assessability!=='assessable'){{missing.forEach(key=>criteria[key]='abstain');}}
 const out={{schema:manifest.schema,review_id:manifest.review_id,public_manifest_sha256:manifest.sha256,created_at:new Date().toISOString(),assessability,criteria,hard_failures:{{A:[...document.querySelectorAll('input[name="failure_a"]:checked')].map(x=>x.value),B:[...document.querySelectorAll('input[name="failure_b"]:checked')].map(x=>x.value)}},left_notes:document.querySelector('#left_notes').value,right_notes:document.querySelector('#right_notes').value,notes:document.querySelector('#notes').value}};
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`skin_finish_review_${{manifest.review_id}}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}};
</script></body></html>"""


def build_review(source: Path, candidate: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    candidate = candidate.resolve()
    output = output.resolve()
    if not source.is_file() or not candidate.is_file():
        raise FileNotFoundError("source and candidate MP4 files are both required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"review output is not empty: {output}")
    source_decode = base._strict_decode(source, "video")
    candidate_decode = base._strict_decode(candidate, "video")
    if not source_decode["passed"] or not candidate_decode["passed"]:
        raise ValueError("source or candidate failed strict video decode")
    source_probe = base._probe(source)
    candidate_probe = base._probe(candidate)
    contract = _validate_pair(source_probe, candidate_probe)
    mapping = _blind_mapping()
    review_id = uuid.uuid4().hex[:12]
    media = output / "media"
    poster = output / "poster"
    media.mkdir(parents=True, exist_ok=True)
    poster.mkdir(parents=True, exist_ok=True)
    paths = {"source": source, "candidate": candidate}
    for label, role in mapping.items():
        copied = media / f"{label}.mp4"
        shutil.copy2(paths[role], copied)
        _write_first_frame(copied, poster / f"{label}.png")
    public: dict[str, Any] = {
        "schema": SCHEMA,
        "review_id": review_id,
        "created_at": base._utc_now(),
        "contract": contract,
        "criteria": [{"id": key, "label": label} for key, label in CRITERIA],
        "boundary": (
            "Anonymous human review of final encoded media. A tie or abstention is valid. "
            "This review cannot replace source-bound float-mask mechanical evidence."
        ),
    }
    public["sha256"] = _hash_json(public)
    private = {
        "schema": f"{SCHEMA}/private-key",
        "review_id": review_id,
        "public_manifest_sha256": public["sha256"],
        "mapping": mapping,
        "source": {"path": str(source), "sha256": base._sha256(source)},
        "candidate": {"path": str(candidate), "sha256": base._sha256(candidate)},
        "copied_media": {
            label: {
                "path": str((media / f"{label}.mp4").resolve()),
                "sha256": base._sha256(media / f"{label}.mp4"),
            }
            for label in ("A", "B")
        },
        "posters": {
            label: {
                "path": str((poster / f"{label}.png").resolve()),
                "sha256": base._sha256(poster / f"{label}.png"),
            }
            for label in ("A", "B")
        },
    }
    private["sha256"] = _hash_json(private)
    base._json_write(output / "public_manifest.json", public)
    base._json_write(output / "private_key.json", private)
    (output / "blind_review.html").write_text(
        _render_html(public), encoding="utf-8", newline="\n"
    )
    return {
        "review": str((output / "blind_review.html").resolve()),
        "public_manifest": str((output / "public_manifest.json").resolve()),
        "private_key": str((output / "private_key.json").resolve()),
        "review_id": review_id,
        "public_manifest_sha256": public["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_review(args.source, args.candidate, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
