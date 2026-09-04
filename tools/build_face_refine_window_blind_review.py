from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path


SCHEMA = "t8.face_refine_window_blind_review.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    streams = json.loads(completed.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    return {
        "video_codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video["avg_frame_rate"],
        "frames": int(video["nb_read_frames"]),
        "audio_codec": audio["codec_name"],
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
    }


def _strict_decode(path: Path) -> None:
    for mapping in ("0:v:0", "0:a:0"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-i",
                str(path),
                "-map",
                mapping,
                "-f",
                "null",
                "NUL",
            ],
            check=True,
        )


def build(source: Path, upstream: Path, window: Path, output: Path) -> dict:
    inputs = {"source": source.resolve(), "fixed_upstream": upstream.resolve(), "window": window.resolve()}
    metadata = {name: _probe(path) for name, path in inputs.items()}
    expected = next(iter(metadata.values()))
    if any(value != expected for value in metadata.values()):
        raise ValueError(f"Blind-review media contracts differ: {metadata}")
    for path in inputs.values():
        _strict_decode(path)

    shas = {name: _sha256(path) for name, path in inputs.items()}
    seed = int(hashlib.sha256("|".join(shas.values()).encode()).hexdigest()[:16], 16)
    methods = list(inputs)
    random.Random(seed).shuffle(methods)
    mapping = dict(zip(("A", "B", "C"), methods, strict=True))
    output.mkdir(parents=True, exist_ok=True)
    for label, method in mapping.items():
        shutil.copy2(inputs[method], output / f"{label}.mp4")

    public = {
        "schema": "t8.face_refine_window_blind_package.v1",
        "review_id": hashlib.sha256("|".join(shas.values()).encode()).hexdigest()[:16],
        "target_range_abs_inclusive": [0, 23],
        "media_contract": expected,
        "sides": {
            label: {"file": f"{label}.mp4", "sha256": _sha256(output / f"{label}.mp4")}
            for label in ("A", "B", "C")
        },
        "mapping_disclosed": False,
    }
    private = {
        "schema": "t8.face_refine_window_blind_key.v1",
        "review_id": public["review_id"],
        "mapping": mapping,
        "inputs": {
            name: {"path": str(path), "sha256": shas[name], "probe": metadata[name]}
            for name, path in inputs.items()
        },
    }
    (output / "public_manifest.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "private_key.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = json.dumps(public["sides"], ensure_ascii=False)
    review_id = json.dumps(public["review_id"])
    page = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Face Refine Window 盲审</title><style>
body{{margin:0;background:#10141c;color:#eef3ff;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}section{{background:#1a2230;border:1px solid #344158;border-radius:10px;padding:10px}}video{{width:100%;background:#000}}button,select,textarea{{font:inherit;padding:8px}}button{{margin:4px}}label{{display:block;margin:8px 0}}textarea{{width:100%;min-height:80px;box-sizing:border-box}}.warn{{color:#ffda74}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Face Refine 局部窗口 A/B/C 盲审</h1><p class=\"warn\">请先看完整视频，再重点反复看第0–23帧（约前1秒）的五官结构。A/B/C中包含原片、固定上游代码和新窗口链，但页面不显示对应关系。本素材静音，不评价声音或口型同步。</p>
<p><button id=\"play\">三路同步播放</button><button id=\"pause\">全部暂停</button><button id=\"start\">回到开头</button><button id=\"one\">定位到第24帧边界（1.0秒）</button></p><div class=\"grid\" id=\"videos\"></div>
<h2>结论</h2><label>总体更好 <select id=\"overall\"><option value=\"pending\">未判断</option><option>A</option><option>B</option><option>C</option><option value=\"tie\">差不多/平</option><option value=\"none\">都不行</option></select></label>
<label>前24帧五官更自然 <select id=\"features\"><option value=\"pending\">未判断</option><option>A</option><option>B</option><option>C</option><option value=\"tie\">差不多/平</option><option value=\"none\">都不行</option></select></label>
<label>身份更稳定 <select id=\"identity\"><option value=\"pending\">未判断</option><option>A</option><option>B</option><option>C</option><option value=\"tie\">差不多/平</option><option value=\"none\">都不行</option></select></label>
<label>时序/接缝更自然 <select id=\"temporal\"><option value=\"pending\">未判断</option><option>A</option><option>B</option><option>C</option><option value=\"tie\">差不多/平</option><option value=\"none\">都不行</option></select></label>
<label>三路是否都完整看过 <input id=\"watched\" type=\"checkbox\"></label><label>备注/问题时间点<textarea id=\"notes\"></textarea></label><button id=\"export\">导出评审JSON</button>
<script>const sides={rows},reviewId={review_id};const box=document.getElementById('videos');for(const label of ['A','B','C']){{const s=document.createElement('section');s.innerHTML='<h2>'+label+'</h2><video controls preload=\"metadata\" src=\"'+sides[label].file+'\"></video>';box.append(s)}}const vids=[...document.querySelectorAll('video')];function sync(t){{for(const v of vids)v.currentTime=t}}document.getElementById('play').onclick=async()=>{{sync(vids[0].currentTime);for(const v of vids)await v.play()}};document.getElementById('pause').onclick=()=>vids.forEach(v=>v.pause());document.getElementById('start').onclick=()=>sync(0);document.getElementById('one').onclick=()=>sync(1);document.getElementById('export').onclick=()=>{{const value={{schema:'{SCHEMA}',review_id:reviewId,exported_at:new Date().toISOString(),review_completed:document.getElementById('watched').checked,overall:document.getElementById('overall').value,features_first24:document.getElementById('features').value,identity:document.getElementById('identity').value,temporal_seam:document.getElementById('temporal').value,notes:document.getElementById('notes').value}};const blob=new Blob([JSON.stringify(value,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='face_refine_window_blind_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};</script></main></body></html>"""
    (output / "blind_review.html").write_text(page, encoding="utf-8")
    return {"public": public, "private": private, "page": str((output / "blind_review.html").resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.upstream, args.window, args.output)
    print(json.dumps({"page": result["page"], "review_id": result["public"]["review_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
