"""Build a playable offline screening page, without inference or source edits."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from string import Template

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "h3_t8"))
from video_interval_screen import screen_video  # noqa: E402


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--region", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--expected-cut", action="append", type=float, default=[])
    parser.add_argument("--cut-threshold", type=float, default=0.22)
    parser.add_argument("--local-threshold", type=float, default=0.08)
    args = parser.parse_args()
    source = args.video.resolve(strict=True)
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    digest = file_hash(source)
    report, rows = screen_video(
        source,
        args.region,
        cut_threshold=args.cut_threshold,
        local_threshold=args.local_threshold,
        expected_cuts=args.expected_cut,
    )
    report.update(source_sha256=digest, source_name=source.name)
    media = "source" + source.suffix.lower()
    shutil.copyfile(source, target / media)
    if file_hash(source) != digest or file_hash(target / media) != digest:
        raise ValueError(
            "Source media changed during screening or copying; no bound review produced"
        )
    (target / "metrics.json").write_text(
        json.dumps(dict(report=report, frames=rows), ensure_ascii=False, indent=2),
        "utf8",
    )
    items = "".join(
        '<li><button data-time="'
        + str(max(0.0, r["focus"] - report["first_pts"] - 1))
        + '">'
        + f"{r['focus'] - report['first_pts']:.3f} 秒 · "
        + html.escape(r["label"])
        + "</button></li>"
        for r in report["intervals"]
    )
    content = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>视频复看提示</title><style>body{background:#111820;color:#e9eef2;max-width:1180px;margin:24px auto;padding:16px;font:16px system-ui}video{width:100%;max-height:70vh;object-fit:contain;background:#000}button{color:inherit;background:#243340;border:1px solid #53697c;border-radius:6px;padding:10px;margin:5px;cursor:pointer}li{list-style:none}ul{padding:0}p{line-height:1.6;color:#c1cfda}</style>
<h1>视频复看提示</h1><p>只定位值得复看的时刻，不自动拒片或重采。正常剪辑、闪光、运动和遮挡也会触发提示；身份与细节请在原视频中比对。没有提示不代表没有问题。</p>
<video controls preload="metadata" src="MEDIA"></video><p>NAME</p><ul>ITEMS</ul>
<script>document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>{let v=document.querySelector('video');v.currentTime=Number(b.dataset.time);v.play().catch(()=>{});});</script></html>"""
    template=Template(content.replace("MEDIA", "$media").replace("NAME", "$name").replace("ITEMS", "$items"))
    content=template.substitute(media=html.escape(media,quote=True), name=html.escape(source.name),
                                items=items or "<li>当前阈值未发现明显突变；仍可完整播放复查。</li>")
    (target / "review.html").write_text(content, "utf8")
    print(
        json.dumps(
            dict(
                path=str(target / "review.html"),
                hints=len(report["intervals"]),
                quality_accepted=False,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
