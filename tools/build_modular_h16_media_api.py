"""Add the saved H16 resume frontend's real AV terminal to private API copies.

The fixed H16 v3/v5 frontends already contain AV Decode -> VHS VideoCombine.
Their sampling-only companion API graphs deliberately omit those nodes. This
builder preserves all source candidates and creates paired private copies with
the media terminal restored only on the completed resume graph.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
PRIVATE = PROJECT / "artifacts/development"
PAIRS = {
    "plain": {
        "source": PRIVATE / "modular-sampling-m4-h16-storage-20260923/candidate-v3",
        "target": PRIVATE / "modular-sampling-m4-h16-storage-20260923/media-api-v1",
        "names": ("01_freeze_after_window_2", "02_resume_windows_3_to_6_DRAFT"),
        "sha": (
            ("44a44417b541afed42c46956472573610dde77a11ef4fff02ca58636e1223d1d",
             "41464d8ae9ad7f9ea50dd79ace53575c273c7ce7634fcb590e623948bae4ddaa"),
            ("736636204c8064aa39570ed95d5a2c71750176c435889eeff4c32533df0bc407",
             "28d4ce202a5d3affb05e5004ee3f9281fb951eeb5ac4b6575c7e389bf93a09bc"),
        ),
        "final_av": (50, "MiniMaxH3H16Pass2WindowEXPT8"),
    },
    "effects": {
        "source": PRIVATE / "modular-sampling-m4-h16-effect-storage-20260924/candidate-v5",
        "target": PRIVATE / "modular-sampling-m4-h16-effect-storage-20260924/media-api-v1",
        "names": ("01_freeze_after_window_2_relay_eav",
                  "02_resume_windows_3_to_6_relay_eav_DRAFT"),
        "sha": (
            ("5e6d6a16ed918ba4e34cd7d3df79f9676e55d584c1cd740ae1cea0e29e73169f",
             "00ed46f56a7c8d79540bb14c8b69d28d5e2cf38d8967c63572105cc977c2ab7b"),
            ("06429413859c7a84155a7d748b18009ea229a7009e5b077f413e2d22deacd93f",
             "cf159537e94e3536de7e58fdcac50707553eb5f28f58f79b8804e5b19f31904b"),
        ),
        "final_av": (80, "MiniMaxH3H16EAVAuditEXPT8"),
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pair(route: str) -> tuple[dict, dict, dict, dict]:
    if route not in PAIRS:
        raise ValueError("Expected plain or effects H16 route")
    config = PAIRS[route]
    loaded = []
    for name, (front_sha, api_sha) in zip(config["names"], config["sha"], strict=True):
        front_path = config["source"] / f"{name}.json"
        api_path = config["source"] / f"{name}.api.json"
        if (_sha(front_path), _sha(api_path)) != (front_sha, api_sha):
            raise ValueError(f"Corrected H16 source changed: {route}/{name}")
        loaded.extend((json.loads(front_path.read_bytes()),
                       json.loads(api_path.read_bytes())))
    return tuple(loaded)


def _connected_source(frontend: dict, target: int, name: str) -> list[str | int]:
    nodes = {node["id"]: node for node in frontend["nodes"]}
    links = {link[0]: link for link in frontend["links"]}
    item = next(item for item in nodes[target]["inputs"] if item["name"] == name)
    link = links[item["link"]]
    if link[3] != target or nodes[target]["inputs"][link[4]]["name"] != name:
        raise ValueError(f"H16 frontend media link changed: {target}/{name}")
    return [str(link[1]), link[2]]


def completed_resume_api(route: str) -> dict:
    _freeze_front, _freeze_api, resume_front, resume_api = source_pair(route)
    config = PAIRS[route]
    nodes = {node["id"]: node for node in resume_front["nodes"]}
    final_id, final_kind = config["final_av"]
    if (nodes[final_id]["type"] != final_kind
            or nodes[20]["type"] != "MiniMaxH3AVDecodeT8"
            or nodes[21]["type"] != "VHS_VideoCombine"
            or "20" in resume_api or "21" in resume_api
            or _connected_source(resume_front, 20, "av_latent")
            != [str(final_id), 0]
            or _connected_source(resume_front, 20, "video_vae") != ["1", 0]
            or _connected_source(resume_front, 20, "audio_vae") != ["2", 0]
            or _connected_source(resume_front, 21, "images") != ["20", 0]
            or _connected_source(resume_front, 21, "audio") != ["20", 1]):
        raise ValueError("H16 resume media terminal no longer matches fixed frontend")
    widgets = nodes[21]["widgets_values"]
    if (len(widgets) != 6 or widgets[:2] != [24, 0]
            or widgets[3:] != ["video/h265-mp4", False, True]
            or not isinstance(widgets[2], str)):
        raise ValueError("H16 VHS saved widget contract changed")
    graph = deepcopy(resume_api)
    graph["20"] = {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {
        name: _connected_source(resume_front, 20, name)
        for name in ("av_latent", "video_vae", "audio_vae")}}
    graph["21"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate": widgets[0], "loop_count": widgets[1],
        "filename_prefix": widgets[2], "format": widgets[3],
        "pingpong": widgets[4], "save_output": widgets[5],
        "images": _connected_source(resume_front, 21, "images"),
        "audio": _connected_source(resume_front, 21, "audio"),
    }}
    if (graph["14"]["inputs"]["width"], graph["14"]["inputs"]["height"]) != (1472, 832):
        raise ValueError("H16 corrected HIGH geometry lost before media attachment")
    return graph


def main() -> None:
    for route, config in PAIRS.items():
        target = config["target"]
        source_pair(route)
        media_api = completed_resume_api(route)
        freeze, resume = config["names"]
        if target.exists():
            originals = (f"{freeze}.json", f"{freeze}.api.json", f"{resume}.json")
            if (any((target / name).read_bytes() != (config["source"] / name).read_bytes()
                    for name in originals)
                    or json.loads((target / f"{resume}.api.json").read_bytes()) != media_api):
                raise FileExistsError(f"Existing H16 media candidate differs: {target}")
            print(json.dumps({"route": route, "target": str(target),
                              "status": "already_exact"}, ensure_ascii=False))
            continue
        target.mkdir(parents=True)
        for name in (f"{freeze}.json", f"{freeze}.api.json", f"{resume}.json"):
            shutil.copyfile(config["source"] / name, target / name)
        (target / f"{resume}.api.json").write_text(
            json.dumps(media_api, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"route": route, "target": str(target),
                          "resume_api_nodes": len(media_api)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
