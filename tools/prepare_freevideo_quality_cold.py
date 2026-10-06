"""Build cold candidates only from real completed native stages; no sampling."""
import argparse
import copy
import json
from pathlib import Path

from prepare_freevideo_quality_workflows import finish, named, write, STAGE


def hide_unexecuted_preview(graph):
    # Cold has not generated a final movie. Do not show an empty/misleading
    # player inherited from Full; the real execution will populate its preview.
    for node in graph["nodes"]:
        if node["type"] == "VHS_VideoCombine":
            named(node, dict(videopreview=dict(hidden=True, paused=True, params={})))


def loader(row):
    name = "MiniMaxH3FreeVideoQualityStageLoadEXPT8"
    values = dict(manifest_path=row["path"], manifest_sha256=row["sha256"])
    return dict(id=22, type=name, pos=[700, 240], size=[480, 180], flags={}, order=0, mode=0,
        inputs=[dict(name=k, type="STRING", widget=dict(name=k), link=None) for k in values],
        outputs=[dict(name=k, type=t, links=None, slot_index=i) for i, (k, t) in enumerate(
            [("av_latent", "LATENT"), ("completed_stage", STAGE), ("report_json", "STRING")])],
        properties={"Node name for S&R": name}, widgets_values=list(values.values()), widgets_values_named=values)


def cold_high(full, low):
    graph = copy.deepcopy(full)
    keep = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21}
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in keep] + [loader(low)]
    graph["links"] = [edge for edge in graph["links"] if edge[1] in keep and edge[3] in keep]
    upscaler = next(n for n in graph["nodes"] if n["id"] == 8)
    high = next(n for n in graph["nodes"] if n["id"] == 10)
    av_slot = next(i for i, v in enumerate(upscaler["inputs"]) if v["name"] == "av_latent")
    low_slot = next(i for i, v in enumerate(high["inputs"]) if v["name"] == "completed_low")
    graph["links"] += [[101, 22, 0, 8, av_slot, "LATENT"], [102, 22, 1, 10, low_slot, STAGE]]
    hide_unexecuted_preview(graph)
    return finish(graph, "FreeVideo Quality Light · 冷恢复HIGH3 · 不重跑LOW")


def cold_decode(full, terminal):
    graph = copy.deepcopy(full)
    keep = {1, 2, 12, 13, 14}
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in keep] + [loader(terminal)]
    graph["links"] = [edge for edge in graph["links"] if edge[1] in keep and edge[3] in keep]
    decode = next(n for n in graph["nodes"] if n["id"] == 12)
    av_slot = next(i for i, v in enumerate(decode["inputs"]) if v["name"] == "av_latent")
    graph["links"] += [[101, 22, 0, 12, av_slot, "LATENT"]]
    hide_unexecuted_preview(graph)
    return finish(graph, "FreeVideo Quality · 冷恢复已完成单采 · 0扩散步")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--light", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    light, medium = [json.loads((path / "audit.json").read_text(encoding="utf8")) for path in (args.light, args.medium)]
    expected = "native_completed_stages_CPU_cold_pass_media_human_pending"
    if any(row.get("status") != expected or row.get("cold_CUDA_initialized") is not False for row in (light, medium)):
        raise ValueError("Cold candidates require actual completed native stages independently loaded on CPU")
    full_light, full_medium = [json.loads((path / "NativeSaved.json").read_text(encoding="utf8")) for path in (args.light, args.medium)]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "FVQ_Light_Cold_HIGH3.json", cold_high(full_light, light["stages"]["LOW"]))
    write(args.output / "FVQ_Medium_Cold_Decode.json", cold_decode(full_medium, medium["stages"]["SINGLE"]))
    write(args.output / "preparation.json", dict(native_reopened=False, generated=False, extra_NFE=0,
        light_source=str(args.light), medium_source=str(args.medium), human_qualified=False))
    print(args.output)


if __name__ == "__main__":
    main()
