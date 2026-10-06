"""Create-only native UI candidates from preserved accepted legacy graphs. No Queue."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid

MODEL = "H3_T8_FREEVIDEO_QUALITY_MODEL"
STAGE = "H3_T8_FREEVIDEO_QUALITY_STAGE"
LABELS = ["Light / 轻量8+3（先LOW8，再独立HIGH3）", "Medium / 标准单采12", "High / 精细单采16", "Max / 极致单采20"]
GLOBAL = "A woman in a plain blue blouse stands in a quiet studio with a neutral gray background. Static medium close-up, natural soft light, no flashing light, no scene cuts. Natural Mandarin voice, synchronized mouth movement, no music, no other speakers."
LOCAL = "The woman looks at the camera, smiles gently and does not speak.\nShe clearly says in Mandarin Chinese: “你好，今天真不错。”"
PROMPT = GLOBAL + " She looks at the camera, smiles gently, then clearly says in Mandarin Chinese: “你好，今天真不错。”"


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def replace_type(node, name):
    node["type"] = name
    node.setdefault("properties", {})["Node name for S&R"] = name
    for socket in node.get("inputs", []) + node.get("outputs", []):
        if socket["type"] == "H3_T8_FREEVIDEO_MODEL":
            socket["type"] = MODEL
        if socket["type"] == "H3_T8_FREEVIDEO_STAGE":
            socket["type"] = STAGE


def named(node, changes):
    mapping = node.setdefault("widgets_values_named", {})
    if isinstance(node.get("widgets_values"), list):
        keys = list(mapping)
        for key, val in changes.items():
            if key not in keys:
                raise ValueError(f"Missing original widget {node['id']} {key}")
            node["widgets_values"][keys.index(key)] = val
    elif isinstance(node.get("widgets_values"), dict):
        node["widgets_values"].update(changes)
    mapping.update(changes)


def quality(node, label, width, height):
    replace_type(node, "MiniMaxH3FreeVideoQualitySamplerEXPT8")
    node["inputs"].insert(3, dict(name="quality", type="COMBO", widget=dict(name="quality"), link=None))
    node["widgets_values"] = [label, width, height, 124, 171, "fixed"]
    node["widgets_values_named"] = dict(quality=label, width=width, height=height, frames=124, seed=171, control_after_generate="fixed")


def finish(graph, title):
    graph["id"] = str(uuid.uuid4())
    graph["revision"] = 0
    graph["extra"]["workflow_title"] = title
    for node in graph["nodes"]:
        for socket in node.get("inputs", []):
            socket["link"] = None
        for socket in node.get("outputs", []):
            socket["links"] = None
    nodes = {n["id"]: n for n in graph["nodes"]}
    for link in graph["links"]:
        source, target = nodes[link[1]], nodes[link[3]]
        output, entry = source["outputs"][link[2]], target["inputs"][link[4]]
        if link[-1] == "H3_T8_FREEVIDEO_MODEL":
            link[-1] = MODEL
        if link[-1] == "H3_T8_FREEVIDEO_STAGE":
            link[-1] = STAGE
        if output["type"] != entry["type"] or link[-1] != output["type"]:
            raise ValueError(f"Graph edge mismatch {link}: {output} -> {entry}")
        output["links"] = (output["links"] or []) + [link[0]]
        entry["link"] = link[0]
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = max(link[0] for link in graph["links"])
    return graph


def light(original, config):
    graph = copy.deepcopy(original)
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != 15]
    graph["links"] = [edge for edge in graph["links"] if edge[0] not in (6, 17)]
    nodes = {n["id"]: n for n in graph["nodes"]}
    replace_type(nodes[4], "MiniMaxH3FreeVideoQualityLoaderEXPT8")
    named(nodes[4], dict(runtime_config=str(config)))
    for index in (5, 9):
        replace_type(nodes[index], "MiniMaxH3FreeVideoQualityPromptRelayEXPT8")
        named(nodes[index], dict(task_type="T2VA", audio_mode="native", add_source_as_reference=False, prompt_primary_audio_ordinal=0))
    quality(nodes[6], LABELS[0], 448, 256)
    replace_type(nodes[10], "MiniMaxH3FreeVideoCommunityHIGH3EXPT8")
    nodes[10]["inputs"] = [i for i in nodes[10]["inputs"] if i["name"] != "tail_steps"]
    nodes[10]["widgets_values"] = [172, "fixed"]
    nodes[10]["widgets_values_named"] = dict(seed=172, control_after_generate="fixed")
    for index in (7, 11):
        replace_type(nodes[index], "MiniMaxH3FreeVideoQualityStageSaveEXPT8")
    for index in (16, 19):
        named(nodes[index], dict(global_prompt=GLOBAL, local_prompts=LOCAL))
    for index in (18, 21):
        replace_type(nodes[index], "MiniMaxH3FreeVideoQualityEAVEXPT8")
        named(nodes[index], dict(mode="report_only"))
    named(nodes[14], dict(filename_prefix="FreeVideo/Quality_Light_8plus3_5s"))
    return finish(graph, "FreeVideo Quality Light · 分离8+3 · 外置Relay／EAV报告")


def single(original, config):
    graph = copy.deepcopy(original)
    keep = {1, 2, 3, 4, 5, 6, 7, 12, 13, 14}
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in keep]
    graph["links"] = [edge for edge in graph["links"] if edge[1] in keep and edge[3] in keep]
    nodes = {n["id"]: n for n in graph["nodes"]}
    replace_type(nodes[4], "MiniMaxH3FreeVideoQualityLoaderEXPT8")
    named(nodes[4], dict(runtime_config=str(config)))
    replace_type(nodes[6], "MiniMaxH3FreeVideoQualitySamplerEXPT8")
    quality(nodes[6], LABELS[1], 512, 288)
    replace_type(nodes[7], "MiniMaxH3FreeVideoQualityStageSaveEXPT8")
    named(nodes[5], dict(prompt=PROMPT, width=512, height=288, task_type="T2VA"))
    graph["links"].append([100, 6, 0, 12, 0, "LATENT"])
    named(nodes[14], dict(filename_prefix="FreeVideo/Quality_Medium_12_5s"))
    return finish(graph, "FreeVideo Quality Medium · 单采12 · 16／20未GPU人审")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-workflows", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.old_workflows / "FreeVideo_Effects_Full_8plus2.json", args.old_workflows / "FreeVideo_Basic_Full_8plus2.json"]
    raw = [p.read_bytes() for p in paths]
    graphs = [json.loads(b) for b in raw]
    write(args.output / "FVQ_Light_Full.json", light(graphs[0], args.runtime_config))
    write(args.output / "FVQ_Single_Full.json", single(graphs[1], args.runtime_config))
    assert all(p.read_bytes() == b for p, b in zip(paths, raw))
    write(args.output / "preparation.json", dict(old_original_sha256={str(p): hashlib.sha256(b).hexdigest() for p, b in zip(paths, raw)}, queued=False, GPU_qualified=False, native_saved=False))
    print(args.output)


if __name__ == "__main__":
    main()
