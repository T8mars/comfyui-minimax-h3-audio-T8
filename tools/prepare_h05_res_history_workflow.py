"""Build a NEW explicit RES boundary canvas; no queue or old graph mutation."""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def graph(info, *, image, mode, checkpoint_path, model_contract_id, run_contract_json):
    from tools.prepare_h05_res_workflow import graph as complete_graph
    from tools.api_to_frontend_workflow import convert
    from tools.repair_frontend_workflow_order import canonical_entries, _is_widget_spec, has_seed_control
    from tools.validate_freevideo_quality_canvas import check_serialized

    if mode not in ("checkpoint", "resume"):
        raise ValueError("Choose explicit checkpoint or read-only resume")
    _, api, _ = complete_graph(info, image=image)
    api["11"] = dict(class_type="MiniMaxH3RESHistoryEXPT8", inputs=dict(
        model=["7", 0], av_latent=["10", 1], steps=8, shift_video=12., shift_audio=3.,
        mode=mode, checkpoint_step=4, checkpoint_path=checkpoint_path,
        model_contract_id=model_contract_id, run_contract_json=run_contract_json,
        confirm_checkpoint_write=mode == "checkpoint", hash_chunk_megabytes=8))
    api["18"]["inputs"]["source"] = ["11", 5]
    api["17"]["inputs"]["filename_prefix"] = f"H05/RES_{mode}"
    for node_id, slot in (("21", 0), ("22", 1)):
        api[node_id] = dict(class_type="MiniMaxH3NativeLatentCheckpointSaveT8Advanced", inputs=dict(
            av_latent=["14", slot], filename_prefix=f"H05_RES_{mode}_{slot}",
            checkpoint_id=f"H05_RES_output_{slot}", confirm_save=True,
            verify_after_write=True, hash_chunk_megabytes=8))
    native = convert(api, info, f"H05_RES_{mode}")
    rows = {row["id"]: row for row in native["nodes"]}
    for row in rows.values():
        values = api[str(row["id"])]["inputs"]
        entries, unknown = canonical_entries(info[row["type"]], list(values))
        assert not unknown, unknown
        named, cursor = {}, 0
        for key, spec, _ in entries:
            if not _is_widget_spec(spec):
                continue
            value = row["widgets_values"][cursor]
            cursor += 1
            named[key] = value
            if key not in values:
                values[key] = value
            elif not isinstance(values[key], list):
                assert values[key] == value, (row["id"], key)
            if has_seed_control(key, spec):
                assert row["widgets_values"][cursor] == "fixed"
                named["control_after_generate"] = "fixed"
                cursor += 1
        assert cursor == len(row["widgets_values"])
        row["widgets_values_named"] = named
        row["size"][0] = 470
        if row["type"] == "PreviewAny":
            row["inputs"][0]["type"] = "STRING"
    for edge in native["links"]:
        if rows[edge[3]]["type"] == "PreviewAny":
            edge[5] = "STRING"
    native["nodes"].append(dict(id=23, type="Note", pos=[0, -320], size=[1300, 260],
        flags={}, order=len(api), mode=0, inputs=[], outputs=[], properties={},
        title="RES history EXP · same trajectory only",
        widgets_values=["NEW explicit RES history boundary graph; old workflows unchanged.\n"
            "Checkpoint: full8 steps, save POST step4 x/old_denoised/old_sigma_down, continue.\n"
            "Resume: read the SAME boundary and original input/seed/condition, execute remaining4 steps.\n"
            "Create-only boundary file: choose a new short name for each new render.\n"
            "MODEL raw storage, ordered LoRA and AV coordinates are checked at execution; unknown owners are not portable.\n"
            "124 preparation frames ->120 delivered at24fps =5s, actual generated video AND audio.\n"
            "Slots0/1 persist full final AV tensors for exact comparison; NOT partial x0 or LOW/HIGH.\n"
            "EAV/Relay NOT applied here. Portable Stage and human quality remain pending."]))
    native["last_node_id"] = 23
    return native, api, check_serialized(native, api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("checkpoint", "resume"))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-contract-id", required=True)
    parser.add_argument("--run-contract-json", required=True)
    parser.add_argument("--image", default="replace_with_legal_performer.png")
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.server)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.path not in ("", "/") or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Use an explicit local ComfyUI origin")
    if args.output.exists():
        raise FileExistsError("Never overwrite an old graph or earlier evidence")
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    native, _, serial = graph(info, image=args.image, mode=args.mode,
        checkpoint_path=args.checkpoint_path, model_contract_id=args.model_contract_id,
        run_contract_json=args.run_contract_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.output), serialization=serial, queued=False)))


if __name__ == "__main__":
    main()
