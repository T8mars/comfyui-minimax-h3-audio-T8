"""Build one explicit pinned Load/Route canvas; no VAE, Qwen, sampling or writes."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

try:
    from .api_to_frontend_workflow import convert
    from .build_reference_asset_workflow import choices
    from .check_windows_paths import validate_paths
except ImportError:
    from api_to_frontend_workflow import convert
    from build_reference_asset_workflow import choices
    from check_windows_paths import validate_paths


LOAD = "MiniMaxH3ReferenceLoadEXPT8"
SAVE = "MiniMaxH3ReferenceSaveEXPT8"
ROUTE = "MiniMaxH3ReferenceRouteEXPT8"
ROLES = '[{"role_id":"A","visual":false,"voice":true},{"role_id":"B","visual":true,"voice":false}]'


def make_workflow(info, *, voice_name, image_name, voice_sha, image_sha):
    installed = choices(info[LOAD]["input"]["required"]["filename"])
    if voice_name == image_name or voice_name not in installed or image_name not in installed:
        raise ValueError("Choose two exact installed packages with explicit role A voice / B image")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (voice_sha, image_sha)):
        raise ValueError("Bind the actual saved file SHA256; no inferred or empty SHA")
    if "source" not in info["PreviewAny"]["input"]["required"]:
        raise ValueError("Use the actual native Preview as Text source socket")
    graph = {
        "1": {"class_type": LOAD, "inputs": {"filename": voice_name, "expected_sha256": voice_sha}},
        "2": {"class_type": LOAD, "inputs": {"filename": image_name, "expected_sha256": image_sha}},
        "3": {"class_type": SAVE, "inputs": {"reference_package": ["1", 0],
            "filename": "readback_A.safetensors", "confirm_save": False}},
        "4": {"class_type": SAVE, "inputs": {"reference_package": ["2", 0],
            "filename": "readback_B.safetensors", "confirm_save": False}},
        "5": {"class_type": ROUTE, "inputs": {"packages.package_0": ["3", 0],
            "packages.package_1": ["4", 0], "roles_json": ROLES}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 1]}},
    }
    workflow = convert(graph, info, "Pinned reference readback and explicit offscreen route EXP")
    positions = {1: [0, 0], 2: [0, 370], 3: [460, 0], 4: [460, 370], 5: [920, 0], 6: [920, 370]}
    for node in workflow["nodes"]:
        node["pos"], node["size"] = positions[node["id"]], [430, 320]
    note = ("Load exact installed voice A / image B, pinned to actual saved file SHA256.\n"
            "Save confirmation remains FALSE: passthrough only, no output files or overwrites.\n"
            "Route explicitly keeps A voice offscreen and B image onscreen; preview the mapping.\n"
            "No asset encode, Qwen, sampler, NFE, or drive/final audio replacement.\n"
            "This is actual readback/routing validation, not generated identity/voice quality.")
    workflow["nodes"].append({"id": 7, "type": "Note", "title": "Pinned readback · contract",
        "pos": [0, 760], "size": [860, 240], "flags": {}, "order": 6, "mode": 0,
        "inputs": [], "outputs": [], "properties": {"text": note}, "widgets_values": [note]})
    workflow["last_node_id"] = 7
    workflow["extra"]["radar_r6_readback"] = {"schema": "t8.r6.reference-readback.canvas/v1",
        "new_sampling_NFE": 0, "automatic_accept": False, "roles": json.loads(ROLES)}
    return workflow, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--voice-sha", required=True)
    parser.add_argument("--image-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.output.resolve()
    owned = root / "artifacts/development/radar-r6-20261005"
    if not path.is_relative_to(owned) or path.exists():
        raise ValueError("Use one new owned RADAR canvas; do not overwrite")
    errors = validate_paths([path.relative_to(root).as_posix()])
    if errors:
        raise ValueError(errors)
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    workflow, _ = make_workflow(info, voice_name=args.voice_name, image_name=args.image_name,
                              voice_sha=args.voice_sha, image_sha=args.image_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(workflow, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
