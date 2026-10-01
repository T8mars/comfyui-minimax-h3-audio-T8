"""Correct fixed H16 cold-resume HIGH geometry in new private candidates.

The historical v2/v4 pairs stay byte-for-byte intact. Their freeze graphs use
736x416 LOW and the production learned upscaler's exact 2x geometry, while
the resume drafts accidentally hard-code 1344x768 HIGH. This builder changes
only those stale HIGH widgets/literals in new resume copies.
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
        "source": PRIVATE / "modular-sampling-m4-h16-storage-20260923/candidate-v2",
        "target": PRIVATE / "modular-sampling-m4-h16-storage-20260923/candidate-v3",
        "names": ("01_freeze_after_window_2", "02_resume_windows_3_to_6_DRAFT"),
        "sha": (
            ("44a44417b541afed42c46956472573610dde77a11ef4fff02ca58636e1223d1d",
             "41464d8ae9ad7f9ea50dd79ace53575c273c7ce7634fcb590e623948bae4ddaa"),
            ("efcf6a38ed5fbb263ada5013dee36e9a459582eded2749df4025d05c6c215a1b",
             "9a779997f99db4cb5f5a74e4aedafbf28dcf64f77a57e6757c0261cad8909c6b"),
        ),
        "high_conditioners": (14,),
    },
    "effects": {
        "source": PRIVATE / "modular-sampling-m4-h16-effect-storage-20260924/candidate-v4",
        "target": PRIVATE / "modular-sampling-m4-h16-effect-storage-20260924/candidate-v5",
        "names": ("01_freeze_after_window_2_relay_eav",
                  "02_resume_windows_3_to_6_relay_eav_DRAFT"),
        "sha": (
            ("5e6d6a16ed918ba4e34cd7d3df79f9676e55d584c1cd740ae1cea0e29e73169f",
             "00ed46f56a7c8d79540bb14c8b69d28d5e2cf38d8967c63572105cc977c2ab7b"),
            ("d3430b0e4632039ec1360aaff3b8ec91264ccc1b900f27feb0f7950407e661ea",
             "7a1da7291e8caa7c74748206b7b4d5b03c59c972878918537ba630a600247d1e"),
        ),
        "high_conditioners": (14, 52),
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pair(route: str) -> tuple[dict, dict, dict, dict]:
    config = PAIRS[route]
    graphs = []
    for name, (frontend_sha, api_sha) in zip(config["names"], config["sha"], strict=True):
        frontend_path = config["source"] / f"{name}.json"
        api_path = config["source"] / f"{name}.api.json"
        if _sha(frontend_path) != frontend_sha or _sha(api_path) != api_sha:
            raise ValueError(f"Historical H16 {route} source changed: {name}")
        graphs.append((json.loads(frontend_path.read_bytes()),
                       json.loads(api_path.read_bytes())))
    return graphs[0][0], graphs[0][1], graphs[1][0], graphs[1][1]


def expected_high_geometry(freeze_api: dict) -> tuple[int, int]:
    low = freeze_api["7"]["inputs"]
    upscale = freeze_api["13"]["inputs"]
    if (freeze_api["7"]["class_type"] != "MiniMaxH3AudioConditioningT8"
            or freeze_api["13"]["class_type"] != "MiniMaxH3LearnedLatentUpscaleT8Advanced"
            or (low["width"], low["height"], low["length"]) != (736, 416, 124)
            or upscale["av_latent"] != ["12", 1]
            or upscale["size_mode"] != "scale_by"
            or upscale["scale_by"] != 2.0
            or upscale["aspect_policy"] != "preserve_source"
            or low["width"] % 32 or low["height"] % 32):
        raise ValueError("H16 LOW/upscaler geometry contract changed")
    # Exact for this fixed, 32-aligned 2x recipe. Test separately against
    # production learned_upscale_geometry, not a generic sizing replacement.
    return low["width"] * 2, low["height"] * 2


def corrected_pair(route: str) -> tuple[dict, dict, dict, dict]:
    if route not in PAIRS:
        raise ValueError("Expected plain or effects H16 route")
    freeze_front, freeze_api, resume_front, resume_api = source_pair(route)
    high_width, high_height = expected_high_geometry(freeze_api)
    if (high_width, high_height) != (1472, 832):
        raise ValueError("Fixed H16 high geometry unexpectedly changed")
    config = PAIRS[route]
    fixed_front = deepcopy(resume_front)
    fixed_api = deepcopy(resume_api)
    nodes = {node["id"]: node for node in fixed_front["nodes"]}
    for node_id in config["high_conditioners"]:
        frontend = nodes[node_id]
        api = fixed_api[str(node_id)]
        expected_kind = ("MiniMaxH3AudioConditioningT8" if node_id == 14 else
                         "MiniMaxH3PromptRelayConditioningT8Advanced")
        if (frontend["type"] != expected_kind or api["class_type"] != expected_kind
                or frontend["widgets_values"][1 if node_id == 14 else 0] != 1344
                or frontend["widgets_values"][2 if node_id == 14 else 1] != 768
                or api["inputs"].get("width") != 1344
                or api["inputs"].get("height") != 768):
            raise ValueError(f"Historical H16 HIGH conditioner changed: {route}/{node_id}")
        width_slot, height_slot = ((1, 2) if node_id == 14 else (0, 1))
        frontend["widgets_values"][width_slot] = high_width
        frontend["widgets_values"][height_slot] = high_height
        api["inputs"]["width"] = high_width
        api["inputs"]["height"] = high_height
    if (fixed_api["51" if route == "plain" else "88"]["inputs"]
            .get("checkpoint_path") != ""):
        raise ValueError("H16 resume draft unexpectedly contains a receipt")
    return freeze_front, freeze_api, fixed_front, fixed_api


def main() -> None:
    for route, config in PAIRS.items():
        target = config["target"]
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite H16 geometry candidate: {target}")
        _freeze_front, _freeze_api, fixed_front, fixed_api = corrected_pair(route)
        target.mkdir(parents=True)
        freeze, resume = config["names"]
        for suffix in (".json", ".api.json"):
            shutil.copyfile(config["source"] / f"{freeze}{suffix}",
                            target / f"{freeze}{suffix}")
        for suffix, graph in ((".json", fixed_front), (".api.json", fixed_api)):
            (target / f"{resume}{suffix}").write_text(
                json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"route": route, "target": str(target),
                          "high_geometry": [1472, 832]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
