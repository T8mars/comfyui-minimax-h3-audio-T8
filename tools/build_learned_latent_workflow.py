#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .api_to_frontend_workflow import _get_json, convert
except ImportError:  # Direct execution puts tools/ on sys.path.
    from api_to_frontend_workflow import _get_json, convert


NOTES = (
    (
        "1 · Current upstream parity / 当前上游基线",
        "## 1 · Current upstream parity / 当前上游基线\n"
        "This graph follows LBH-123-AI workflow commit `64fc9d4`: corrected LightX2V "
        "FL2V Turbo `comfyui_alpha8` conversion at loader strength 1.0, "
        "I2V shift `12/3`, Comfy `simple` 8-step schedule split after four low-resolution "
        "calls, then the published raw 3-call refine sigmas `0.9035, 0.6316, 0.3158, 0`. "
        "Do not remap those values through a guessed shift-6 reference.",
    ),
    (
        "2 · Change only one size / 只改一个倍率",
        "## 2 · Automatic size synchronization / 尺寸自动同步\n"
        "Change only Learned Latent Upscale `scale_by`. Its `width` and `height` outputs are "
        "already connected to HIGH Conditioning. Do not create a second manual width/height. "
        "Default `2.0x` turns the LOW `736x416` latent into `1472x832`.",
    ),
    (
        "3 · Required pass handoff / 二采接线",
        "## 3 · Required pass handoff / 二采接线\n"
        "Pass 1 must feed `denoised_output` into the learned upscaler. HIGH Conditioning is "
        "rebuilt after the upscaler reports its actual aligned size, then Reconcile verifies "
        "the contract. Pass 2 uses fresh RandomNoise; AV Decode receives sampler `output`, "
        "not `denoised_output`.",
    ),
    (
        "4 · Optional detail and memory / 可选细节与显存",
        "## 4 · Optional detail and memory / 可选细节与显存\n"
        "Keep Tail/Bias/STG/Restart OFF for upstream parity. They alter the HIGH refine path "
        "and require a separate A/B review. Learned latent upscale saves first-pass compute, "
        "not the high-resolution peak VRAM. `offload_after` releases only the learned resizer.",
    ),
    (
        "5 · Starting values / 建议参数",
        "## 5 · Known-good starting values / 建议参数\n"
        "LOW `736x416x124`; scale mode `scale_by`; scale `2.0`; output `1472x832`; "
        "video/audio shift `12/3`; base/coarse/refine `8/4/3`; Euler; all optional detail "
        "toggles OFF. Use only the `_comfyui_alpha8` LightX2V conversion; the superseded "
        "plain `_comfyui` file applies 16x excessive LoRA strength and destroys frames.",
    ),
)


def append_notes(workflow: dict) -> None:
    next_id = max(node["id"] for node in workflow["nodes"]) + 1
    for index, (title, text) in enumerate(NOTES):
        workflow["nodes"].append(
            {
                "id": next_id + index,
                "type": "MarkdownNote",
                "title": title,
                "pos": [index * 540, -440],
                "size": [520, 280],
                "flags": {},
                "order": len(workflow["nodes"]),
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {"Node name for S&R": "MarkdownNote"},
                "widgets_values": [text],
            }
        )
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the importable learned-latent two-pass H3 workflow."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--api",
        type=Path,
        default=Path("tests/fixtures/api/learned_latent_two_pass_i2va_advanced_api.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "examples/workflows/13-latent-upscale/"
            "2026-08-19_H3_Learned_Latent_TwoPass_I2VA_Advanced_EXP.json"
        ),
    )
    args = parser.parse_args()
    prompt = json.loads(args.api.read_text(encoding="utf-8"))
    object_info = _get_json(f"{args.server.rstrip('/')}/object_info")
    workflow = convert(
        prompt,
        object_info,
        "MiniMax H3 learned latent two-pass I2VA · upstream parity",
    )
    append_notes(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
