from __future__ import annotations

import json
from pathlib import Path

from h3_audio_t8_pkg.tools import build_dlss_nr_workflows as builder


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "workflows" / "25-dlss-nr"
MIRROR = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "25-dlss-nr"
)


def _node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(matches) == 1
    return matches[0]


def test_four_independent_dlss_nr_workflows_match_builder_and_user_mirror():
    expected = builder.build_workflows()
    assert {path.name for path in SOURCE.glob("*.json")} == set(
        builder.FILENAMES.values()
    )
    for name, filename in builder.FILENAMES.items():
        source = SOURCE / filename
        mirror = MIRROR / filename
        actual = json.loads(source.read_text(encoding="utf-8"))
        assert actual == expected[name]
        assert source.read_bytes() == mirror.read_bytes()
        assert actual["version"] == 0.4
        assert actual["last_node_id"] == max(node["id"] for node in actual["nodes"])
        assert actual["last_link_id"] == max(
            (link[0] for link in actual["links"]), default=0
        )


def test_every_execution_workflow_has_its_own_fail_closed_runtime_audit():
    for name, workflow in builder.build_workflows().items():
        runtime = _node(workflow, builder.RUNTIME)
        assert runtime["widgets_values"] == [
            "1.3",
            False,
            "feature_probe_1_frame",
            0,
            0,
        ]
        note = _node(workflow, "MarkdownNote")["widgets_values"][0]
        for required in (
            "Windows + NVIDIA RTX",
            "ComfyUI/models/DLSS-NR/1.3/",
            "不下载",
            "不会修复",
        ):
            assert required in note, name
        if name != "runtime":
            assert runtime["outputs"][0]["links"]


def test_image_workflow_pins_standard_two_x_and_previews_source_and_candidate():
    workflow = builder.build_workflows()["image"]
    node = _node(workflow, builder.IMAGE)
    assert node["widgets_values"][:4] == ["sr_nr", "2.0", "standard", "default"]
    assert sum(item["type"] == "PreviewImage" for item in workflow["nodes"]) == 2


def test_video_frames_workflow_preserves_source_fps_and_audio_connections():
    workflow = builder.build_workflows()["video_frames"]
    node = _node(workflow, builder.VIDEO_FRAMES)
    inputs = {item["name"]: item for item in node["inputs"]}
    assert inputs["fps"]["widget"] == {"name": "fps"}
    assert inputs["audio"]["type"] == "AUDIO"
    assert node["widgets_values"][1:5] == [
        "sr_nr",
        "2.0",
        "standard",
        "default",
    ]
    create = _node(workflow, "CreateVideo")
    create_inputs = {item["name"]: item for item in create["inputs"]}
    assert create_inputs["audio"]["link"] is not None
    assert create_inputs["fps"]["link"] is not None


def test_video_file_workflow_is_direct_file_backed_and_output_owned():
    workflow = builder.build_workflows()["video_file"]
    node = _node(workflow, builder.VIDEO_FILE)
    assert [item["name"] for item in node["inputs"]] == [
        "dlss_nr_runtime",
        "source_video",
    ]
    assert node["widgets_values"][:4] == ["sr_nr", "2.0", "standard", "default"]
    assert node["widgets_values"][-2:] == [
        "MiniMaxH3/DLSS-NR/video_file_standard_2x",
        18.0,
    ]
    assert all(item["type"] != "GetVideoComponents" for item in workflow["nodes"])


def test_readmes_document_openvdn_bundle_and_complete_dlss_runtime_contract():
    chinese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_EN.md").read_text(encoding="utf-8")
    workflow = (SOURCE / "README.md").read_text(encoding="utf-8")

    for readme in (chinese, english):
        assert "https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy" in readme
        assert "https://github.com/DaniilSokolyuk/video2dlssnr/releases/tag/v1.3" in readme
        assert "video2dlssnr_release.zip" in readme
        assert "t8-runtime-manifest.json" in readme
        assert "nvngx_dlssnr.dll" in readme
        assert "616.56" in readme
        assert "ffprobe" in readme.lower()

    for required in (
        "video2dlssnr_release.zip",
        "t8-runtime-manifest.json",
        "nvngx_dlss.dll",
        "nvngx_dlssnr.dll",
        "nvngx.dll_dlssnr.dll",
        "616.56",
        "ffprobe",
    ):
        assert required in workflow
