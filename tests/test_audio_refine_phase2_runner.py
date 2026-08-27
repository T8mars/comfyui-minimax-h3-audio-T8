from __future__ import annotations

from array import array
import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_audio_refine_phase2.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_audio_refine_phase2", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_allows_exactly_one_missing_arm_per_invocation():
    tool = _load_tool()

    ordinary = tool.build_prompt(
        arm="base_ordinary8", run_id="ordinary", seed=2608260404
    )
    refine = tool.build_prompt(
        arm="base_refine4", run_id="refine", seed=2608260404
    )

    assert tool.MAX_PROMPTS_PER_INVOCATION == 1
    assert tool.ALLOWED_ARMS == ("base_ordinary8", "base_refine4")
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced"
        for node in ordinary.values()
    ) == 1
    assert not any(
        node["class_type"] == "MiniMaxH3AudioRefineModelRouteT8Advanced"
        for node in ordinary.values()
    )
    assert ordinary["9"]["inputs"]["model"] == ["6", 0]
    assert ordinary["9"]["inputs"]["steps"] == 8
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced" for node in refine.values()
    ) == 2
    assert refine["17"]["inputs"]["route_strategy"] == "base_without_turbo"
    assert refine["17"]["inputs"]["refine_model"] == ["6", 0]
    assert refine["18"]["inputs"]["refine_steps"] == 4
    assert refine["18"]["inputs"]["audio_denoise"] == 0.5
    assert refine["25"]["inputs"]["accept_candidate"] is False


def test_runner_rejects_existing_or_unregistered_arms():
    tool = _load_tool()

    for arm in ("turbo4_original", "same_turbo_stack", "unknown"):
        with pytest.raises(ValueError, match="unsupported Phase 2 arm"):
            tool.build_prompt(arm=arm, run_id="blocked", seed=1)


def test_runner_binds_review_geometry_prompt_and_output_prefixes():
    tool = _load_tool()
    prompt = tool.build_prompt(
        arm="base_refine4",
        run_id="fixed-run",
        seed=77,
        width=1056,
        height=608,
        frames=124,
        audio_denoise=0.35,
    )

    assert prompt["8"]["inputs"]["prompt"] == tool.PROMPT
    assert prompt["8"]["inputs"]["width"] == 1056
    assert prompt["8"]["inputs"]["height"] == 608
    assert prompt["8"]["inputs"]["length"] == 124
    assert prompt["18"]["inputs"]["audio_denoise"] == 0.35
    prefixes = {
        node["inputs"]["filename_prefix"]
        for node in prompt.values()
        if node["class_type"] == "SaveVideo"
    }
    assert prefixes == {
        "MiniMaxH3_AudioRefine_Phase2/fixed-run_base_refine4_original",
        "MiniMaxH3_AudioRefine_Phase2/fixed-run_base_refine4_candidate",
        "MiniMaxH3_AudioRefine_Phase2/fixed-run_base_refine4_selected",
    }


def test_checkpointed_refine_prompt_saves_three_native_av_latents_without_video_encode():
    tool = _load_tool()
    prompt = tool.build_prompt(
        arm="base_refine4",
        run_id="checkpoint-run",
        seed=77,
        checkpointed=True,
    )

    assert not any(node["class_type"] == "SaveVideo" for node in prompt.values())
    saves = {
        node_id: node
        for node_id, node in prompt.items()
        if node["class_type"]
        == "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
    }
    assert set(saves) == {"15", "24", "28"}
    assert saves["15"]["inputs"]["av_latent"] == ["12", 0]
    assert saves["24"]["inputs"]["av_latent"] == ["21", 0]
    assert saves["28"]["inputs"]["av_latent"] == ["25", 0]
    assert all(node["inputs"]["confirm_save"] is True for node in saves.values())
    assert all(node["inputs"]["verify_after_write"] is True for node in saves.values())
    assert prompt["29"]["inputs"]["source"] == ["15", 5]
    assert prompt["32"]["inputs"]["source"] == ["15", 4]
    assert prompt["25"]["inputs"]["original_audio"] == ["13", 1]
    assert prompt["25"]["inputs"]["candidate_audio"] == ["22", 1]


def test_checkpoint_decode_prompt_requires_external_manifest_and_hash():
    tool = _load_tool()
    record = {
        "checkpoint_path": "audio_refine_phase2/example.h3latent.safetensors",
        "file_sha256": "a" * 64,
        "manifest_json": '{"schema":"example"}',
    }

    prompt = tool.build_checkpoint_decode_prompt(
        record=record, run_id="decode-run", label="candidate"
    )

    assert prompt["3"]["class_type"] == (
        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
    )
    assert prompt["3"]["inputs"]["expected_manifest_json"] == record["manifest_json"]
    assert prompt["3"]["inputs"]["expected_file_sha256"] == record["file_sha256"]
    assert prompt["4"]["inputs"]["av_latent"] == ["3", 0]
    assert prompt["5"]["class_type"] == "SaveImage"
    assert prompt["6"]["class_type"] == "SaveAudio"
    assert prompt["7"]["inputs"]["source"] == ["3", 7]


def test_bounded_file_fingerprint_is_deterministic_and_detects_content_change(
    tmp_path,
):
    tool = _load_tool()
    path = tmp_path / "asset.bin"
    path.write_bytes(bytes(range(256)) * 40)

    first = tool.bounded_file_fingerprint(path, sample_bytes=128)
    second = tool.bounded_file_fingerprint(path, sample_bytes=128)
    path.write_bytes(b"changed!" + path.read_bytes()[8:])
    changed = tool.bounded_file_fingerprint(path, sample_bytes=128)

    assert first["bounded_sample_sha256"] == second["bounded_sample_sha256"]
    assert first["sample_offsets"] == second["sample_offsets"]
    assert first["full_file_sha256"] is None
    assert first["bounded_sample_sha256"] != changed["bounded_sample_sha256"]
    assert first["claim"] == "bounded identity sample; not a full-file hash"


def test_decoded_pcm_contract_rejects_nonfinite_clipping_and_channel_collapse():
    tool = _load_tool()

    healthy = tool._pcm_values_contract(array("f", [0.10, -0.08] * 1000))
    collapsed = tool._pcm_values_contract(array("f", [0.10, 0.0] * 1000))
    clipped = tool._pcm_values_contract(array("f", [1.0, -1.0] * 1000))
    nonfinite = tool._pcm_values_contract(array("f", [float("nan"), 0.1]))

    assert healthy["passed"] is True
    assert collapsed["passed"] is False
    assert collapsed["channel_collapse_suspected"] is True
    assert clipped["passed"] is False
    assert clipped["clipping_suspected"] is True
    assert nonfinite["passed"] is False
    assert nonfinite["finite"] is False


def test_parse_args_is_dry_run_and_locks_profile():
    tool = _load_tool()

    args = tool.parse_args(["--arm", "base_refine4"])

    assert args.confirm_run is False
    assert args.arm == "base_refine4"
    assert (args.width, args.height, args.frames) == (1056, 608, 124)
    assert args.audio_denoise == 0.5
    assert args.scenario == "baseline_dialogue"
    assert args.task_type == "T2VA"
    assert args.reference_image is None
    assert args.min_free_vram_mib == 12000
    assert args.reserve_vram_gib == 4.0
    with pytest.raises(SystemExit):
        tool.parse_args(["--arm", "same_turbo_stack"])


def test_scenario_args_and_i2va_graph_are_explicit(tmp_path):
    tool = _load_tool()
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"test fixture path only")
    args = tool.parse_args(
        [
            "--arm",
            "base_refine4",
            "--scenario",
            "i2va_speech",
            "--reference-image",
            str(reference),
        ]
    )
    prompt = tool.build_prompt(
        arm=args.arm,
        run_id="i2va-run",
        seed=args.seed,
        checkpointed=True,
        prompt_text=args.prompt_text,
        task_type=args.task_type,
        first_image_name="reference.png",
    )

    assert args.task_type == "I2VA"
    assert args.prompt_text == tool.I2VA_SPEECH_PROMPT
    assert args.reference_image == reference
    assert prompt["8"]["inputs"]["task_type"] == "I2VA"
    assert prompt["8"]["inputs"]["first_frame"] == ["35", 0]
    assert prompt["35"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "reference.png"},
    }


def test_history_json_accepts_v3_executed_output_when_history_is_null():
    tool = _load_tool()
    report = {"decision": "ALLOW", "source": "v3_executed"}
    phase = {
        "history": None,
        "executed_outputs": {"17": {"text": [tool.json.dumps(report)]}},
    }

    assert tool._history_json(phase, "17") == report


def test_phase2_server_command_uses_conservative_private_vram_reserve(tmp_path):
    tool = _load_tool()
    args = tool.parse_args(["--arm", "base_refine4"])
    command = tool.smoke._server_command(args, tmp_path)

    reserve_index = command.index("--reserve-vram")
    assert command[reserve_index + 1] == "4.0"


def test_default_main_never_starts_real_run(monkeypatch, tmp_path):
    tool = _load_tool()
    called = []
    monkeypatch.setattr(
        tool,
        "preflight",
        lambda _args: {"status": "READY", "ready_for_real_run": True},
    )
    monkeypatch.setattr(tool, "run_real_arm", lambda *_args: called.append(True))

    result = tool.main(
        [
            "--arm",
            "base_ordinary8",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert result == 0
    assert called == []
