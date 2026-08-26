from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_pdd_real_validation.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_pdd_real_validation", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase_text_prefers_current_comfy_v3_executed_output():
    tool = _load_tool()
    phase = {
        "executed_outputs": {"14": {"text": ["v3-report"]}},
        "history": {"outputs": {"14": {"text": ["legacy-report"]}}},
    }

    assert tool._phase_text(phase, "14") == "v3-report"


def test_phase_text_keeps_legacy_history_fallback():
    tool = _load_tool()
    phase = {"history": {"outputs": {"14": {"text": ["legacy-report"]}}}}

    assert tool._phase_text(phase, "14") == "legacy-report"


def test_phase_text_accepts_current_comfy_null_history():
    tool = _load_tool()
    phase = {
        "executed_outputs": {"14": {"text": ["v3-report"]}},
        "history": None,
    }

    assert tool._phase_text(phase, "14") == "v3-report"


def test_real_prompts_use_one_pdd_setup_and_official_eight_nfe_contract():
    tool = _load_tool()
    args = tool._parser().parse_args(["--variant", "FL2VA"])

    prompt = tool._prompt(args, "unit-test")

    assert prompt["7"]["inputs"]["task_type"] == "FL2VA"
    assert prompt["7"]["inputs"]["width"] == 736
    assert prompt["7"]["inputs"]["height"] == 416
    assert prompt["7"]["inputs"]["length"] == 124
    assert prompt["8"]["class_type"] == "MiniMaxH3PDD8StepSetupT8Advanced"
    assert prompt["8"]["inputs"]["strength"] == 1.0
    assert prompt["11"]["inputs"]["sampler"] == ["8", 1]
    assert prompt["11"]["inputs"]["sigmas"] == ["8", 2]
    assert prompt["14"]["inputs"]["source"] == ["8", 3]


def test_ref2va_prompt_uses_matching_base_adapter_and_reference_slot():
    tool = _load_tool()
    args = tool._parser().parse_args(["--variant", "Ref2VA"])

    prompt = tool._prompt(args, "unit-test")

    assert prompt["4"]["inputs"]["unet_name"] == tool.BASE_NAME["Ref2VA"]
    assert prompt["7"]["inputs"]["task_type"] == "Ref2VA"
    assert prompt["7"]["inputs"]["ref_images.ref_image_0"] == ["5", 0]
    assert prompt["8"]["inputs"]["base_variant"] == "Ref2VA"
    assert prompt["8"]["inputs"]["pdd_lora_name"] == tool.EXPECTED_LORA["Ref2VA"][0]


def test_real_prompt_accepts_explicit_0p7mp_dimensions_without_changing_defaults():
    tool = _load_tool()
    args = tool._parser().parse_args(
        [
            "--variant",
            "Ref2VA",
            "--width",
            "1152",
            "--height",
            "640",
            "--frame-count",
            "124",
        ]
    )
    tool._validate_contract(args)

    prompt = tool._prompt(args, "unit-test")

    assert prompt["7"]["inputs"]["width"] == 1152
    assert prompt["7"]["inputs"]["height"] == 640
    assert prompt["7"]["inputs"]["length"] == 124
    assert "1152x640_124f" in prompt["13"]["inputs"]["filename_prefix"]


def test_real_prompt_rejects_non_native_grid_dimensions():
    tool = _load_tool()
    args = tool._parser().parse_args(
        ["--variant", "Ref2VA", "--width", "1151", "--height", "640"]
    )

    try:
        tool._validate_contract(args)
    except ValueError as exc:
        assert "divisible by 32" in str(exc)
    else:
        raise AssertionError("non-native-grid dimensions were accepted")


def test_resource_gate_failure_is_distinct_from_media_or_setup_failure(
    monkeypatch, tmp_path
):
    tool = _load_tool()
    args = tool._parser().parse_args(["--variant", "FL2VA"])
    phase = {
        "executed_outputs": {
            "14": {
                "text": [
                    "{\"adapter\":{\"base_variant\":\"FL2VA\"},"
                    "\"lora\":{\"mapped_adapters\":258,\"bypass_hooks\":258,"
                    "\"eject_policy\":\"move_adapter_weights_to_model_offload_device\"},"
                    "\"sampling\":{\"block_indices\":[0,1,2,3,4,5,6,7],\"nfe\":8}}"
                ]
            }
        }
    }
    report = {"run_id": "unit-test", "gpu_monitor": {"minimum_free_mib": 510}}
    output = tmp_path / "output" / "MiniMaxH3_PDD_Validation"
    output.mkdir(parents=True)
    video = output / "unit-test_fl2va_736x416_124f_00001-audio.mp4"
    video.write_bytes(b"fixture")
    media = {
        "strict_decode_passed": True,
        "probe": {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 736,
                    "height": 416,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "32000",
                    "channels": 2,
                },
            ]
        },
        "decoded_video": {"bytes": 124 * 736 * 416 * 3},
    }
    monkeypatch.setattr(tool.shared, "media_report", lambda *_args, **_kwargs: media)
    monkeypatch.setattr(
        tool, "_audio_numeric", lambda *_args, **_kwargs: {"sample_values": 8, "all_finite": True}
    )
    monkeypatch.setattr(tool, "_contact_sheet", lambda *_args, **_kwargs: None)

    result = tool._finalize_completed_run(
        args=args,
        run_root=tmp_path,
        phase=phase,
        report=report,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )

    assert result == 1
    assert report["status"] == (
        "MEDIA_SETUP_PASS_RESOURCE_GATE_FAIL_HUMAN_REVIEW_PENDING"
    )
