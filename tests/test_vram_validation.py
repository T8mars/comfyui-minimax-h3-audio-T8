from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools.validate_h3_vram import (
    MIB,
    ValidationError,
    analyze_prompt,
    compare_reports,
    dynamic_vram_evidence,
    load_api_prompt,
    make_ab_prompts,
    summarize_samples,
)


def make_prompt(*, steps=4, width=1024, seed=123):
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "LoraLoaderBypassModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_turbo_4step_comfyui.safetensors",
                "strength_model": 1.0,
            },
        },
        "3": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "width": width,
                "height": 608,
                "length": 362,
                "task_type": "T2VA",
                "audio_mode": "native",
            },
        },
        "4": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["2", 0],
                "av_latent": ["3", 1],
                "steps": steps,
                "shift_video": 12.0,
                "shift_audio": 3.0,
            },
        },
        "5": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "6": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "sampler": ["4", 1],
                "sigmas": ["4", 2],
                "noise": ["5", 0],
                "guider": ["7", 0],
            },
        },
        "7": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["4", 0], "conditioning": ["3", 0]},
        },
    }


def make_report(prompt, *, label, peak_delta, status="success"):
    return {
        "label": label,
        "workflow": {"analysis": analyze_prompt(prompt)},
        "runtime": {
            "status": status,
            "summary": {"peak_vram_delta_from_baseline_bytes": peak_delta},
        },
    }


def test_analysis_identifies_control_inputs_treatment_and_non_four_step_warning():
    analysis = analyze_prompt(make_prompt(steps=12))

    assert analysis["node_count"] == 7
    assert analysis["controls"]["unets"][0]["name"].endswith("int8_convrot.safetensors")
    assert analysis["controls"]["loras"][0]["class_type"] == "LoraLoaderBypassModelOnly"
    assert analysis["controls"]["conditioning"][0]["pixel_area"] == 1024 * 608
    assert analysis["treatment"]["sampling"][0]["steps"] == 12
    assert {item["code"] for item in analysis["risks"]} == {
        "bypass_lora_gpu_residency",
        "dual_clock_non_turbo_step_count",
    }


def test_load_api_prompt_rejects_frontend_workflow(tmp_path):
    path = tmp_path / "frontend.json"
    path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")

    with pytest.raises(ValidationError, match=r"Save \(API Format\)"):
        load_api_prompt(path)


def test_dynamic_vram_requires_log_marker_for_proven_enabled_status():
    stats = {
        "system": {
            "argv": ["main.py", "--disable-smart-memory"],
            "comfy_package_versions": [
                {"name": "comfy-aimdo", "installed": "0.4.13"},
            ],
        },
        "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4060 Ti"}],
    }

    available = dynamic_vram_evidence(stats)
    enabled = dynamic_vram_evidence(
        stats, "DynamicVRAM support detected and enabled\ncomfy-aimdo version: 0.4.13"
    )
    disabled = dynamic_vram_evidence(
        {**stats, "system": {**stats["system"], "argv": ["main.py", "--novram"]}}
    )

    assert available["status"] == "available_not_proven"
    assert enabled["status"] == "enabled"
    assert enabled["source"] == "log"
    assert disabled["status"] == "disabled_by_cli"


def test_make_ab_prompts_rewires_all_dual_outputs_and_preserves_controls():
    prompt = make_prompt(steps=12)
    stock, dual = make_ab_prompts(prompt, steps=4)

    assert dual["4"]["inputs"]["steps"] == 4
    assert "4" not in stock
    stock_types = {node["class_type"] for node in stock.values()}
    assert {
        "MiniMaxH3SigmaShift",
        "KSamplerSelect",
        "BasicScheduler",
    } <= stock_types
    assert "MiniMaxH3DualClockSamplerT8" not in stock_types

    stock_analysis = analyze_prompt(stock)
    dual_analysis = analyze_prompt(dual)
    assert stock_analysis["controls"] == dual_analysis["controls"]
    assert stock_analysis["treatment"] != dual_analysis["treatment"]

    guider = stock["7"]
    sampler = stock["6"]
    assert stock[guider["inputs"]["model"][0]]["class_type"] == "MiniMaxH3SigmaShift"
    assert stock[sampler["inputs"]["sampler"][0]]["class_type"] == "KSamplerSelect"
    assert stock[sampler["inputs"]["sigmas"][0]]["class_type"] == "BasicScheduler"


def test_sample_summary_attributes_peak_to_node_and_uses_median_baseline():
    samples = [
        {
            "phase": "baseline",
            "vram_used_bytes": 100 * MIB,
            "torch_pool_used_bytes": 20 * MIB,
        },
        {
            "phase": "baseline",
            "vram_used_bytes": 120 * MIB,
            "torch_pool_used_bytes": 20 * MIB,
        },
        {
            "phase": "running",
            "node_id": "6",
            "node_type": "SamplerCustomAdvanced",
            "progress_value": 2,
            "progress_max": 4,
            "vram_used_bytes": 900 * MIB,
            "torch_pool_used_bytes": 700 * MIB,
        },
        {
            "phase": "running",
            "node_id": "6",
            "node_type": "SamplerCustomAdvanced",
            "progress_value": 3,
            "progress_max": 4,
            "vram_used_bytes": 1000 * MIB,
            "torch_pool_used_bytes": 800 * MIB,
        },
    ]

    summary = summarize_samples(samples)

    assert summary["baseline_vram_used_bytes"] == 110 * MIB
    assert summary["peak_vram_delta_from_baseline_bytes"] == 890 * MIB
    assert summary["peak_vram_node_type"] == "SamplerCustomAdvanced"
    assert summary["peak_vram_progress_value"] == 3
    assert summary["per_node"][0]["sample_count"] == 2


def test_comparison_accepts_sampler_treatment_change_but_rejects_control_change():
    first = make_report(make_prompt(steps=4), label="four", peak_delta=8 * 1024 * MIB)
    second = make_report(make_prompt(steps=12), label="twelve", peak_delta=9 * 1024 * MIB)
    controlled = compare_reports(first, second)

    assert controlled["comparable"] is True
    assert controlled["treatment_changed"] is True
    assert controlled["verdict"] == "second_run_has_higher_peak"

    changed_model_input = make_report(
        make_prompt(steps=12, width=960),
        label="changed-width",
        peak_delta=7 * 1024 * MIB,
    )
    invalid = compare_reports(first, changed_model_input)

    assert invalid["comparable"] is False
    assert invalid["verdict"] == "not_comparable_control_inputs_changed"
    assert invalid["control_differences"][0]["field"] == "conditioning"
