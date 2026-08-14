from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools.validate_h3_vram import (
    MIB,
    ValidationError,
    analyze_prompt,
    compare_reports,
    _counter_delta,
    dynamic_vram_evidence,
    load_api_prompt,
    make_activation_chunk_prompts,
    make_ab_prompts,
    make_vram_policy_prompts,
    summarize_samples,
)


def make_prompt(*, steps=8, width=1024, seed=123):
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


def test_analysis_identifies_control_inputs_treatment_and_non_eight_step_warning():
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


def test_analysis_resolves_sampling_literals_projected_by_long_video_orchestrator():
    prompt = make_prompt(steps=12)
    prompt["0"] = {
        "class_type": "MiniMaxH3LongVideoOrchestratorT8",
        "inputs": {
            "steps": 8,
            "shift_video": 12.0,
            "shift_audio": 3.0,
            "sampler_name": "dual_clock_euler",
            "scheduler": "native_flow",
        },
    }
    prompt["4"]["inputs"].update({
        "steps": ["0", 16],
        "shift_video": ["0", 17],
        "shift_audio": ["0", 18],
        "sampler_name": ["0", 19],
        "scheduler": ["0", 20],
    })

    analysis = analyze_prompt(prompt)
    sampler = next(
        item for item in analysis["treatment"]["sampling"]
        if item["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )

    assert sampler == {
        "node_id": "4",
        "class_type": "MiniMaxH3DualClockSamplerT8",
        "steps": 8,
        "video_steps": None,
        "audio_steps": None,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "scheduler": "native_flow",
        "sampler_name": "dual_clock_euler",
    }
    assert "dual_clock_non_turbo_step_count" not in {
        item["code"] for item in analysis["risks"]
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
    stock, dual = make_ab_prompts(prompt, steps=8)

    assert dual["4"]["inputs"]["steps"] == 8
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


def make_hybrid_prompt():
    prompt = make_prompt(steps=20, width=736, seed=2608125201)
    prompt["1"] = {
        "class_type": "MiniMaxH3HybridModelLoaderT8Advanced",
        "inputs": {
            "quality_base": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "mode": "base_only",
            "weight_dtype": "default",
        },
    }
    return prompt


def make_activation_prompt():
    prompt = make_prompt(steps=8, width=736, seed=2608131801)
    prompt["8"] = {
        "class_type": "MiniMaxH3ActivationChunkT8Advanced",
        "inputs": {
            "model": ["2", 0],
            "mode": "report_only",
            "chunk_rows": 512,
            "block_start": 0,
            "block_end": 49,
            "preserve_short_path": True,
            "expected_width": 736,
            "expected_height": 608,
            "expected_length": 362,
            "expected_single_image_references": 0,
        },
    }
    prompt["4"]["inputs"]["model"] = ["8", 0]
    return prompt


def test_make_activation_chunk_prompts_isolates_mode_as_treatment():
    baseline, treatment = make_activation_chunk_prompts(
        make_activation_prompt(), chunk_rows=256, block_start=5, block_end=44
    )

    assert baseline["8"]["inputs"]["mode"] == "report_only"
    assert treatment["8"]["inputs"]["mode"] == "apply_exp"
    assert treatment["8"]["inputs"]["chunk_rows"] == 256
    assert treatment["8"]["inputs"]["block_start"] == 5
    assert treatment["8"]["inputs"]["block_end"] == 44

    baseline_analysis = analyze_prompt(baseline)
    treatment_analysis = analyze_prompt(treatment)
    assert baseline_analysis["controls"] == treatment_analysis["controls"]
    assert baseline_analysis["treatment"] != treatment_analysis["treatment"]
    assert treatment_analysis["treatment"]["activation_chunk"][0]["mode"] == (
        "apply_exp"
    )

    first = make_report(baseline, label="baseline", peak_delta=12 * 1024 * MIB)
    second = make_report(treatment, label="chunked", peak_delta=11 * 1024 * MIB)
    comparison = compare_reports(first, second)
    assert comparison["comparable"] is True
    assert comparison["treatment_changed"] is True
    assert comparison["verdict"] == "second_run_has_lower_peak"


def test_make_activation_chunk_prompts_rejects_ambiguous_or_unused_node():
    prompt = make_activation_prompt()
    prompt["9"] = duplicate = json.loads(json.dumps(prompt["8"]))
    duplicate["inputs"]["model"] = ["2", 0]
    with pytest.raises(ValidationError, match="exactly one"):
        make_activation_chunk_prompts(prompt)

    unused = make_activation_prompt()
    unused["4"]["inputs"]["model"] = ["2", 0]
    with pytest.raises(ValidationError, match="MODEL output is unused"):
        make_activation_chunk_prompts(unused)


def test_make_vram_policy_prompts_preserves_controls_and_wires_loader():
    baseline, policy = make_vram_policy_prompts(
        make_hybrid_prompt(),
        fixed_total_reserved_gib=2.0,
        clean_before_load=False,
    )

    policy_nodes = [
        (node_id, node)
        for node_id, node in policy.items()
        if node["class_type"] == "MiniMaxH3VRAMPolicyT8Advanced"
    ]
    assert len(policy_nodes) == 1
    policy_id, policy_node = policy_nodes[0]
    assert policy_node["inputs"]["mode"] == "fixed_total_reserved_exp"
    assert policy_node["inputs"]["fixed_total_reserved_gib"] == 2.0
    assert policy_node["inputs"]["clean_before_load"] is False
    assert policy["1"]["inputs"]["vram_policy"] == [policy_id, 0]
    assert "vram_policy" not in baseline["1"]["inputs"]

    baseline_analysis = analyze_prompt(baseline)
    policy_analysis = analyze_prompt(policy)
    assert baseline_analysis["controls"] == policy_analysis["controls"]
    assert baseline_analysis["treatment"]["vram_policy"] == []
    assert policy_analysis["treatment"]["vram_policy"][0]["mode"] == (
        "fixed_total_reserved_exp"
    )
    assert baseline_analysis["treatment"] != policy_analysis["treatment"]


def test_vram_policy_pair_is_comparable_and_rejects_ambiguous_sources():
    baseline, policy = make_vram_policy_prompts(make_hybrid_prompt())
    first = make_report(baseline, label="baseline", peak_delta=15 * 1024 * MIB)
    second = make_report(policy, label="policy", peak_delta=14 * 1024 * MIB)
    comparison = compare_reports(first, second)
    assert comparison["comparable"] is True
    assert comparison["treatment_changed"] is True
    assert comparison["verdict"] == "second_run_has_lower_peak"

    existing = make_hybrid_prompt()
    existing["1"]["inputs"]["vram_policy"] = ["99", 0]
    with pytest.raises(ValidationError, match="already has a vram_policy"):
        make_vram_policy_prompts(existing)

    ambiguous = make_hybrid_prompt()
    ambiguous["8"] = {
        "class_type": "MiniMaxH3HybridModelLoaderT8Advanced",
        "inputs": {
            "quality_base": "other.safetensors",
            "mode": "base_only",
            "weight_dtype": "default",
        },
    }
    with pytest.raises(ValidationError, match="requires exactly one"):
        make_vram_policy_prompts(ambiguous)

    with pytest.raises(ValidationError, match="requires clean_before_load=true"):
        make_vram_policy_prompts(
            make_hybrid_prompt(),
            mode="external_usage_plus_margin_exp",
            clean_before_load=False,
        )


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


def test_sample_summary_classifies_high_run_io_as_thrashing():
    samples = [
        {
            "phase": "baseline",
            "elapsed_seconds": 0.0,
            "vram_used_bytes": 100 * MIB,
            "torch_pool_used_bytes": 20 * MIB,
            "ram_free_bytes": 32 * 1024**3,
            "process_read_bytes": 1024,
            "process_page_faults": 10,
            "process_private_bytes": 4 * 1024**3,
            "gpu_temperature_c": 50,
            "gpu_power_mw": 100000,
            "gpu_sm_clock_mhz": 1800,
        },
        {
            "phase": "running",
            "elapsed_seconds": 10.0,
            "vram_used_bytes": 900 * MIB,
            "torch_pool_used_bytes": 700 * MIB,
            "ram_free_bytes": 24 * 1024**3,
            "process_read_bytes": 65 * 1024**3 + 1024,
            "process_page_faults": 2010,
            "process_private_bytes": 8 * 1024**3,
            "gpu_temperature_c": 70,
            "gpu_power_mw": 200000,
            "gpu_sm_clock_mhz": 1500,
        },
    ]
    summary = summarize_samples(samples)
    assert summary["resource_behavior"] == "fits_with_thrashing"
    assert summary["process_read_delta_bytes"] == 65 * 1024**3
    assert summary["process_page_fault_delta"] == 2000
    assert summary["process_peak_private_bytes"] == 8 * 1024**3
    assert summary["maximum_gpu_temperature_c"] == 70
    assert summary["maximum_gpu_power_w"] == 200.0
    assert _counter_delta(samples, "missing") is None


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
