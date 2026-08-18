from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .build_h3_speed_multimodal_validation import build_multimodal_speed_prompts
    from .build_h3_speed_reference_validation import build_reference_speed_prompts
except ImportError:  # Direct script execution keeps the tools directory on sys.path.
    from build_h3_speed_multimodal_validation import build_multimodal_speed_prompts
    from build_h3_speed_reference_validation import build_reference_speed_prompts


def _conditioning_inputs(
    speed_prompt: Mapping[str, Mapping[str, Any]], *, width: int, height: int
) -> dict[str, Any]:
    source_node = speed_prompt.get("6")
    if not isinstance(source_node, Mapping) or source_node.get("class_type") != (
        "MiniMaxH3SPEEDSourceT8Advanced"
    ):
        raise ValueError("Expected node 6 to be the frozen SPEED stage source")
    inputs = copy.deepcopy(dict(source_node.get("inputs", {})))
    inputs.pop("checkpoint_fingerprint", None)
    inputs.pop("vae_fingerprint", None)
    inputs["width"] = int(width)
    inputs["height"] = int(height)
    return inputs


def build_full_resolution_baseline(
    speed_prompt: Mapping[str, Mapping[str, Any]],
    *,
    width: int,
    height: int,
    filename_prefix: str,
) -> dict[str, dict[str, Any]]:
    """Build the exact Stock20 control for one frozen multimodal SPEED prompt.

    Media loaders, conditioning values, model files, seed, shifts, trim policy and
    output encoding are retained. The only treatment removed is staged SPEED:
    conditioning is built once at the final canvas and sampled for all 20 Euler
    calls with the same modality-stable AV noise contract.
    """

    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    prompt = copy.deepcopy(dict(speed_prompt))
    speed_sampler = prompt.get("7")
    if not isinstance(speed_sampler, Mapping) or speed_sampler.get("class_type") != (
        "MiniMaxH3SPEEDSamplerT8Advanced"
    ):
        raise ValueError("Expected node 7 to be the frozen SPEED whole-chain sampler")
    sampler_inputs = dict(speed_sampler.get("inputs", {}))
    seed = int(sampler_inputs["seed"])
    shift_audio = float(sampler_inputs["shift_audio"])

    plan_node = prompt.get("5")
    if not isinstance(plan_node, Mapping) or plan_node.get("class_type") != (
        "MiniMaxH3SPEEDPlanT8Advanced"
    ):
        raise ValueError("Expected node 5 to be the frozen SPEED plan")
    plan_inputs = dict(plan_node.get("inputs", {}))
    steps = int(plan_inputs["steps"])
    shift_video = float(plan_inputs["shift_video"])
    if steps != 20:
        raise ValueError("SPEED quality controls are frozen to Stock20")

    prompt["5"] = {
        "class_type": "MiniMaxH3AudioConditioningT8",
        "inputs": _conditioning_inputs(speed_prompt, width=width, height=height),
        "_meta": {"title": "Full-resolution conditioning control"},
    }
    prompt["6"] = {
        "class_type": "MiniMaxH3DualClockSamplerT8",
        "inputs": {
            "model": ["1", 0],
            "av_latent": ["5", 1],
            "steps": steps,
            "shift_video": shift_video,
            "shift_audio": shift_audio,
            "sampler_name": "dual_clock_euler",
            "scheduler": "native_flow",
        },
        "_meta": {"title": "Full-resolution Stock20 H3 Euler"},
    }
    prompt["7"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": ["6", 0], "conditioning": ["5", 0]},
        "_meta": {"title": "CFG-false BasicGuider control"},
    }
    prompt["19"] = {
        "class_type": "MiniMaxH3SPEEDModalityStableNoiseT8Advanced",
        "inputs": {"seed": seed},
        "_meta": {"title": "Identical modality-stable AV seed"},
    }
    prompt["20"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["19", 0],
            "guider": ["7", 0],
            "sampler": ["6", 1],
            "sigmas": ["6", 2],
            "latent_image": ["5", 1],
        },
        "_meta": {"title": "Full-resolution same-NFE control"},
    }
    prompt["13"]["inputs"]["av_latent"] = ["20", 0]
    prompt["16"]["inputs"]["filename_prefix"] = filename_prefix
    prompt["16"]["_meta"]["title"] = "Save full-resolution quality control"
    prompt["17"] = {
        "class_type": "SaveText",
        "inputs": {
            "text": ["5", 5],
            "filename_prefix": f"{filename_prefix}_conditioning_report",
            "format": "json",
        },
        "_meta": {"title": "Save full-resolution conditioning report"},
    }
    prompt.pop("18", None)
    return prompt


def _controlled_pair_contract(
    baseline: Mapping[str, Mapping[str, Any]],
    speed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    speed_source = dict(speed["6"]["inputs"])
    baseline_conditioning = dict(baseline["5"]["inputs"])
    shared_conditioning = {
        key: value
        for key, value in speed_source.items()
        if key not in {"checkpoint_fingerprint", "vae_fingerprint"}
    }
    baseline_shared = {
        key: value
        for key, value in baseline_conditioning.items()
        if key not in {"width", "height"}
    }
    return {
        "model_equal": baseline["1"]["inputs"] == speed["1"]["inputs"],
        "clip_equal": baseline["2"]["inputs"] == speed["2"]["inputs"],
        "vae_equal": (
            baseline["3"]["inputs"] == speed["3"]["inputs"]
            and baseline["4"]["inputs"] == speed["4"]["inputs"]
        ),
        "source_media_equal": all(
            baseline[node_id]["inputs"] == speed[node_id]["inputs"]
            for node_id in ("8", "9", "10", "11", "12")
        ),
        "conditioning_equal_except_canvas_and_fingerprints": (
            baseline_shared == shared_conditioning
        ),
        "seed_equal": baseline["19"]["inputs"]["seed"]
        == speed["7"]["inputs"]["seed"],
        "steps_equal": baseline["6"]["inputs"]["steps"]
        == speed["5"]["inputs"]["steps"],
        "shifts_equal": (
            baseline["6"]["inputs"]["shift_video"]
            == speed["5"]["inputs"]["shift_video"]
            and baseline["6"]["inputs"]["shift_audio"]
            == speed["7"]["inputs"]["shift_audio"]
        ),
        "trim_equal": baseline["14"]["inputs"] == speed["14"]["inputs"],
    }


def build_quality_pairs(
    *,
    source_video: str,
    reference_image: str,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    steps: int = 20,
    multimodal_seed: int = 2608192001,
    reference_seed: int = 2608193001,
    output_prefix: str = "MiniMaxH3/SPEED_quality_v1",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    multimodal, multimodal_manifest = build_multimodal_speed_prompts(
        source_video=source_video,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=multimodal_seed,
    )
    references, reference_manifest = build_reference_speed_prompts(
        source_video=source_video,
        reference_image=reference_image,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=reference_seed,
    )
    fl2va_speed = multimodal["fl2va_remix_source"]
    ref2va_speed = references["ref_image_native"]
    fl2va_baseline = build_full_resolution_baseline(
        fl2va_speed,
        width=width,
        height=height,
        filename_prefix=f"{output_prefix}/fl2va_baseline_stock20",
    )
    ref2va_baseline = build_full_resolution_baseline(
        ref2va_speed,
        width=width,
        height=height,
        filename_prefix=f"{output_prefix}/ref2va_baseline_stock20",
    )
    prompts = {
        "fl2va_baseline": fl2va_baseline,
        "fl2va_speed": fl2va_speed,
        "ref2va_baseline": ref2va_baseline,
        "ref2va_speed": ref2va_speed,
    }
    contracts = {
        "fl2va": _controlled_pair_contract(fl2va_baseline, fl2va_speed),
        "ref2va": _controlled_pair_contract(ref2va_baseline, ref2va_speed),
    }
    if not all(all(contract.values()) for contract in contracts.values()):
        raise RuntimeError("Generated SPEED quality pair changed a controlled input")
    manifest = {
        "schema": "minimax_h3_speed_quality_pairs_v1",
        "controlled": {
            "source_video": source_video,
            "reference_image": reference_image,
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "sampler": "Euler",
            "scheduler": "native H3 flow",
            "noise_contract": "modality_stable_nested_av_v1",
            "multimodal_seed": multimodal_seed,
            "reference_seed": reference_seed,
        },
        "contracts": contracts,
        "source_manifests": {
            "multimodal_schema": multimodal_manifest["schema"],
            "reference_schema": reference_manifest["schema"],
        },
        "treatment_only": (
            "SPEED uses a 0.5-to-1.0 spatial stage transition with official DCT/kappa/sigma "
            "alignment; baseline performs all 20 calls at the final canvas"
        ),
        "claims": {
            "quality_validated": False,
            "speedup_validated": False,
            "audio_noninferiority_validated": False,
            "reference_noninferiority_validated": False,
            "memory_safe_16gb": False,
        },
    }
    return prompts, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one controlled FL2VA and one controlled Ref2VA full-resolution "
            "Stock20 versus SPEED quality pair."
        )
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source-video", default="face_refine_validation_dance_362_736x416.mp4"
    )
    parser.add_argument("--reference-image", default="t8_dynamic_guidance_1v1_736x416.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--multimodal-seed", type=int, default=2608192001)
    parser.add_argument("--reference-seed", type=int, default=2608193001)
    parser.add_argument("--output-prefix", default="MiniMaxH3/SPEED_quality_v1")
    args = parser.parse_args()
    prompts, manifest = build_quality_pairs(
        source_video=args.source_video,
        reference_image=args.reference_image,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        multimodal_seed=args.multimodal_seed,
        reference_seed=args.reference_seed,
        output_prefix=args.output_prefix,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in prompts.items():
        (args.output_dir / f"{name}_api.json").write_text(
            json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
