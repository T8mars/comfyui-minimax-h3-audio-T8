#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
COMFY_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "skin-finish-oily-lora8-source-20260825"
    / "20260825-094343-e7088277"
    / "output"
    / "MiniMaxH3_SkinFinish"
    / "oily_lora8_speaking_00001_.mp4"
)
EXPECTED_SOURCE_SHA256 = (
    "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "skin-finish-surface-localized-stream-20260825-v3"
DICHROMATIC_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-dichromatic-stream-20260825-v1"
)
SURFACE_PARAMETERS = {
    "amount": 0.90,
    "surface_smoothing": 0.25,
    "texture_keep": 0.96,
    "highlight_compression": 0.90,
    "broad_highlight_compression": 0.90,
    "broad_highlight_start": 0.68,
    "broad_highlight_end": 0.94,
    "blemish_balance": 0.10,
    "surface_radius_percent": 2.5,
}
DICHROMATIC_PARAMETERS = {
    "amount": 0.90,
    "specular_strength": 0.85,
    "diffuse_radius_percent": 2.5,
    "specular_threshold_linear": 0.003,
    "specular_softness_linear": 0.025,
    "chroma_dilution_threshold": 0.001,
    "chroma_dilution_softness": 0.015,
    "minimum_diffuse_chroma": 0.006,
    "diffuse_chroma_softness": 0.035,
    "minimum_direction_cosine": 0.70,
    "maximum_surface_delta": 0.08,
    "minimum_texture_ratio": 0.82,
    "maximum_texture_ratio": 1.10,
}

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_skin_finish_human_review as human_review  # noqa: E402
import validate_skin_finish_quality_stream_representative as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one bounded localized Guided Surface stream validation on the pinned "
            "MiniMax H3 v1.0 eight-step oily speaking source."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("surface", "dichromatic"),
        default="surface",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    is_dichromatic = args.mode == "dichromatic"
    stage_name = "dichromatic" if is_dichromatic else "surface"
    stage_parameters = (
        DICHROMATIC_PARAMETERS if is_dichromatic else SURFACE_PARAMETERS
    )
    default_output = DICHROMATIC_OUTPUT if is_dichromatic else DEFAULT_OUTPUT
    requested_output = args.output if args.output is not None else default_output
    stage_rejected_key = f"{stage_name}_rejected_frame_count"
    plan = {
        "source": str(args.source.resolve()),
        "output": str(requested_output.resolve()),
        "contract": "960x544x124@24fps",
        "mode": stage_name,
        "stage_parameters": stage_parameters,
        "chunk_frames": 2,
        "loads_h3": False,
        "loads_sam": False,
        "stress_or_repeat": False,
        "torch_cpu_threads": 2,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0

    source = args.source.resolve()
    output_root = requested_output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite validation evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or common._sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the pinned v1.0 eight-step oily speaking source is missing or changed")
    if not common.FFMPEG.is_file() or not common.FFPROBE.is_file():
        raise FileNotFoundError("bundled FFmpeg and FFprobe are required")
    os.environ["PATH"] = str(common.FFMPEG.parent) + os.pathsep + os.environ.get(
        "PATH", ""
    )

    common._strict_decode(source)
    source_probe = common._probe(source)
    video_stream = next(
        item for item in source_probe["streams"] if item["codec_type"] == "video"
    )
    source_contract = (
        int(video_stream["width"]),
        int(video_stream["height"]),
        int(video_stream["nb_frames"]),
        str(video_stream["r_frame_rate"]),
    )
    if source_contract != (960, 544, 124, "24/1"):
        raise RuntimeError("surface representative must be 960x544x124 at 24fps")

    common._load_package()
    import folder_paths
    import torch
    from comfy_api.latest import InputImpl
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish import (
        _prepare_mask,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p1 import (
        stream_skin_finish_video,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_p2 import (
        guard_skin_finish_candidate,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_safety_audit import (
        audit_skin_finish_candidate,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_stream_quality import (
        _QualityChunkProcessor,
        _quality_stream_ram_preflight,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_surface import (
        finish_skin_surface,
    )
    from h3_audio_t8_skin_finish_quality_stream_validation.skin_finish_dichromatic import (
        attenuate_skin_specular_dichromatic,
    )

    stage_function = (
        attenuate_skin_specular_dichromatic
        if is_dichromatic
        else finish_skin_surface
    )

    class _SurfaceChunkProcessor(_QualityChunkProcessor):
        def __init__(self) -> None:
            super().__init__(
                preset="subtle",
                amount=0.0,
                texture_keep=1.0,
                shine_control=0.0,
                crop_expansion=1.45,
                minimum_class_probability=0.55,
                feature_protection_px=4,
                mask_feather_px=0,
                proxy_long_side=640,
                low_frequency_strength=0.0,
                source_detail_gain=1.0,
                separation_radius_percent=1.0,
                maximum_radius_px=32,
                shadow_protection=0.10,
                highlight_protection=0.94,
                minimum_texture_ratio=0.78,
                maximum_temporal_effect_jump=0.04,
            )
            self.stats[stage_rejected_key] = 0

        def __call__(
            self,
            source_chunk: torch.Tensor,
            face_records: list[list[dict]],
            absolute_start_frame: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del absolute_start_frame
            frame_count, height, width, _ = map(int, source_chunk.shape)
            if len(face_records) != frame_count:
                raise RuntimeError("surface stream face metadata does not match frame chunk")
            self.stats["chunk_count"] += 1
            self.stats["peak_chunk_frames"] = max(
                int(self.stats["peak_chunk_frames"]), frame_count
            )
            self.stats["source_frame_count"] += frame_count
            raw_mask = self._semantic_mask(source_chunk, face_records)
            used_mask, _, mask_report = _prepare_mask(
                raw_mask,
                frame_count=frame_count,
                height=height,
                width=width,
                minimum_area=0.00005,
                maximum_area=0.35,
                feather_px=0,
                temporal_radius=0,
                chunk_frames=frame_count,
            )
            if int(mask_report["accepted_frame_count"]) == 0:
                output = source_chunk
                effective_mask = torch.zeros_like(used_mask)
            else:
                (
                    surface_candidate,
                    surface_source,
                    surface_selected,
                    _,
                    surface_mask,
                    _,
                    _,
                    surface_json,
                ) = stage_function(
                    source_chunk,
                    used_mask,
                    **stage_parameters,
                    minimum_mask_area=0.00005,
                    maximum_mask_area=0.35,
                    chunk_frames=frame_count,
                    accept_candidate=False,
                    audio=None,
                )
                surface_report = json.loads(surface_json)
                self.stats[stage_rejected_key] += int(
                    surface_report["rejected_frame_count"]
                )
                if surface_source is not source_chunk or surface_selected is not source_chunk:
                    raise RuntimeError("surface stage changed source selection")
                (
                    guarded_candidate,
                    guard_source,
                    guard_selected,
                    _,
                    guard_mask,
                    _,
                    _,
                    guard_json,
                ) = guard_skin_finish_candidate(
                    source_chunk,
                    surface_candidate,
                    surface_mask,
                    shadow_protection=0.10,
                    highlight_protection=0.94,
                    transition_width=0.06,
                    minimum_texture_ratio=0.78,
                    minimum_reference_texture=0.003,
                    maximum_new_clipped_fraction=0.0005,
                    texture_radius=1,
                    chunk_frames=frame_count,
                    accept_candidate=False,
                    audio=None,
                )
                guard_report = json.loads(guard_json)
                self.stats["texture_guard_rejected_frame_count"] += int(
                    guard_report["rejected_frame_count"]
                )
                if guard_source is not source_chunk or guard_selected is not source_chunk:
                    raise RuntimeError("surface guard changed source selection")
                if self.previous_source is None:
                    audit_source = source_chunk
                    audit_candidate = guarded_candidate
                    audit_mask = guard_mask
                    leading = 0
                else:
                    audit_source = torch.cat((self.previous_source, source_chunk), dim=0)
                    audit_candidate = torch.cat(
                        (self.previous_candidate, guarded_candidate), dim=0
                    )
                    audit_mask = torch.cat((self.previous_mask, guard_mask), dim=0)
                    leading = 1
                if bool((audit_mask > 0).any()):
                    (
                        _,
                        gated_candidate,
                        _,
                        _,
                        hard_gate_pass,
                        _,
                        _,
                        audit_json,
                    ) = audit_skin_finish_candidate(
                        audit_source,
                        audit_candidate,
                        audit_mask,
                        audit_scope="mask_only",
                        temporal_policy="hard_gate",
                        maximum_mean_abs_change=0.08,
                        maximum_peak_abs_change=0.30,
                        maximum_temporal_effect_jump=0.04,
                        minimum_temporal_pixels=64,
                        scene_cut_reset_threshold=0.20,
                        accept_candidate=False,
                        audio_source=None,
                        audio_passthrough=None,
                    )
                    audit_report = json.loads(audit_json)
                    observed_jump = float(
                        audit_report["summary"][
                            "maximum_observed_temporal_effect_jump"
                        ]
                    )
                    self.stats["maximum_temporal_effect_jump"] = max(
                        float(self.stats["maximum_temporal_effect_jump"]), observed_jump
                    )
                    current_failed = sum(
                        int(index) >= leading
                        for index in audit_report["summary"]["failed_frame_indices"]
                    )
                    self.stats["safety_audit_failed_frame_count"] += current_failed
                    if not hard_gate_pass:
                        self.stats["safety_audit_failed_chunk_count"] += 1
                        output = source_chunk
                        effective_mask = torch.zeros_like(guard_mask)
                    else:
                        output = gated_candidate[leading:]
                        effective_mask = guard_mask
                else:
                    output = source_chunk
                    effective_mask = torch.zeros_like(guard_mask)

            outside = effective_mask <= 0
            if not torch.equal(
                output[..., :3][outside], source_chunk[..., :3][outside]
            ):
                raise RuntimeError("surface stream changed pixels outside its mask")
            self.previous_source = source_chunk[-1:].detach().clone()
            self.previous_candidate = output[-1:].detach().clone()
            self.previous_mask = effective_mask[-1:].detach().clone()
            gc.collect()
            return output, effective_mask

        def report(self) -> dict:
            report = super().report()
            report["method"] = f"parsenet_{stage_name}_guard_safety"
            report["stage_parameters"] = dict(stage_parameters)
            return report

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    folder_paths.models_dir = str(COMFY_ROOT / "models")
    folder_paths.set_output_directory(str(output_root))
    preflight = _quality_stream_ram_preflight()
    if not bool(preflight["allowed"]):
        raise RuntimeError("surface stream host-RAM preflight did not pass")
    processor = _SurfaceChunkProcessor()
    source_video = InputImpl.VideoFromFile(str(source))
    before = common._memory_mib()
    started = time.perf_counter()
    result = None
    try:
        result = stream_skin_finish_video(
            source_video,
            preset="subtle",
            amount=0.0,
            texture_keep=1.0,
            shine_control=0.0,
            detection_threshold=0.35,
            minimum_face_height_px=32.0,
            minimum_detail=0.010,
            bbox_ema_alpha=0.55,
            scene_cut_threshold=0.28,
            maximum_faces=1,
            mask_feather_px=0,
            proxy_long_side=640,
            chunk_frames=2,
            filename_prefix=(
                "node-output/oily_lora8_dichromatic_candidate"
                if is_dichromatic
                else "node-output/oily_lora8_guided_surface_candidate"
            ),
            crf=16.0,
            accept_candidate=True,
            _chunk_processor=processor,
            _stream_report_label=f"parsenet_{stage_name}_guard_safety",
        )
    finally:
        processor.close()
    assert result is not None
    output_video, saved_path, report_json, saved = result
    elapsed = time.perf_counter() - started
    after = common._memory_mib()
    candidate = Path(saved_path).resolve()
    node_report = json.loads(report_json)
    quality = processor.report()
    node_report["quality_pipeline"] = quality
    node_report["resource_preflight"] = preflight
    expected_chunks = math.ceil(124 / 2)
    if saved is None or not candidate.is_file() or output_video.get_frame_count() != 124:
        raise RuntimeError("surface stream did not publish an exact 124-frame VIDEO")
    common._strict_decode(candidate)
    candidate_probe = common._probe(candidate)
    candidate_frames = int(
        next(
            item for item in candidate_probe["streams"] if item["codec_type"] == "video"
        )["nb_frames"]
    )
    source_pcm = common._pcm_sha256(source)
    candidate_pcm = common._pcm_sha256(candidate)
    summary = quality["summary"]
    parser_report = quality["parser"]
    execution = node_report["execution"]
    semantic_ready = int(summary["semantic_ready_frame_count"])
    gate_checks = {
        "two_pass_stream": execution["passes"] == 2,
        "bounded_two_frame_chunks": execution["peak_chunk_frames"] <= 2,
        "expected_chunk_calls": execution["chunk_processor_calls"] == expected_chunks,
        "no_full_image_batch": execution["full_image_batch_materialized"] is False,
        "source_stable_between_passes": (
            execution["source_proxy_equal_between_passes"] is True
        ),
        "outside_mask_exact_before_encode": (
            execution["outside_mask_bit_exact_before_encode"] is True
        ),
        "audio_packet_payload_exact": node_report["audio"]["packet_payload_exact"]
        is True,
        "parser_loaded": parser_report["loaded"] is True,
        "parser_released": parser_report["released_after_execute"] is True,
        "no_persistent_parser_cache": parser_report["persistent_cache"] is False,
        "source_frame_count": summary["source_frame_count"] == 124,
        "chunk_count": summary["chunk_count"] == expected_chunks,
        "semantic_candidate_exists": semantic_ready > 0,
        "stage_candidate_exists": int(summary[stage_rejected_key])
        < semantic_ready,
        "texture_guard_candidate_exists": int(
            summary["texture_guard_rejected_frame_count"]
        )
        < semantic_ready,
        "safety_audit_zero_failures": summary["safety_audit_failed_frame_count"] == 0,
        "decoded_pcm_exact": source_pcm == candidate_pcm,
        "candidate_frame_count": candidate_frames == 124,
    }
    mechanical_pass = all(gate_checks.values())
    diagnostic = {
        "schema": f"h3_t8_skin_finish_{stage_name}_stream_diagnostics/v1",
        "source": str(source),
        "candidate": str(candidate),
        "stage_parameters": stage_parameters,
        "gate_checks": gate_checks,
        "mechanical_pass": mechanical_pass,
        "quality_pipeline": quality,
        "execution": execution,
        "audio": node_report["audio"],
        "source_pcm_sha256": source_pcm,
        "candidate_pcm_sha256": candidate_pcm,
    }
    diagnostic_path = output_root / "mechanical_diagnostics.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    if not mechanical_pass:
        failed = [name for name, passed in gate_checks.items() if not passed]
        raise RuntimeError(
            f"{stage_name} stream failed its mechanical gate: " + ", ".join(failed)
        )

    review = human_review.build_review(source, candidate, output_root / "blind-review")
    report = {
        "schema": f"h3_t8_skin_finish_{stage_name}_stream_validation/v1",
        "status": (
            "PASS_MECHANICAL_WITH_SOURCE_FALLBACKS_HUMAN_REVIEW_PENDING"
            if int(summary["source_only_frame_count"])
            or int(summary[stage_rejected_key])
            or int(summary["texture_guard_rejected_frame_count"])
            else "PASS_MECHANICAL_HUMAN_REVIEW_PENDING"
        ),
        "source": {
            "path": str(source),
            "sha256": common._sha256(source),
            "probe": source_probe,
            "generation_contract": {
                "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                "lora_strength": 1.0,
                "sampling_steps": 8,
                "dialogue": "你在干嘛呢，我在这里呀，看看效果如何。",
            },
        },
        "stage_contract": {
            "mode": stage_name,
            **stage_parameters,
            "chunk_frames": 2,
            "automatic_accept": False,
            "comparison_requires_human_review": True,
        },
        "node_report": node_report,
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "memory_mib_before": before,
            "memory_mib_after": after,
            "torch_cpu_threads": 2,
            "h3_model_loaded": False,
            "sam_model_loaded": False,
            "cuda_processing_requested": False,
            "stress_or_repeated_run": False,
        },
        "media_gates": {
            "source_strict_decode": True,
            "candidate_strict_decode": True,
            "source_frame_count": 124,
            "candidate_frame_count": candidate_frames,
            "decoded_pcm_exact": source_pcm == candidate_pcm,
            "decoded_pcm_sha256": source_pcm,
            "packet_payload_exact": node_report["audio"]["packet_payload_exact"],
        },
        "outputs": {
            "candidate": str(candidate),
            "candidate_sha256": common._sha256(candidate),
            "blind_review": review,
            "mechanical_diagnostics": str(diagnostic_path),
        },
        "mechanical_pass": True,
        "human_review_required": True,
        "claim_boundary": (
            "One pinned speaking source validates bounded mechanics and media preservation only. "
            "The selected post-process is not calibrated physical inverse rendering, and human "
            "review must decide oil control, pumping, halos and naturalness."
        ),
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "candidate": str(candidate),
                "review": review["review"],
                "review_id": review["review_id"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
