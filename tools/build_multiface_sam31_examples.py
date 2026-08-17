#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _node(class_type: str, inputs: dict, title: str) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def build_prompt(character_count: int) -> dict:
    if character_count not in (2, 3):
        raise ValueError("Only the reviewed two- and three-person examples are supported")

    prompt: dict[str, dict] = {}
    next_id = 1

    def add(class_type: str, inputs: dict, title: str) -> str:
        nonlocal next_id
        node_id = str(next_id)
        next_id += 1
        prompt[node_id] = _node(class_type, inputs, title)
        return node_id

    source = add(
        "LoadVideo",
        {"file": f"replace_with_{character_count}_person_exact_24fps_source.mp4"},
        "Replace with an exact-24fps source; keep the original aspect ratio",
    )
    components = add(
        "GetVideoComponents",
        {"video": [source, 0]},
        "Source frames and untouched final soundtrack",
    )

    names = ["Character_A", "Character_B", "Character_C"][:character_count]
    references = [
        add(
            "LoadImage",
            {"image": f"replace_with_authorized_{name}_reference.png"},
            f"Authorized single-person reference for {name}",
        )
        for name in names
    ]

    sam_loader = add(
        "CheckpointLoaderSimple",
        {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"},
        "Official ComfyUI native SAM3.1 multiplex checkpoint",
    )
    sam_text = add(
        "CLIPTextEncode",
        {"text": "front-facing person with a visible face", "clip": [sam_loader, 1]},
        "SAM3.1 prompt biased toward people whose faces can actually be repaired",
    )
    track = add(
        "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
        {
            "frames": [components, 0],
            "model": [sam_loader, 0],
            "conditioning": [sam_text, 0],
            "fps": 24.0,
            "maximum_people": character_count,
            "detection_threshold": 0.53,
            "detect_interval": 3,
            "scene_cut_threshold": 0.28,
            "analysis_max_side": 512,
            "preview_stride": 8,
            "release_policy": "offload_sam31_after_track",
        },
        "Track people per shot, color preview them, then selectively unload SAM3.1",
    )

    profiles = [
        add(
            "MiniMaxH3FaceCharacterProfileT8Advanced",
            {
                "character_id": name,
                "reference_images": [reference, 0],
                "rights_confirmed": False,
                "reference_face_policy": "require_single_face",
            },
            f"Enable rights_confirmed only after verifying the {name} reference",
        )
        for name, reference in zip(names, references)
    ]
    cast = None
    for index, profile in enumerate(profiles):
        inputs = {"profile": [profile, 0]}
        if cast is not None:
            inputs["previous_cast"] = [cast, 0]
        cast = add(
            "MiniMaxH3FaceCastMergeT8Advanced",
            inputs,
            f"Merge authorized character {index + 1} of {character_count}",
        )

    manual = {f"0:{index}": name for index, name in enumerate(names)}
    assignment = add(
        "MiniMaxH3FaceTrackAssignT8Advanced",
        {
            "frames": [components, 0],
            "track_plan": [track, 0],
            "face_cast": [cast, 0],
            "identity_mode": "sface_cpu_suggest",
            "manual_assignments_json": json.dumps(manual, ensure_ascii=False),
            "minimum_similarity": 0.40,
            "minimum_margin": 0.05,
            "identity_samples_per_track": 3,
            "strict_identity": True,
            "preview_stride": 8,
        },
        "Review colored shot-local IDs; edit the JSON for every shot before repair",
    )

    jobs = [
        add(
            "MiniMaxH3MultiFaceRepairJobT8Advanced",
            {
                "frames": [components, 0],
                "identity_assignment": [assignment, 0],
                "character_id": name,
                "shot_id": 0,
                "window_start_in_shot": 0,
                "window_frame_count": 73,
                "crop_factor": 2.5,
                "canvas_mode": "manual_512",
                "center_smooth_window": 21,
                "size_smooth_window": 51,
                "identity_guard": "sface_cpu",
                "minimum_similarity": 0.36,
                "analysis_chunk_frames": 4,
                "crop_scale_mode": "target_face_px",
                "target_face_px": 300.0,
            },
            f"One 73-frame / 3.04-second MANUAL512 repair job for {name}; duplicate per shot/window",
        )
        for name in names
    ]

    video_vae = add(
        "VAELoader",
        {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        "Official MiniMax H3 video VAE",
    )
    audio_vae = add(
        "VAELoader",
        {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        "Official MiniMax H3 audio VAE",
    )
    clip = add(
        "CLIPLoader",
        {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax",
            "device": "default",
        },
        "MiniMax H3 Qwen3-VL text encoder",
    )
    unet = add(
        "UNETLoader",
        {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        },
        "Reviewed Ref2VA pruned model",
    )
    lora = add(
        "LoraLoaderModelOnly",
        {
            "lora_name": "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
            "strength_model": 0.75,
            "model": [unet, 0],
        },
        "Reviewed FL2V Turbo 4-step LoRA at 0.75",
    )
    shifted_model = add(
        "MiniMaxH3SigmaShift",
        {"shift_video": 12.0, "shift_audio": 3.0, "model": [lora, 0]},
        "Native MiniMax H3 video/audio sigma shift",
    )

    direction = (
        "Preserve the assigned character identity, expression, gaze, head pose, timing, "
        "lighting and camera motion. Restore only plausible facial structure and fine detail. "
        "Do not beautify, change makeup, age, hairstyle, mouth timing, background, music or "
        "sound effects. No additional speech."
    )
    previous_composite = None
    for index, (name, job) in enumerate(zip(names, jobs)):
        conditioning = add(
            "MiniMaxH3AudioConditioningT8",
            {
                "prompt": direction,
                "width": 512,
                "height": 512,
                "length": 73,
                "task_type": "Ref2VA",
                "audio_mode": "lock_source",
                "audio_denoise_strength": 0.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "clip": [clip, 0],
                "video_vae": [video_vae, 0],
                "audio_vae": [audio_vae, 0],
                "drive_audio": [components, 1],
                "ref_images.ref_image_0": [job, 3],
                "ref_images.ref_image_1": [job, 4],
            },
            f"Ref2VA conditioning for {name}; lock the original soundtrack",
        )
        latent = add(
            "MiniMaxH3FaceRefineParityLatentT8Advanced",
            {
                "audio_policy": "require_locked",
                "allow_multi_shot_exp": False,
                "positive": [conditioning, 0],
                "av_latent": [conditioning, 1],
                "crops": [job, 2],
                "video_vae": [video_vae, 0],
                "face_plan": [job, 0],
            },
            f"Inject only the {name} crop video latent; keep audio locked",
        )
        denoise = add(
            "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced",
            {
                "strength_small_face": 0.8,
                "strength_large_face": 0.35,
                "scale_mode": "relative_to_clip",
                "face_px_small": 30.0,
                "face_px_large": 120.0,
                "gamma": 1.0,
                "smooth_frames": 9,
                "video_mask_mode": "replace_video_parity",
                "require_locked_audio": True,
                "av_latent": [latent, 1],
                "face_plan": [job, 0],
            },
            f"Human-reviewed relative-to-clip denoise for {name}",
        )
        noise = add("RandomNoise", {"noise_seed": 42 + index}, f"Fixed seed for {name}")
        guider = add(
            "BasicGuider",
            {"model": [shifted_model, 0], "conditioning": [latent, 0]},
            f"Basic guider for {name}",
        )
        sampler = add("KSamplerSelect", {"sampler_name": "er_sde"}, "Reviewed er_sde sampler")
        sigmas = add(
            "BasicScheduler",
            {
                "scheduler": "simple",
                "steps": 8,
                "denoise": 0.45,
                "model": [shifted_model, 0],
            },
            "Reviewed simple 8-step base denoise 0.45",
        )
        sampled = add(
            "SamplerCustomAdvanced",
            {
                "noise": [noise, 0],
                "guider": [guider, 0],
                "sampler": [sampler, 0],
                "sigmas": [sigmas, 0],
                "latent_image": [denoise, 0],
            },
            f"Generate {name} sequentially, never in parallel with another character",
        )
        decoded = add(
            "MiniMaxH3AVDecodeT8",
            {"av_latent": [sampled, 0], "video_vae": [video_vae, 0], "audio_vae": [audio_vae, 0]},
            f"Decode the {name} crop video; generated audio is discarded",
        )
        stitched = add(
            "MiniMaxH3FaceRefineParityStitchT8Advanced",
            {
                "paste_region": "face_only",
                "mask_dilation": 24,
                "feather_source_px": 24.0,
                "colour_match": 1.0,
                "blend": 1.0,
                "undetected_frames": "fade_out",
                "max_face_mean_abs_delta": 1.0,
                "processing_device": "cpu_memory_safe",
                "base_frames": [job, 1],
                "refined_crops": [decoded, 0],
                "face_plan": [job, 0],
            },
            f"Face-only audited stitch for {name}; preview before accepting",
        )
        add(
            "PreviewImage",
            {"images": [stitched, 0]},
            f"Review the complete {name} candidate window before accepting",
        )
        composite_inputs = {
            "base_frames": [components, 0],
            "candidate_window": [stitched, 0],
            "changed_mask": [stitched, 1],
            "face_plan": [job, 0],
            "accept_candidate": False,
            "overlap_policy": "reject",
        }
        if previous_composite is not None:
            composite_inputs["previous_composite"] = [previous_composite, 1]
        previous_composite = add(
            "MiniMaxH3MultiFaceCompositeT8Advanced",
            composite_inputs,
            f"After review, toggle accept for {name}; overlaps fail closed",
        )

    final_video = add(
        "CreateVideo",
        {
            "images": [previous_composite, 0],
            "audio": [components, 1],
            "fps": 24.0,
            "bit_depth": 8,
        },
        "Mux the untouched original soundtrack after all accepted face jobs",
    )
    add(
        "SaveVideo",
        {
            "video": [final_video, 0],
            "filename_prefix": f"MiniMaxH3/multiface_sam31_{character_count}person_reviewed",
            "format": "mp4",
            "codec": "h264",
        },
        "Save to a new file; never overwrite the source",
    )
    return prompt


def main() -> int:
    for count in (2, 3):
        output = ROOT / "examples" / f"multiface_sam31_{count}person_advanced_api.json"
        output.write_text(
            json.dumps(build_prompt(count), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
