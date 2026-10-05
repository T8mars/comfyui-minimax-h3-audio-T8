"""Build ordinary / separated Full / HIGH-only Cold reference candidates, no queue.

Exact installed files and SHA pins only. One small 5-second recipe, not a GPU
matrix or quality acceptance. EAV/Relay are explicit separate nodes per stage.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import urllib.request

try:
    from .api_to_frontend_workflow import convert
    from .build_reference_asset_workflow import choices
    from .build_reference_readback_workflow import make_workflow as readback
    from .check_windows_paths import validate_paths
except ImportError:
    from api_to_frontend_workflow import convert
    from build_reference_asset_workflow import choices
    from build_reference_readback_workflow import make_workflow as readback
    from check_windows_paths import validate_paths


APPLY = "MiniMaxH3ReferenceConditioningEXPT8"
RELAY = "MiniMaxH3ReferenceRelayConditioningEXPT8"
MODEL = "minimax_h3_ref2va_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
GLOBAL = ("Quiet daylight room, a continuous medium shot. <Picture 1> is the on-screen woman B. "
          "A speaks off-screen with the reference voice in <Audio 1>; A never appears. "
          "Natural human speech, quiet room tone, no music or subtitles.")
LOCAL = ('Off-screen A asks in Mandarin, "今天去公园吗？" B listens quietly.\n'
         'B nods and answers in Mandarin, "好啊，我们走吧。" A is quiet.')


def make_graph(info, *, voice_name, image_name, voice_sha, image_sha,
               variant="split", low_path="REPLACE_WITH_SAVED_LOW_PATH/manifest.json", low_sha="0" * 64):
    if variant not in {"ordinary", "split", "cold"}:
        raise ValueError("Choose ordinary, split or explicit HIGH-only cold")
    _workflow, graph = readback(info, voice_name=voice_name, image_name=image_name,
                               voice_sha=voice_sha, image_sha=image_sha)
    graph = {key: item for key, item in graph.items() if key in {"1", "2", "5"}}
    graph["5"]["inputs"].update({"packages.package_0": ["1", 0], "packages.package_1": ["2", 0]})
    for node, field, value in (("UNETLoader", "unet_name", MODEL), ("CLIPLoader", "clip_name", CLIP),
            ("VAELoader", "vae_name", VIDEO_VAE), ("VAELoader", "vae_name", AUDIO_VAE),
            ("MiniMaxH3LoRACompatibilityLoaderT8Advanced", "lora_name", LORA)):
        if value not in choices(info[node]["input"]["required"][field]):
            raise ValueError(f"Explicit installed selection missing: {node}.{field}={value}")
    if variant != "ordinary" and UPSCALER not in choices(
            info["MiniMaxH3LearnedLatentUpscaleT8Advanced"]["input"]["required"]["model_name"]):
        raise ValueError("Explicit installed learned upscaler missing")

    def node(key, class_type, **values):
        if class_type not in info:
            raise ValueError(f"Actual service has not loaded {class_type}; do not invent its schema")
        graph[key] = {"class_type": class_type, "inputs": values}

    node("6", "CLIPLoader", clip_name=CLIP, type="minimax", device="default")
    node("7", "VAELoader", vae_name=VIDEO_VAE)
    node("8", "VAELoader", vae_name=AUDIO_VAE)
    common = dict(reference_set=["5", 0], clip=["6", 0], video_vae=["7", 0], audio_vae=["8", 0],
        width=448, height=256, task_type="auto", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=True, prompt_primary_audio_ordinal=0, strict_prompt_tags=True)

    def model_lane(prefix, *, after=None):
        if after is None:
            node(prefix+"_model", "UNETLoader", unet_name=MODEL, weight_dtype="default")
        else:
            node(prefix+"_model", "MiniMaxH3StageUNETLoaderAfterEXPT8", completed_stage=after,
                 unet_name=MODEL, weight_dtype="default")
        node(prefix+"_lora", "MiniMaxH3LoRACompatibilityLoaderT8Advanced", model=[prefix+"_model", 0],
             lora_name=LORA, strength_model=1.)
        node(prefix+"_head", "MiniMaxH3LowVRAMAttentionT8Advanced", model=[prefix+"_lora", 0], head_chunks=4)
        node(prefix+"_ffn", "MiniMaxH3ChunkFeedForwardT8Advanced", model=[prefix+"_head", 0], chunks=2, seq_threshold=4096)
        return [prefix+"_ffn", 0]

    def relay_lane(prefix, model, *, width=448, height=256):
        node(prefix+"_plan", "MiniMaxH3PromptRelayPlanT8Advanced", global_prompt=GLOBAL, local_prompts=LOCAL,
             length=124, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1", epsilon=.1,
             allow_gaps=False, allow_overlaps=False)
        node(prefix+"_apply", RELAY, **{**common, "model": model, "width": width, "height": height,
             "prompt_relay_plan": [prefix+"_plan", 0], "ref_image_size": "match",
             "reference_video_policy": "official_2_to_15s", "execution_mode": "apply_exp", "query_chunk_rows": 256})

    def stage(prefix, *, stage_name, latent, positive):
        node(prefix+"_setup", "MiniMaxH3NativeDualStageSetupEXPT8", model=[prefix+"_apply", 0],
             av_latent=latent, stage=stage_name, shift_video=12., shift_audio=3.)
        node(prefix+"_eav_config", "MiniMaxH3StageEAVConfigEXPT8", mode="report_only", tau=4.,
             start_video_progress=.15, end_video_progress=.9, max_workspace_mib=32, g_hard_limit=1.5)
        node(prefix+"_eav", "MiniMaxH3StageEAVApplyEXPT8", model=[prefix+"_setup", 0],
             sigmas=[prefix+"_setup", 2], av_latent=latent, stage_context=[prefix+"_setup", 3],
             eav_config=[prefix+"_eav_config", 0])
        node(prefix+"_noise", "RandomNoise", noise_seed=2610050101 if prefix == "low" else 2610050102)
        node(prefix+"_guider", "BasicGuider", model=[prefix+"_eav", 0], conditioning=positive)
        node(prefix+"_sample", "MiniMaxH3StageSamplerEXPT8", noise=[prefix+"_noise", 0],
             guider=[prefix+"_guider", 0], sampler=[prefix+"_setup", 1], sigmas=[prefix+"_setup", 2],
             latent_image=latent, stage_context=[prefix+"_setup", 3])
        node(prefix+"_save", "MiniMaxH3StageSaveEXPT8", stage_result=[prefix+"_sample", 2], prefix="R6_Ref_"+prefix)
        node(prefix+"_audit", "MiniMaxH3StageEAVAuditEXPT8",
             av_latent=[prefix+"_save", 1 if prefix == "low" else 0], runtime=[prefix+"_eav", 1])
        return [prefix+"_audit", 0]

    if variant == "ordinary":
        model = model_lane("single")
        node("single_apply", APPLY, **common, prompt=GLOBAL+"\n"+LOCAL, length=124)
        node("single_setup", "MiniMaxH3DualClockSamplerT8", model=model, av_latent=["single_apply", 1],
             steps=8, shift_video=12., shift_audio=3., sampler_name="dual_clock_euler", scheduler="native_flow")
        node("single_noise", "RandomNoise", noise_seed=2610050101)
        node("single_guider", "BasicGuider", model=["single_setup", 0], conditioning=["single_apply", 0])
        node("single_sample", "SamplerCustomAdvanced", noise=["single_noise", 0], guider=["single_guider", 0],
             sampler=["single_setup", 1], sigmas=["single_setup", 2], latent_image=["single_apply", 1])
        final = ["single_sample", 0]
    else:
        if variant == "split":
            relay_lane("low", model_lane("low"))
            low = stage("low", stage_name="dual_low_4", latent=["low_apply", 2], positive=["low_apply", 1])
            high_model = model_lane("high", after=["low_sample", 2])
        else:
            node("low_load", "MiniMaxH3StageLoadEXPT8", artifact_path=low_path,
                 artifact_sha256=low_sha, expected_stage="dual_low_4")
            low = ["low_load", 1]
            high_model = model_lane("high")
        node("upscale", "MiniMaxH3LearnedLatentUpscaleT8Advanced", av_latent=low, model_name=UPSCALER,
             size_mode="scale_by", scale_by=1.2, target_megapixels=.15, target_width=512, target_height=288,
             aspect_policy="preserve_source", max_anisotropy=1.05, precision="fp16", release_policy="offload_after")
        relay_lane("high", high_model, width=["upscale", 1], height=["upscale", 2])
        node("handoff", "MiniMaxH3NativeDualHandoffEXPT8", learned_latent=["upscale", 0],
             highres_template=["high_apply", 2], positive=["high_apply", 1], first_pass_steps="4",
             second_audio_source="auto", second_audio_strength=0.)
        final = stage("high", stage_name="dual_high_4", latent=["handoff", 0], positive=["handoff", 1])
    node("decode", "MiniMaxH3AVDecodeT8", av_latent=final, video_vae=["7", 0], audio_vae=["8", 0])
    node("trim", "MiniMaxH3OutputTrimT8", frames=["decode", 0], audio=["decode", 1],
         start_seconds=0., duration_seconds=5., fps=24.)
    node("save", "MiniMaxH3SafeAVSaveT8Advanced", images=["trim", 0], audio=["trim", 1],
         filename_prefix="R6_Reference_"+variant, crf=18)
    return graph


def make_workflow(info, **kwargs):
    graph = make_graph(info, **kwargs)
    variant = kwargs.get("variant", "split")
    workflow = convert(graph, info, "R6 Reference · "+variant+" · EXP candidate")
    for index, item in enumerate(workflow["nodes"]):
        item["pos"] = [(index % 7)*440, (index // 7)*480]
        item["size"] = [400, 410]
    note = ("EXP candidate; no GPU/human acceptance inherited. Exact package file SHA pins.\n"
            "A voice only/off-screen, B image on-screen; no two-person/voice-clone claim.\n"
            "Reference VAE identity includes selected precision/device: use matching producers.\n"
            "Split LOW4 partial x0 -> existing learned1.2 -> HIGH4 joint audio; never freeze partial voice.\n"
            "LOW/HIGH Relay Plans and EAV configs are separate; EAV report_only is audit, not enhancement.\n"
            "Cold contains no LOW sampler or LOW model; explicitly fill the actual saved LOW path/SHA.\n"
            "124 native frames -> final trim120/5s at24fps; native generated audio, no replacement.\n"
            "Save/Load only assets: not fresh Qwen/Cold diffusion/identity or quality proof.")
    last = workflow["last_node_id"]+1
    workflow["nodes"].append({"id": last, "type": "Note", "title": "Reference candidate · boundaries",
        "pos": [0, ((len(graph)+6)//7)*480], "size": [1250, 330], "flags": {}, "order": len(graph),
        "mode": 0, "inputs": [], "outputs": [], "properties": {"text": note}, "widgets_values": [note]})
    workflow["last_node_id"] = last
    workflow["extra"]["radar_r6_reference_sampling"] = {"schema": "t8.r6.reference-sampling.canvas/v1",
        "variant": variant, "automatic_accept": False, "gpu_verified": False, "human_accepted": False}
    return workflow, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    for field in ("voice-name", "image-name", "voice-sha", "image-sha"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination.exists() or not destination.is_relative_to(root / "artifacts/development/radar-r6-20261005"):
        raise ValueError("Use a new owned RADAR candidate directory, never overwrite")
    with urllib.request.urlopen(args.server.rstrip("/")+"/object_info", timeout=30) as response:
        info = json.load(response)
    records = {}
    for variant, name in (("ordinary", "Ref_Ordinary_EXP"), ("split", "Ref_Full_EXP"), ("cold", "Ref_Cold_EXP")):
        workflow, graph = make_workflow(info, voice_name=args.voice_name, image_name=args.image_name,
            voice_sha=args.voice_sha, image_sha=args.image_sha, variant=variant)
        records[name+".json"] = workflow
        records[name+".api.json"] = graph
    errors = validate_paths([(destination / name).relative_to(root).as_posix() for name in records])
    if errors:
        raise ValueError(errors)
    destination.mkdir(parents=True)
    for name, value in records.items():
        with (destination / name).open("x", encoding="utf8", newline="\n") as stream:
            json.dump(deepcopy(value), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
