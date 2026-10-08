"""Create one separate same-source timed Guide canvas; never installs or queues."""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def graph(info, *, source):
    from tools.api_to_frontend_workflow import convert
    from tools.repair_frontend_workflow_order import canonical_entries, _is_widget_spec, has_seed_control
    from tools.validate_freevideo_quality_canvas import check_serialized

    def node(node_type, title, **inputs):
        return dict(class_type=node_type, inputs=inputs, _meta=dict(title=title))

    prompt = (
        "The adult woman in <Picture 1> speaks naturally to the camera. Preserve her face, "
        "hair, glasses, clothing and the plain background. Follow the complete aligned "
        "dialogue in <Audio 1>; <Audio 2> is only her separate voice reference, not a second "
        "speaker or a second script. The same-source image guide anchors her pose at 3 seconds. "
        "No extra words, repeated dialogue, additional people, music or scene cuts."
    )
    common = dict(clip=["9", 0], video_vae=["1", 0], audio_vae=["2", 0], prompt=prompt,
        length=124, task_type="Ref2VA", audio_mode="lock_source", audio_denoise_strength=0.0,
        add_source_as_reference=True, prompt_primary_audio_ordinal=1, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        drive_audio=["12", 1], final_audio=["12", 1],
        **{"ref_images.ref_image_0": ["18", 0], "ref_audios.ref_audio_0": ["19", 0]})
    api = {
        "1": node("VAELoader", "Native video VAE", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "2": node("VAELoader", "Native audio VAE", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "3": node("UNETLoader", "Existing Ref2VA model", unet_name="minimax_h3_ref2va_int8_convrot.safetensors", weight_dtype="default"),
        "4": node("MiniMaxH3ChunkFeedForwardT8Advanced", "Existing FFN budget", model=["3", 0], chunks=2, seq_threshold=4096),
        "5": node("MiniMaxH3MemoryEfficientSageAttentionPatch", "Existing KJ memory path", model=["4", 0]),
        "6": node("ModelAttentionBackend", "Explicit existing backend", model=["5", 0], attention="pytorch attention"),
        "7": node("SolAttnMiniMax", "Existing optional sparse policy", model=["6", 0], tau=1.3,
            start_percent=0.2, end_percent=0.9, min_tokens=12288, sink_conditioning="exact_kv_and_rows",
            morton=False, morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
            reuse_qkv_memory=False, verbose=False, dense_blocks=""),
        "8": node("LoraLoaderBypassModelOnly", "Existing 4+4 adapter", model=["7", 0],
            lora_name="minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors", strength_model=0.7),
        "9": node("CLIPLoader", "Native Qwen", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "10": node("LoadVideo", "One CFR source video with complete audio", file=source),
        "11": node("MiniMaxH3SourceClockEXPT8", "Observe source RGB / PCM / CFR clock together", video=["10", 0]),
        "12": node("MiniMaxH3SourceConformEXPT8", "Prepare source ONCE · 124/24 seconds", frames=["11", 0],
            source_fps=["11", 2], width=448, height=256, length=124, start_seconds=0.0,
            short_video_policy="hold_last_frame", short_audio_policy="pad_silence",
            source_audio=["11", 1], source_clock=["11", 3]),
        "13": node("MiniMaxH3SourceControlDeriveEXPT8", "Owned identity RGB · BEFORE Conform", source_frames=["11", 0], operation="identity_rgb"),
        "14": node("MiniMaxH3SourceControlMapEXPT8", "LOW geometry · shared exact source indices", frame_map=["12", 4],
            kind="image", width=448, height=256, control_image=["13", 0], lineage=["13", 2]),
        "15": node("ImageFromBatch", "LOW guide · target frame72 = 3 seconds", image=["14", 0], batch_index=72, length=1),
        "16": node("MiniMaxH3SourceControlMapEXPT8", "HIGH actual learned geometry · SAME map", frame_map=["12", 4],
            kind="image", width=["27", 1], height=["27", 2], control_image=["13", 0], lineage=["13", 2]),
        "17": node("ImageFromBatch", "HIGH guide · SAME target frame72", image=["16", 0], batch_index=72, length=1),
        "18": node("ImageFromBatch", "Performer reference · original source first frame", image=["11", 0], batch_index=0, length=1),
        "19": node("TrimAudioDuration", "Separate voice reference · original source 2–5s", audio=["11", 1], start_index=2.0, duration=3.0),
        "20": node("MiniMaxH3AudioConditioningT8", "LOW · preserve complete drive and separate voice ref", width=448, height=256, **common),
        "21": node("MiniMaxH3DualClockSamplerT8", "LOW own actual geometry", model=["8", 0], av_latent=["20", 1],
            steps=8, shift_video=12.0, shift_audio=3.0, sampler_name="dual_clock_euler", scheduler="native_flow"),
        "22": node("MiniMaxH3LearnedTwoPassParityPlanT8Advanced", "Existing shared sigma plan · 4+4", model=["21", 0], base_steps=8, coarse_steps=4, refine_steps=4),
        "23": node("MiniMaxH3AddGuide", "LOW AddGuide · frame72, append not replace", positive=["20", 0], latent=["20", 1],
            frame_idx=72, vae=["1", 0], image=["15", 0]),
        "24": node("RandomNoise", "LOW fixed seed", noise_seed=2610070401),
        "25": node("BasicGuider", "LOW consumes guided conditioning", model=["21", 0], conditioning=["23", 0]),
        "26": node("SamplerCustomAdvanced", "LOW4 · consume denoised_output", noise=["24", 0], guider=["25", 0],
            sampler=["21", 1], sigmas=["22", 0], latent_image=["20", 1]),
        "27": node("MiniMaxH3LearnedLatentUpscaleT8Advanced", "Existing learned1.2x · video only", av_latent=["26", 1],
            model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors", size_mode="scale_by", scale_by=1.2,
            target_megapixels=0.7, target_width=512, target_height=288, aspect_policy="preserve_source",
            max_anisotropy=1.05, precision="fp16", release_policy="offload_after"),
        "28": node("MiniMaxH3AudioConditioningT8", "Fresh HIGH · same drive and voice roles", width=["27", 1], height=["27", 2], **common),
        "29": node("MiniMaxH3TwoPassLatentReconcileT8Advanced", "Keep COMPLETE external audio, not partial LOW", learned_latent=["27", 0],
            highres_template=["28", 1], positive=["28", 0], audio_policy="auto",
            second_pass_audio_source="highres_template", second_pass_audio_strength=0.0),
        "30": node("MiniMaxH3AddGuide", "HIGH AddGuide · same frame72, fresh stage VAE", positive=["29", 1], latent=["29", 0],
            frame_idx=72, vae=["1", 0], image=["17", 0]),
        "31": node("MiniMaxH3DualClockSamplerT8", "HIGH own actual lifted geometry", model=["8", 0], av_latent=["29", 0],
            steps=8, shift_video=12.0, shift_audio=3.0, sampler_name="dual_clock_euler", scheduler="native_flow"),
        "32": node("RandomNoise", "HIGH fixed seed", noise_seed=2610070402),
        "33": node("BasicGuider", "HIGH consumes guided conditioning", model=["31", 0], conditioning=["30", 0]),
        "34": node("SamplerCustomAdvanced", "HIGH4 · refined denoised_output", noise=["32", 0], guider=["33", 0],
            sampler=["31", 1], sigmas=["22", 1], latent_image=["29", 0]),
        "35": node("MiniMaxH3TwoPassAudioAuditT8Advanced", "Audit complete-source audio lock", second_pass_input=["29", 0],
            second_pass_output=["34", 1], expected_audio_strength=0.0, fail_on_locked_mismatch=True, locked_atol=1e-5),
        "36": node("MiniMaxH3AVDecodeT8", "Decode full refined AV", av_latent=["35", 0], video_vae=["1", 0], audio_vae=["2", 0]),
        "37": node("MiniMaxH3OutputTrimT8", "5 seconds · ORIGINAL complete source audio", frames=["36", 0],
            audio=["12", 1], start_seconds=0.0, duration_seconds=5.0, fps=24.0),
        "38": node("MiniMaxH3SafeAVSaveT8Advanced", "One independent timed-guide export", images=["37", 0], audio=["37", 1], filename_prefix="H05/Q04_Timed_Guide", crf=18),
        "39": node("PreviewAny", "Actual source clock", source=["11", 4]),
        "40": node("PreviewAny", "Actual preparation / source frame-map", source=["12", 5]),
        "41": node("PreviewAny", "Actual LOW mapping receipt", source=["14", 2]),
        "42": node("PreviewAny", "Actual HIGH mapping receipt", source=["16", 2]),
        "43": node("PreviewAny", "Actual LOW Audio1 drive / Audio2 voice", source=["20", 4]),
        "44": node("PreviewAny", "Actual HIGH Audio1 drive / Audio2 voice", source=["28", 4]),
        "45": node("PreviewImage", "Actual LOW image guide at3s", images=["15", 0]),
        "46": node("PreviewImage", "Actual HIGH image guide at3s", images=["17", 0]),
        "47": node("PreviewAudio", "Original complete source dialogue · final5s", audio=["37", 1]),
    }
    native = convert(api, info, "H05_Q04_Timed_Guide_4plus4")
    for row in native["nodes"]:
        inputs = api[str(row["id"])]["inputs"]
        entries, unknown = canonical_entries(info[row["type"]], list(inputs))
        assert not unknown, unknown
        named, cursor = {}, 0
        for key, spec, _ in entries:
            if not _is_widget_spec(spec):
                continue
            value = row["widgets_values"][cursor]
            cursor += 1
            named[key] = value
            if key not in inputs:
                inputs[key] = value
            elif not isinstance(inputs[key], list):
                assert inputs[key] == value, (row["id"], key)
            if has_seed_control(key, spec):
                assert row["widgets_values"][cursor] == "fixed"
                named["control_after_generate"] = row["widgets_values"][cursor]
                cursor += 1
        assert cursor == len(row["widgets_values"]), row["id"]
        row["widgets_values_named"] = named
        if row["type"] == "PreviewAny":
            row["inputs"][0]["type"] = "STRING"
        row["size"][0] = 440
    lookup = {row["id"]: row for row in native["nodes"]}
    for link in native["links"]:
        if lookup[link[3]]["type"] == "PreviewAny":
            link[5] = "STRING"
    native["nodes"].append(dict(id=48, type="Note", pos=[0, -320], size=[1400, 280], flags={}, order=47,
        mode=0, inputs=[], outputs=[], properties={}, title="One source / shared time / stage-specific Guide",
        widgets_values=["NEW opt-in template; no old workflow, Core Guide or sampler algorithm changed.\n"
            "Use one legal 5s/24fps source. SourceClock observes actual CFR; Conform once holds4 tail frames to124 and reports audio padding.\n"
            "LOW448x256 and HIGH actual learned1.2x geometry share the SAME complete frame-map, not the same pixel hash.\n"
            "Frame72=3s in the target. Both Core AddGuide nodes append to fresh conditioning before their BasicGuider.\n"
            "Audio1=complete source dialogue; Audio2=separate voice sample; final mux=original source AUDIO, not decoded latent.\n"
            "This is not arbitrary VFR/clock, hard pose/identity/voice lock or PCM-exact AAC qualification.\n"
            "EAV / Relay remain externally connectable, neither is applied by default. KJ and existing assets required.\n"
            "Human picture/pose/lip-sync/voice review remains necessary; no automatic PASS."]))
    native["last_node_id"] = 48
    clean = {key: {name: value[name] for name in ("class_type", "inputs")} for key, value in api.items()}
    return native, clean, check_serialized(native, clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="replace_with_5s_CFR_source.mp4")
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.server)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.path not in ("", "/") or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Use an explicit local ComfyUI origin")
    if args.output.exists():
        raise FileExistsError("Never overwrite user workflows or earlier evidence")
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    native, _, serial = graph(info, source=args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.output), serialization=serial, queued=False)))


if __name__ == "__main__":
    main()
