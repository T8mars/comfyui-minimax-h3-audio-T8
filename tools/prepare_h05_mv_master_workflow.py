"""Create a separate three-audio-role MV canvas; no download, install or Queue."""
import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def graph(info, *, image, vocal, master, source_start=5.0, voice_start=16.0):
    from tools.api_to_frontend_workflow import convert
    from tools.repair_frontend_workflow_order import canonical_entries, _is_widget_spec, has_seed_control
    from tools.validate_freevideo_quality_canvas import check_serialized

    def node(kind, title, **inputs):
        return dict(class_type=kind, inputs=inputs, _meta=dict(title=title))

    prompt = (
        "One adult performer from <Picture 1>, a steady frontal medium close-up in a quiet studio. "
        "Keep the same face, hair and clothing. Perform in sync with the exact vocal timing of "
        "<Audio 1>. <Audio 2> is a voice-character reference only, not additional words, a second "
        "timeline or a second performer. No added dialogue, subtitles, cuts or other speakers. "
        "The actual source vocal recording supplies the content; no lyric transcript was provided."
    )
    common = dict(clip=["9", 0], video_vae=["1", 0], audio_vae=["2", 0], prompt=prompt,
        length=124, task_type="Ref2VA", audio_mode="lock_source", audio_denoise_strength=0.0,
        add_source_as_reference=True, prompt_primary_audio_ordinal=1, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        drive_audio=["15", 0], final_audio=["16", 0],
        **{"ref_images.ref_image_0": ["10", 0], "ref_audios.ref_audio_0": ["17", 0]})
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
        "8": node("LoraLoaderBypassModelOnly", "Existing 4+4 adapter recipe", model=["7", 0],
            lora_name="minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors", strength_model=0.7),
        "9": node("CLIPLoader", "Native Qwen", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "10": node("LoadImage", "Performer reference", image=image),
        "11": node("LoadAudio", "Aligned vocal stem · complete source", audio=vocal),
        "12": node("LoadAudio", "Original complete mixed master · delivery only", audio=master),
        "13": node("PrimitiveFloat", "Shared GLOBAL source origin in seconds", value=source_start),
        "14": node("PrimitiveFloat", "Shared preparation duration · 124/24", value=124/24),
        "15": node("TrimAudioDuration", "DRIVE · vocal at global origin", audio=["11", 0], start_index=["13", 0], duration=["14", 0]),
        "16": node("TrimAudioDuration", "FINAL · original mix at SAME origin", audio=["12", 0], start_index=["13", 0], duration=["14", 0]),
        "17": node("TrimAudioDuration", "VOICE reference · separate anchor, NOT target timeline", audio=["11", 0], start_index=voice_start, duration=3.0),
        "18": node("MiniMaxH3AudioConditioningT8", "LOW · Audio1 drive / Audio2 voice / final mix", width=448, height=256, **common),
        "19": node("MiniMaxH3DualClockSamplerT8", "Existing joint clock", model=["8", 0], av_latent=["18", 1],
            steps=8, shift_video=12.0, shift_audio=3.0, sampler_name="dual_clock_euler", scheduler="native_flow"),
        "20": node("MiniMaxH3LearnedTwoPassParityPlanT8Advanced", "Original split4+4", model=["19", 0], base_steps=8, coarse_steps=4, refine_steps=4),
        "21": node("RandomNoise", "LOW fixed seed", noise_seed=2610070301),
        "22": node("BasicGuider", "LOW guider", model=["19", 0], conditioning=["18", 0]),
        "23": node("SamplerCustomAdvanced", "LOW4 · consume denoised_output", noise=["21", 0], guider=["22", 0],
            sampler=["19", 1], sigmas=["20", 0], latent_image=["18", 1]),
        "24": node("MiniMaxH3LearnedLatentUpscaleT8Advanced", "Video-only learned1.2x · audio preserved", av_latent=["23", 1],
            model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors", size_mode="scale_by", scale_by=1.2,
            target_megapixels=0.7, target_width=512, target_height=288, aspect_policy="preserve_source",
            max_anisotropy=1.05, precision="fp16", release_policy="offload_after"),
        "25": node("MiniMaxH3AudioConditioningT8", "Fresh HIGH · same three audio roles", width=["24", 1], height=["24", 2], **common),
        "26": node("MiniMaxH3TwoPassLatentReconcileT8Advanced", "Explicit complete-source audio lock", learned_latent=["24", 0],
            highres_template=["25", 1], positive=["25", 0], audio_policy="auto",
            second_pass_audio_source="highres_template", second_pass_audio_strength=0.0),
        "27": node("RandomNoise", "HIGH fixed seed", noise_seed=2610070302),
        "28": node("BasicGuider", "HIGH guider · own actual geometry", model=["39", 0], conditioning=["26", 1]),
        "29": node("SamplerCustomAdvanced", "HIGH4 · no partial LOW audio freeze", noise=["27", 0], guider=["28", 0],
            sampler=["39", 1], sigmas=["20", 1], latent_image=["26", 0]),
        "30": node("MiniMaxH3TwoPassAudioAuditT8Advanced", "Verify explicit source lock before decode", second_pass_input=["26", 0],
            second_pass_output=["29", 1], expected_audio_strength=0.0, fail_on_locked_mismatch=True, locked_atol=1e-5),
        "31": node("MiniMaxH3AVDecodeT8", "Decode native complete result", av_latent=["30", 0], video_vae=["1", 0], audio_vae=["2", 0]),
        "32": node("MiniMaxH3OutputTrimT8", "5 seconds · mux ORIGINAL mix, not decoded vocal", frames=["31", 0],
            audio=["25", 2], start_seconds=0.0, duration_seconds=5.0, fps=24.0),
        "33": node("MiniMaxH3SafeAVSaveT8Advanced", "Final full mix · one export", images=["32", 0], audio=["32", 1], filename_prefix="H05/Q03_MV_Master", crf=18),
        "34": node("PreviewAny", "Actual LOW media ordinals", source=["18", 4]),
        "35": node("PreviewAny", "Actual HIGH media ordinals", source=["25", 4]),
        "36": node("PreviewAny", "Actual HIGH contract", source=["25", 5]),
        "37": node("PreviewAudio", "Hear aligned DRIVE, not final mix", audio=["15", 0]),
        "38": node("PreviewAudio", "Hear aligned FINAL original mix", audio=["32", 1]),
        "39": node("MiniMaxH3DualClockSamplerT8", "HIGH own model/sampler · actual lifted geometry", model=["8", 0], av_latent=["26", 0],
            steps=8, shift_video=12.0, shift_audio=3.0, sampler_name="dual_clock_euler", scheduler="native_flow"),
    }
    native = convert(api, info, "H05_Q03_MV_Master_4plus4")
    for row in native["nodes"]:
        source = api[str(row["id"])]["inputs"]
        # Linked widget sockets keep positional defaults in native saves. Decode
        # ALL actual schema widgets; linked values still resolve from their edge.
        entries, unknown = canonical_entries(info[row["type"]], list(source))
        assert not unknown, unknown
        named, cursor = {}, 0
        for key, spec, _ in entries:
            if not _is_widget_spec(spec):
                continue
            value = row["widgets_values"][cursor]
            cursor += 1
            named[key] = value
            if key not in source:
                source[key] = value
            elif not isinstance(source[key], list):
                assert source[key] == value, (row["id"], key)
            if has_seed_control(key, spec):
                assert row["widgets_values"][cursor] == "fixed"
                named["control_after_generate"] = row["widgets_values"][cursor]
                cursor += 1
        assert cursor == len(row["widgets_values"]), row["id"]
        row["widgets_values_named"] = named
        if row["type"] == "PreviewAny":
            row["inputs"][0]["type"] = "STRING"
        row["size"][0] = 440
    for link in native["links"]:
        if native["nodes"][link[3]-1]["type"] == "PreviewAny":
            link[5] = "STRING"
    native["nodes"].append(dict(id=40, type="Note", pos=[0, -340], size=[1300, 300], flags={}, order=39,
        mode=0, inputs=[], outputs=[], properties={}, title="Three audio roles / source time / old workflows unchanged",
        widgets_values=["NEW opt-in MV example; no old workflow or audio default changed.\n"
            "Drive=vocal stem at shared global source origin; Audio2=separate voice reference only; final=original mix at the same origin.\n"
            "124 preparation frames at24fps, then120 output frames/5s. Keep stems time-aligned; no automatic lyrics/beat/role inference.\n"
            "This source-audio lock is opt-in for COMPLETE external audio, not freezing partial LOW native generation.\n"
            "AAC MP4 is not original PCM exact. Input AUDIO slices can be exact; export is separately qualified.\n"
            "KJNodes and existing T8 assets required; no automatic download. External EAV/Relay may be added explicitly; neither is applied here.\n"
            "One native representative and human review required; no promise of hard voice identity or lip-sync quality."]))
    native["last_node_id"] = 40
    clean = {key: {name: value[name] for name in ("class_type", "inputs")} for key, value in api.items()}
    serial = check_serialized(native, clean)
    return native, clean, serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="replace_with_performer.png")
    parser.add_argument("--vocal", default="replace_with_aligned_vocal.wav")
    parser.add_argument("--master", default="replace_with_original_mix.wav")
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.server)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.path not in ("", "/") or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Use an explicit local ComfyUI origin")
    if args.output.exists():
        raise FileExistsError("Never overwrite a user workflow or previous evidence")
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    native, _, serial = graph(info, image=args.image, vocal=args.vocal, master=args.master)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.output), serialization=serial, queued=False)))


if __name__ == "__main__":
    main()
