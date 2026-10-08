"""Create an isolated complete native RES AV canvas; never installs or queues."""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def graph(info, *, image):
    from tools.api_to_frontend_workflow import convert
    from tools.repair_frontend_workflow_order import canonical_entries, _is_widget_spec, has_seed_control
    from tools.validate_freevideo_quality_canvas import check_serialized

    def node(kind, title, **inputs):
        return dict(class_type=kind, inputs=inputs, _meta=dict(title=title))

    api = {
        "1": node("VAELoader", "Native video VAE", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "2": node("VAELoader", "Native audio VAE", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "3": node("UNETLoader", "Ref2VA base · NO Turbo or distilled LoRA",
            unet_name="minimax_h3_ref2va_int8_convrot.safetensors", weight_dtype="default"),
        "4": node("MiniMaxH3ChunkFeedForwardT8Advanced", "Existing FFN budget", model=["3", 0], chunks=2, seq_threshold=4096),
        "5": node("MiniMaxH3MemoryEfficientSageAttentionPatch", "Existing KJ memory path", model=["4", 0]),
        "6": node("ModelAttentionBackend", "Explicit existing backend", model=["5", 0], attention="pytorch attention"),
        "7": node("SolAttnMiniMax", "Existing optional sparse policy", model=["6", 0], tau=1.3,
            start_percent=0.2, end_percent=0.9, min_tokens=12288, sink_conditioning="exact_kv_and_rows",
            morton=False, morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
            reuse_qkv_memory=False, verbose=False, dense_blocks=""),
        "8": node("CLIPLoader", "Native Qwen", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "9": node("LoadImage", "One legal adult performer reference", image=image),
        "10": node("MiniMaxH3AudioConditioningT8", "Fresh native VIDEO + GENERATED AUDIO",
            clip=["8", 0], video_vae=["1", 0], audio_vae=["2", 0],
            prompt='One adult woman from <Picture 1>, a steady frontal medium close-up in a quiet studio. '
                'Preserve her face, hair, glasses and clothing. She looks at the camera and says once '
                'in clear natural Mandarin: "你好，今天天气真好。" Then she smiles quietly. '
                'One speaker only. No additional words, repeated speech, music, subtitles or cuts.',
            width=512, height=288, length=124, task_type="Ref2VA", audio_mode="native",
            audio_denoise_strength=1.0, add_source_as_reference=False, prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
            **{"ref_images.ref_image_0": ["9", 0]}),
        "11": node("MiniMaxH3DualClockSamplerT8", "Existing Core RES · complete 8-step history",
            model=["7", 0], av_latent=["10", 1], steps=8, shift_video=12.0, shift_audio=3.0,
            sampler_name="res_multistep", scheduler="native_flow"),
        "12": node("RandomNoise", "Fixed seed", noise_seed=2610070101),
        "13": node("BasicGuider", "Complete native AV conditioning", model=["11", 0], conditioning=["10", 0]),
        "14": node("SamplerCustomAdvanced", "ONE complete RES invocation · not latent-only 4+4",
            noise=["12", 0], guider=["13", 0], sampler=["11", 1], sigmas=["11", 2], latent_image=["10", 1]),
        "15": node("MiniMaxH3AVDecodeT8", "Decode actual COMPLETE generated video AND audio",
            av_latent=["14", 1], video_vae=["1", 0], audio_vae=["2", 0]),
        "16": node("MiniMaxH3OutputTrimT8", "Explicit 120 frames / 5 seconds · generated audio",
            frames=["15", 0], audio=["15", 1], start_seconds=0.0, duration_seconds=5.0, fps=24.0),
        "17": node("MiniMaxH3SafeAVSaveT8Advanced", "Independent generated AV export",
            images=["16", 0], audio=["16", 1], filename_prefix="H05/Q01_RES8", crf=18),
        "18": node("PreviewAny", "Actual media roles", source=["10", 4]),
        "19": node("PreviewAny", "Actual native conditioning contract", source=["10", 5]),
        "20": node("PreviewAudio", "Actual generated speech, NOT replacement source", audio=["16", 1]),
    }
    native = convert(api, info, "H05_Q01_RES8_Complete_AV")
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
        assert cursor == len(row["widgets_values"])
        row["widgets_values_named"] = named
        if row["type"] == "PreviewAny":
            row["inputs"][0]["type"] = "STRING"
        row["size"][0] = 430
    rows = {row["id"]: row for row in native["nodes"]}
    for edge in native["links"]:
        if rows[edge[3]]["type"] == "PreviewAny":
            edge[5] = "STRING"
    native["nodes"].append(dict(id=21, type="Note", pos=[0, -330], size=[1300, 280],
        flags={}, order=20, mode=0, inputs=[], outputs=[], properties={},
        title="Experimental native RES / complete history / generated AV",
        widgets_values=["NEW opt-in qualification template. Old samplers, defaults and workflows unchanged.\n"
            "Existing Core res_multistep + native_flow; shifts12/3; one complete8-step invocation.\n"
            "Ref2VA INT8 ConvRot BASE, no Turbo/DMAD/PDD/FastH3 LoRA or distilled sigma substitution.\n"
            "124 preparation frames, explicitly trim120 at24fps =5s. Audio is ACTUAL generated audio.\n"
            "The packed joint AV solver uses one common coordinate; not vLLM independent solver parity.\n"
            "Not continuous4+4 resume: old_denoised/old_sigma_down are NOT persisted by this template.\n"
            "No portable Stage/history resume or speed/quality promise. Existing Euler-only resume guards stay.\n"
            "External EAV/Prompt Relay can be inserted explicitly on MODEL/conditioning; neither applied here.\n"
            "KJNodes and existing T8 assets required. Human voice/picture review remains pending."]))
    native["last_node_id"] = 21
    clean = {key: {name: value[name] for name in ("class_type", "inputs")} for key, value in api.items()}
    return native, clean, check_serialized(native, clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="replace_with_legal_performer.png")
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.server)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.path not in ("", "/") or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Use an explicit local ComfyUI origin")
    if args.output.exists():
        raise FileExistsError("Never overwrite user workflows or earlier evidence")
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    native, _, serial = graph(info, image=args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.output), serialization=serial, queued=False)))


if __name__ == "__main__":
    main()
