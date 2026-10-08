"""Create a separate external-RGB/LTX canvas; no Queue, install or overwrite."""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_ltx_x2_canvas(width, height):
    """The selected full video VAE has a 32-pixel grid BEFORE the x2 lift.

    This only validates this new template. It does not change the older RGB
    preparation node's contract, or silently round/crop a user's canvas.
    """
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value < 64 or value % 64 for value in (width, height)):
        raise ValueError("This full LTX VAE + half-canvas x2 template needs target dimensions divisible by 64")
    return width, height


def graph(info, *, source, load_mode="legacy_default", target_width=256, target_height=448,
          output_backend="vhs_external"):
    from tools.api_to_frontend_workflow import convert
    from tools.repair_frontend_workflow_order import canonical_entries, _is_widget_spec, has_seed_control
    from tools.validate_freevideo_quality_canvas import check_serialized

    if load_mode not in ("legacy_default", "consistent_streaming_exp"):
        raise ValueError("Use an existing explicit LTX load policy")
    width, height = validate_ltx_x2_canvas(target_width, target_height)
    if output_backend not in ("vhs_external", "safe_av"):
        raise ValueError("Select the explicit external VHS or strict H3 Safe AV output")

    def node(kind, title, **inputs):
        return dict(class_type=kind, inputs=inputs, _meta=dict(title=title))

    api = {
        "1": node("LoadVideo", "User-selected external video, not a native H3 checkpoint", file=source),
        "2": node("MiniMaxH3SourceClockEXPT8", "Actual bounded source PTS and original audio", video=["42", 0]),
        "3": node("MiniMaxH3SourceConformEXPT8", "One 24fps RGB conform, source origin 1 second",
            frames=["2", 0], source_fps=["2", 2], source_audio=["2", 1], source_clock=["2", 3],
            width=width, height=height, length=124, start_seconds=1.0,
            short_video_policy="strict", short_audio_policy="pad_silence"),
        "4": node("TrimAudioDuration", "Original-rate audio uses the SAME source origin",
            audio=["2", 1], start_index=1.0, duration=124/24),
        "5": node("MiniMaxH3SolEngineDraftToLTXT8Advanced", "Existing 8n+1 RGB preparation: 124 to121",
            frames=["3", 0], target_width=width, target_height=height, frame_policy="trim_to_8n_plus_1", fps=24.0),
        "6": node("VAELoader", "Actual full LTX video VAE, not TAEHV encode",
            vae_name="ltx-2.5-video-vae-conv-bf16.safetensors"),
        "7": node("VAEEncode", "RGB re-encoding is NOT native H3 ancestry", pixels=["5", 0], vae=["6", 0]),
        "8": node("LatentUpscaleModelLoader", "Existing LTX x2 latent model",
            model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        "9": node("LTXVLatentUpsampler", "Existing half-canvas to target LTX lift",
            samples=["7", 0], upscale_model=["8", 0], vae=["6", 0]),
        "10": node("UNETLoader", "Existing actual LTX-2.5 transformer",
            unet_name="ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", weight_dtype="default"),
        "11": node("LoraLoaderModelOnly", "Existing refiner LoRA, unchanged 0.8",
            model=["10", 0], lora_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors", strength_model=0.8),
        "12": node("MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced", "Existing independent three-step Identity preset",
            model=["11", 0], enabled=True, schedule_mode="identity_preserve_0p5",
            manual_sigmas="0.5, 0.412, 0.350, 0", attention_backend="dense_reference",
            min_tokens=4096, kernel_precision="bf16_official", verbose=False),
        "13": node("MiniMaxH3LTXLoadPolicyEXPT8", "Explicit load policy, no shared Core change", model=["12", 0], mode=load_mode),
        "14": node("CLIPLoader", "Existing LTX Gemma encoder",
            clip_name="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", type="ltxv", device="default"),
        "15": node("MiniMaxH3LTXPromptRelayPlanEXPT8", "External Relay observes actual 121-frame LTX grid",
            ltx_latent=["9", 0], global_prompt="Preserve the adult subject, clothing, scene, camera and visible actions of the source video. Continuous natural motion.",
            local_prompts="Preserve the source performance and scene throughout.", timing_mode="seconds",
            time_ranges="0-5.041666666666667", fps=24.0, epsilon=0.1,
            allow_gaps=False, allow_overlaps=False),
        "16": node("MiniMaxH3LTXPromptRelayEncodeEXPT8", "Fresh global and event text",
            clip=["14", 0], ltx_relay_plan=["15", 0], max_text_tokens=2048),
        "17": node("MiniMaxH3StageEAVConfigEXPT8", "External EAV, report only by default",
            mode="report_only", tau=4.0, start_video_progress=0.15, end_video_progress=0.9,
            max_workspace_mib=32, g_hard_limit=1.5),
        "18": node("MiniMaxH3LTXEAVApplyEXPT8", "External video-only EAV",
            model=["13", 0], ltx_latent=["9", 0], sigmas=["12", 2], eav_config=["17", 0]),
        "19": node("MiniMaxH3LTXPromptRelayApplyEXPT8", "External Relay, report only by default",
            model=["18", 0], ltx_latent=["9", 0], sigmas=["12", 2], positive=["16", 0],
            text_binding=["16", 1], mode="report_only", max_workspace_mib=32),
        "20": node("CLIPTextEncode", "Original empty negative", clip=["14", 0], text=""),
        "21": node("LTXVConditioning", "Actual LTX frame rate", positive=["19", 1], negative=["20", 0], frame_rate=24.0),
        "22": node("CFGGuider", "Same final MODEL as stage audit", model=["19", 0], positive=["21", 0], negative=["21", 1], cfg=1.0),
        "23": node("RandomNoise", "Independent fixed refiner seed", noise_seed=2610070501),
        "24": node("MiniMaxH3LTXRGBStageBindEXPT8", "Bind exact prepared RGB, original-rate audio and controls",
            source_frames=["3", 0], source_audio=["4", 0], prepared_frames=["5", 0], prep_report_json=["5", 6],
            ltx_latent=["9", 0], model=["19", 0], noise=["23", 0], guider=["22", 0], sampler=["12", 1], sigmas=["12", 2], setup_report_json=["12", 4]),
        "25": node("SamplerCustomAdvanced", "Separate actual three-step LTX sampler", noise=["24", 0], guider=["24", 1], sampler=["24", 2], sigmas=["24", 3], latent_image=["24", 4]),
        "26": node("MiniMaxH3LTXRGBStageAuditEXPT8", "Candidate/source audit, not portable cache",
            stage_boundary=["24", 5], source_frames=["3", 0], source_audio=["4", 0], prepared_frames=["5", 0], prep_report_json=["5", 6],
            ltx_latent=["24", 4], model=["19", 0], noise=["24", 0], guider=["24", 1], sampler=["24", 2], sigmas=["24", 3], setup_report_json=["12", 4], candidate_latent=["25", 0]),
        "27": node("MiniMaxH3LTXEAVAuditEXPT8", "Observe actual EAV calls", model=["19", 0], candidate_latent=["26", 0], runtime=["18", 1]),
        "28": node("MiniMaxH3LTXPromptRelayAuditEXPT8", "Observe actual Relay calls", model=["19", 0], candidate_latent=["27", 0], runtime=["19", 2]),
        "29": node("MiniMaxH3LTXLoadPolicyAuditEXPT8", "Observe preparation, not completion", model=["19", 0], candidate_latent=["28", 0], runtime=["13", 1]),
        "30": node("MiniMaxH3SolEngineTAEHVLoaderT8Advanced", "Existing approximate decoder", model_name="taeltx2_3_wide.pth"),
        "31": node("MiniMaxH3SolEngineTAEHVDecodeT8Advanced", "Separate final video decode", latent=["29", 0], taehv=["30", 0], execution_mode="auto_official", precision="bf16_official"),
        "32": node("MiniMaxH3OutputTrimT8", "121 prepared frames to exactly120 /5s, original audio",
            frames=["31", 0], audio=["26", 1], start_seconds=0.0, duration_seconds=5.0, fps=24.0),
        "33": node("MiniMaxH3SafeAVSaveT8Advanced", "Independent external-RGB output", images=["32", 0], audio=["32", 1], filename_prefix="H05/Q05_LTX_Source", crf=18),
        "34": node("PreviewAny", "Observed source clock", source=["2", 4]),
        "35": node("PreviewAny", "Actual nearest map and conform facts", source=["3", 5]),
        "36": node("PreviewAny", "Actual LTX trim and half-canvas", source=["5", 6]),
        "37": node("PreviewAny", "Actual stage source boundary", source=["26", 2]),
        "38": node("PreviewAny", "Actual EAV coverage", source=["27", 1]),
        "39": node("PreviewAny", "Actual Relay coverage", source=["28", 1]),
        "40": node("PreviewAny", "Actual load-policy observations", source=["29", 1]),
        "41": node("PreviewAudio", "Final original-rate audio", audio=["32", 1]),
        "42": node("Video Slice", "Native lazy prefix trim BEFORE resident decode",
            video=["1", 0], start_time=0.0, duration=6.2, strict_duration=True),
    }
    if output_backend == "vhs_external":
        # VHS feeds the original floating PCM directly to FFmpeg. A decoded
        # external AAC source can exceed unity; do not normalize/clip it or
        # weaken the stricter H3 Safe AV exporter to disguise that fact.
        api["33"] = node("VHS_VideoCombine", "External float-PCM output (VHS), not H3 Safe AV certification",
            images=["32", 0], audio=["32", 1], frame_rate=24.0, loop_count=0,
            filename_prefix="H05/Q05_LTX_Source", format="video/h264-mp4", pix_fmt="yuv420p",
            crf=18, save_metadata=False, trim_to_audio=False, pingpong=False, save_output=True)
    native = convert(api, info, "H05_External_RGB_LTX_Source")
    for row in native["nodes"]:
        inputs = api[str(row["id"])]["inputs"]
        entries, unknown = canonical_entries(info[row["type"]], list(inputs))
        if unknown:
            raise ValueError((row["type"], unknown))
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
                named["control_after_generate"] = "fixed"
                cursor += 1
        assert cursor == len(row["widgets_values"])
        row["widgets_values_named"] = named
        if row["type"] == "VHS_VideoCombine":
            # VHS restores dictionaries by widget NAME. Its legacy list
            # migration uses another order and would silently move pix_fmt /
            # crf into pingpong / save_output. Do not change shared converters.
            row["widgets_values"] = dict(named)
        row["size"][0] = 440
        if row["type"] == "PreviewAny":
            row["inputs"][0]["type"] = "STRING"
    rows = {row["id"]: row for row in native["nodes"]}
    for link in native["links"]:
        if rows[link[3]]["type"] == "PreviewAny":
            link[5] = "STRING"
    native["nodes"].append(dict(id=43, type="Note", pos=[0, -360], size=[1440, 280],
        flags={}, order=42, mode=0, inputs=[], outputs=[], properties={},
        title="External RGB source, not a native H3 ancestor or completed cache",
        widgets_values=["NEW independent template; no old graph or Core sampling math changes.\n"
            "Use a legal CFR video longer than6.2s, with real stereo source audio. Local test media are NOT distributed.\n"
            "Native Video Slice selects0-6.2s before resident decode; SourceClock observes actual PTS. Conform maps source origin1s to124 frames at24fps; LTX trims to121.\n"
            f"Final trim delivers120frames /5s at{width}x{height}. Full LTX VAE encodes half-canvas RGB; existing x2 latent lift then three independent Identity updates.\n"
            "This selected VAE needs a32-pixel encoder grid, so the x2 target must be divisible by64; no silent rounding. Aspect-preserving preparation may center-crop.\n"
            "Original AUDIO uses the SAME origin1s and its original sample rate, bypassing the Conform32k and LTX audio paths. AAC is not PCM-exact.\n"
            f"Explicit output backend: {output_backend}. VHS preserves floating PCM input, including source AAC overshoot; it is NOT H3 Safe AV certification. No silent exporter fallback or normalization.\n"
            "Source SHA/geometry is evidence, not a native H3 teacher latent or Cold cache. Both external effects default report_only.\n"
            "Load policy is explicit; actual trained GPU, picture/voice review and general VFR/audio-PTS sync qualification are separate."]))
    native["last_node_id"] = 43
    clean = {key: {field: value[field] for field in ("class_type", "inputs")} for key, value in api.items()}
    return native, clean, check_serialized(native, clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="replace_with_legal_CFR_stereo.mp4")
    parser.add_argument("--load-mode", choices=("legacy_default", "consistent_streaming_exp"), default="legacy_default")
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--target-height", type=int, default=448)
    parser.add_argument("--output-backend", choices=("vhs_external", "safe_av"), default="vhs_external")
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.server)
    if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost", "::1")
            or origin.path not in ("", "/") or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Use an explicit local ComfyUI origin")
    if args.output.exists():
        raise FileExistsError("Never overwrite user workflows or earlier evidence")
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    native, _, serial = graph(info, source=args.source, load_mode=args.load_mode,
                             target_width=args.target_width, target_height=args.target_height,
                             output_backend=args.output_backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.output), serialization=serial, queued=False)))


if __name__ == "__main__":
    main()
