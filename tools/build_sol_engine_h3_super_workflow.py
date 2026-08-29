from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "workflows" / "22-sol-engine-h3-super"
OUT_FILE = OUT_DIR / "2026-08-29_H3_Sol_Engine_Super_Acceleration_LTX25_Advanced_EXP.json"
USER_DIR = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "22-sol-engine-h3-super"
)


def socket(name: str, kind: str, link=None, *, widget: bool = False, shape=None):
    value = {"name": name, "type": kind, "link": link}
    if widget:
        value["widget"] = {"name": name}
    if shape is not None:
        value["shape"] = shape
    return value


def output(name: str, kind: str, links=None):
    return {"name": name, "type": kind, "links": links}


def node(
    node_id: int,
    node_type: str,
    title: str,
    pos: tuple[int, int],
    size: tuple[int, int],
    order: int,
    *,
    inputs=None,
    outputs=None,
    widgets=None,
    properties=None,
    color=None,
    bgcolor=None,
):
    # Native ComfyUI frontend saves widget-backed values in widgets_values and
    # serializes an input socket only after that widget has been converted to
    # an input or linked. Keeping unlinked widget descriptors here breaks
    # strict third-party workflow importers and shifts later linked sockets.
    serialized_inputs = [
        item
        for item in (inputs or [])
        if not ("widget" in item and item.get("link") is None)
    ]
    value = {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": serialized_inputs,
        "outputs": outputs or [],
        "properties": properties or {
            "cnr_id": "comfyui-minimax-h3-audio-T8",
            "Node name for S&R": node_type,
        },
        "widgets_values": widgets or [],
    }
    if color:
        value["color"] = color
    if bgcolor:
        value["bgcolor"] = bgcolor
    return value


def note(node_id: int, title: str, text: str, pos, size, color="#243", bgcolor="#354"):
    return node(
        node_id,
        "MarkdownNote",
        title,
        pos,
        size,
        0,
        widgets=[text],
        properties={},
        color=color,
        bgcolor=bgcolor,
    )


def build_workflow():
    links = [
        [1, 1, 0, 2, 0, "VIDEO"],
        [2, 2, 0, 3, 0, "IMAGE"],
        [3, 2, 2, 3, 1, "FLOAT"],
        [4, 3, 0, 5, 0, "IMAGE"],
        [5, 4, 0, 5, 1, "VAE"],
        [6, 5, 0, 7, 0, "LATENT"],
        [7, 6, 0, 7, 1, "LATENT_UPSCALE_MODEL"],
        [8, 4, 0, 7, 2, "VAE"],
        [9, 8, 0, 9, 0, "MODEL"],
        [10, 9, 0, 10, 0, "MODEL"],
        [11, 10, 0, 14, 0, "MODEL"],
        [12, 10, 1, 16, 2, "SAMPLER"],
        [13, 10, 2, 16, 3, "SIGMAS"],
        [14, 11, 0, 12, 0, "CLIP"],
        [15, 11, 0, 13, 0, "CLIP"],
        [16, 12, 0, 26, 0, "CONDITIONING"],
        [17, 13, 0, 26, 1, "CONDITIONING"],
        [18, 14, 0, 16, 1, "GUIDER"],
        [19, 15, 0, 16, 0, "NOISE"],
        [20, 7, 0, 16, 4, "LATENT"],
        [21, 16, 0, 17, 0, "LATENT"],
        [22, 25, 0, 17, 1, "T8_SOL_ENGINE_TAEHV"],
        [23, 17, 0, 18, 0, "IMAGE"],
        [24, 3, 5, 18, 1, "FLOAT"],
        [25, 2, 2, 18, 2, "FLOAT"],
        [26, 2, 1, 18, 3, "AUDIO"],
        [27, 18, 0, 19, 0, "IMAGE"],
        [28, 18, 1, 19, 1, "AUDIO"],
        [29, 2, 2, 19, 2, "FLOAT"],
        [30, 2, 3, 19, 3, "COMBO"],
        [31, 19, 0, 20, 0, "VIDEO"],
        [32, 26, 0, 14, 1, "CONDITIONING"],
        [33, 26, 1, 14, 2, "CONDITIONING"],
    ]
    nodes = [
        node(
            1,
            "LoadVideo",
            "1. Load the decoded MiniMax H3 draft / 载入H3草稿视频",
            (0, 0),
            (400, 124),
            0,
            outputs=[output("VIDEO", "VIDEO", [1])],
            widgets=["replace_with_h3_draft_24fps.mp4"],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "LoadVideo"},
        ),
        node(
            2,
            "GetVideoComponents",
            "2. Split frames; audio stays outside LTX / 拆帧且音频旁路",
            (450, 0),
            (410, 126),
            1,
            inputs=[socket("video", "VIDEO", 1)],
            outputs=[
                output("images", "IMAGE", [2]),
                output("audio", "AUDIO", [26]),
                output("fps", "FLOAT", [3, 25, 29]),
                output("bit_depth", "COMBO", [30]),
                output("color_space", "COMBO", None),
            ],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "GetVideoComponents"},
        ),
        node(
            3,
            "MiniMaxH3SolEngineDraftToLTXT8Advanced",
            "3. H3 draft → LTX-2.5 refiner input (official defaults)",
            (920, 0),
            (460, 250),
            2,
            inputs=[
                socket("frames", "IMAGE", 2),
                socket("target_width", "INT", None, widget=True),
                socket("target_height", "INT", None, widget=True),
                socket("frame_policy", "COMBO", None, widget=True),
                socket("fps", "FLOAT", 3, widget=True),
            ],
            outputs=[
                output("ltx_encoder_frames", "IMAGE", [4]),
                output("encoder_width", "INT", None),
                output("encoder_height", "INT", None),
                output("kept_frames", "INT", None),
                output("dropped_tail_frames", "INT", None),
                output("output_duration_seconds", "FLOAT", [24]),
                output("report_json", "STRING", None),
            ],
            widgets=[1920, 1088, "trim_to_8n_plus_1", 24.0],
        ),
        node(
            4,
            "VAELoader",
            "LTX-2.5 original video VAE (required)",
            (920, 330),
            (460, 70),
            3,
            outputs=[output("VAE", "VAE", [5, 8])],
            widgets=["ltx-2.5-video-vae-conv-bf16.safetensors"],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": "VAELoader",
                "models": [{
                    "name": "ltx-2.5-video-vae-conv-bf16.safetensors",
                    "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                    "directory": "vae",
                }],
            },
        ),
        node(
            5,
            "VAEEncode",
            "4. Full LTX-2.5 Video VAE encode (required)",
            (1440, 0),
            (380, 132),
            4,
            inputs=[
                socket("pixels", "IMAGE", 4),
                socket("vae", "VAE", 5),
            ],
            outputs=[output("LATENT", "LATENT", [6])],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "VAEEncode"},
        ),
        node(
            6,
            "LatentUpscaleModelLoader",
            "Official LTX-2.5 x2 latent upscaler",
            (1440, 180),
            (420, 82),
            5,
            outputs=[output("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL", [7])],
            widgets=["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": "LatentUpscaleModelLoader",
                "models": [{
                    "name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                    "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                    "directory": "latent_upscale_models",
                }],
            },
        ),
        node(
            7,
            "LTXVLatentUpsampler",
            "5. Learned latent x2 upscale",
            (1910, 40),
            (320, 100),
            6,
            inputs=[
                socket("samples", "LATENT", 6),
                socket("upscale_model", "LATENT_UPSCALE_MODEL", 7),
                socket("vae", "VAE", 8),
            ],
            outputs=[output("LATENT", "LATENT", [20])],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "LTXVLatentUpsampler"},
        ),
        node(
            8,
            "UNETLoader",
            "LTX-2.5 dev transformer (official Comfy INT8 local default)",
            (0, 530),
            (470, 86),
            7,
            outputs=[output("MODEL", "MODEL", [9])],
            widgets=["ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", "default"],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": "UNETLoader",
                "models": [{
                    "name": "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
                    "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
                    "directory": "diffusion_models",
                }],
            },
        ),
        node(
            9,
            "LoraLoaderModelOnly",
            "Official distilled refiner LoRA — fixed strength 0.8",
            (530, 530),
            (450, 92),
            8,
            inputs=[socket("model", "MODEL", 9)],
            outputs=[output("MODEL", "MODEL", [10])],
            widgets=["ltx-2.5-22b-distilled-lora-450-bf16.safetensors", 0.8],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": "LoraLoaderModelOnly",
                "models": [{
                    "name": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
                    "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
                    "directory": "loras",
                }],
            },
        ),
        node(
            10,
            "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced",
            "6. Official 3-step Euler + per-step Sol-Attn tau",
            (1040, 510),
            (500, 238),
            9,
            inputs=[
                socket("model", "MODEL", 10),
                socket("enabled", "BOOLEAN", None, widget=True),
                socket("attention_backend", "COMBO", None, widget=True),
                socket("min_tokens", "INT", None, widget=True),
                socket("kernel_precision", "COMBO", None, widget=True),
                socket("verbose", "BOOLEAN", None, widget=True),
            ],
            outputs=[
                output("model", "MODEL", [11]),
                output("sampler", "SAMPLER", [12]),
                output("sigmas", "SIGMAS", [13]),
                output("refiner_lora_strength", "FLOAT", None),
                output("report_json", "STRING", None),
            ],
            widgets=[True, "auto_sol_attn", 4096, "bf16_official", False],
        ),
        node(
            11,
            "CLIPLoader",
            "LTX-2.5 text encoder",
            (0, 730),
            (470, 110),
            10,
            outputs=[output("CLIP", "CLIP", [14, 15])],
            widgets=["gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "ltxv", "default"],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": "CLIPLoader",
                "models": [{
                    "name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                    "url": "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                    "directory": "text_encoders",
                }],
            },
        ),
        node(
            12,
            "CLIPTextEncode",
            "Use the same positive prompt as the H3 draft",
            (530, 720),
            (500, 160),
            11,
            inputs=[socket("clip", "CLIP", 14), socket("text", "STRING", None, widget=True)],
            outputs=[output("CONDITIONING", "CONDITIONING", [16])],
            widgets=["Replace this text with the exact prompt used to generate the MiniMax H3 draft."],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "CLIPTextEncode"},
            color="#232",
            bgcolor="#353",
        ),
        node(
            13,
            "CLIPTextEncode",
            "Negative prompt (official default: empty)",
            (530, 910),
            (500, 120),
            12,
            inputs=[socket("clip", "CLIP", 15), socket("text", "STRING", None, widget=True)],
            outputs=[output("CONDITIONING", "CONDITIONING", [17])],
            widgets=[""],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "CLIPTextEncode"},
            color="#322",
            bgcolor="#533",
        ),
        node(
            14,
            "CFGGuider",
            "CFG 1.0",
            (1600, 510),
            (320, 150),
            13,
            inputs=[
                socket("model", "MODEL", 11),
                socket("positive", "CONDITIONING", 32),
                socket("negative", "CONDITIONING", 33),
                socket("cfg", "FLOAT", None, widget=True),
            ],
            outputs=[output("GUIDER", "GUIDER", [18])],
            widgets=[1.0],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "CFGGuider"},
        ),
        node(
            15,
            "RandomNoise",
            "Stage-2 noise",
            (1600, 700),
            (320, 90),
            14,
            inputs=[socket("noise_seed", "INT", None, widget=True)],
            outputs=[output("NOISE", "NOISE", [19])],
            widgets=[42, "fixed"],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "RandomNoise"},
        ),
        node(
            16,
            "SamplerCustomAdvanced",
            "7. LTX-2.5 refiner — exactly 3 Euler updates",
            (2290, 430),
            (360, 210),
            15,
            inputs=[
                socket("noise", "NOISE", 19),
                socket("guider", "GUIDER", 18),
                socket("sampler", "SAMPLER", 12),
                socket("sigmas", "SIGMAS", 13),
                socket("latent_image", "LATENT", 20),
            ],
            outputs=[output("output", "LATENT", [21]), output("denoised_output", "LATENT", None)],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "SamplerCustomAdvanced"},
        ),
        node(
            17,
            "MiniMaxH3SolEngineTAEHVDecodeT8Advanced",
            "8. Official TAEHV Wide decode",
            (2720, 430),
            (360, 190),
            16,
            inputs=[
                socket("latent", "LATENT", 21),
                socket("taehv", "T8_SOL_ENGINE_TAEHV", 22),
                socket("execution_mode", "COMBO", None, widget=True),
                socket("precision", "COMBO", None, widget=True),
            ],
            outputs=[
                output("frames", "IMAGE", [23]),
                output("report_json", "STRING", None),
            ],
            widgets=["auto_official", "bf16_official"],
        ),
        node(
            18,
            "MiniMaxH3OutputTrimT8",
            "9. Trim bypassed H3 audio to the retained video duration",
            (3140, 400),
            (470, 235),
            17,
            inputs=[
                socket("frames", "IMAGE", 23),
                socket("start_seconds", "FLOAT", None, widget=True),
                socket("duration_seconds", "FLOAT", 24, widget=True),
                socket("fps", "FLOAT", 25, widget=True),
                socket("audio", "AUDIO", 26, shape=7),
            ],
            outputs=[
                output("frames", "IMAGE", [27]),
                output("audio", "AUDIO", [28]),
                output("report_json", "STRING", None),
            ],
            widgets=[0.0, 10.041667, 24.0],
        ),
        node(
            19,
            "CreateVideo",
            "10. Mux refined frames with untouched H3 audio",
            (3680, 410),
            (410, 180),
            18,
            inputs=[
                socket("images", "IMAGE", 27),
                socket("audio", "AUDIO", 28, shape=7),
                socket("fps", "FLOAT", 29, widget=True),
                socket("bit_depth", "COMBO", 30, widget=True, shape=7),
            ],
            outputs=[output("VIDEO", "VIDEO", [31])],
            widgets=[24.0, 8],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "CreateVideo"},
        ),
        node(
            20,
            "SaveVideo",
            "11. Save H3 Super candidate",
            (4160, 410),
            (470, 165),
            19,
            inputs=[socket("video", "VIDEO", 31)],
            outputs=[output("video", "VIDEO", None)],
            widgets=["MiniMaxH3/sol_engine_h3_super_candidate", "auto", "auto"],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "SaveVideo"},
        ),
        node(
            25,
            "MiniMaxH3SolEngineTAEHVLoaderT8Advanced",
            "TAEHV Wide codec used by NVIDIA Stage 2",
            (1440, 310),
            (430, 90),
            20,
            inputs=[socket("model_name", "COMBO", None, widget=True)],
            outputs=[output("taehv", "T8_SOL_ENGINE_TAEHV", [22])],
            widgets=["taeltx2_3_wide.pth"],
            properties={
                "cnr_id": "comfyui-minimax-h3-audio-T8",
                "Node name for S&R": "MiniMaxH3SolEngineTAEHVLoaderT8Advanced",
                "models": [{
                    "name": "taeltx2_3_wide.pth",
                    "url": "https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4/taeltx2_3_wide.pth",
                    "directory": "taehv",
                }],
            },
        ),
        node(
            26,
            "LTXVConditioning",
            "LTX frame-rate conditioning — official 24 fps",
            (1090, 820),
            (390, 130),
            21,
            inputs=[
                socket("positive", "CONDITIONING", 16),
                socket("negative", "CONDITIONING", 17),
                socket("frame_rate", "FLOAT", None, widget=True),
            ],
            outputs=[
                output("positive", "CONDITIONING", [32]),
                output("negative", "CONDITIONING", [33]),
            ],
            widgets=[24.0],
            properties={"cnr_id": "comfy-core", "Node name for S&R": "LTXVConditioning"},
        ),
        note(
            21,
            "What this workflow is / 这是什么",
            "## NVIDIA H3 Super Acceleration — ComfyUI adaptation\n\n"
            "This is a **two-model, two-stage pipeline**: MiniMax H3 makes the draft in 4 steps; "
            "LTX-2.5 refines the decoded video in 3 Euler steps. Total: 7 model updates. It is not "
            "a seven-step H3 sampler and it is not lossless. The published 22.2× number is a fixed "
            "4×GB200 benchmark, not a consumer-GPU promise.",
            (0, -360),
            (920, 280),
        ),
        note(
            22,
            "Official fixed settings / 官方固定设置",
            "## Do not change these for the parity route\n\n"
            "- H3 draft: 896×512, 4 steps, 24 fps (the published demo later center-crops 864×480).\n"
            "- Stage 2 target: 1920×1088; input is encoded at 960×544 with the **full LTX-2.5 Video VAE**, then latent-upscaled ×2.\n"
            "- TAEHV is used only for the final fast decode; never feed a TAEHV-encoded latent into the Refiner.\n"
            "- LTX sigmas: `0.909375 → 0.725 → 0.421875 → 0`.\n"
            "- Euler, CFG 1, refiner LoRA strength 0.8.\n"
            "- Self-attention layer 0 dense; layers 1–47 use tau 1.0 / 1.25 / 1.5.\n"
            "- H3 audio bypasses LTX Stage 2 and is only trimmed/muxed.",
            (980, -360),
            (980, 300),
        ),
        note(
            23,
            "Required models / 必需模型",
            "## Required LTX-2.5 assets\n\n"
            "1. `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors`\n"
            "2. `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`\n"
            "3. `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`\n"
            "4. `ltx-2.5-video-vae-conv-bf16.safetensors` (**required encoder and upscaler statistics**)\n"
            "5. `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`\n"
            "6. `taeltx2_3_wide.pth` (**final decode only**; put in `models/taehv`)\n\n"
            "Optional acceleration: Kijai `ComfyUI-SolAttn_triton`. If it is absent or not eligible, "
            "the node keeps dense attention and does not block execution. Model names, hashes, byte "
            "sizes and output pixel area are never used as execution gates. The default Transformer and "
            "text encoder are official Comfy INT8 releases for local feasibility; selecting BF16 recreates "
            "the unquantized NVIDIA model policy but requires substantially more disk and memory.",
            (2020, -360),
            (1040, 300),
        ),
        note(
            24,
            "Usage and limits / 用法与边界",
            "## Use\n\n"
            "Generate the H3 draft first, load that MP4 here, and paste the **same positive prompt** "
            "into the LTX text encoder. The official reference input has 243 frames; this workflow "
            "drops the last two to 241 (8n+1). Other sizes use the same formula but are not the "
            "published benchmark. Large outputs need substantial VRAM/RAM; choose geometry yourself. "
            "No project pixel ceiling is imposed.",
            (3120, -360),
            (1030, 280),
        ),
    ]
    return {
        "id": "6f0cf708-3d83-47bc-bd2d-79b30dcc7bbb",
        "revision": 0,
        "last_node_id": 26,
        "last_link_id": 33,
        "nodes": nodes,
        "links": links,
        "groups": [
            {"id": 1, "title": "H3 draft input and deterministic handoff", "bounding": [-40, -60, 2310, 500], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 2, "title": "LTX-2.5 official 3-step refiner", "bounding": [-40, 460, 3140, 620], "color": "#8b5a9e", "font_size": 24, "flags": {}},
            {"id": 3, "title": "Decode, exact audio bypass trim and delivery", "bounding": [2660, 340, 2020, 360], "color": "#4b8b57", "font_size": 24, "flags": {}},
        ],
        "config": {},
        "extra": {"ds": {"scale": 0.72, "offset": [120, 390]}, "frontendVersion": "1.26.7"},
        "version": 0.4,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n"
    OUT_FILE.write_text(payload, encoding="utf-8")
    shutil.copy2(OUT_FILE, USER_DIR / OUT_FILE.name)
    print(OUT_FILE)
    print(USER_DIR / OUT_FILE.name)


if __name__ == "__main__":
    main()
