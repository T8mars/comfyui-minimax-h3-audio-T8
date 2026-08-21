#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import uuid


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
CATEGORY = ROOT / "examples" / "workflows" / "13-latent-upscale"
USER_CATEGORY = (
    COMFY_ROOT
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "13-latent-upscale"
)
BASE = CATEGORY / "2026-08-19_H3_Learned_Latent_TwoPass_I2VA_Advanced_EXP.json"

SPEECH = "All the time he was talking to me, his angry little eyes were following Lake."
SOURCE_AUDIO = "h3_twopass_voice_5683_5p152s.flac"
SEED = 2608215001

VARIANTS = (
    {
        "filename": "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Lock_Source_Advanced_EXP.json",
        "label": "Hybrid lock_source",
        "task_type": "Hybrid",
        "audio_mode": "lock_source",
        "strength": 0.0,
        "add_source": True,
        "ordinal": 1,
        "load_audio": True,
        "save_mux_audio": True,
        "prefix": "MiniMaxH3/learned_twopass_hybrid_lock_source_speech",
        "prompt": (
            "Locked-off medium close-up of the woman facing camera. She speaks in precise "
            f"synchronization with <Audio 1>: <d>{SPEECH}</d> Preserve the exact source timing, "
            "phoneme rhythm, pauses, and natural lip movement. Minimal head motion, no music, "
            "no subtitles."
        ),
        "audio_contract": (
            "The source is encoded into both LOW and HIGH Conditioning with a zero audio mask. "
            "The final Save node must use HIGH Conditioning `mux_audio`, not AV Decode audio, so "
            "the delivered waveform remains the loaded source."
        ),
    },
    {
        "filename": "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Remix_Source_020_Advanced_EXP.json",
        "label": "Hybrid remix_source 0.20",
        "task_type": "Hybrid",
        "audio_mode": "remix_source",
        "strength": 0.20,
        "add_source": True,
        "ordinal": 1,
        "load_audio": True,
        "save_mux_audio": False,
        "prefix": "MiniMaxH3/learned_twopass_hybrid_remix_source_020_speech",
        "prompt": (
            "Locked-off medium close-up of the woman facing camera. Remix the source voice lightly "
            f"while preserving exact timing, phoneme rhythm, and pauses from <Audio 1>: <d>{SPEECH}</d> "
            "Keep visible lip movements precisely synchronized. Minimal head motion, no music, "
            "no subtitles."
        ),
        "audio_contract": (
            "The source initializes audio in both passes with denoise strength 0.20. Save the "
            "generated AV Decode audio. Strength 0.20 is still a model remix, not a transparent "
            "source bypass, so listen for timbre and high-frequency loss."
        ),
    },
    {
        "filename": "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Reference_Only_Advanced_EXP.json",
        "label": "Hybrid reference_only",
        "task_type": "Hybrid",
        "audio_mode": "reference_only",
        "strength": 1.0,
        "add_source": True,
        "ordinal": 1,
        "load_audio": True,
        "save_mux_audio": False,
        "prefix": "MiniMaxH3/learned_twopass_hybrid_reference_only_speech",
        "prompt": (
            "Locked-off medium close-up of the woman facing camera. Use <Audio 1> only as rhythm "
            f"and pacing reference and generate a new voice saying exactly: <d>{SPEECH}</d> Keep "
            "visible lip movements precisely synchronized to the generated speech. Minimal head "
            "motion, no music, no subtitles."
        ),
        "audio_contract": (
            "The loaded source is registered as `<Audio 1>` reference, while the target audio latent "
            "starts blank and is regenerated. Save AV Decode audio. This is not equivalent to "
            "`remix_source=1`; the strength widget is not the remix control in this mode."
        ),
    },
    {
        "filename": "2026-08-21_H3_Learned_Latent_TwoPass_I2VA_Native_Speech_Advanced_EXP.json",
        "label": "I2VA native speech",
        "task_type": "I2VA",
        "audio_mode": "native",
        "strength": 1.0,
        "add_source": False,
        "ordinal": 0,
        "load_audio": False,
        "save_mux_audio": False,
        "prefix": "MiniMaxH3/learned_twopass_i2va_native_speech",
        "prompt": (
            "Locked-off medium close-up of the woman facing camera. She says clearly: "
            f"<d>{SPEECH}</d> Natural, precise synchronized lip movements. Minimal head motion, "
            "no music, no subtitles."
        ),
        "audio_contract": (
            "No drive/reference audio is connected. LOW and HIGH Conditioning both use native "
            "audio generation, pass 2 remains unmasked, and Save receives generated AV Decode audio."
        ),
    },
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


def _set_conditioning(node: dict, variant: dict) -> None:
    values = node["widgets_values"]
    values[0] = variant["prompt"]
    values[4] = variant["task_type"]
    values[5] = variant["audio_mode"]
    values[6] = variant["strength"]
    values[7] = variant["add_source"]
    values[8] = variant["ordinal"]
    node["title"] = f"{node['title'].split(' Conditioning')[0]} Conditioning · {variant['label']}"


def _audio_node(order: int) -> dict:
    return {
        "id": 27,
        "type": "LoadAudio",
        "title": "Drive/reference audio · replace as needed",
        "pos": [0, 1360],
        "size": [390, 120],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {
                "name": "audio",
                "type": "COMBO",
                "widget": {"name": "audio"},
                "link": None,
            }
        ],
        "outputs": [{"name": "AUDIO", "type": "AUDIO", "links": [41, 42]}],
        "properties": {"Node name for S&R": "LoadAudio"},
        "widgets_values": [SOURCE_AUDIO],
    }


def _set_note(node: dict, title: str, text: str) -> None:
    node["title"] = title
    node["widgets_values"] = [text]


def build_variant(base: dict, variant: dict) -> dict:
    workflow = copy.deepcopy(base)
    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"minimax-h3-audio-t8/{variant['filename']}")
    )
    workflow["revision"] = 0

    low = _node(workflow, 7)
    high = _node(workflow, 14)
    _set_conditioning(low, variant)
    _set_conditioning(high, variant)

    _node(workflow, 6)["title"] = "I2VA/Hybrid first frame"
    _node(workflow, 11)["widgets_values"][0] = SEED
    _node(workflow, 18)["widgets_values"][0] = SEED
    _node(workflow, 16)["widgets_values"][-1] = SEED
    _node(workflow, 15)["title"] = "Validate HIGH contract · pass 2 owns final AV"
    _node(workflow, 21)["widgets_values"][2] = variant["prefix"]

    if variant["load_audio"]:
        workflow["nodes"].append(
            _audio_node(max(node.get("order", 0) for node in workflow["nodes"]) + 1)
        )
        low["inputs"][15]["link"] = 41
        high["inputs"][15]["link"] = 42
        workflow["links"].extend(
            ([41, 27, 0, 7, 15, "AUDIO"], [42, 27, 0, 14, 15, "AUDIO"])
        )
        workflow["last_node_id"] = 27
        workflow["last_link_id"] = 42
    else:
        workflow["last_node_id"] = 26
        workflow["last_link_id"] = 40

    if variant["save_mux_audio"]:
        workflow["links"] = [
            [40, 14, 2, 21, 7, "AUDIO"] if link[0] == 40 else link
            for link in workflow["links"]
        ]
        _node(workflow, 14)["outputs"][2]["links"] = [40]
        _node(workflow, 20)["outputs"][1]["links"] = None
        _node(workflow, 21)["title"] = "Save synchronized MP4 · source mux_audio"
    else:
        _node(workflow, 14)["outputs"][2]["links"] = None
        _node(workflow, 20)["outputs"][1]["links"] = [40]
        _node(workflow, 21)["title"] = "Save synchronized MP4 · generated AV audio"

    _set_note(
        _node(workflow, 22),
        f"1 · {variant['label']} · tested route",
        (
            f"## 1 · {variant['label']} / 已审核路线\n"
            "This workflow preserves the reviewed 4+3 learned two-pass route: LOW 736x416x124, "
            "2x learned video-latent upscale, HIGH 1472x832x124, shift 12/3, and seed 2608215001. "
            "Replace the first image and, where present, the loaded audio for your own material."
        ),
    )
    _set_note(
        _node(workflow, 23),
        "2 · Automatic size synchronization",
        (
            "## 2 · Automatic size synchronization / 尺寸自动同步\n"
            "Change only Learned Latent Upscale `scale_by`. Its width/height outputs feed HIGH "
            "Conditioning. Do not maintain a second manual canvas."
        ),
    )
    _set_note(
        _node(workflow, 24),
        "3 · Audio ownership and final connection",
        f"## 3 · Audio ownership / 音频归属\n{variant['audio_contract']}",
    )
    _set_note(
        _node(workflow, 25),
        "4 · Keep optional detail controls off",
        (
            "## 4 · Optional detail controls / 可选细节控制\n"
            "Keep Tail, model-time Bias, STG, and Restart OFF to preserve the reviewed route. "
            "Enabling any of them changes joint AV prediction and requires a fresh full audio/video review."
        ),
    )
    _set_note(
        _node(workflow, 26),
        "5 · Bounded validation result",
        (
            "## 5 · Bounded validation / 有限验证\n"
            "The four saved variants completed real HEVC/AAC generation and one reviewer approved "
            "their complete audio and visible lip motion. This is evidence for one image, utterance, "
            "seed, model, and reviewer—not a universal lip-sync, voice-identity, quality, or 16GB claim."
        ),
    )
    return workflow


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    USER_CATEGORY.mkdir(parents=True, exist_ok=True)
    user_base = USER_CATEGORY / BASE.name
    shutil.copyfile(BASE, user_base)
    print(user_base)
    for variant in VARIANTS:
        workflow = build_variant(base, variant)
        payload = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
        project_path = CATEGORY / variant["filename"]
        user_path = USER_CATEGORY / variant["filename"]
        project_path.write_text(payload, encoding="utf-8")
        user_path.write_text(payload, encoding="utf-8")
        print(project_path)
        print(user_path)
    user_readme = USER_CATEGORY / "README.md"
    shutil.copyfile(CATEGORY / "README.md", user_readme)
    print(user_readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
