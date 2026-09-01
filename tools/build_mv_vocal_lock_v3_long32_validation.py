from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "artifacts/mv-vocal-lock-v2-long32-validation-20260901/"
    "mv_vocal_lock_v2_long32_api.json"
)
OUTPUT = (
    ROOT
    / "artifacts/mv-vocal-lock-v3-long32-validation-20260901/"
    "mv_vocal_lock_v3_long32_api.json"
)
SCENE02_OUTPUT = (
    ROOT
    / "artifacts/mv-vocal-lock-v3-long32-validation-20260901/"
    "mv_vocal_lock_v3_scene02_official_ref2v_api.json"
)

OFFICIAL_REF2V_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


SCENE_DIRECTIONS = [
    {
        "camera": "locked frontal medium close-up at eye level",
        "lighting": "soft cool key light over a plain dark blue-gray studio wall",
        "performance": "maintains direct eye-line with restrained natural head movement",
        "emotion": "calm and attentive",
    },
    {
        "camera": "locked left three-quarter medium close-up at eye level",
        "lighting": "soft neutral key light over a plain non-figurative studio wall",
        "performance": "turns only slightly while keeping both lips continuously visible",
        "emotion": "focused and sincere",
    },
    {
        "camera": "locked tighter frontal medium close-up at eye level",
        "lighting": "clean soft frontal light over a simple dark blue-gray studio wall",
        "performance": "keeps shoulders steady and uses subtle facial expression",
        "emotion": "clear and engaged",
    },
    {
        "camera": "locked right three-quarter medium close-up at eye level",
        "lighting": "soft warm-neutral key light over a plain non-figurative studio wall",
        "performance": "keeps the complete mouth visible with minimal head rotation",
        "emotion": "confident and composed",
    },
    {
        "camera": "locked frontal medium close-up at eye level",
        "lighting": "balanced soft key and fill over a plain blue-gray studio wall",
        "performance": "returns to direct eye-line with restrained natural expression",
        "emotion": "resolved and present",
    },
]


def build() -> dict:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow["2"]["inputs"]["lora_name"] = OFFICIAL_REF2V_LORA
    workflow["2"]["inputs"]["strength_model"] = 1.0
    workflow["2"]["_meta"]["title"] = (
        "Official Ref2V Turbo v0.1 ComfyUI LoRA · strength 1.0"
    )
    director = workflow["10"]
    director["class_type"] = "MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced"
    director["_meta"]["title"] = "V3 explicit five-scene single-subject visual direction"
    director["inputs"] = {
        "scene_plan": ["9", 0],
        "global_creative_prompt": (
            "A coherent five-shot performance film of the exact same lead performer. Every scene "
            "is a clean intentional hard-cut studio setup, never a morph or dissolve. Identity, "
            "hair, black top, facial geometry, lighting continuity, and crisp subject boundaries "
            "remain consistent across the film."
        ),
        "performer_description": (
            "the exact same front-facing young woman with long straight dark-brown hair and a black "
            "top shown in the reference picture"
        ),
        "visual_style": (
            "cinematic realistic performance portrait, natural skin texture, crisp eyelashes and "
            "hair strands, clean face contour, high local contrast"
        ),
        "scene_directions_json": json.dumps(SCENE_DIRECTIONS, ensure_ascii=False),
        "vocal_content_type": "spoken_dialogue",
        "vocal_language": "Chinese",
        "exact_vocal_text_json": "",
        "non_vocal_action": "keeps the lips naturally closed and breathes softly with the rhythm",
    }

    workflow["12"] = {
        "inputs": {
            "clip": ["3", 0],
            "mode": "memory_lru_exp",
            "max_entries": 1,
            "maximum_cache_mib": 1024.0,
            "cache_epoch": 260901,
        },
        "class_type": "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
        "_meta": {"title": "Bounded same-reference Qwen prefix cache"},
    }

    renderer = workflow["11"]
    renderer["class_type"] = "MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced"
    renderer["_meta"]["title"] = "New V3 serial 32-second five-scene stage gate"
    renderer["inputs"]["clip"] = ["12", 0]
    # r1-r3 used the wrong generic LarryVrh EMA adapter and an 8-step/6:3 schedule.
    # r4 starts a new contract with the official Ref2V Turbo4 adapter and recipe.
    renderer["inputs"]["chain_id"] = "mv_vocal_lock_v3_long32_5scene_20260901_r4_official_ref2v"
    renderer["inputs"]["width"] = 1024
    renderer["inputs"]["height"] = 768
    renderer["inputs"]["steps"] = 4
    renderer["inputs"]["shift_video"] = 12.0
    renderer["inputs"]["shift_audio"] = 3.0
    renderer["inputs"]["sampler_name"] = "euler"
    renderer["inputs"]["scheduler"] = "simple"
    renderer["inputs"]["filename_prefix"] = "MiniMaxH3_MV_VocalLock_V3_Long32_5Scene"
    renderer["inputs"]["model_id"] = (
        "minimax_h3_ref2va_int8_convrot+official-ref2v-turbo4-v0.1-vocal-lock-v3-long32"
    )
    return workflow


def build_scene02_probe() -> dict:
    """Reproduce the failed second shot with only the corrected model recipe changed."""
    workflow = build()
    workflow["7"]["inputs"]["audio"] = "mv_vocal_lock_scene02_full_mix_zh.wav"
    workflow["8"]["inputs"]["audio"] = "mv_vocal_lock_scene02_vocal_zh.wav"
    workflow["9"]["inputs"]["manual_boundaries_json"] = ""
    workflow["10"]["inputs"]["global_creative_prompt"] = (
        "A single-shot performance test of the exact same lead performer. The scene is a clean "
        "studio setup with no morph, dissolve, duplicate face, reflection or secondary person."
    )
    workflow["10"]["inputs"]["scene_directions_json"] = json.dumps(
        [SCENE_DIRECTIONS[1]], ensure_ascii=False
    )
    renderer = workflow["11"]
    renderer["inputs"]["chain_id"] = (
        "mv_vocal_lock_v3_scene02_20260901_r1_official_ref2v_same_seed"
    )
    renderer["inputs"]["base_seed"] = 2609013202
    renderer["inputs"]["filename_prefix"] = (
        "MiniMaxH3_MV_VocalLock_V3_Scene02_OfficialRef2V"
    )
    renderer["inputs"]["model_id"] = (
        "minimax_h3_ref2va_int8_convrot+official-ref2v-turbo4-v0.1-scene02-isolation"
    )
    return workflow


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    SCENE02_OUTPUT.write_text(
        json.dumps(build_scene02_probe(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(SCENE02_OUTPUT)


if __name__ == "__main__":
    main()
