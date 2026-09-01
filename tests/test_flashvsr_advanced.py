from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.flashvsr_advanced import (
    FlashVSRModelHandle,
    build_flashvsr_plan,
    next_8n_plus_5,
    restore_flashvsr,
)
from h3_audio_t8_pkg.flashvsr_vendor.attention import (
    generate_sparge_mask,
    uses_split_k_mask,
)


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.offloads = 0

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["LQ_video"][0]

    def offload_model(self):
        self.offloads += 1


def _handle(pipe):
    return FlashVSRModelHandle(
        pipe=pipe,
        model_dir=Path("X:/models/FlashVSR-v1.1"),
        model_name="FlashVSR-v1.1",
        mode="tiny",
        device=torch.device("cpu"),
        dtype=torch.float32,
        precision="fp16",
        cache_key=("fake",),
    )


def test_padding_uses_the_public_8n_plus_5_contract():
    assert next_8n_plus_5(1) == 21
    assert next_8n_plus_5(21) == 21
    assert next_8n_plus_5(22) == 29
    assert next_8n_plus_5(124) == 125


def test_split_k_lcsa_mask_keeps_exact_topk_per_head_and_time_group():
    generator = torch.Generator().manual_seed(42)
    q_windows = torch.randn(4, 128, 64, generator=generator)
    k_windows = torch.randn(6, 128, 64, generator=generator)
    local = torch.ones(2, 2, dtype=torch.bool)
    mask = generate_sparge_mask(
        1,
        4,
        2,
        q_windows,
        k_windows,
        topk=3,
        local_mask=local,
    )
    assert uses_split_k_mask(k_windows)
    assert mask.shape == (1, 4, 4, 12)
    assert mask.dtype == torch.int8
    assert torch.equal(mask.sum(dim=(-1, -2)), torch.full((1, 4), 6))


def test_quality_locked_plan_never_reduces_the_published_budget():
    frames = torch.rand(22, 32, 48, 3)
    plan, report = build_flashvsr_plan(frames)
    assert plan["quality_profile"] == "quality_locked"
    assert plan["spatial_strategy"] == "full_frame"
    assert plan["memory_policy"] == "resident"
    assert set(map(tuple, plan["budget_values"])) == {(2.0, 3.0, 11)}
    assert json.loads(report)["quality_contract"].startswith("quality_locked")


def test_dynamic_plan_is_explicit_and_guards_first_and_last_chunks():
    frames = torch.zeros(61, 32, 48, 3)
    frames[20:40] = 1.0
    plan, _ = build_flashvsr_plan(frames, quality_profile="balanced_dynamic_exp")
    assert plan["denoise_chunks"] > 2
    assert tuple(plan["budget_values"][0]) == (2.0, 3.0, 11)
    assert tuple(plan["budget_values"][-1]) == (2.0, 3.0, 11)
    assert any("reduced_exp" in row["tier"] for row in plan["chunk_report"][1:-1])


def test_restore_trims_padding_and_passes_the_exact_audio_object():
    frames = torch.rand(5, 64, 64, 3)
    audio = {"waveform": torch.randn(1, 2, 100), "sample_rate": 32000}
    plan, _ = build_flashvsr_plan(frames)
    pipe = _FakePipe()
    result, source, returned_audio, report_text = restore_flashvsr(
        _handle(pipe),
        plan,
        frames,
        audio,
        scale=2,
        color_fix=False,
        release_policy="offload_after",
    )
    report = json.loads(report_text)
    assert result.shape == (5, 128, 128, 3)
    assert torch.equal(source, frames)
    assert returned_audio is audio
    assert report["audio"]["exact_object_passthrough"] is True
    assert report["output"]["frames"] == 5
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["num_frames"] == 25
    assert pipe.offloads == 1


def test_memory_safe_tiles_share_one_seed_and_preserve_audio_identity():
    frames = torch.rand(5, 128, 192, 3)
    audio = {"waveform": torch.randn(1, 2, 100), "sample_rate": 32000}
    plan, _ = build_flashvsr_plan(
        frames,
        quality_profile="memory_safe",
        tile_size=128,
        tile_overlap=16,
    )
    pipe = _FakePipe()
    result, _, returned_audio, report_text = restore_flashvsr(
        _handle(pipe),
        plan,
        frames,
        audio,
        scale=2,
        seed=26083001,
        color_fix=False,
        release_policy="offload_after",
    )
    report = json.loads(report_text)
    assert result.shape == (5, 256, 384, 3)
    assert returned_audio is audio
    assert len(pipe.calls) == 2
    assert {call["seed"] for call in pipe.calls} == {26083001}
    assert report["execution"]["tile_count"] == 2


def test_three_flashvsr_frontend_workflows_are_importable_and_profile_pinned():
    root = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "23-flashvsr"
    expected = {
        "2026-08-30_H3_FlashVSR_Quality_Locked_Advanced_EXP.json": "quality_locked",
        "2026-08-30_H3_FlashVSR_Balanced_Dynamic_EXP_Advanced_EXP.json": (
            "balanced_dynamic_exp"
        ),
        "2026-08-30_H3_FlashVSR_Memory_Safe_Advanced_EXP.json": "memory_safe",
    }
    assert {path.name for path in root.glob("*.json")} == set(expected)
    for filename, profile in expected.items():
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = [node["type"] for node in workflow["nodes"]]
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        assert types.count("MiniMaxH3FlashVSRModelT8Advanced") == 1
        assert types.count("MiniMaxH3FlashVSRExecutionPlanT8Advanced") == 1
        assert types.count("MiniMaxH3FlashVSRRestoreT8Advanced") == 1
        plan_node = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "MiniMaxH3FlashVSRExecutionPlanT8Advanced"
        )
        restore_node = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "MiniMaxH3FlashVSRRestoreT8Advanced"
        )
        assert plan_node["widgets_values"][0] == profile
        assert restore_node["widgets_values"][:3] == [2, 26083001, True]
        components = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "GetVideoComponents"
        )
        assert [(item["name"], item["type"]) for item in components["outputs"]] == [
            ("images", "IMAGE"),
            ("audio", "AUDIO"),
            ("fps", "FLOAT"),
            ("bit_depth", "COMBO"),
            ("color_space", "COMBO"),
        ]
        note = next(
            node["widgets_values"][0]
            for node in workflow["nodes"]
            if node["type"] == "MarkdownNote"
        )
        for required in ("FlashVSR v1.1", "AUDIO", "2.0/3.0/11"):
            assert required in note


def test_flashvsr_nodes_are_append_only_at_the_registration_tail():
    node_classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in node_classes]
    assert ids[-14:-11] == [
        "MiniMaxH3FlashVSRModelT8Advanced",
        "MiniMaxH3FlashVSRExecutionPlanT8Advanced",
        "MiniMaxH3FlashVSRRestoreT8Advanced",
    ]
    assert ids[-11] == "MiniMaxH3LongVideoSamplingPlanT8Advanced"
    assert ids[-10] == "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
    assert ids[-9] == "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
    assert ids[-8] == "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
    assert ids[-7] == "MiniMaxH3SubjectSafeRGBCompositeT8Advanced"
