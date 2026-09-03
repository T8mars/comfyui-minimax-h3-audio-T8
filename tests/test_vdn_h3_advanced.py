from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import nodes_vdn_h3_advanced as nodes_vdn
from h3_audio_t8_pkg import vdn_h3_advanced as vdn


def _layout(*, frames: int = 8, per_frame: int = 2) -> vdn.VDNSequenceLayout:
    video_start = 3
    return vdn.VDNSequenceLayout(
        seq_len=video_start + frames * per_frame,
        video_start=video_start,
        num_frames=frames,
        tokens_per_frame=per_frame,
        frame_height=1,
        frame_width=per_frame,
        text_start=0,
        text_len=1,
    )


def _dense_mask_reference(q, k, value, layout, bounds, scale):
    allowed = torch.ones((layout.seq_len, layout.seq_len), dtype=torch.bool)
    vs = layout.video_start
    for query_frame in range(layout.num_frames):
        qa = vs + query_frame * layout.tokens_per_frame
        qb = qa + layout.tokens_per_frame
        for key_frame in range(layout.num_frames):
            if (
                query_frame in (0, layout.num_frames - 1)
                or key_frame in (0, layout.num_frames - 1)
                or bounds[query_frame][0] <= key_frame <= bounds[query_frame][1]
            ):
                continue
            ka = vs + key_frame * layout.tokens_per_frame
            kb = ka + layout.tokens_per_frame
            allowed[qa:qb, ka:kb] = False
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
    probs = scores.softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, value)


def test_window_bounds_match_published_chunk_geometry():
    assert vdn.window_bounds(12, radius=1, chunk=5) == [
        (-5, 9),
        (-5, 9),
        (-5, 9),
        (-5, 9),
        (-5, 9),
        (0, 14),
        (0, 14),
        (0, 14),
        (0, 14),
        (0, 14),
        (5, 19),
        (5, 19),
    ]


@pytest.mark.parametrize("frames,per_frame", [(4, 1), (8, 2), (12, 3)])
def test_grouped_sdpa_is_exactly_the_both_anchor_reference(frames, per_frame):
    torch.manual_seed(260903)
    layout = _layout(frames=frames, per_frame=per_frame)
    q = torch.randn(layout.seq_len, 2, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    value = torch.randn_like(q)
    bounds = vdn.window_bounds(frames, radius=1, chunk=5)
    actual = vdn.window_softmax_sdpa(q, k, value, layout, bounds, 0.5)
    expected = _dense_mask_reference(q, k, value, layout, bounds, 0.5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_plain_t2va_layout_is_accepted():
    packed = SimpleNamespace(
        segments=[(0, 2, "text"), (2, 8, "audio"), (8, 20, "video")],
        signature=(2, 3, 4, 4, 3),
        seq_len=20,
    )
    layout = vdn.layout_from_packed(packed)
    assert layout.num_frames == 3
    assert layout.tokens_per_frame == 4
    assert layout.global_index("cpu").tolist() == list(range(8))
    assert layout.condition_kinds == ()


@pytest.mark.parametrize(
    "middle_kinds",
    [
        ("cond",),
        ("cond", "cond"),
        ("ref_img",),
        ("ref_img", "ref_img"),
        ("ref_audio",),
        ("ref_audio", "ref_img"),
        ("cond", "ref_img", "ref_audio"),
        ("cond_audio", "ref_img", "ref_audio"),
    ],
)
def test_every_native_h3_condition_and_reference_layout_is_accepted(middle_kinds):
    segments = []
    offset = 0
    for kind in ("text", *middle_kinds, "audio", "video"):
        length = {"text": 2, "audio": 6, "video": 12}.get(kind, 1)
        segments.append((offset, offset + length, kind))
        offset += length
    packed = SimpleNamespace(
        segments=segments,
        signature=(2, 3, 4, 4, 3),
        seq_len=offset,
    )
    layout = vdn.layout_from_packed(packed)
    assert layout.condition_kinds == middle_kinds
    assert layout.global_index("cpu").numel() == offset - 12


@pytest.mark.parametrize(
    "segments",
    [
        [(0, 2, "text"), (2, 8, "audio"), (8, 20, "unknown"), (20, 32, "video")],
        [(0, 2, "text"), (2, 14, "video"), (14, 20, "audio")],
        [(0, 2, "cond"), (2, 8, "audio"), (8, 20, "video")],
        [(0, 2, "text"), (2, 2, "ref_img"), (2, 8, "audio"), (8, 20, "video")],
        [(0, 2, "text"), (3, 9, "audio"), (9, 21, "video")],
    ],
)
def test_non_native_or_malformed_packed_layouts_fail_closed(segments):
    packed = SimpleNamespace(
        segments=segments,
        signature=(2, 3, 4, 4, 3),
        seq_len=segments[-1][1],
    )
    with pytest.raises(RuntimeError, match="OpenVDN"):
        vdn.layout_from_packed(packed)


def test_branch_manifest_shape_and_published_key_mapping_are_exact():
    branch = vdn.VDNBranchModel()
    keys = set(branch.state_dict())
    assert len(keys) == 800
    assert "blocks.49.linear_attention.alpha.A_log" in keys
    assert "blocks.49.to_out_linear.weight" in keys
    assert len(vdn.BRANCH_TENSOR_SUFFIXES) == 16
    assert (
        vdn._branch_key(
            "transformer_blocks.17.attn.linear_attention.short_conv.k_sp.weight"
        )
        == "blocks.17.linear_attention.short_conv.k_sp.weight"
    )
    with pytest.raises(ValueError, match="unsupported OpenVDN branch key"):
        vdn._branch_key("token_refiner.0.attn.softmax_gate.up.weight")


def test_named_openvdn_adapter_normalizes_and_converts_root_norm_and_fused_qkv():
    state = {}
    for name in ("q", "k", "v"):
        state[f"transformer_blocks.0.attn.orig.to_{name}.lora_A.default.weight"] = (
            torch.randn(2, 4)
        )
        state[f"transformer_blocks.0.attn.orig.to_{name}.lora_B.default.weight"] = (
            torch.randn(4, 2)
        )
    state["norm_out.linear.lora_A.default.weight"] = torch.randn(2, 4)
    state["norm_out.linear.lora_B.default.weight"] = torch.randn(8, 2)

    normalized = vdn._normalize_adapter_state(state, "default")
    assert all(
        ".default." not in key and ".attn.orig." not in key for key in normalized
    )
    converted, report = vdn.convert_fastvideo_h3_adapter(normalized)
    assert report["fused_qkv_groups"] == 1
    assert report["direct_lora_modules"] == 1
    assert converted["blocks.0.attn.qkv_proj.lora_A.weight"].shape == (6, 4)
    assert converted["blocks.0.attn.qkv_proj.lora_B.weight"].shape == (12, 6)
    assert "final_layer.adaln_proj.linear.lora_A.weight" in converted


def test_openvdn_adapter_shapes_fail_closed_for_curve_basis_adaln():
    converted = {
        "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(2, 4),
        "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(8, 2),
    }
    key_map = {
        "blocks.0.adaln_proj.linear": (
            "diffusion_model.blocks.0.adaln_proj.linear.weight"
        )
    }
    target = {
        "diffusion_model.blocks.0.adaln_proj.linear.weight": torch.empty(8, 4)
    }
    report = vdn._validate_adapter_target_shapes(converted, key_map, target)
    assert report == {
        "checked_targets": 1,
        "adaln_targets": 1,
        "all_shapes_exact": True,
    }

    target["diffusion_model.blocks.0.adaln_proj.linear.weight"] = torch.empty(8, 1)
    with pytest.raises(ValueError, match="curve-basis/pruned"):
        vdn._validate_adapter_target_shapes(converted, key_map, target)


def test_small_bidirectional_branch_runs_finite_and_keeps_anchor_rows_zero():
    torch.manual_seed(11)
    branch = vdn.BidirectionalLinearBranch(4, 1, 2, dtype=torch.float32)
    for parameter in branch.parameters():
        if parameter.numel():
            torch.nn.init.uniform_(parameter, -0.05, 0.05)
    layout = vdn.VDNSequenceLayout(
        seq_len=7,
        video_start=3,
        num_frames=4,
        tokens_per_frame=1,
        frame_height=1,
        frame_width=1,
        text_start=0,
        text_len=1,
    )
    video_x = torch.randn(4, 4)
    text_x = torch.randn(1, 4)
    video_qkv = tuple(torch.randn(4, 1, 2) for _ in range(3))
    text_qkv = tuple(torch.randn(1, 1, 2) for _ in range(3))
    with torch.no_grad():
        output = branch(
            video_x,
            layout,
            vdn.window_bounds(4, radius=1, chunk=5),
            video_qkv,
            text_x,
            text_qkv,
        )
    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[[0, -1]], torch.zeros(2, 2))


def test_short_full_cover_clip_skips_linear_branch(monkeypatch):
    layout = vdn.VDNSequenceLayout(
        seq_len=13,
        video_start=3,
        num_frames=10,
        tokens_per_frame=1,
        frame_height=1,
        frame_width=1,
        text_start=0,
        text_len=1,
    )
    block = SimpleNamespace(attn=SimpleNamespace(head_dim=2))
    branch = SimpleNamespace()
    branch.softmax_gate = lambda x: torch.ones(x.shape[0], 1, 1)
    branch.linear_attention = lambda *_args: pytest.fail(
        "linear branch must be skipped"
    )
    branch.to_out_linear = lambda value: value
    block.attn.out_proj = lambda value: value
    monkeypatch.setattr(
        vdn,
        "_qkv",
        lambda *_args: (
            torch.zeros(13, 1, 2),
            torch.zeros(13, 1, 2),
            torch.zeros(13, 1, 2),
            tuple(torch.zeros(13, 1, 2) for _ in range(3)),
        ),
    )
    output = vdn._vdn_attention(block, branch, torch.zeros(13, 2), None, layout)
    assert output.shape == (13, 2)


def test_execution_plan_owns_exact_stage_nfe_and_dual_clock(monkeypatch):
    captured = {}

    class Model:
        def get_attachment(self, key):
            assert key == vdn.ATTACHMENT_KEY
            return {"status": "configured", "stage": "stage_dmd_8nfe", "steps": 8}

    def fake_setup(*args):
        captured["args"] = args
        return "planned", "sampler", torch.ones(9)

    monkeypatch.setattr(vdn, "setup_dual_clock_sampling", fake_setup)
    planned, sampler, sigmas, report_json = vdn.setup_vdn_execution(Model(), "latent")
    assert (planned, sampler) == ("planned", "sampler")
    assert sigmas.numel() == 9
    assert captured["args"][2:] == (8, 12.0, 3.0, "euler", "native_flow")
    report = json.loads(report_json)
    assert report["nfe"] == 8
    assert report["scheduler"] == "native_flow"


def test_vdn_node_schemas_are_append_only_advanced_contracts():
    classes = nodes_vdn.VDN_H3_ADVANCED_NODE_CLASSES
    schemas = [node.define_schema() for node in classes]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3VDNRuntimeAuditT8Advanced",
        "MiniMaxH3VDNModelComposerT8Advanced",
        "MiniMaxH3VDNExecutionPlanT8Advanced",
    ]
    assert all(not schema.is_experimental for schema in schemas)
    assert all(schema.category == nodes_vdn.CATEGORY for schema in schemas)
    composer = {item.id: item for item in schemas[1].inputs}
    assert composer["stage"].default == "stage_dmd_8nfe"
    assert composer["verify_hashes"].default is True
    assert composer["allow_structural_base"].default is False


def test_downloaded_dmd_headers_match_when_assets_are_present():
    root = Path(
        r"F:\AI-T8-video-onekey\ComfyUI\models\diffusion_models\OpenVDN\vdn-minimax-h3"
    )
    if not root.is_dir():
        pytest.skip("OpenVDN integration assets are optional and not installed")
    report, errors = vdn._asset_report(root, "stage_dmd_8nfe", verify_hashes=False)
    assert errors == []
    assert report["linear_branch/model.safetensors"]["tensor_count"] == 800
    assert report["adapters/default/adapter_model.safetensors"]["tensor_count"] == 416
    assert report["adapters/turbo/adapter_model.safetensors"]["tensor_count"] == 726


def test_openvdn_frontend_workflow_is_pinned_wired_and_reproducible():
    from tools.build_openvdn_h3_workflow import build

    root = Path(__file__).resolve().parents[1]
    speed = root / "examples" / "workflows" / "10-speed"
    source = json.loads(
        (
            speed
            / "2026-08-30_H3_FastH3_VSA_T2VA_4Step_0p4MP_Advanced_EXP.json"
        ).read_text(encoding="utf-8")
    )
    path = (
        speed
        / "2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert build(source) == workflow
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert "MiniMaxH3LoRACompatibilityLoaderT8Advanced" not in by_type
    composer = by_type["MiniMaxH3VDNModelComposerT8Advanced"]
    execution = by_type["MiniMaxH3VDNExecutionPlanT8Advanced"]
    conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    assert by_type["UNETLoader"]["widgets_values"] == [
        "minimax_h3_fl2va_int8_convrot.safetensors",
        "default",
    ]
    assert composer["widgets_values"] == [
        "OpenVDN/vdn-minimax-h3",
        "stage_dmd_8nfe",
        True,
        True,
    ]
    assert execution["widgets_values"] == []
    assert conditioning["widgets_values"][1:5] == [960, 512, 73, "T2VA"]
    links = {link[0]: link for link in workflow["links"]}
    assert links[composer["inputs"][0]["link"]][1] == by_type["UNETLoader"]["id"]
    assert links[execution["inputs"][0]["link"]][1] == composer["id"]
    assert links[execution["inputs"][1]["link"]][1] == conditioning["id"]
    note = by_type["MarkdownNote"]["widgets_values"]
    for required in (
        "18be6bcc4ee72585eee322ba28b5ccac2cf85ef0",
        "不要再接 EMA_B",
        "v1 只允许普通 T2VA",
        "allow_structural_base",
        "adaln_t_table",
        "50 NFE",
        "512MiB",
    ):
        assert required in note


def test_validation_media_report_retries_only_a_transient_strict_failure(monkeypatch):
    from tools import run_openvdn_h3_validation as validation

    calls = []

    def fake_report(*_args, **_kwargs):
        calls.append(len(calls) + 1)
        passed = len(calls) >= 2
        return {
            "strict_decode_passed": passed,
            "strict_decode": {"video": {"passed": passed}},
        }

    monkeypatch.setattr(validation.shared, "media_report", fake_report)
    monkeypatch.setattr(validation.time, "sleep", lambda _seconds: None)
    report = validation.stable_media_report(
        Path("candidate.mp4"), ffmpeg="ffmpeg", ffprobe="ffprobe"
    )
    assert calls == [1, 2]
    assert report["strict_decode_passed"] is True
    assert report["strict_decode_transient_recovered"] is True
    assert len(report["strict_decode_attempts"]) == 2


def test_openvdn_license_and_notice_are_preserved_with_the_weight_boundary():
    root = Path(__file__).resolve().parents[1]
    license_text = (root / "THIRD_PARTY_NOTICES" / "OpenVDN-LICENSE.txt").read_text(
        encoding="utf-8"
    )
    notice_text = (root / "THIRD_PARTY_NOTICES" / "OpenVDN-NOTICE.txt").read_text(
        encoding="utf-8"
    )
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "VDN — Video DeltaNet" in notice_text
    assert "MiniMax H3 Community License Agreement" in notice_text
    assert "European Union" in notice_text
    assert "United States of America" in notice_text
    assert vdn.SOURCE_LICENSE == "Apache-2.0"
    assert "Applicable Territory excludes" in vdn.WEIGHT_TERRITORY_NOTICE
