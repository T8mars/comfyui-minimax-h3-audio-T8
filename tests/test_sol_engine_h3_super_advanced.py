from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from h3_audio_t8_pkg import sol_engine_h3_super_advanced as super_h3
from h3_audio_t8_pkg import sol_engine_taehv as taehv_impl


def test_official_stage2_schedule_and_tau_mapping_are_exact():
    assert super_h3.official_stage2_sigmas().tolist() == pytest.approx([
        0.909375,
        0.725,
        0.421875,
        0.0,
    ])
    assert super_h3.stage2_tau_for_sigma(torch.tensor([0.909375])) == 1.0
    assert super_h3.stage2_tau_for_sigma(torch.tensor([0.725])) == 1.25
    assert super_h3.stage2_tau_for_sigma(torch.tensor([0.421875])) == 1.5
    assert super_h3.stage2_tau_for_sigma(torch.tensor([0.0])) is None


def test_h3_draft_handoff_uses_half_geometry_and_ltx_frame_grid():
    frames = torch.linspace(0, 1, 243 * 24 * 43 * 3).reshape(243, 24, 43, 3)
    prepared, width, height, kept, dropped, duration, report_json = (
        super_h3.prepare_h3_draft_for_ltx_refiner(
            frames,
            192,
            128,
            "trim_to_8n_plus_1",
        )
    )
    report = json.loads(report_json)
    assert prepared.shape == (241, 64, 96, 3)
    assert (width, height, kept, dropped) == (96, 64, 241, 2)
    assert duration == pytest.approx(241 / 24)
    assert torch.isfinite(prepared).all()
    assert report["pixel_limit_policy"] == "no_project_pixel_area_limit"
    assert report["audio_policy"] == "bypass_stage2_and_preserve_original_h3_audio_object"
    assert report["output_duration_seconds"] == pytest.approx(241 / 24)


def test_h3_draft_handoff_has_no_two_megapixel_gate():
    frames = torch.zeros((1, 32, 32, 3))
    prepared, width, height, *_ = super_h3.prepare_h3_draft_for_ltx_refiner(
        frames,
        2560,
        1440,
    )
    assert width * 2 == 2560
    assert height * 2 == 1440
    assert prepared.shape == (1, 720, 1280, 3)


class _FakeSolBackend:
    def __init__(self):
        self.calls = []

    def make_override(self, *, tau, **settings):
        self.calls.append((tau, settings))

        def override(_func, _q, _k, _v, _heads, *_args, **_kwargs):
            return {"route": "sol", "tau": tau}

        return override


class _FakePatcher:
    def __init__(self, blocks=48, model_options=None):
        self.diffusion_model = SimpleNamespace(
            transformer_blocks=[object() for _ in range(blocks)]
        )
        self.model_options = model_options or {"transformer_options": {}}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion_model

    def clone(self):
        cloned = _FakePatcher.__new__(_FakePatcher)
        cloned.diffusion_model = self.diffusion_model
        cloned.model_options = deepcopy(self.model_options)
        return cloned

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        assert transformer_index is None
        transformer_options = self.model_options.setdefault("transformer_options", {})
        patches = transformer_options.setdefault("patches_replace", {})
        patches.setdefault(name, {})[(block_name, number)] = patch


def test_ltx_stage2_setup_preserves_existing_block_patch_and_routes_official_tau():
    def previous_patch(args, _extra_args):
        return {"img": args["img"], "tag": args["transformer_options"]["sol_block"]}

    original_options = {
        "transformer_options": {
            "patches_replace": {"dit": {("double_block", 7): previous_patch}}
        }
    }
    model = _FakePatcher(model_options=original_options)
    backend = _FakeSolBackend()
    patched, sigmas, strength, report_json = super_h3.setup_ltx_stage2_refiner(
        model,
        sol_backend=backend,
    )
    report = json.loads(report_json)
    replacements = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert len(replacements) == 48
    assert replacements[("double_block", 7)](
        {"img": "x", "transformer_options": {}},
        {"original_block": lambda args: {"img": args["img"]}},
    ) == {"img": "x", "tag": 7}
    assert strength == 0.8
    assert sigmas.tolist() == pytest.approx([0.909375, 0.725, 0.421875, 0.0])
    assert report["attention"] == "sol_attn_active"
    assert report["composition"]["existing_dit_replacements_preserved"] == 1

    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    dense = override(
        lambda *_args, **_kwargs: {"route": "dense"},
        object(),
        object(),
        object(),
        32,
        transformer_options={"sol_block": 0, "sigmas": torch.tensor([0.725])},
    )
    sparse = override(
        lambda *_args, **_kwargs: {"route": "dense"},
        object(),
        object(),
        object(),
        32,
        transformer_options={"sol_block": 12, "sigmas": torch.tensor([0.725])},
    )
    assert dense == {"route": "dense"}
    assert sparse == {"route": "sol", "tau": 1.25}


def test_missing_sol_backend_is_dense_passthrough_not_a_hard_error():
    model = _FakePatcher()
    patched, _, _, report_json = super_h3.setup_ltx_stage2_refiner(
        model,
        sol_backend=None,
    )
    report = json.loads(report_json)
    assert report["attention"] == "dense_fallback_sol_attn_not_loaded"
    assert "optimized_attention_override" not in patched.model_options["transformer_options"]
    assert report["model_identity_policy"] == (
        "no_filename_hash_byte_size_or_pixel_area_execution_gate"
    )


class _FakeTAEHV:
    def __init__(self):
        self.encode_input = None
        self.decode_input = None
        self.moves = []

    def to(self, *, device, dtype=None):
        self.moves.append((torch.device(device).type, dtype))
        return self

    def encode_video(self, pixels, *, parallel):
        self.encode_input = (tuple(pixels.shape), parallel, pixels.dtype)
        return torch.zeros((1, 2, 128, 1, 2), device=pixels.device, dtype=pixels.dtype)

    def decode_video(self, latent, *, parallel):
        self.decode_input = (tuple(latent.shape), parallel, latent.dtype)
        return torch.full(
            (1, 9, 3, 32, 64),
            0.5,
            device=latent.device,
            dtype=latent.dtype,
        )


def test_taehv_nodes_preserve_comfy_and_upstream_axis_contracts(monkeypatch):
    monkeypatch.setattr(
        super_h3,
        "_taehv_device_and_dtype",
        lambda _precision: (torch.device("cpu"), torch.device("cpu"), torch.float32),
    )
    model = _FakeTAEHV()
    handle = super_h3.T8TAEHVHandle(model=model, source_path="taeltx2_3_wide.pth")
    frames = torch.rand((9, 32, 64, 3))

    latent, encode_report_json = super_h3.encode_h3_frames_with_taehv(
        frames,
        handle,
        "auto_official",
        "fp32_reference",
    )
    decoded, decode_report_json = super_h3.decode_ltx_latent_with_taehv(
        latent,
        handle,
        "auto_official",
        "fp32_reference",
    )

    assert model.encode_input == ((1, 9, 3, 32, 64), True, torch.float32)
    assert latent["samples"].shape == (1, 128, 2, 1, 2)
    assert model.decode_input == ((1, 2, 128, 1, 2), True, torch.float32)
    assert decoded.shape == (9, 32, 64, 3)
    assert decoded.mean().item() == pytest.approx(0.5)
    assert json.loads(encode_report_json)["latent_shape_bcthw"] == [1, 128, 2, 1, 2]
    assert json.loads(decode_report_json)["output_shape_fhwc"] == [9, 32, 64, 3]


def test_taehv_auto_execution_matches_official_element_threshold():
    assert super_h3._taehv_parallel("auto_official", 99_999_999) is True
    assert super_h3._taehv_parallel("auto_official", 100_000_000) is False
    assert super_h3._taehv_parallel("sequential_low_vram", 1) is False
    assert super_h3._taehv_parallel("parallel_high_vram_exp", 1_000_000_000) is True


def test_taehv_temporal_parallel_and_sequential_paths_match_on_tiny_network():
    torch.manual_seed(29)
    network = torch.nn.Sequential(
        taehv_impl.MemBlock(2, 2),
        taehv_impl.TPool(2, 2),
        taehv_impl.MemBlock(2, 2),
        taehv_impl.TGrow(2, 2),
        taehv_impl.MemBlock(2, 2),
    ).eval()
    tensor = torch.randn((1, 4, 2, 3, 5))
    with torch.inference_mode():
        parallel = taehv_impl._apply_parallel(network, tensor)
        sequential = taehv_impl._apply_sequential(network, tensor)
    torch.testing.assert_close(sequential, parallel, rtol=1e-5, atol=1e-6)


def test_frontend_workflow_is_complete_official_stage2_and_keeps_audio_external():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "22-sol-engine-h3-super"
        / "2026-08-29_H3_Sol_Engine_Super_Acceleration_LTX25_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in workflow["nodes"]}
    assert {
        "LoadVideo",
        "GetVideoComponents",
        "MiniMaxH3SolEngineDraftToLTXT8Advanced",
        "MiniMaxH3SolEngineTAEHVLoaderT8Advanced",
        "MiniMaxH3SolEngineTAEHVEncodeT8Advanced",
        "MiniMaxH3SolEngineTAEHVDecodeT8Advanced",
        "LTXVLatentUpsampler",
        "LTXVConditioning",
        "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced",
        "SamplerCustomAdvanced",
        "MiniMaxH3OutputTrimT8",
        "CreateVideo",
        "SaveVideo",
    } <= types

    setup = next(
        node for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced"
    )
    assert setup["widgets_values"] == [
        True,
        "auto_sol_attn",
        4096,
        "bf16_official",
        False,
    ]
    lora = next(node for node in workflow["nodes"] if node["type"] == "LoraLoaderModelOnly")
    assert lora["widgets_values"] == [
        "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        0.8,
    ]
    unet = next(node for node in workflow["nodes"] if node["type"] == "UNETLoader")
    assert unet["widgets_values"][0] == (
        "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
    )
    clip = next(node for node in workflow["nodes"] if node["type"] == "CLIPLoader")
    assert clip["widgets_values"][0] == (
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    )
    assert "VAEEncode" not in types
    assert "VAEDecodeTiled" not in types
    ltx_conditioning = next(
        node for node in workflow["nodes"] if node["type"] == "LTXVConditioning"
    )
    assert ltx_conditioning["widgets_values"] == [24.0]

    links = workflow["links"]
    get_video = next(node for node in workflow["nodes"] if node["type"] == "GetVideoComponents")
    sampler = next(node for node in workflow["nodes"] if node["type"] == "SamplerCustomAdvanced")
    trim = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3OutputTrimT8")
    vae = next(node for node in workflow["nodes"] if node["type"] == "VAELoader")
    upsampler = next(node for node in workflow["nodes"] if node["type"] == "LTXVLatentUpsampler")
    vae_targets = {link[3] for link in links if link[1] == vae["id"]}
    assert vae_targets == {upsampler["id"]}
    assert not any(link[1] == get_video["id"] and link[4] < 5 and link[3] == sampler["id"] for link in links)
    assert any(
        link[1] == get_video["id"]
        and link[2] == 1
        and link[3] == trim["id"]
            and link[4] == 3
        and link[5] == "AUDIO"
        for link in links
    )
    assert len(nodes) == len(workflow["nodes"])
