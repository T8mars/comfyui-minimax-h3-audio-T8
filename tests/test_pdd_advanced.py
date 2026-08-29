from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch
from torch import nn

from h3_audio_t8_pkg import pdd_advanced as pdd
from h3_audio_t8_pkg import nodes_pdd_advanced as pdd_nodes
from h3_audio_t8_pkg.nodes_pdd_advanced import PDD_ADVANCED_NODE_CLASSES


def test_native_pdd_core_probe_is_semantic_and_reports_current_capabilities(monkeypatch):
    class _NativeFinal:
        def forward(
            self,
            x,
            t_emb,
            video_seg,
            audio_seg,
            sigma,
            sample_sigmas,
            shifts,
        ):
            del x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts

    class _Diffusion:
        final_layer = _NativeFinal()

    def fake_load(self, device_to=None, lowvram_model_memory=0):
        del device_to, lowvram_model_memory
        weight_key = "probe.weight"
        bias_key = "probe.bias"
        comfy = pdd.comfy
        comfy.lora.calculate_shape(self.patches[weight_key], None, weight_key)
        comfy.lora.calculate_shape(self.patches[bias_key], None, bias_key)

    def fake_partially_unload(self, device_to=None, memory_to_free=0):
        del device_to, memory_to_free
        weight_key = "probe.weight"
        bias_key = "probe.bias"
        comfy = pdd.comfy
        comfy.lora.calculate_shape(self.patches[weight_key], None, weight_key)
        comfy.lora.calculate_shape(self.patches[bias_key], None, bias_key)

    monkeypatch.setattr(pdd.comfy.model_patcher.ModelPatcher, "load", fake_load)
    monkeypatch.setattr(
        pdd.comfy.model_patcher.ModelPatcher,
        "partially_unload",
        fake_partially_unload,
    )
    monkeypatch.setattr(
        pdd.comfy.lora,
        "calculate_shape",
        lambda _patches, _weight, _key: torch.Size((3, 2)),
    )

    report = pdd.probe_native_pdd_core(_Diffusion())

    assert report["available"] is True
    assert report["policy"] == "semantic_capability_probe_no_version_or_hash_gate"
    assert report["shape_changing_padded_diff"] is True
    assert "sigma" in report["final_layer_parameters"]


def test_native_pdd_core_probe_rejects_partial_core_support():
    class _LegacyFinal:
        def forward(self, x, t_emb, video_seg, audio_seg):
            del x, t_emb, video_seg, audio_seg

    report = pdd.probe_native_pdd_core(type("_Diffusion", (), {"final_layer": _LegacyFinal()})())

    assert report["available"] is False
    assert report["final_layer_schedule_args"] is False


def test_native_pdd_state_keeps_only_backbone_adapters(monkeypatch):
    monkeypatch.setattr(pdd.comfy.lora_convert, "convert_lora", lambda value: dict(value))
    state = {
        "diffusion_model.blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 3),
        "pdd.final_layer.video_out.weight": torch.zeros(2, 3, 4),
        "pdd.final_layer.video_out.bias": torch.zeros(2, 3),
        "pdd.final_layer.audio_out.weight": torch.zeros(2, 5, 4),
        "pdd.final_layer.audio_out.bias": torch.zeros(2, 5),
    }

    converted = pdd._native_pdd_lora_state(state)

    assert set(converted) == {
        "diffusion_model.blocks.0.attn.to_q.lora_A.weight"
    }
    assert state["pdd.final_layer.video_out.weight"].shape == (2, 3, 4)


def test_native_pdd_head_bank_encodes_first_head_plus_offsets():
    absolute = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(32, 2, 3)
    encoded = pdd._encode_native_pdd_head_bank(absolute)
    rows = encoded.reshape_as(absolute)

    assert rows[0] == pytest.approx(absolute[0])
    assert rows[1:] == pytest.approx(absolute[1:] - absolute[:1])


@pytest.mark.parametrize("shift", [pdd.PDD_SHIFT_VIDEO, pdd.PDD_SHIFT_AUDIO])
def test_native_and_fallback_head_math_are_equivalent_for_every_block(shift):
    absolute = torch.arange(32 * 2 * 3, dtype=torch.float64).reshape(32, 2, 3)
    encoded = pdd._encode_native_pdd_head_bank(absolute).reshape_as(absolute)
    fallback, _ = pdd._fuse_head_bank(
        absolute,
        torch.zeros((32, 2), dtype=torch.float64),
        shift,
    )

    for block in range(pdd.PDD_NFE):
        start = block * pdd.PDD_BLOCK_SIZE
        stop = start + pdd.PDD_BLOCK_SIZE
        plan = pdd.pdd_plan(shift, block)[start:stop]
        first = max(start, 1)
        native = encoded[0] + torch.einsum(
            "n,noi->oi",
            plan[first - start :],
            encoded[first:stop],
        )
        assert native == pytest.approx(fallback[block])


def test_native_pdd_head_patches_use_padded_diff_for_all_four_targets():
    state = {
        "pdd.final_layer.video_out.weight": torch.zeros(32, 2, 3),
        "pdd.final_layer.video_out.bias": torch.zeros(32, 2),
        "pdd.final_layer.audio_out.weight": torch.zeros(32, 1, 3),
        "pdd.final_layer.audio_out.bias": torch.zeros(32, 1),
    }
    patches = pdd._native_pdd_head_patches(state)

    assert set(patches) == {
        "diffusion_model.final_layer.video_out.weight",
        "diffusion_model.final_layer.video_out.bias",
        "diffusion_model.final_layer.audio_out.weight",
        "diffusion_model.final_layer.audio_out.bias",
    }
    for patch_type, values in patches.values():
        assert patch_type == "diff"
        assert values[1] == {"pad_weight": True}


def test_pdd_plan_and_runtime_schedule_cover_the_eight_blocks_exactly():
    plans = torch.stack([pdd.pdd_plan(12.0, index) for index in range(8)])
    assert plans.shape == (8, 32)
    assert plans.sum(dim=1).tolist() == pytest.approx([1.0] * 8)
    assert torch.count_nonzero(plans, dim=1).tolist() == [4] * 8
    assert pdd.validate_pdd_sigmas(pdd.pdd_runtime_sigmas())["block_indices"] == list(
        range(8)
    )


def test_pdd_runtime_schedule_can_be_split_into_exact_four_plus_four_blocks():
    sigmas = pdd.pdd_runtime_sigmas()
    low_pass = sigmas[:5]
    high_pass = sigmas[4:]
    assert [pdd.pdd_block_index(value) for value in low_pass[:-1]] == [0, 1, 2, 3]
    assert [pdd.pdd_block_index(value) for value in high_pass[:-1]] == [4, 5, 6, 7]
    assert low_pass[-1] == high_pass[0]
    assert high_pass[-1] == 0


def test_pdd_schedule_rejects_wrong_nfe_and_wrong_grid():
    with pytest.raises(ValueError, match="exactly 8 model evaluations"):
        pdd.validate_pdd_sigmas(torch.linspace(1.0, 0.0, 5))
    wrong = pdd.pdd_runtime_sigmas().clone()
    wrong[3] -= 0.01
    with pytest.raises(ValueError, match="official Euler/simple 8-step sigma"):
        pdd.validate_pdd_sigmas(wrong)


def test_pdd_head_fusion_matches_four_interval_weighted_sum():
    weights = torch.arange(32 * 2 * 3, dtype=torch.float64).reshape(32, 2, 3)
    biases = torch.arange(32 * 2, dtype=torch.float64).reshape(32, 2)
    fused_weight, fused_bias = pdd._fuse_head_bank(weights, biases, 12.0)
    for block in range(8):
        plan = pdd.pdd_plan(12.0, block)
        assert fused_weight[block] == pytest.approx(
            torch.einsum("n,noi->oi", plan, weights)
        )
        assert fused_bias[block] == pytest.approx(
            torch.einsum("n,no->o", plan, biases)
        )


def test_pdd_dtype_names_normalize_safetensors_and_torch_spellings():
    assert pdd._dtype_name("BF16") == pdd._dtype_name(torch.bfloat16) == "BF16"
    assert pdd._dtype_name("F32") == pdd._dtype_name(torch.float32) == "F32"


class _FakeSlice:
    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype


class _FakeSafeOpen:
    def __init__(self, metadata, slices, keys):
        self._metadata = metadata
        self._slices = slices
        self._keys = keys

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def metadata(self):
        return self._metadata

    def keys(self):
        return self._keys

    def get_slice(self, key):
        return self._slices[key]


def _fake_adapter_contract(variant: str):
    metadata = dict(pdd.PDD_REQUIRED_METADATA)
    metadata["base_variant"] = variant
    slices = {
        key: _FakeSlice(shape, "BF16")
        for key, (shape, _dtype) in pdd.PDD_HEAD_SPECS.items()
    }
    keys = set(slices)
    for index in range(258):
        prefix = f"diffusion_model.fake.{index}"
        keys.update(
            {
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
                f"{prefix}.alpha",
            }
        )
    return metadata, slices, keys


def test_adapter_header_contract_reports_variant_without_blocking(monkeypatch, tmp_path):
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"header")
    metadata, slices, keys = _fake_adapter_contract("FL2VA")
    monkeypatch.setattr(
        pdd,
        "safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(metadata, slices, keys),
    )
    report = pdd.inspect_pdd_adapter(path, "FL2VA")
    assert report["adapter_count"] == 258
    assert report["tensor_count"] == 778
    mismatch = pdd.inspect_pdd_adapter(path, "Ref2VA")
    assert mismatch["base_variant_reference_match"] is False
    assert mismatch["model_identity_policy"] == "diagnostic_only_not_a_load_gate"


class _TinyAdaLN(nn.Module):
    def forward(self, t_emb):
        shape = (t_emb.shape[0], 3)
        return torch.zeros(shape), torch.zeros(shape)


class _TinyFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()
        self.adaln_proj = _TinyAdaLN()
        self.video_out = nn.Linear(3, 2)
        self.audio_out = nn.Linear(3, 1)

    def forward(self, *_args, **_kwargs):
        return "native-final"


def test_final_layer_selects_each_official_block(monkeypatch):
    monkeypatch.setattr(
        pdd,
        "PDD_HEAD_SPECS",
        {
            "pdd.final_layer.video_out.weight": ((32, 2, 3), torch.bfloat16),
            "pdd.final_layer.video_out.bias": ((32, 2), torch.bfloat16),
            "pdd.final_layer.audio_out.weight": ((32, 1, 3), torch.bfloat16),
            "pdd.final_layer.audio_out.bias": ((32, 1), torch.bfloat16),
        },
    )
    layer = pdd.PDDHeadFinalLayer(
        _TinyFinal(),
        torch.zeros((32, 2, 3), dtype=torch.bfloat16),
        torch.zeros((32, 2), dtype=torch.bfloat16),
        torch.zeros((32, 1, 3), dtype=torch.bfloat16),
        torch.zeros((32, 1), dtype=torch.bfloat16),
        strength=1.0,
        variant="FL2VA",
    )
    assert [layer.select_for_sigma(float(value)) for value in pdd.pdd_runtime_sigmas()[:-1]] == list(
        range(8)
    )
    with pytest.raises(ValueError, match="video shift"):
        layer.select_for_sigma(1.0, shift_video=6.0)
    with pytest.raises(ValueError, match="outside its official"):
        layer.select_for_sigma(0.5)


def test_final_layer_supports_current_comfy_per_token_mod_rows(monkeypatch):
    monkeypatch.setattr(
        pdd,
        "PDD_HEAD_SPECS",
        {
            "pdd.final_layer.video_out.weight": ((32, 2, 3), torch.bfloat16),
            "pdd.final_layer.video_out.bias": ((32, 2), torch.bfloat16),
            "pdd.final_layer.audio_out.weight": ((32, 1, 3), torch.bfloat16),
            "pdd.final_layer.audio_out.bias": ((32, 1), torch.bfloat16),
        },
    )
    video_weight = torch.ones((32, 2, 3), dtype=torch.bfloat16)
    video_bias = torch.zeros((32, 2), dtype=torch.bfloat16)
    audio_weight = torch.full((32, 1, 3), 2.0, dtype=torch.bfloat16)
    audio_bias = torch.zeros((32, 1), dtype=torch.bfloat16)
    layer = pdd.PDDHeadFinalLayer(
        _TinyFinal(),
        video_weight,
        video_bias,
        audio_weight,
        audio_bias,
        strength=1.0,
        variant="FL2VA",
    )
    layer.select_for_sigma(float(pdd.pdd_runtime_sigmas()[0]))
    hidden = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=torch.bfloat16,
    )
    t_emb = torch.zeros((4, 1), dtype=torch.bfloat16)
    video, audio = layer(
        hidden,
        t_emb,
        (0, 2, torch.tensor([0, 1], dtype=torch.long)),
        (2, 4, torch.tensor([2, 3], dtype=torch.long)),
    )
    assert video.float() == pytest.approx(
        torch.tensor([[6.0, 6.0], [6.0, 6.0]])
    )
    assert audio.float() == pytest.approx(torch.tensor([[6.0], [12.0]]))


def test_final_layer_injection_preserves_native_parameter_paths_and_restores_forward(
    monkeypatch,
):
    monkeypatch.setattr(
        pdd,
        "PDD_HEAD_SPECS",
        {
            "pdd.final_layer.video_out.weight": ((32, 2, 3), torch.bfloat16),
            "pdd.final_layer.video_out.bias": ((32, 2), torch.bfloat16),
            "pdd.final_layer.audio_out.weight": ((32, 1, 3), torch.bfloat16),
            "pdd.final_layer.audio_out.bias": ((32, 1), torch.bfloat16),
        },
    )
    base = _TinyFinal()
    diffusion = nn.Module()
    diffusion.final_layer = base
    model = nn.Module()
    model.diffusion_model = diffusion
    native_paths = set(dict(model.named_parameters()))
    assert native_paths == {
        "diffusion_model.final_layer.video_out.weight",
        "diffusion_model.final_layer.video_out.bias",
        "diffusion_model.final_layer.audio_out.weight",
        "diffusion_model.final_layer.audio_out.bias",
    }

    layer = pdd.PDDHeadFinalLayer(
        base,
        torch.ones((32, 2, 3), dtype=torch.bfloat16),
        torch.zeros((32, 2), dtype=torch.bfloat16),
        torch.ones((32, 1, 3), dtype=torch.bfloat16),
        torch.zeros((32, 1), dtype=torch.bfloat16),
        strength=1.0,
        variant="Ref2VA",
    )
    injection = pdd._create_pdd_final_layer_injection(base, layer)

    class _Patcher:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")

    assert base() == "native-final"
    injection.inject(_Patcher())
    assert model.diffusion_model.final_layer is base
    assert set(dict(model.named_parameters())) == native_paths
    assert not any(".base." in key for key in model.state_dict())
    assert base.forward.__self__ is layer

    injection.eject(_Patcher())
    assert base() == "native-final"
    assert set(dict(model.named_parameters())) == native_paths


def test_pdd_runtime_injection_rolls_back_final_forward_when_backbone_inject_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        pdd,
        "PDD_HEAD_SPECS",
        {
            "pdd.final_layer.video_out.weight": ((32, 2, 3), torch.bfloat16),
            "pdd.final_layer.video_out.bias": ((32, 2), torch.bfloat16),
            "pdd.final_layer.audio_out.weight": ((32, 1, 3), torch.bfloat16),
            "pdd.final_layer.audio_out.bias": ((32, 1), torch.bfloat16),
        },
    )
    base = _TinyFinal()
    layer = pdd.PDDHeadFinalLayer(
        base,
        torch.zeros((32, 2, 3), dtype=torch.bfloat16),
        torch.zeros((32, 2), dtype=torch.bfloat16),
        torch.zeros((32, 1, 3), dtype=torch.bfloat16),
        torch.zeros((32, 1), dtype=torch.bfloat16),
        strength=1.0,
        variant="FL2VA",
    )

    class _FailingBackbone:
        def inject(self, _patcher):
            raise RuntimeError("synthetic backbone failure")

        def eject(self, _patcher):
            return None

    class _Patcher:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")

    injection = pdd._create_pdd_runtime_injection(
        _FailingBackbone(), base, layer
    )
    with pytest.raises(RuntimeError, match="synthetic backbone failure"):
        injection.inject(_Patcher())
    assert base() == "native-final"


def test_node_schema_is_one_append_only_advanced_setup_node():
    assert len(PDD_ADVANCED_NODE_CLASSES) == 1
    schema = PDD_ADVANCED_NODE_CLASSES[0].define_schema()
    assert schema.node_id == "MiniMaxH3PDD8StepSetupT8Advanced"
    assert schema.node_id.endswith("Advanced")
    assert [output.id for output in schema.outputs] == [
        "model",
        "sampler",
        "sigmas",
        "report_json",
    ]


def test_pdd_file_picker_does_not_offer_ordinary_loras(monkeypatch):
    monkeypatch.setattr(
        pdd_nodes.folder_paths,
        "get_filename_list",
        lambda _folder: [
            "ordinary_turbo.safetensors",
            "custom\\my_pdd_adapter.sft",
            pdd_nodes.DEFAULT_REF2VA_PDD,
        ],
    )
    assert pdd_nodes._pdd_lora_options() == [
        pdd_nodes.DEFAULT_FL2VA_PDD,
        pdd_nodes.DEFAULT_REF2VA_PDD,
        "custom\\my_pdd_adapter.sft",
    ]


def test_bypass_injection_offloads_on_eject_and_partial_inject_failure(monkeypatch):
    calls = []

    class _Injection:
        def __init__(self, fail=False):
            self.fail = fail

        def inject(self, _patcher):
            calls.append("inject")
            if self.fail:
                raise RuntimeError("synthetic partial injection failure")

        def eject(self, _patcher):
            calls.append("eject")

    class _Manager:
        def __init__(self, fail=False):
            self.injection = _Injection(fail=fail)

        def create_injections(self, _model):
            return [self.injection]

    class _Patcher:
        offload_device = torch.device("cpu")

    monkeypatch.setattr(
        pdd,
        "_move_adapter_weights_to_device",
        lambda adapters, device: calls.append(
            ("offload", tuple(adapters), torch.device(device))
        ),
    )
    adapter = object()
    injection = pdd._create_offloading_bypass_injections(
        _Manager(), object(), (adapter,)
    )[0]
    injection.inject(_Patcher())
    injection.eject(_Patcher())
    assert calls == [
        "inject",
        "eject",
        ("offload", (adapter,), torch.device("cpu")),
    ]

    calls.clear()
    failing = pdd._create_offloading_bypass_injections(
        _Manager(fail=True), object(), (adapter,)
    )[0]
    with pytest.raises(RuntimeError, match="synthetic partial injection failure"):
        failing.inject(_Patcher())
    assert calls == [
        "inject",
        "eject",
        ("offload", (adapter,), torch.device("cpu")),
    ]


def test_pdd_examples_are_frontend_workflows_with_local_notes():
    root = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "19-pdd-acceleration"
    expected = {
        "2026-08-27_H3_PDD_FL2VA_8Step_Advanced_EXP.json": (
            "FL2VA",
            "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors",
            False,
        ),
        "2026-08-27_H3_PDD_Ref2VA_8Step_Advanced_EXP.json": (
            "Ref2VA",
            "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors",
            False,
        ),
        "2026-08-27_H3_PDD_FL2VA_Learned_Latent_TwoPass_4Plus4_Advanced_EXP.json": (
            "FL2VA",
            "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors",
            True,
        ),
        "2026-08-27_H3_PDD_Ref2VA_Learned_Latent_TwoPass_4Plus4_Stable.json": (
            "Ref2VA",
            "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors",
            True,
        ),
    }
    for filename, (variant, adapter, two_pass) in expected.items():
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert isinstance(workflow["nodes"], list)
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        types = [node["type"] for node in workflow["nodes"]]
        assert types.count("MiniMaxH3PDD8StepSetupT8Advanced") == 1
        assert "LoraLoaderModelOnly" not in types
        assert "LoraLoaderBypassModelOnly" not in types
        assert types.count("MarkdownNote") >= (5 if two_pass else 3)
        node = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "MiniMaxH3PDD8StepSetupT8Advanced"
        )
        assert node["widgets_values"] == [adapter, variant, 1.0]
        assert all(input_["link"] is not None for input_ in node["inputs"])
        assert all(output["links"] for output in node["outputs"][:3])
        if not two_pass:
            assert "MiniMaxH3DualClockSamplerT8" not in types
            assert "SplitSigmas" not in types
            continue

        assert types.count("MiniMaxH3DualClockSamplerT8") == 1
        assert types.count("SplitSigmas") == 1
        assert types.count("SamplerCustomAdvanced") == 2
        assert types.count("MiniMaxH3LearnedLatentUpscaleT8Advanced") == 1
        assert types.count("MiniMaxH3TwoPassLatentReconcileT8Advanced") == 1
        split = next(item for item in workflow["nodes"] if item["type"] == "SplitSigmas")
        assert split["widgets_values"] == [4]
        assert split["inputs"][0]["link"] is not None
        assert all(output["links"] for output in split["outputs"])

        links = {(link[1], link[2], link[3], link[4]) for link in workflow["links"]}
        pdd_id = node["id"]
        split_id = split["id"]
        pass1 = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "SamplerCustomAdvanced" and "PASS 1" in item.get("title", "")
        )
        pass2 = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "SamplerCustomAdvanced" and "PASS 2" in item.get("title", "")
        )
        upscaler = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced"
        )
        high_sampler = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "MiniMaxH3DualClockSamplerT8"
        )
        assert (pdd_id, 2, split_id, 0) in links
        assert (split_id, 0, pass1["id"], 3) in links
        assert (split_id, 1, pass2["id"], 3) in links
        assert (pass1["id"], 1, upscaler["id"], 0) in links
        assert (pdd_id, 0, high_sampler["id"], 0) in links
        if variant == "Ref2VA":
            low_conditioning = next(item for item in workflow["nodes"] if item["id"] == 7)
            high_conditioning = next(item for item in workflow["nodes"] if item["id"] == 14)
            assert low_conditioning["widgets_values"][1:4] == [864, 480, 22]
            assert high_conditioning["widgets_values"][1:4] == [1312, 736, 22]
            assert upscaler["widgets_values"][2] == 1.5
            assert workflow["extra"]["workflow_title"].endswith("Stable")
