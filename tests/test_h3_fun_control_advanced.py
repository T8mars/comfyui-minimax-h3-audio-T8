from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from h3_audio_t8_pkg import h3_fun_control_advanced as fun


def test_fit_control_video_holds_last_frame_and_obeys_17n_plus_5():
    frames = torch.stack(
        [torch.zeros(32, 32, 3), torch.ones(32, 32, 3)], dim=0
    )
    fitted = fun._fit_control_video(
        frames, width=32, height=32, length=22, fit_mode="exact"
    )
    assert fitted.shape == (22, 32, 32, 3)
    assert torch.equal(fitted[-1], frames[-1])
    with pytest.raises(ValueError, match=r"17n\+5"):
        fun._fit_control_video(
            frames, width=32, height=32, length=21, fit_mode="exact"
        )


def test_patchify_builds_control_mask_and_masked_width_without_batch_axis():
    latent = torch.arange(1 * 24 * 2 * 4 * 6, dtype=torch.float32).reshape(
        1, 24, 2, 4, 6
    )
    rows = fun._patchify_control(latent, 196)
    assert rows.shape == (12, 196)
    assert torch.count_nonzero(rows[:, 96:]) == 0


class _EntryControl:
    def __init__(self):
        self.control_proj_in = nn.Linear(3, 2, bias=False)
        self.control_blocks = [type("_Block", (), {})()]
        self.control_blocks[0].before_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.control_proj_in.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
            self.control_blocks[0].before_proj.weight.copy_(torch.eye(2))


def test_control_entry_changes_video_rows_only():
    control = _EntryControl()
    seed = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    before = seed.clone()
    rows = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    stream = fun._enter_control_stream(control, rows, seed, 2, 4)
    assert torch.equal(stream[:2], before[:2])
    assert torch.equal(stream[4:], before[4:])
    assert torch.equal(stream[2:4], before[2:4] + rows[:, :2])


def test_strength_zero_is_true_identity_and_does_not_encode():
    class _Model:
        model_options = {}

    class _VAE:
        def encode(self, _frames):
            raise AssertionError("strength-zero bypass must not encode")

    model = _Model()
    positive = [[torch.zeros(1), {}]]
    bundle = fun.H3FunControlBundle(
        "compatibility", {}, "control.safetensors", "control.safetensors", {}
    )
    result_model, result_positive, report_json = fun.apply_h3_fun_control(
        model,
        positive,
        bundle,
        _VAE(),
        torch.zeros(5, 32, 32, 3),
        32,
        32,
        5,
        "depth",
        "exact",
        0.0,
        0.0,
        0.75,
    )
    assert result_model is model
    assert result_positive is positive
    assert json.loads(report_json)["status"] == "bypass"


def test_sol_morton_is_rejected_before_control_encode():
    class _Model:
        model_options = {"transformer_options": {"sol_morton": True}}

    bundle = fun.H3FunControlBundle(
        "compatibility", {}, "control.safetensors", "control.safetensors", {}
    )
    with pytest.raises(RuntimeError, match="raster-order"):
        fun.apply_h3_fun_control(
            _Model(),
            [],
            bundle,
            object(),
            torch.zeros(5, 32, 32, 3),
            32,
            32,
            5,
            "pose",
            "exact",
            0.7,
            0.0,
            0.75,
        )


def test_native_multi_control_chains_previous_control_and_reports_budget():
    created = []

    class _NativeControl:
        def __init__(self):
            self.previous = None
            self.hint = None

        def copy(self):
            child = _NativeControl()
            created.append(child)
            return child

        def set_cond_hint(self, hint, strength, window, *, vae):
            self.hint = (hint, strength, window, vae)
            return self

        def set_previous_controlnet(self, previous):
            self.previous = previous
            return self

    class _Model:
        model_options = {}

        def __init__(self, attachments=None):
            self.attachments = dict(attachments or {})

        def clone(self):
            return _Model(self.attachments)

        def get_attachment(self, key):
            return self.attachments.get(key)

        def set_attachments(self, key, value):
            self.attachments[key] = value

    frames = torch.zeros(5, 32, 32, 3)
    positive = [[torch.zeros(1), {}]]
    vae = object()
    bundle = fun.H3FunControlBundle(
        "native", _NativeControl(), "union.safetensors", "union.safetensors", {}
    )
    first_model, first_positive, _ = fun.apply_h3_fun_control(
        _Model(), positive, bundle, vae, frames, 32, 32, 5,
        "depth", "exact", 0.55, 0.0, 0.75,
    )
    second_model, second_positive, report_json = fun.apply_h3_fun_control(
        first_model, first_positive, bundle, vae, frames, 32, 32, 5,
        "pose", "exact", 0.60, 0.1, 0.8,
    )

    first_control = first_positive[0][1]["control"]
    second_control = second_positive[0][1]["control"]
    assert len(created) == 2
    assert first_control.previous is None
    assert second_control.previous is first_control
    attachment = second_model.get_attachment(fun.FUN_CONTROL_ATTACHMENT_KEY)
    assert [item["kind"] for item in attachment["controls"]] == ["depth", "pose"]
    assert attachment["combined_strength"] == pytest.approx(1.15)
    report = json.loads(report_json)
    assert report["control_count"] == 2
    assert report["combined_strength"] == pytest.approx(1.15)
    assert report["warnings"] == [
        "combined control strength exceeds 1.0; saturation is possible"
    ]


def test_loader_uses_framework_structure_not_filename_size_or_hash(monkeypatch, tmp_path):
    path = tmp_path / "arbitrary-user-name.safetensors"
    path.write_bytes(b"not-a-real-checkpoint")
    fake_control = object()
    fake_patcher = object()
    monkeypatch.setattr(fun.folder_paths, "get_full_path_or_raise", lambda *_: str(path))
    monkeypatch.setattr(fun, "_native_fun_control_available", lambda: False)
    monkeypatch.setattr(
        fun,
        "_load_compatibility_control",
        lambda _path: (fake_control, fake_patcher, {"block_count": 5}),
    )
    bundle, report_json = fun.load_h3_fun_control(path.name)
    report = json.loads(report_json)
    assert bundle.filename == path.name
    assert bundle.control == {"model": fake_control, "patcher": fake_patcher}
    assert report["fingerprint_policy"].startswith("diagnostic_only")


def test_loader_prefers_new_official_model_patch_contract(monkeypatch, tmp_path):
    path = tmp_path / "new-official-contract.safetensors"
    path.write_bytes(b"placeholder")
    patcher = SimpleNamespace(model=object())
    monkeypatch.setattr(
        fun, "_resolve_fun_control_path", lambda _name: (str(path), "model_patches")
    )
    monkeypatch.setattr(
        fun,
        "_load_official_model_patch_control",
        lambda _path: (patcher, {"block_count": 5}),
    )
    monkeypatch.setattr(
        fun,
        "_native_fun_control_available",
        lambda: (_ for _ in ()).throw(AssertionError("old loader must not run")),
    )
    bundle, report_json = fun.load_h3_fun_control(path.name)
    report = json.loads(report_json)
    assert bundle.backend == "official_model_patch"
    assert bundle.control is patcher
    assert report["source_folder"] == "model_patches"


def test_new_official_model_patch_apply_preserves_conditioning(monkeypatch):
    registrations = []

    class _Patch:
        def __init__(self, *args):
            self.args = args

        def register(self, model):
            registrations.append((self, model))

    class _Sampling:
        def percent_to_sigma(self, value):
            return 1.0 - value

    class _Model:
        model_options = {"transformer_options": {}}

        def __init__(self):
            self.attachments = {}

        def clone(self):
            return _Model()

        def get_model_object(self, name):
            assert name == "model_sampling"
            return _Sampling()

        def get_attachment(self, key):
            return self.attachments.get(key)

        def set_attachments(self, key, value):
            self.attachments[key] = value

    monkeypatch.setattr(
        fun,
        "_official_model_patch_fun_control_modules",
        lambda: (object(), object(), _Patch),
    )
    monkeypatch.setattr(fun, "_assert_compatible_adaln_pair", lambda *_: None)
    positive = [[torch.zeros(1), {}]]
    patcher = SimpleNamespace(model=object())
    bundle = fun.H3FunControlBundle(
        "official_model_patch", patcher, "official.safetensors", "", {}
    )
    result_model, result_positive, report_json = fun.apply_h3_fun_control(
        _Model(),
        positive,
        bundle,
        object(),
        torch.zeros(5, 32, 32, 3),
        32,
        32,
        5,
        "depth",
        "exact",
        0.7,
        0.0,
        0.75,
    )
    assert result_positive is positive
    assert len(registrations) == 1
    assert registrations[0][1] is result_model
    assert json.loads(report_json)["backend"] == "official_model_patch"


def test_curve_control_rejects_full_width_base_before_vae_encode():
    class _VAE:
        def encode(self, _frames):
            raise AssertionError("AdaLN mismatch must fail before VAE encode")

    base_block = SimpleNamespace(
        adaln_proj=SimpleNamespace(linear=nn.Linear(2688, 2, bias=False))
    )
    control_block = SimpleNamespace(
        adaln_proj=SimpleNamespace(linear=nn.Linear(8, 2, bias=False))
    )
    model = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=SimpleNamespace(blocks=[base_block])),
        model_options={"transformer_options": {}},
    )
    bundle = fun.H3FunControlBundle(
        "compatibility",
        {"model": SimpleNamespace(control_blocks=[control_block])},
        "arbitrary-control.safetensors",
        "arbitrary-control.safetensors",
        {},
    )
    with pytest.raises(RuntimeError, match="2688.*8"):
        fun.apply_h3_fun_control(
            model,
            [[torch.zeros(1), {}]],
            bundle,
            _VAE(),
            torch.zeros(5, 32, 32, 3),
            32,
            32,
            5,
            "depth",
            "exact",
            0.8,
            0.0,
            0.85,
        )


def test_adaln_pairing_uses_live_shapes_not_model_names_or_hashes():
    base = SimpleNamespace(adaln_t_table=torch.zeros(1025, 8))
    control_block = SimpleNamespace(
        adaln_proj=SimpleNamespace(linear=nn.Linear(8, 2, bias=False))
    )
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=base))
    bundle = fun.H3FunControlBundle(
        "compatibility",
        {"model": SimpleNamespace(control_blocks=[control_block])},
        "user-renamed-anything.safetensors",
        "user-renamed-anything.safetensors",
        {},
    )
    fun._assert_compatible_adaln_pair(model, bundle)


def test_registration_is_append_only_and_schema_is_experimental():
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension

    nodes = asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    ids = [node.define_schema().node_id for node in nodes]
    assert ids[226:228] == [
        "MiniMaxH3FunControlLoaderT8Advanced",
        "MiniMaxH3FunControlApplyT8Advanced",
    ]
    for node in nodes[226:228]:
        assert node.define_schema().is_experimental is True
