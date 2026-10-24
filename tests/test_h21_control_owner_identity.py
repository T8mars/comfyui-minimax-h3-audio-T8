"""Desired genuine regional-owner persistence; initially red until implemented.

All sampling uses real native random tiny50+2 DiT, actual mini video Encoder,
native VAE wrapper and real official/compat control towers. Adapter/text are
synthetic. An unknown wrapper remains executable, never implicitly portable.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.ops
import comfy.sample
import comfy.sd
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
import pytest
import torch

from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.control_region_runtime import apply_h3_fun_control_region
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg.progressive_producers import native_producer_identity
from test_h21_control_regions import initialized, source_frames
from test_modular_hyperflow import inputs


def real_native_encoder():
    # Empty native construction initializes its real standard wrapper settings
    # without loading a full architecture. Supply an actual reproducible mini
    # native H3 Encoder and native patcher, never replace encode with a fake.
    value = comfy.sd.VAE(sd={}, device=torch.device("cpu"), dtype=torch.float32)
    network = initialized(
        MiniMaxH3VideoVAE(
            ch=32,
            ch_mult=(1, 1, 1, 1, 1, 1),
            num_res_blocks=0,
            decoder=False,
            tiling=False,
        ),
        211021101,
    )
    value.first_stage_model = network
    value.patcher = ModelPatcher(network, torch.device("cpu"), torch.device("cpu"))
    value.device = value.output_device = torch.device("cpu")
    value.vae_dtype = torch.float32
    value.latent_dim, value.latent_channels = 3, 24
    value.upscale_ratio = value.downscale_ratio = 16
    value.disable_offload = True
    return value


def real_tower(backend, seed):
    arguments = dict(
        control_in_dim=49,
        injection_layers=(0,),
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=16,
        time_embed_dim=4,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    tower = (
        MiniMaxH3FunControl(**arguments)
        if backend == "official_model_patch"
        else fun._MiniMaxH3FunControl(**arguments, use_adaln_curves=False)
    )
    initialized(tower, seed)
    patcher = ModelPatcher(tower, torch.device("cpu"), torch.device("cpu"))
    owner = (
        patcher
        if backend == "official_model_patch"
        else {"model": tower, "patcher": patcher}
    )
    return fun.H3FunControlBundle(
        backend, owner, "arbitrary-user-name.safetensors", "", {}
    ), tower


def build_controls(monkeypatch, backend, count=1, *, unknown_stack=False):
    low, high, source, positive = inputs(monkeypatch)
    if unknown_stack:
        from h3_audio_t8_pkg.modular_sampling.control_owner_identity import KEY

        low.set_attachments(KEY, {"foreign_owner": True})
    encoder = real_native_encoder()
    frames = source_frames(5, width=32, height=32)
    crop = json.dumps(
        {
            "schema": "t8.h3.control_crop.v1",
            "source_size": [32, 32],
            "boxes": [[3, 2, 29, 30]],
            "fit": "stretch",
        }
    )
    mask = torch.ones(1, 32, 32)
    mask[:, :, 16:] = 0.25
    selected, low_states, high_states, towers = low, [], [], []
    for index in range(count):
        bundle, tower = real_tower(backend, 211021103 + index)
        towers.append(tower)
        selected, _, state, _ = apply_h3_fun_control_region(
            selected,
            positive,
            bundle,
            encoder,
            frames * (1 - index * 0.1),
            32,
            32,
            5,
            "pose",
            "exact",
            0.7 - index * 0.2,
            0.0,
            1.0,
            region_mask=mask,
            crop_plan_json=crop,
            stage="LOW",
        )
        low_states.append(state)
        high, _, state, _ = apply_h3_fun_control_region(
            high,
            positive,
            bundle,
            encoder,
            frames * 0.9,
            32,
            32,
            5,
            "pose",
            "exact",
            -0.35 - index * 0.1,
            0.0,
            1.0,
            region_mask=mask.flip(-1),
            crop_plan_json=crop,
            stage="HIGH",
        )
        high_states.append(state)
    return SimpleNamespace(
        model=selected,
        high=high,
        source=source,
        positive=positive,
        encoder=encoder,
        low_states=low_states,
        high_states=high_states,
        towers=towers,
    )


def sample_head(case, *, callback=None):
    return stages.sample_head(
        case.model,
        case.source,
        case.positive,
        [],
        comfy.sample.prepare_noise(case.source["samples"], 211021103),
        seed=211021103,
        split_interval=4,
        callback=callback,
    )


def test_real_native_wrapper_and_actual_encoder_are_not_a_fake_producer():
    encoder = real_native_encoder()
    before = native_producer_identity(encoder, "video_vae")
    assert before.get("portable_cache_reuse") is not False
    assert type(encoder) is comfy.sd.VAE and "encode" not in vars(encoder)
    output = encoder.encode(source_frames(5, width=32, height=32))
    assert output.shape == (1, 24, 2, 2, 2) and torch.isfinite(output).all()
    assert native_producer_identity(encoder, "video_vae") == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_returned_owner_identity_is_detached_from_live_geometry(monkeypatch, backend):
    from h3_audio_t8_pkg.modular_sampling.control_owner_identity import (
        project,
        integrity_snapshot,
    )

    case = build_controls(monkeypatch, backend)
    state = case.low_states[0]
    before = deepcopy(state.geometry)
    _, contract = project(case.model)
    snapshot = integrity_snapshot(state.owner)
    assert contract["controls"][0]["geometry"] is not state.geometry
    assert snapshot["geometry"] is not state.geometry
    contract["controls"][0]["geometry"]["target"]["width"] *= 2
    snapshot["geometry"]["crop_spec"]["boxes"][0][0] += 1
    assert state.geometry == before
    assert sample_head(case).verify()["execution"]["portable_identity"] is True


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("count", [1, 2])
def test_exact_own_control_head_has_real_content_portability_without_counter_instability(
    monkeypatch, backend, count
):
    case = build_controls(monkeypatch, backend, count)
    wrappers = {
        kind: {key: list(values) for key, values in group.items()}
        for kind, group in case.model.wrappers.items()
    }
    hooks = dict(
        case.model.model_options["transformer_options"]["patches_replace"]["dit"]
    )
    boundary = sample_head(case)
    receipt = boundary.verify()
    assert receipt["execution"]["actual_apply_intervals"] == [0, 1, 2, 3]
    assert all(state.gated_calls == 4 for state in case.low_states)
    assert receipt["execution"]["portable_identity"] is True, receipt["request"][
        "model"
    ]
    assert case.model.wrappers == wrappers
    assert (
        case.model.model_options["transformer_options"]["patches_replace"]["dit"]
        == hooks
    )


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_genuine_control_head_roundtrip_really_replays_new_process_tail_only(
    tmp_path, monkeypatch, backend
):
    case = build_controls(monkeypatch, backend, 2)
    boundary = sample_head(case)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    expected, _ = stages.sample_tail(
        boundary, case.high, case.positive, [], seed=211021103
    )
    code = r"""
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import comfy.sample, pytest, torch
from test_h21_control_owner_identity import build_controls
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling.hyperflow_storage import load_boundary
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy
def forbidden(*args, **kwargs):
    raise AssertionError('HEAD, old whole split or new noise in cold TAIL-only replay')
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(stages, 'sample_head', forbidden)
    patch.setattr(legacy, 'sample_hyperflow_split', forbidden)
    patch.setattr(comfy.sample, 'prepare_noise', forbidden)
    case = build_controls(patch, sys.argv[4], 2)
    boundary, loaded = load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
    assert json.loads(loaded)['sampling_calls'] == 0
    output, raw = stages.sample_tail(boundary, case.high, case.positive, [], seed=211021103)
report = json.loads(raw)
assert report['execution']['actual_apply_intervals'] == [4,5,6,7]
assert report['execution']['portable_identity'] is True
assert report['head_executed'] is report['fresh_noise'] is False
assert all(state.gated_calls == 4 for state in case.high_states)
assert not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(output)))
"""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            "-c",
            code,
            str(tmp_path),
            path,
            digest,
            backend,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(
        next(
            line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")
        )
    )
    assert actual == stages.snapshot(expected)


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_foreign_control_wrapper_still_executes_without_portable_promotion(
    monkeypatch, backend
):
    case = build_controls(monkeypatch, backend)
    calls = []

    def foreign(executor, *args, **kwargs):
        calls.append(1)
        return executor(*args, **kwargs)

    case.model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL, "foreign-control-wrapper", foreign
    )
    boundary = sample_head(case)
    assert calls and case.low_states[0].gated_calls == 4
    assert boundary.verify()["execution"]["portable_identity"] is False
    assert case.model.get_wrappers(
        WrappersMP.DIFFUSION_MODEL, "foreign-control-wrapper"
    ) == [foreign]


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("target", ["control_weights", "geometry", "region_mask"])
def test_same_model_actual_control_content_mutation_never_issues_completed_receipt(
    monkeypatch, backend, target
):
    case = build_controls(monkeypatch, backend)
    state = case.low_states[0]
    geometry_before = deepcopy(state.geometry)
    mask_before = state.gate.mask.clone()
    weight = next(case.towers[0].parameters())
    weight_before = weight.detach().clone()

    def mutate(index, *_args):
        if index != 0:
            return
        if target == "control_weights":
            with torch.no_grad():
                weight.add_(0.125)
        elif target == "geometry":
            state.geometry["crop_spec"]["boxes"][0][0] += 1
        else:
            state.gate.mask[0, 0, 0] = 0

    try:
        with pytest.raises(
            ValueError, match="changed|geometry|mask|owner|control|Control|MODEL"
        ):
            sample_head(case, callback=mutate)
    finally:
        with torch.no_grad():
            weight.copy_(weight_before)
        state.geometry = geometry_before
        state.gate.mask.copy_(mask_before)
    assert not torch.cuda.is_initialized()


def add_foreign_wrapper(case):
    calls = []

    def foreign(executor, *args, **kwargs):
        calls.append(1)
        return executor(*args, **kwargs)

    case.model.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL, "foreign-integrity-probe", foreign
    )
    return calls


def closure_cell(function, name):
    return dict(
        zip(function.__code__.co_freevars, function.__closure__ or (), strict=True)
    )[name]


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize(
    "target",
    [
        "strength",
        "window",
        "prepared_RGB",
        "helper_code",
        "predecessor",
        "region_cache",
    ],
)
def test_nonportable_foreign_wrapper_does_not_hide_actual_own_control_mutations(
    monkeypatch, backend, target
):
    case = build_controls(monkeypatch, backend)
    calls = add_foreign_wrapper(case)
    owner = case.low_states[0].owner
    _, hook, previous = owner.hooks[0]
    restore = []

    def mutate(index, *_args):
        if index != 0:
            return
        if target in {"strength", "window"}:
            if backend == "compatibility":
                function = (
                    hook
                    if target == "strength"
                    else closure_cell(hook, "active").cell_contents
                )
                cell = closure_cell(
                    function, "strength" if target == "strength" else "end_sigma"
                )
                original = cell.cell_contents
                restore.append(lambda: setattr(cell, "cell_contents", original))
                cell.cell_contents = 0.875 if target == "strength" else 0.8
            else:
                name = "strength" if target == "strength" else "sigma_end"
                original = getattr(owner.patch, name)
                restore.append(lambda: setattr(owner.patch, name, original))
                setattr(owner.patch, name, 0.875 if target == "strength" else 0.8)
        elif target == "prepared_RGB":
            original = owner.prepared.clone()
            restore.append(lambda: owner.prepared.copy_(original))
            owner.prepared.add_(0.01)
        elif target == "helper_code":
            function = (
                fun._enter_control_stream
                if backend == "compatibility"
                else type(owner.patch).after_block
            )
            original = function.__code__
            restore.append(lambda: setattr(function, "__code__", original))
            # Same numerical behavior, changed actual executing code contents.
            function.__code__ = original.replace(
                co_consts=(*original.co_consts, "changed_control_owner_code")
            )
        elif target == "predecessor":

            def replacement(args, extra):
                return (
                    extra["original_block"](args)
                    if previous is None
                    else previous(args, extra)
                )

            if backend == "compatibility":
                cell = closure_cell(hook, "inner")
                restore.append(lambda: setattr(cell, "cell_contents", previous))
                cell.cell_contents = replacement
            else:
                restore.append(lambda: setattr(hook, "previous", previous))
                hook.previous = replacement
        else:
            cache = owner.runtime.gate._cache
            assert cache  # Actual first denoiser populated its native row gate.
            value = next(iter(cache.values()))
            original = value.clone()
            restore.append(lambda: value.copy_(original))
            value.add_(0.01)

    try:
        with pytest.raises(
            ValueError, match="changed|geometry|mask|owner|control|Control|MODEL"
        ):
            sample_head(case, callback=mutate)
    finally:
        with torch.inference_mode():
            for action in reversed(restore):
                action()
    assert calls and not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "backend,target",
    [("compatibility", "encoded_rows"), ("official_model_patch", "derived_latent")],
)
def test_actual_derived_control_data_mutation_is_rejected_with_nonportable_wrapper(
    monkeypatch, backend, target
):
    case = build_controls(monkeypatch, backend)
    add_foreign_wrapper(case)
    owner = case.low_states[0].owner
    saved = []

    def mutate(index, *_args):
        if index == 0:
            value = (
                owner.rows if target == "encoded_rows" else owner.patch.control_latent
            )
            assert isinstance(value, torch.Tensor)
            saved.append((value, value.clone()))
            value.add_(0.01)

    try:
        with pytest.raises(ValueError, match="changed|owner|control|Control|MODEL"):
            sample_head(case, callback=mutate)
    finally:
        with torch.inference_mode():
            for value, original in saved:
                value.copy_(original)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_identity_and_derived_validation_do_not_add_encoder_forwards(
    monkeypatch, backend
):
    case = build_controls(monkeypatch, backend)
    calls = []
    handle = case.encoder.first_stage_model.encoder.register_forward_hook(
        lambda *_args: calls.append(1)
    )
    try:
        add_foreign_wrapper(case)
        boundary = sample_head(case)
        assert boundary.verify()["execution"]["portable_identity"] is False
    finally:
        handle.remove()
    assert len(calls) == (0 if backend == "compatibility" else 1)
    assert case.low_states[0].gated_calls == 4


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_unknown_existing_control_owner_stack_keeps_actual_execution_nonportable(
    monkeypatch, caplog, backend
):
    case = build_controls(monkeypatch, backend, unknown_stack=True)
    boundary = sample_head(case)
    assert boundary.verify()["execution"]["portable_identity"] is False
    assert case.low_states[0].gated_calls == 4
    assert case.low_states[0].owner is None
    assert "control_owner_stack_unknown_portable_identity_unavailable" in caplog.text
    assert not torch.cuda.is_initialized()
