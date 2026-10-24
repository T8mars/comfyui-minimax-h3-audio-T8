"""Actual node schemas, temporal RGB execution and typed HEAD dependencies."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
import uuid

import comfy.ops
import comfy.sample
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy.model_patcher import ModelPatcher
import pytest
import torch

from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.control_region_runtime import (
    ControlRegionRuntime,
    apply_h3_fun_control_region,
)
from h3_audio_t8_pkg.control_regions import canonical
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling.hyperflow_nodes import BOUNDARY
from test_h21_control_regions import RealTinyEncoderOwner, initialized, source_frames
from test_modular_hyperflow import inputs, head


def module():
    return importlib.import_module("h3_audio_t8_pkg.nodes_director_controls")


def row(**extra):
    return {"id": str(uuid.uuid4()), "asset_id": None, **extra}


def source_node(frames, spec=None, **extra):
    values = dict(
        control_video=frames,
        spec_json=json.dumps(row() if spec is None else spec),
        width=64,
        height=32,
        length=5,
        source_kind="image",
        source_fps=24.0,
        stage="SINGLE",
    )
    values.update(extra)
    return module().MiniMaxH3DirectorControlSourceEXPT8.execute(**values).result


def test_two_node_schemas_are_append_only_typed_and_have_no_new_geometry_caps():
    api = module()
    assert len(api.NODES) == 2
    source, audit = [cls.GET_SCHEMA() for cls in api.NODES]
    assert source.node_id == "MiniMaxH3DirectorControlSourceEXPT8"
    assert [value.id for value in source.inputs] == [
        "control_video",
        "spec_json",
        "width",
        "height",
        "length",
        "source_kind",
        "source_fps",
        "stage",
    ]
    assert [value.id for value in source.outputs] == [
        "control_video",
        "region_mask",
        "crop_plan_json",
        "report_json",
    ]
    fields = {value.id: value for value in source.inputs}
    assert fields["width"].step == fields["height"].step == 32
    assert fields["width"].max is fields["height"].max is fields["length"].max is None
    assert fields["source_fps"].max is None and fields["length"].step == 17
    assert audit.is_output_node and audit.is_experimental and source.is_experimental
    assert [value.id for value in audit.inputs] == [
        "region_state",
        "continuous_boundary",
    ]
    assert [value.id for value in audit.outputs] == [
        "continuous_boundary",
        "report_json",
    ]
    data = api.MiniMaxH3DirectorControlHeadAuditEXPT8.GET_NODE_INFO_V1()
    assert BOUNDARY in json.dumps(data) and "H3_T8_CONTROL_REGION_STATE" in json.dumps(
        data
    )
    assert "LATENT" not in json.dumps(data)


def test_actual_source_node_returns_full_none_mask_and_preserves_original_rgb():
    frames, spec = source_frames(3, width=75, height=41), row()
    original, saved = frames.clone(), deepcopy(spec)
    image, mask, crop, raw = source_node(frames, spec)
    report = json.loads(raw)
    assert image.shape == (1, 41, 75, 3) and torch.equal(image, frames[:1])
    assert mask is None and crop == ""
    assert report["source_kind"] == "image" and report["source_fps"] is None
    assert report["quality_accepted"] is report["portable_cache_identity"] is False
    assert torch.equal(frames, original) and spec == saved


def test_actual_source_node_connects_real_video_fps_and_stage_geometry():
    frames = source_frames(5)
    spec = row(region={"mode": "rect", "xyxy_normalized": [0, 0, 0.5, 1]})
    image, mask, crop, raw = source_node(
        frames,
        spec,
        source_kind="video",
        source_fps=12.0,
        width=128,
        height=64,
        stage="HIGH",
    )
    report = json.loads(raw)
    assert report["source_indices"] == [0, 0, 1, 2, 2]
    assert torch.equal(image, frames[torch.tensor([0, 0, 1, 2, 2])])
    assert mask.shape == (1, 64, 128) and mask.sum() == 4096 and crop == ""
    assert report["stage"] == "HIGH" and report["source_fps"] == 12.0


@pytest.mark.parametrize(
    "invalid",
    ["not-json", "[]", "null", '{"version":2,"items":[]}', '{"version":1,"items":[]}'],
)
def test_source_node_bad_json_or_envelope_cannot_silently_ignore_the_version(invalid):
    frames = source_frames(1)
    before = frames.clone()
    with pytest.raises(ValueError, match="spec_json|单条|JSON|版本"):
        source_node(frames, spec_json=invalid)
    assert torch.equal(frames, before)


@pytest.fixture(scope="module")
def encoder():
    return RealTinyEncoderOwner()


def real_head_with_control(
    monkeypatch, encoder, backend, *, strength=0.7, via_node=False
):
    low, _, source, positive = inputs(monkeypatch)
    if backend == "official_model_patch":
        tower = MiniMaxH3FunControl(
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
    else:
        tower = fun._MiniMaxH3FunControl(
            control_in_dim=49,
            injection_layers=(0,),
            hidden_size=8,
            num_attention_heads=2,
            attention_head_dim=8,
            ffn_hidden_size=16,
            time_embed_dim=4,
            use_adaln_curves=False,
            dtype=torch.float32,
            device="cpu",
            operations=comfy.ops.disable_weight_init,
        )
    initialized(tower, 21102138)
    patcher = ModelPatcher(tower, torch.device("cpu"), torch.device("cpu"))
    holder = (
        patcher
        if backend == "official_model_patch"
        else {"model": tower, "patcher": patcher}
    )
    bundle = fun.H3FunControlBundle(
        backend, holder, "unknown-source-name.safetensors", "", {}
    )
    frames = source_frames(5, width=32, height=32)
    apply = apply_h3_fun_control_region
    if via_node:
        from h3_audio_t8_pkg.nodes_control_region import (
            MiniMaxH3FunControlRegionApplyEXPT8,
        )

        def apply(*args, **kwargs):
            return MiniMaxH3FunControlRegionApplyEXPT8.execute(*args, **kwargs).result

    selected, positive, runtime, configuration = apply(
        low,
        positive,
        bundle,
        encoder,
        frames,
        32,
        32,
        5,
        "pose",
        "exact",
        strength,
        0.0,
        1.0,
        region_mask=torch.ones(1, 32, 32),
        stage="HEAD",
    )
    assert json.loads(configuration)["control"]["strength"] == strength
    assert (
        selected.get_attachment(fun.FUN_CONTROL_ATTACHMENT_KEY)["combined_strength"]
        == strength
    )
    boundary = head(selected, source, positive)
    receipt = boundary.verify()
    assert receipt["execution"]["actual_apply_intervals"] == [0, 1, 2, 3]
    assert receipt["execution"]["portable_identity"] is (backend == "compatibility")
    if backend == "compatibility":
        contract = receipt["request"]["model"]["stage_effects"]["regional_control"]
        assert contract["schema"] == "t8.h3.control-owner-content.v1"
        assert (
            len(contract["controls"]) == 1
            and contract["controls"][0]["encoded_rows"] is not None
        )
    assert runtime.report()["gated_calls"] == 4
    return boundary, runtime


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("strength", [7.0, -3.5])
def test_actual_region_node_and_tower_keep_positive_and_negative_finite_strengths(
    monkeypatch, encoder, backend, strength
):
    from h3_audio_t8_pkg.nodes_control_region import MiniMaxH3FunControlRegionApplyEXPT8
    from h3_audio_t8_pkg.nodes_h3_fun_control_advanced import (
        MiniMaxH3FunControlApplyT8Advanced,
    )

    current = {
        item.id: item
        for item in MiniMaxH3FunControlRegionApplyEXPT8.GET_SCHEMA().inputs
    }
    legacy = {
        item.id: item for item in MiniMaxH3FunControlApplyT8Advanced.GET_SCHEMA().inputs
    }
    assert current["strength"].min is current["strength"].max is None
    assert (legacy["strength"].min, legacy["strength"].max) == (0.0, 2.0)
    assert current["strength"].default == legacy["strength"].default == 0.7
    boundary, state = real_head_with_control(
        monkeypatch, encoder, backend, strength=strength, via_node=True
    )
    actual, raw = (
        module().MiniMaxH3DirectorControlHeadAuditEXPT8.execute(state, boundary).result
    )
    assert actual is boundary and torch.isfinite(actual.x_sigma).all()
    report = json.loads(raw)
    assert report["region"]["gated_calls"] == 4
    assert report["head_actual_apply_intervals"] == [0, 1, 2, 3]
    assert report["portable_cache_identity"] is report["quality_accepted"] is False


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_head_audit_observes_real_control_and_head_without_resampling_or_changing_receipt(
    monkeypatch, encoder, backend
):
    boundary, runtime = real_head_with_control(monkeypatch, encoder, backend)
    before = boundary.receipt_json
    digest = boundary.verify()["receipt_sha256"]
    raw_state = boundary.x_sigma.clone()
    runtime_before = runtime.report()

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must never run HEAD or prepare noise")

    monkeypatch.setattr(stages, "sample_head", forbidden)
    monkeypatch.setattr(comfy.sample, "prepare_noise", forbidden)
    actual, raw = (
        module()
        .MiniMaxH3DirectorControlHeadAuditEXPT8.execute(runtime, boundary)
        .result
    )
    report = json.loads(raw)
    assert actual is boundary and actual.receipt_json == before
    assert actual.verify()["receipt_sha256"] == digest and torch.equal(
        actual.x_sigma, raw_state
    )
    assert runtime.report() == runtime_before
    assert report["region"]["gated_calls"] == 4 and report["sampling_calls"] == 0
    assert report["head_portable_identity"] is (backend == "compatibility")
    assert report["portable_cache_identity"] is report["quality_accepted"] is False
    assert report["runtime_binding"] == "dependency_only_not_immutable_execution_owner"


def test_unused_runtime_reports_not_executed_without_claiming_a_portable_head_effect(
    monkeypatch,
):
    low, _, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    state = ControlRegionRuntime(stage="HEAD")
    actual, raw = (
        module().MiniMaxH3DirectorControlHeadAuditEXPT8.execute(state, boundary).result
    )
    report = json.loads(raw)
    assert actual is boundary and report["region"]["status"] == "not_executed"
    assert report["portable_cache_identity"] is False
    assert report["head_portable_identity"] is True


@pytest.mark.parametrize("target", ["state", "boundary"])
def test_wrong_typed_inputs_fail_before_any_sampling_or_claim(target):
    state = ControlRegionRuntime(stage="HEAD")
    with pytest.raises(TypeError):
        module().MiniMaxH3DirectorControlHeadAuditEXPT8.execute(
            {} if target == "state" else state, {} if target == "boundary" else None
        )


@pytest.mark.parametrize(
    "fault",
    ["bad_receipt", "negative_calls", "gated_above_calls", "rows", "geometry", "mask"],
)
def test_real_head_integrity_and_runtime_statistics_cannot_be_promoted_after_tamper(
    monkeypatch, encoder, fault
):
    boundary, state = real_head_with_control(monkeypatch, encoder, "compatibility")
    if fault == "bad_receipt":
        boundary = replace(
            boundary,
            receipt_json=boundary.receipt_json.replace(
                '"callbacks":[0,1,2,3]', '"callbacks":[]'
            ),
        )
    elif fault == "negative_calls":
        state.calls = -1
    elif fault == "gated_above_calls":
        state.gated_calls = state.calls + 1
    elif fault == "rows":
        state.target_row_evaluations = 0
    elif fault == "geometry":
        state.geometry["target"]["width"] *= 2
        payload = {
            key: value
            for key, value in state.geometry.items()
            if key != "geometry_sha256"
        }
        state.geometry["geometry_sha256"] = hashlib.sha256(
            canonical(payload).encode()
        ).hexdigest()
    else:
        state.gate.mask[0, 0, 0] = 0
    with pytest.raises(ValueError):
        module().MiniMaxH3DirectorControlHeadAuditEXPT8.execute(state, boundary)


def test_audit_fingerprint_is_fresh_json_safe_not_nan():
    cls = module().MiniMaxH3DirectorControlHeadAuditEXPT8
    first, second = cls.fingerprint_inputs(), cls.fingerprint_inputs()
    assert isinstance(first, str) and len(first) == 32 and first != second
    assert json.dumps([first, second], allow_nan=False)
