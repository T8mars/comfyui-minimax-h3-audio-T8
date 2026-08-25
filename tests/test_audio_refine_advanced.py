from __future__ import annotations

import asyncio
import json
from pathlib import Path

import comfy.nested_tensor
import pytest
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

import h3_audio_t8_pkg
from h3_audio_t8_pkg.audio_refine_advanced import (
    AUDIO_REFINE_AUDIT_SCHEMA,
    AUDIO_REFINE_AUDIT_TYPE,
    AUDIO_REFINE_PLAN_SCHEMA,
    AUDIO_REFINE_PLAN_TYPE,
    AudioRefineBasicGuider,
    AudioRefineBypassNoise,
    AudioRefineRandomNoise,
    audit_audio_refine,
    canonical_json,
    classify_audio_refine_latent,
    plan_audio_refine,
    setup_audio_refine,
)
from h3_audio_t8_pkg.nodes_audio_refine_advanced import (
    AUDIO_REFINE_ADVANCED_NODE_CLASSES,
    MiniMaxH3AudioRefineAuditT8Advanced,
    MiniMaxH3AudioRefineDualClockSetupT8Advanced,
    MiniMaxH3AudioRefinePlanT8Advanced,
)


def _latent(*, audio_offset: float = 0.0) -> dict:
    video = torch.arange(1 * 24 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1, 24, 2, 2, 2
    )
    audio = torch.arange(1 * 32 * 2 * 3, dtype=torch.float32).reshape(
        1, 32, 2, 3
    )
    audio = audio + float(audio_offset)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
    }


def _positive(*, offset: float = 0.0):
    return [
        [
            torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + offset,
            {"pooled_output": torch.arange(4, dtype=torch.float32).reshape(1, 4)},
        ]
    ]


class MiniMaxH3DiffusionModel:
    pass


class MiniMaxH3BaseModel:
    def __init__(self):
        self.diffusion_model = MiniMaxH3DiffusionModel()


class ModelSamplingFlow:
    pass


class FakeModelPatcher:
    def __init__(
        self,
        *,
        patches_replace=None,
        wrappers=None,
        object_patches=None,
        shift_video=None,
        shift_audio=None,
    ):
        self.model = MiniMaxH3BaseModel()
        self.model_sampling = ModelSamplingFlow()
        transformer_options = {
            "patches_replace": patches_replace or {},
            "wrappers": wrappers or {},
        }
        if shift_video is not None:
            transformer_options["minimax_h3_sigma_shift_video"] = shift_video
        if shift_audio is not None:
            transformer_options["minimax_h3_sigma_shift_audio"] = shift_audio
        self.model_options = {
            "transformer_options": transformer_options
        }
        self.patches = {}
        self.object_patches = dict(object_patches or {})
        self.attachments = {}

    def is_dynamic(self):
        return False

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.model_sampling
        raise KeyError(name)

    def clone(self):
        clone = FakeModelPatcher()
        clone.model = self.model
        clone.model_sampling = self.model_sampling
        clone.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        clone.patches = dict(self.patches)
        clone.object_patches = dict(self.object_patches)
        clone.attachments = dict(self.attachments)
        return clone


def _runtime(*, free_mib: float = 4096.0, commit_gib: float = 64.0):
    return {
        "gpu": {"whole_device_free_mib": free_mib},
        "host": {
            "ram_available_gib": 64.0,
            "commit_headroom_gib": commit_gib,
        },
    }


def _audit(**overrides):
    values = {
        "model": FakeModelPatcher(),
        "positive": _positive(),
        "av_latent": _latent(),
        "conditioned_prompt": "A woman speaks clearly.",
        "media_map_json": json.dumps(
            {"pictures": {}, "videos": {}, "audios": {}}
        ),
        "conditioning_report": "task=I2VA\naudio_mode=native\nframes=22",
        "minimum_free_vram_mib": 512,
        "minimum_commit_headroom_gib": 16.0,
        "hash_chunk_megabytes": 1,
        "runtime_snapshot_fn": _runtime,
    }
    values.update(overrides)
    runtime = values.get("runtime_snapshot_fn")
    if isinstance(runtime, dict):
        values["runtime_snapshot_fn"] = lambda: runtime
    return audit_audio_refine(**values)


def test_audio_refine_contract_constants_are_stable():
    assert AUDIO_REFINE_AUDIT_TYPE == "H3_T8_AUDIO_REFINE_AUDIT"
    assert AUDIO_REFINE_PLAN_TYPE == "H3_T8_AUDIO_REFINE_PLAN"
    assert AUDIO_REFINE_AUDIT_SCHEMA == "t8.minimax_h3.audio_refine.audit.v1"
    assert AUDIO_REFINE_PLAN_SCHEMA == "t8.minimax_h3.audio_refine.plan.v1"


def test_joint_av_manifest_is_content_bound_and_deterministic():
    first = classify_audio_refine_latent(_latent())
    second = classify_audio_refine_latent(_latent())

    assert canonical_json(first["manifest"]) == canonical_json(second["manifest"])
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest"]["video"]["shape"] == [1, 24, 2, 2, 2]
    assert first["manifest"]["audio"]["shape"] == [1, 32, 2, 3]

    changed = classify_audio_refine_latent(_latent(audio_offset=1.0))
    assert changed["manifest_sha256"] != first["manifest_sha256"]


def test_audit_allows_native_full_audio_with_clean_stack():
    audit, decision, report = _audit()

    assert decision == "ALLOW"
    assert audit["schema"] == AUDIO_REFINE_AUDIT_SCHEMA
    assert audit["decision"] == "ALLOW"
    assert audit["reason_codes"] == []
    assert json.loads(report)["decision"] == "ALLOW"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("lock_source", "ABSTAIN_AUDIO_LOCKED"),
        ("remix_source", "ABSTAIN_REMIX_SOURCE_NOT_VALIDATED"),
    ],
)
def test_audit_abstains_for_protected_source_modes(mode, expected):
    audit, decision, _ = _audit(
        conditioning_report=f"task=I2VA\naudio_mode={mode}\nframes=22"
    )

    assert decision == "ABSTAIN"
    assert expected in audit["reason_codes"]


def test_audit_abstains_when_headroom_is_unknown_or_below_floor():
    assert _audit(runtime_snapshot_fn={"gpu": {}, "host": {}})[1] == "ABSTAIN"
    assert _audit(runtime_snapshot_fn=_runtime(free_mib=511))[1] == "ABSTAIN"
    assert _audit(runtime_snapshot_fn=_runtime(commit_gib=15.99))[1] == "ABSTAIN"


def test_audit_resource_floors_cannot_be_lowered():
    audit, decision, _ = _audit(
        minimum_free_vram_mib=1,
        minimum_commit_headroom_gib=0.0,
        runtime_snapshot_fn=_runtime(free_mib=511, commit_gib=15.99),
    )

    assert decision == "ABSTAIN"
    assert audit["resource_gates"]["minimum_free_vram_mib"] == 512.0
    assert audit["resource_gates"]["minimum_commit_headroom_gib"] == 16.0


def test_audit_abstains_for_transformer_patch_replacement():
    model = FakeModelPatcher(
        patches_replace={"attention": {("double_block", 0): object()}}
    )
    audit, decision, _ = _audit(model=model)

    assert decision == "ABSTAIN"
    assert "ABSTAIN_PATCH_STACK_UNVALIDATED" in audit["reason_codes"]


def test_audit_abstains_for_transformer_wrapper():
    audit, decision, _ = _audit(
        model=FakeModelPatcher(wrappers={"diffusion_model": [object()]})
    )

    assert decision == "ABSTAIN"
    assert "ABSTAIN_PATCH_STACK_UNVALIDATED" in audit["reason_codes"]


def test_audit_allows_the_exact_existing_t8_dual_clock_model_patch():
    model_sampling = ModelSamplingFlow()
    model = FakeModelPatcher(
        object_patches={"model_sampling": model_sampling},
        shift_video=12.0,
        shift_audio=3.0,
    )
    model.model_sampling = model_sampling

    audit, decision, _ = _audit(model=model)

    assert decision == "ALLOW"
    assert audit["model_manifest"]["t8_dual_clock_patch"] == "validated_12_3"


@pytest.mark.parametrize(
    "model",
    [
        FakeModelPatcher(
            object_patches={"model_sampling": ModelSamplingFlow()},
            shift_video=6.0,
            shift_audio=3.0,
        ),
        FakeModelPatcher(object_patches={"unknown_runtime_patch": object()}),
    ],
)
def test_audit_abstains_for_nonstandard_or_unknown_object_patches(model):
    audit, decision, _ = _audit(model=model)

    assert decision == "ABSTAIN"
    assert "ABSTAIN_PATCH_STACK_UNVALIDATED" in audit["reason_codes"]


def test_audit_classifies_audio_masks_without_mutating_latent():
    locked = _latent()
    video, audio = locked["samples"].unbind()
    locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.zeros_like(video), torch.zeros_like(audio))
    )
    locked_audit, locked_decision, _ = _audit(av_latent=locked)
    assert locked_decision == "ABSTAIN"
    assert "ABSTAIN_AUDIO_LOCKED" in locked_audit["reason_codes"]

    fractional = _latent()
    video, audio = fractional["samples"].unbind()
    fractional["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.zeros_like(video), torch.full_like(audio, 0.2))
    )
    fractional_audit, fractional_decision, _ = _audit(av_latent=fractional)
    assert fractional_decision == "ABSTAIN"
    assert (
        "ABSTAIN_PARTIAL_AUDIO_MASK_UNSUPPORTED"
        in fractional_audit["reason_codes"]
    )


def test_audit_allows_full_audio_mask_and_native_legacy_video_mask():
    full = _latent()
    video, audio = full["samples"].unbind()
    full["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.zeros_like(video), torch.ones_like(audio))
    )
    assert _audit(av_latent=full)[1] == "ALLOW"

    legacy = _latent()
    legacy["noise_mask"] = torch.zeros_like(video)
    audit, decision, _ = _audit(av_latent=legacy)
    assert decision == "ALLOW"
    assert "WARN_LEGACY_VIDEO_ONLY_MASK" in audit["warning_codes"]


def test_audit_abstains_for_protected_final_audio():
    audit, decision, _ = _audit(
        protected_audio={"waveform": torch.zeros((1, 2, 16)), "sample_rate": 32000}
    )

    assert decision == "ABSTAIN"
    assert "ABSTAIN_PROTECTED_FINAL_AUDIO" in audit["reason_codes"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"media_map_json": "not-json"}, "REJECT_CONDITIONING_CONTRACT_INVALID"),
        (
            {"conditioning_report": "task=I2VA\nframes=22"},
            "REJECT_AUDIO_MODE_AMBIGUOUS",
        ),
    ],
)
def test_audit_rejects_malformed_conditioning_contract(overrides, reason):
    audit, decision, _ = _audit(**overrides)

    assert decision == "REJECT"
    assert reason in audit["reason_codes"]


def test_audit_rejects_invalid_latent_or_wrong_model():
    invalid = _latent()
    video, audio = invalid["samples"].unbind()
    audio[0, 0, 0, 0] = float("nan")
    audit, decision, _ = _audit(av_latent=invalid)
    assert decision == "REJECT"
    assert "REJECT_INVALID_AV_LATENT" in audit["reason_codes"]

    wrong_model = object()
    audit, decision, _ = _audit(model=wrong_model)
    assert decision == "REJECT"
    assert "REJECT_NOT_MINIMAX_H3_MODEL" in audit["reason_codes"]


def test_audit_conditioning_binding_changes_with_tensor_content():
    first = _audit()[0]
    changed = _audit(positive=_positive(offset=1.0))[0]

    assert first["bindings"]["run_contract_sha256"] != changed["bindings"][
        "run_contract_sha256"
    ]


def _allow_audit():
    audit, decision, _ = _audit()
    assert decision == "ALLOW"
    return audit


def _abstain_audit():
    audit, decision, _ = _audit(
        conditioning_report="task=I2VA\naudio_mode=lock_source\nframes=22"
    )
    assert decision == "ABSTAIN"
    return audit


def test_default_partial_tail_matches_reference():
    plan, decision, report = plan_audio_refine(_allow_audit(), 4, 0.5, 42)

    assert decision == "ALLOW"
    assert plan["full_steps"] == 8
    assert plan["actual_refine_nfe"] == 4
    assert plan["base_sigmas"] == pytest.approx([0.5, 0.375, 0.25, 0.125, 0])
    assert plan["video_sigmas"] == pytest.approx(
        [12 / 13, 4.5 / 5.125, 0.8, 1.5 / 2.375, 0]
    )
    assert plan["audio_sigmas"] == pytest.approx([0.75, 9 / 14, 0.5, 0.3, 0])
    assert json.loads(report)["payload_sha256"] == plan["payload_sha256"]


@pytest.mark.parametrize("steps,denoise", [(1, 0.35), (4, 0.35), (6, 0.5), (8, 1.0)])
def test_partial_tail_uses_current_ksampler_integer_rule(steps, denoise):
    plan, decision, _ = plan_audio_refine(_allow_audit(), steps, denoise, 1)

    assert decision == "ALLOW"
    assert plan["full_steps"] == int(steps / denoise)
    assert len(plan["video_sigmas"]) == steps + 1
    assert plan["effective_audio_denoise"] == pytest.approx(
        steps / int(steps / denoise)
    )


def test_plan_never_upgrades_audit_decision():
    plan, decision, _ = plan_audio_refine(_abstain_audit(), 4, 0.5, 1)

    assert decision == "ABSTAIN"
    assert plan["decision"] == "ABSTAIN"
    assert "ABSTAIN_AUDIO_LOCKED" in plan["reason_codes"]


@pytest.mark.parametrize(
    ("steps", "denoise", "seed", "strategy"),
    [
        (0, 0.5, 1, "connected_model_explicit"),
        (9, 0.5, 1, "connected_model_explicit"),
        (4, 0.0, 1, "connected_model_explicit"),
        (4, 1.01, 1, "connected_model_explicit"),
        (4, float("nan"), 1, "connected_model_explicit"),
        (4, 0.5, -1, "connected_model_explicit"),
        (4, 0.5, 1, "implicit_model_switch"),
    ],
)
def test_plan_rejects_invalid_or_implicit_parameters(steps, denoise, seed, strategy):
    with pytest.raises(ValueError):
        plan_audio_refine(
            _allow_audit(),
            steps,
            denoise,
            seed,
            model_strategy=strategy,
        )


def test_plan_rejects_tampered_audit_descriptor():
    audit = _allow_audit()
    audit["audio_mode"] = "lock_source"

    with pytest.raises(ValueError, match="REJECT_DESCRIPTOR_TAMPERED"):
        plan_audio_refine(audit, 4, 0.5, 1)


def _allow_bundle(*, seed: int = 42):
    model = FakeModelPatcher()
    positive = _positive()
    latent = _latent()
    audit, decision, _ = _audit(
        model=model,
        positive=positive,
        av_latent=latent,
    )
    assert decision == "ALLOW"
    plan, decision, _ = plan_audio_refine(audit, 4, 0.5, seed)
    assert decision == "ALLOW"
    return model, positive, latent, plan


def _abstain_bundle():
    model = FakeModelPatcher()
    positive = _positive()
    latent = _latent()
    audit, decision, _ = _audit(
        model=model,
        positive=positive,
        av_latent=latent,
        conditioning_report="task=I2VA\naudio_mode=lock_source\nframes=22",
    )
    assert decision == "ABSTAIN"
    plan, decision, _ = plan_audio_refine(audit, 4, 0.5, 42)
    assert decision == "ABSTAIN"
    return model, positive, latent, plan


def _full_video_sigmas(steps: int):
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return 12.0 * base / (1.0 + 11.0 * base)


def test_setup_allow_uses_full_schedule_then_exact_tail():
    model, positive, latent, plan = _allow_bundle()
    called = {}
    sentinel_sampler = object()

    def fake_setup(
        connected_model,
        connected_latent,
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    ):
        called.update(
            model=connected_model,
            latent=connected_latent,
            steps=steps,
            shifts=(shift_video, shift_audio),
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        return connected_model.clone(), sentinel_sampler, _full_video_sigmas(steps)

    result = setup_audio_refine(
        plan=plan,
        model=model,
        positive=positive,
        av_latent=latent,
        setup_sampling_fn=fake_setup,
        runtime_snapshot_fn=_runtime,
    )

    assert called == {
        "model": model,
        "latent": latent,
        "steps": 8,
        "shifts": (12.0, 3.0),
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
    }
    assert result.sampler is sentinel_sampler
    assert result.sigmas.tolist() == pytest.approx(plan["video_sigmas"])


def test_setup_preserves_samples_and_builds_exact_nested_masks():
    model, positive, latent, plan = _allow_bundle()
    samples = latent["samples"]

    result = setup_audio_refine(
        plan=plan,
        model=model,
        positive=positive,
        av_latent=latent,
        setup_sampling_fn=lambda *args: (
            model.clone(),
            object(),
            _full_video_sigmas(plan["full_steps"]),
        ),
        runtime_snapshot_fn=_runtime,
    )

    assert result.latent is not latent
    assert result.latent["samples"] is samples
    video_mask, audio_mask = result.latent["noise_mask"].unbind()
    assert video_mask.dtype == torch.float32
    assert audio_mask.dtype == torch.float32
    assert torch.count_nonzero(video_mask).item() == 0
    assert torch.all(audio_mask == 1)
    assert result.guider.cfg == 1.0
    assert set(result.guider.original_conds) == {"positive"}


def test_audio_refine_noise_is_deterministic_and_seed_bound():
    latent = _latent()
    first = AudioRefineRandomNoise(7).generate_noise(latent)
    second = AudioRefineRandomNoise(7).generate_noise(latent)
    changed = AudioRefineRandomNoise(8).generate_noise(latent)

    first_parts = first.unbind()
    second_parts = second.unbind()
    changed_parts = changed.unbind()
    assert all(torch.equal(left, right) for left, right in zip(first_parts, second_parts))
    assert any(not torch.equal(left, right) for left, right in zip(first_parts, changed_parts))


@pytest.mark.parametrize("binding", ["model", "positive", "latent"])
def test_setup_rejects_changed_bindings_before_sampling_setup(binding):
    model, positive, latent, plan = _allow_bundle()
    if binding == "model":
        model = FakeModelPatcher()
    elif binding == "positive":
        positive = _positive(offset=1.0)
    else:
        latent = _latent(audio_offset=1.0)
    calls = []

    with pytest.raises(ValueError, match="REJECT_CONTRACT_MISMATCH"):
        setup_audio_refine(
            plan=plan,
            model=model,
            positive=positive,
            av_latent=latent,
            setup_sampling_fn=lambda *args: calls.append(args),
            runtime_snapshot_fn=_runtime,
        )
    assert calls == []


def test_setup_abstains_without_allocation_when_runtime_gate_falls():
    model, positive, latent, plan = _allow_bundle()
    calls = []

    result = setup_audio_refine(
        plan=plan,
        model=model,
        positive=positive,
        av_latent=latent,
        setup_sampling_fn=lambda *args: calls.append(args),
        runtime_snapshot_fn=lambda: _runtime(free_mib=511),
    )

    assert calls == []
    assert result.model is model
    assert result.latent is latent
    assert isinstance(result.noise, AudioRefineBypassNoise)
    assert result.noise.generate_noise(latent) is latent["samples"]
    assert result.sigmas.shape == (0,)
    assert json.loads(result.report_json)["bypassed"] is True


def test_existing_abstain_plan_returns_empty_sigma_bypass():
    model, positive, latent, plan = _abstain_bundle()
    result = setup_audio_refine(
        plan=plan,
        model=model,
        positive=positive,
        av_latent=latent,
        setup_sampling_fn=lambda *args: pytest.fail("ABSTAIN must not clone sampling"),
        runtime_snapshot_fn=_runtime,
    )

    assert result.latent is latent
    assert result.sigmas.numel() == 0
    assert isinstance(result.noise, AudioRefineBypassNoise)


def test_real_sampler_custom_advanced_empty_sigmas_skips_prepare_sampling(
    monkeypatch,
):
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy_extras.nodes_custom_sampler as custom_sampler_nodes

    model, positive, latent, _ = _abstain_bundle()
    guider = AudioRefineBasicGuider(model, positive)
    noise = AudioRefineBypassNoise()

    monkeypatch.setattr(
        comfy.sampler_helpers,
        "prepare_sampling",
        lambda *args, **kwargs: pytest.fail("prepare_sampling must not run"),
    )
    monkeypatch.setattr(
        custom_sampler_nodes.latent_preview,
        "prepare_callback",
        lambda *args, **kwargs: (lambda *callback_args: None),
    )
    monkeypatch.setattr(
        comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
    )

    output = SamplerCustomAdvanced.execute(
        noise,
        guider,
        object(),
        torch.empty((0,), dtype=torch.float32),
        latent,
    ).result[0]

    actual = output["samples"].unbind()
    expected = latent["samples"].unbind()
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_audio_refine_node_schemas_are_append_only_experimental_contracts():
    schemas = [node.define_schema() for node in AUDIO_REFINE_ADVANCED_NODE_CLASSES]

    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3AudioRefineAuditT8Advanced",
        "MiniMaxH3AudioRefinePlanT8Advanced",
        "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
    ]
    assert all(schema.is_experimental is True for schema in schemas)
    assert all(schema.category == "T8/MiniMax H3/Audio/Experimental" for schema in schemas)
    assert [item.id for item in schemas[0].inputs] == [
        "model",
        "positive",
        "av_latent",
        "conditioned_prompt",
        "media_map_json",
        "conditioning_report",
        "protected_audio",
        "minimum_free_vram_mib",
        "minimum_commit_headroom_gib",
        "hash_chunk_megabytes",
    ]
    assert [item.id for item in schemas[0].outputs] == [
        "audit",
        "decision",
        "report_json",
    ]
    assert [item.id for item in schemas[1].inputs] == [
        "audit",
        "refine_steps",
        "audio_denoise",
        "refine_seed",
        "model_strategy",
    ]
    assert [item.id for item in schemas[1].outputs] == [
        "plan",
        "decision",
        "report_json",
    ]
    assert [item.id for item in schemas[2].inputs] == [
        "plan",
        "model",
        "positive",
        "av_latent",
    ]
    assert [item.id for item in schemas[2].outputs] == [
        "model",
        "noise",
        "guider",
        "sampler",
        "sigmas",
        "latent",
        "report_json",
    ]


def test_audio_refine_schema_defaults_lock_the_reviewed_route():
    audit_schema = MiniMaxH3AudioRefineAuditT8Advanced.define_schema()
    plan_schema = MiniMaxH3AudioRefinePlanT8Advanced.define_schema()

    protected_audio = audit_schema.inputs[6]
    minimum_vram = audit_schema.inputs[7]
    minimum_commit = audit_schema.inputs[8]
    hash_chunk = audit_schema.inputs[9]
    assert protected_audio.optional is True
    assert minimum_vram.default == 512
    assert minimum_vram.min == 512
    assert minimum_commit.default == 16.0
    assert minimum_commit.min == 16.0
    assert hash_chunk.default == 8
    assert (hash_chunk.min, hash_chunk.max) == (1, 64)

    assert plan_schema.inputs[1].default == 4
    assert (plan_schema.inputs[1].min, plan_schema.inputs[1].max) == (1, 8)
    assert plan_schema.inputs[2].default == 0.5
    assert plan_schema.inputs[3].default == 0
    assert plan_schema.inputs[4].options == ["connected_model_explicit"]


def test_runtime_sensitive_nodes_disable_comfy_result_cache():
    assert torch.isnan(
        torch.tensor(MiniMaxH3AudioRefineAuditT8Advanced.fingerprint_inputs())
    )
    assert torch.isnan(
        torch.tensor(
            MiniMaxH3AudioRefineDualClockSetupT8Advanced.fingerprint_inputs()
        )
    )


def test_plan_node_execute_is_a_thin_domain_adapter():
    audit = _allow_audit()
    output = MiniMaxH3AudioRefinePlanT8Advanced.execute(
        audit,
        4,
        0.5,
        9,
        "connected_model_explicit",
    )

    plan, decision, report = output.result
    assert decision == "ALLOW"
    assert plan["refine_seed"] == 9
    assert json.loads(report)["payload_sha256"] == plan["payload_sha256"]


def test_audio_refine_registration_appends_after_the_released_211_node_prefix():
    node_classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in node_classes]
    feature_ids = json.loads(
        (Path(__file__).resolve().parents[1] / "features.json").read_text(
            encoding="utf-8"
        )
    )["nodes"]

    assert len(ids) == 214
    assert ids == feature_ids
    assert ids[208:211] == [
        "MiniMaxH3SkinFinishSpecularFrequencyT8Advanced",
        "MiniMaxH3SkinFinishSurfaceT8Advanced",
        "MiniMaxH3SkinFinishDichromaticT8Advanced",
    ]
    assert ids[211:] == [
        "MiniMaxH3AudioRefineAuditT8Advanced",
        "MiniMaxH3AudioRefinePlanT8Advanced",
        "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
    ]
