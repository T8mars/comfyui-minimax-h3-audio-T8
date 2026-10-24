"""Actual Core CPU reference packing; no trained/GPU quality qualification."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import reference_temporal_pack as pack
from h3_audio_t8_pkg import reference_temporal_pack_advanced as feature
from h3_audio_t8_pkg.nodes_reference_temporal_pack import (
    MiniMaxH3ReferenceTemporalPackEXPT8 as Apply,
    MiniMaxH3ReferenceTemporalPackAuditEXPT8 as Audit,
)
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def video(t=12, h=4, w=8, audio=0):
    values = torch.arange(24 * t * h * w).reshape(1, 24, t, h, w).float() / 10000
    return {"kind": "video_audio" if audio else "video", "latent": values,
            "latent_t": t, "latent_h": h, "latent_w": w, "ref_audio_t": audio,
            "audio_latent": torch.ones(1, 32, 2, audio) * .3 if audio else None}


def positive():
    result = conditioning("i2va")
    result[0][1]["minimax_refs"] = [
        {"kind": "audio", "ref_audio_t": 2, "audio_latent": torch.ones(1, 32, 2, 2)},
        {"kind": "image", "latent": torch.ones(1, 24, 1, 2, 6), "latent_h": 2, "latent_w": 6},
        video(audio=60), video(7, 2, 4),
    ]
    return result


def sample(selected, pos, source=None):
    source = latent() if source is None else source
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 2, 12., 3., "dual_clock_euler", "native_flow")
    guider = BasicGuider.execute(prepared, pos).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(211023).result[0], guider,
                                         sampler, sigmas, source).result[0]


@pytest.mark.parametrize("ratio,expected", [
    ("0/5", list(range(12))), ("1/5", [0, 1, 2, 3, 5, 6, 7, 8, 10, 11]),
    ("2/5", [0, 1, 3, 5, 6, 8, 10, 11]), ("3/5", [0, 3, 5, 8, 10])])
def test_fixed_source_recipe_true_indices_not_nominal_keep_percentage(ratio, expected):
    assert pack.retained_indices(12, ratio) == expected


def test_default_and_zero_are_same_objects_and_exact_actual_sampler_baseline():
    bare, pos = model(), positive()
    selected, output, report = feature.configure_reference_temporal_pack(bare, pos)
    assert selected is bare and output is pos and report["status"] == "disabled_identity"
    expected, actual = sample(bare, pos), sample(selected, output)
    for a, b in zip(expected["samples"].unbind(), actual["samples"].unbind(), strict=True):
        assert torch.equal(a, b)
    assert not bare.wrappers and "model_function_wrapper" not in bare.model_options


def test_original_physical_rope_all_non_reference_rows_bitwise_and_each_ref_independent():
    pos, signature = positive(), (2, 2, 4, 8, 8)
    original = deepcopy(pos)
    packed, report = pack.pack_conditioning(pos, "2/5")
    assert pos[0][0] is packed[0][0]
    source_refs, refs = pos[0][1]["minimax_refs"], packed[0][1]["minimax_refs"]
    assert all(refs[i] is source_refs[i] for i in (0, 1))
    assert refs[2]["audio_latent"] is source_refs[2]["audio_latent"]
    assert [refs[i]["latent_t"] for i in (2, 3)] == [8, 5]
    for i in (2, 3):
        indices = pack.verify_ref(refs[i])["indices"]
        assert torch.equal(refs[i]["latent"], source_refs[i]["latent"][:, :, indices])
        assert all(a < b for a, b in zip(indices, indices[1:]))
        assert report["references"][i-2]["descriptor"]["original_seconds_24fps"][-1] > 0
    keyframes = pos[0][1]["minimax_keyframes"]
    full = native.PackedLayout(*signature, keyframes=keyframes, refs=source_refs)
    compact = native.PackedLayout(*signature, keyframes=keyframes, refs=refs)
    before = compact.position_ids.clone()
    payload = {"refs": refs, "keyframes": keyframes, "layout": compact}
    repaired, observed = pack.repair_payload(payload, signature)
    for old, new in zip(full.segments, repaired["layout"].segments, strict=True):
        a, b, kind = old
        c, d, _ = new
        if b-a == d-c:
            assert torch.equal(full.position_ids[a:b], repaired["layout"].position_ids[c:d])
    full_visual = [x for x in full.segments if x[2] == "ref_img"][1:]
    compact_visual = [x for x in compact.segments if x[2] == "ref_img"][1:]
    for source_seg, dest_seg, ref in zip(full_visual, compact_visual, refs[2:], strict=True):
        record = ref[pack.KEY]
        frame_rows = ref["latent_h"] * ref["latent_w"] // 4
        original_rows = full.position_ids[source_seg[0]:source_seg[1]].reshape(record["original_t"], frame_rows, 3)
        assert torch.equal(original_rows[record["indices"]].reshape(-1, 3),
                           repaired["layout"].position_ids[dest_seg[0]:dest_seg[1]])
    assert torch.equal(compact.position_ids, before)
    assert repaired["layout"] is not compact and payload["layout"] is compact
    assert observed["generated_and_audio_rows_removed"] == 0
    for now, then in zip(pos[0][1]["minimax_refs"], original[0][1]["minimax_refs"], strict=True):
        if "latent" in now:
            assert torch.equal(now["latent"], then["latent"])


def test_actual_two_step_core_consumes_fewer_rows_then_plain_request_is_unchanged():
    bare, pos, seen = model(), positive(), []
    baseline = sample(bare, pos)
    block = bare.model.diffusion_model.blocks[0]
    handle = block.register_forward_pre_hook(lambda _module, args: seen.append(args[0].shape[0]))
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, pos, "2/5")
    try:
        result = sample(selected, packed)
    finally:
        handle.remove()
    report = feature.audit_reference_temporal_pack(selected)
    assert report["completed_model_calls"] == 2
    assert report["cleanup_observed"] and not report["request_active"]
    assert seen == [report["last_layout"]["packed_sequence_rows"]] * 2
    assert seen[0] < report["last_layout"]["original_sequence_rows"]
    assert all(torch.isfinite(x).all() for x in result["samples"].unbind())
    assert [x.shape for x in result["samples"].unbind()] == [x.shape for x in baseline["samples"].unbind()]
    for now, before in zip(sample(bare, pos)["samples"].unbind(), baseline["samples"].unbind(), strict=True):
        assert torch.equal(now, before)
    assert not bare.wrappers and pack.KEY not in bare.attachments
    assert not report["portable_cache_identity"]


def test_independent_low_high_shared_base_actual_geometry_and_content_provider_keys():
    bare = model()
    low, lpos, _ = feature.configure_reference_temporal_pack(bare, positive(), "2/5", phase="LOW")
    high_positive = positive()
    high_positive[0][1]["minimax_keyframes"][0]["latent"] = torch.ones(1, 24, 1, 8, 12) * .2
    high, hpos, _ = feature.configure_reference_temporal_pack(bare, high_positive, "1/5", phase="HIGH")
    source = latent()
    from comfy.nested_tensor import NestedTensor
    high_source = {"samples": NestedTensor((torch.zeros(1, 24, 2, 8, 12), source["samples"].unbind()[1]))}
    sample(low, lpos)
    sample(high, hpos, high_source)
    a, b = feature.audit_reference_temporal_pack(low), feature.audit_reference_temporal_pack(high)
    assert a["last_layout"]["signature"][2:4] == [4, 8]
    assert b["last_layout"]["signature"][2:4] == [8, 12]
    assert a["binding_sha256"] != b["binding_sha256"]
    assert a["packed_condition_identity"]["sha256"] != b["packed_condition_identity"]["sha256"]
    sample(low, lpos)
    assert feature.audit_reference_temporal_pack(low)["completed_model_calls"] == 2
    assert feature.audit_reference_temporal_pack(low)["request_number"] == 2
    assert feature.audit_reference_temporal_pack(high)["request_number"] == 1


def test_unknown_model_delegates_original_without_faked_compression():
    class Foreign:
        def __call__(self):
            return "executed"
    foreign, pos = Foreign(), positive()
    selected, output, report = feature.configure_reference_temporal_pack(foreign, pos, "2/5")
    assert selected is foreign and output is pos and selected() == "executed"
    assert report["compression_applied"] is False and not report["portable_cache_identity"]


def test_existing_wrapper_called_once_real_exception_propagates_and_next_request_recovers():
    bare, pos, calls = model(), positive(), []
    state = SimpleNamespace(fail=True)

    def previous(apply_model, args):
        calls.append(1)
        if state.fail:
            raise TypeError("user kernel failure")
        return apply_model(args["input"], args["timestep"], **args["c"])

    bare.set_model_unet_function_wrapper(previous)
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, pos, "2/5")
    with pytest.raises(TypeError, match="user kernel failure"):
        sample(selected, packed)
    assert calls == [1]
    assert bare.model_options["model_function_wrapper"] is previous
    assert feature.audit_reference_temporal_pack(selected)["status"] == "failed"
    state.fail = False
    assert all(torch.isfinite(x).all() for x in sample(selected, packed)["samples"].unbind())
    report = feature.audit_reference_temporal_pack(selected)
    assert report["completed_model_calls"] == 2 and report["failed_model_calls"] == 0
    assert report["cleanup_observed"] and not report["request_active"]


@pytest.mark.parametrize("failure", ["NaN", "shape", "SHA", "index", "version"])
def test_real_bad_shapes_nan_descriptor_and_sha_not_swallowed(failure):
    pos = positive()
    if failure == "NaN":
        pos[0][1]["minimax_refs"][2]["latent"][0, 0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            pack.pack_conditioning(pos)
        return
    if failure == "shape":
        pos[0][1]["minimax_refs"][2]["latent_t"] = 17
        with pytest.raises(ValueError, match="geometry"):
            pack.pack_conditioning(pos)
        return
    packed, _ = pack.pack_conditioning(pos)
    ref = packed[0][1]["minimax_refs"][2]
    if failure == "SHA":
        ref["latent"].add_(1)
    elif failure == "index":
        ref[pack.KEY]["indices"][1] = 100
    else:
        ref[pack.KEY]["schema"] = "v999"
    with pytest.raises(ValueError):
        pack.verify_ref(ref)


def test_last_real_forward_content_mutation_refuses_success_and_keeps_source():
    bare, pos, count = model(), positive(), []
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, pos, "2/5")
    original = pos[0][1]["minimax_refs"][2]["latent"].clone()

    def mutate(*_args):
        count.append(1)
        if len(count) == 2:
            packed[0][1]["minimax_refs"][2]["latent"].add_(1)

    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match="content SHA changed"):
            sample(selected, packed)
    finally:
        handle.remove()
    assert count == [1, 1]
    assert torch.equal(pos[0][1]["minimax_refs"][2]["latent"], original)


def test_real_node_schema_defaults_and_cpu_execute_pair_and_audit():
    schema = Apply.define_schema().get_v1_info(Apply)
    assert schema.input["required"]["ratio"][1]["default"] == "0/5"
    bare, pos = model(), positive()
    selected, packed, raw = Apply.execute(bare, pos, "2/5", "SINGLE").result
    assert json.loads(raw)["completed_model_calls"] == 0
    output = sample(selected, packed)
    report = json.loads(Audit.execute(selected, output).result[0])
    assert report["completed_model_calls"] == 2
    assert Audit.define_schema().is_output_node
    assert not report["quality_accepted"]


@pytest.mark.parametrize("query_route", ["video_only_paper", "joint_av_exp"])
def test_actual_relay_rebinds_physical_layout_preserves_events_times_and_bias(monkeypatch, query_route):
    from h3_audio_t8_pkg import prompt_relay_advanced as relay
    from test_prompt_relay_advanced import NativeLikeFakeClip, _plan, _allow_fixture_core_contract
    from helpers import FakeAudioVAE, FakeVideoVAE
    _allow_fixture_core_contract(monkeypatch)
    plan = _plan(length=22, local_prompts="She looks left.\nShe turns right.")[0]
    plan = relay.configure_prompt_relay_query_route(plan, query_route)[0]
    # Existing policy examines the original RGB count before the ordinary
    # builder limits it to the 22-frame generation window. Do not weaken it in
    # this independent test; only the resulting seven VAE ticks are packed.
    frames = torch.ones(56, 32, 32, 3) * .4
    built = relay.build_prompt_relay_conditioning(
        model=model(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt_relay_plan=plan, width=32, height=32, task_type="auto", audio_mode="native",
        audio_denoise_strength=.35, add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp", query_chunk_rows=64, ref_videos={"ref_video_0": frames})
    original, pos, source = built[:3]
    old = relay.prompt_relay_model_contract(original)
    selected, packed, report = feature.configure_reference_temporal_pack(original, pos, "2/5")
    assert selected is not original and report["contract"]["relay_rebound"]
    new = relay.prompt_relay_model_contract(selected)
    assert new["binding"]["events"] == old["binding"]["events"]
    assert new["binding"]["plan_hash"] == old["binding"]["plan_hash"]
    signature = old["binding"]["layout_contract"]["signature"]
    full = native.PackedLayout(*signature, refs=pos[0][1]["minimax_refs"])
    payload, _ = pack.repair_payload({"refs": packed[0][1]["minimax_refs"]}, signature)
    old_route = relay._runtime_route(full, old["binding"], torch.device("cpu"))
    new_route = relay._runtime_route(payload["layout"], new["binding"], torch.device("cpu"))
    for a, b in zip(old_route["query_segments"], new_route["query_segments"], strict=True):
        assert a["kind"] == b["kind"] and torch.equal(a["query_times"], b["query_times"])
        before = relay.make_prompt_relay_bias(a["query_times"], old_route["seq_len"], old_route["events"], dtype=torch.float32)
        after = relay.make_prompt_relay_bias(b["query_times"], new_route["seq_len"], new_route["events"], dtype=torch.float32)
        assert torch.equal(before[:, :signature[0]], after[:, :signature[0]])
        assert torch.count_nonzero(after) > 0
    result = sample(selected, packed, source)
    actual = relay.prompt_relay_model_contract(selected)
    assert actual["execution_counts"]["completed_forwards"] == 2
    assert actual["execution_counts"]["routed_attention_calls"] >= 2
    assert all(torch.isfinite(x).all() for x in result["samples"].unbind())
    assert relay.prompt_relay_model_contract(original)["binding"] == old["binding"]
    assert relay.prompt_relay_model_contract(original)["execution_counts"]["completed_forwards"] == 0


def test_mismatched_model_condition_pair_and_source_content_mutation_are_real_errors():
    bare = model()
    a, ap, _ = feature.configure_reference_temporal_pack(bare, positive(), "2/5", phase="LOW")
    b, bp, _ = feature.configure_reference_temporal_pack(bare, positive(), "2/5", phase="HIGH")
    with pytest.raises(ValueError, match="pair binding"):
        sample(a, bp)
    bp[0][1]["minimax_refs"][0]["audio_latent"].add_(.1)
    with pytest.raises(ValueError, match="conditioning content SHA changed"):
        sample(b, bp)
    assert all(torch.isfinite(x).all() for x in sample(a, ap)["samples"].unbind())


def test_no_reference_selection_and_unknown_existing_forward_keep_original_path():
    bare, pos = model(), conditioning()
    selected, result, report = feature.configure_reference_temporal_pack(bare, pos, "2/5")
    assert selected is bare and result is pos and "original_retained" in report["status"]
    original = bare.model.diffusion_model._forward
    calls = []
    def unknown(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    bare.model.diffusion_model._forward = unknown
    pos = positive()
    selected, result, report = feature.configure_reference_temporal_pack(bare, pos, "2/5")
    assert selected is bare and result is pos and not report["compression_applied"]
    sample(selected, result)
    assert calls == [1, 1] and bare.model.diffusion_model._forward is unknown


def test_real_sampler_cancel_does_not_retry_or_modify_next_plain_request():
    import comfy.model_management
    bare, pos, calls = model(), positive(), []
    baseline = sample(bare, pos)
    def previous(_apply, _args):
        calls.append(1)
        raise comfy.model_management.InterruptProcessingException()
    branch = bare.clone()
    branch.set_model_unet_function_wrapper(previous)
    selected, packed, _ = feature.configure_reference_temporal_pack(branch, pos, "2/5")
    with pytest.raises(comfy.model_management.InterruptProcessingException):
        sample(selected, packed)
    assert calls == [1]
    report = feature.audit_reference_temporal_pack(selected)
    assert report["failed_model_calls"] == 1 and not report["request_active"]
    for a, b in zip(baseline["samples"].unbind(), sample(bare, pos)["samples"].unbind(), strict=True):
        assert torch.equal(a, b)


def test_existing_stage_identity_rejects_portable_reuse_not_normal_execution():
    from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
    selected, packed, _ = feature.configure_reference_temporal_pack(model(), positive(), "2/5")
    identity = stage_model_identity(selected)
    assert identity["portable_cache_reuse"] is False
    assert all(torch.isfinite(x).all() for x in sample(selected, packed)["samples"].unbind())
    assert feature.audit_reference_temporal_pack(selected)["completed_model_calls"] == 2


def test_later_wrapper_selection_is_preserved_and_audit_reports_lost_coverage():
    bare = model()
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, positive(), "2/5")
    sample(selected, packed)
    def later(_apply, _args):
        return "user-selected-wrapper"
    selected.set_model_unet_function_wrapper(later)
    report = feature.audit_reference_temporal_pack(selected)
    assert selected.model_options["model_function_wrapper"] is later
    assert report["status"] == "observed_partial" and report["partial"]
    assert not report["selected_wrapper_present"] and not report["portable_cache_identity"]


def test_reported_rope_offsets_are_actual_native_float64_clock_not_rounded_product():
    pos = conditioning()
    pos[0][1]["minimax_refs"] = [video(37)]
    packed, _ = pack.pack_conditioning(pos, "2/5")
    record = packed[0][1]["minimax_refs"][0][pack.KEY]
    expected = native._video_t_grid(37, 0)[record["indices"]].tolist()
    assert record["physical_rope_offsets"] == expected


def test_real_core_api_graph_validates_paired_apply_guider_sampler_audit(monkeypatch):
    import asyncio
    import execution
    import nodes
    from comfy_api.latest import io
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced as Sampler
    from comfy_extras.nodes_custom_sampler import BasicGuider as Guider
    class TinyInputs(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="H21TemporalPackTinyInputs", inputs=[], outputs=[
                io.Model.Output("model"), io.Conditioning.Output("positive"), io.Latent.Output("latent"),
                io.Custom("NOISE").Output("noise"), io.Custom("SAMPLER").Output("sampler"),
                io.Custom("SIGMAS").Output("sigmas")])
        @classmethod
        def execute(cls):
            source = latent()
            selected, sampler, sigmas = setup_dual_clock_sampling(
                model(), source, 2, 12., 3., "dual_clock_euler", "native_flow")
            return io.NodeOutput(selected, positive(), source, RandomNoise.execute(211023).result[0],
                                 sampler, sigmas)
    for cls in (TinyInputs, Apply, Audit, Sampler, Guider):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    graph = {
        "1": {"class_type": "H21TemporalPackTinyInputs", "inputs": {}},
        "2": {"class_type": Apply.__name__, "inputs": {"model": ["1", 0], "positive": ["1", 1], "ratio": "2/5", "phase": "LOW"}},
        "3": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["2", 1]}},
        "4": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["1", 3], "guider": ["3", 0],
               "sampler": ["1", 4], "sigmas": ["1", 5], "latent_image": ["1", 2]}},
        "5": {"class_type": Audit.__name__, "inputs": {"model": ["2", 0], "after_sampling": ["4", 0]}},
    }
    result = asyncio.run(execution.validate_prompt("h21-reference-temporal-pack", graph, None))
    assert result[0], result[1:]
    assert "5" in result[2]
