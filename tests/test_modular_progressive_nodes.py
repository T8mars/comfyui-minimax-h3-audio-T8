"""Real public-node method execution; learned weights remain a labelled double."""
import json

import pytest
import torch
import comfy.nested_tensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

from h3_audio_t8_pkg.modular_sampling import progressive_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from test_modular_progressive_stages import inputs, initialized
import test_progressive_sampling_runtime as runtime_fixtures

stub_lifter = runtime_fixtures.stub_lifter


def low_nodes(case):
    plan, source, low_sigmas, high_sigmas, _ = nodes.MiniMaxH3ProgressiveStagePlanEXPT8.execute(
        case["source"], case["sigmas"], case["low"], .5, case["task"], case["mode"]).result
    positive, negative = nodes.MiniMaxH3ProgressiveStageConditioningEXPT8.execute(
        case["positive"], case["negative"], plan, "low").result
    boundary, text = nodes.MiniMaxH3ProgressiveLowStageEXPT8.execute(case["model"], case["sampler"],
        Noise_RandomNoise(7), plan, source, positive, negative).result
    return boundary, low_sigmas, high_sigmas, text


def high_nodes(case, boundary):
    low, width, height, plan, _ = nodes.MiniMaxH3ProgressiveLiftInputEXPT8.execute(
        boundary, case["model"], case["sampler"]).result
    video, audio = low["samples"].unbind()
    lifted, _ = stages.legacy._lift_video(video, audio, plan, {"test_double": True})
    latent = {"samples": comfy.nested_tensor.NestedTensor((lifted, audio))}
    state, high_plan, report = nodes.MiniMaxH3ProgressiveHighHandoffEXPT8.execute(
        boundary, case["model"], case["sampler"], latent, case["source"], Noise_RandomNoise(8)).result
    assert high_plan == plan and (width, height) == (128, 64)
    assert json.loads(report)["sampling_calls"] == 0
    positive, negative = nodes.MiniMaxH3ProgressiveStageConditioningEXPT8.execute(
        case["positive"], case["negative"], high_plan, "high").result
    return nodes.MiniMaxH3ProgressiveHighStageEXPT8.execute(
        state, case["model"], case["sampler"], positive, negative, 7).result


@pytest.mark.parametrize("kind", ["t2va", "i2va", "initialized", "avatar_mask"])
def test_real_nodes_equal_legacy_complete_math(stub_lifter, kind):
    case = inputs(kind) if kind in ("t2va", "i2va") else inputs(
        source=initialized("avatar" if kind == "avatar_mask" else "fractional"), mode="initialized_av_exp")
    expected, _ = stages.legacy.sample_progressive_h3(case["model"], case["positive"], case["negative"],
        case["source"], case["sampler"], case["sigmas"], upscaler_model="test", seed=7,
        low_evaluations=case["low"], task=case["task"], input_mode=case["mode"])
    boundary, low_sigmas, high_sigmas, text = low_nodes(case)
    assert torch.equal(low_sigmas, case["sigmas"][:3]) and torch.equal(high_sigmas, case["sigmas"][2:])
    assert json.loads(text)["execution"]["callbacks"] == [0, 1]
    output, report, completed = high_nodes(case, boundary)
    assert completed.output is output
    assert completed.verify()["sampling"] == json.loads(report)
    assert json.loads(report)["low_executed"] is False
    for a, b in zip(expected["samples"].unbind(), output["samples"].unbind()):
        assert torch.equal(a, b)


def test_save_load_nodes_and_only_high_execute(tmp_path, monkeypatch, stub_lifter):
    monkeypatch.setattr(nodes, "_store_root", lambda: tmp_path)
    case = inputs()
    boundary = low_nodes(case)[0]
    kept, path, digest, _ = nodes.MiniMaxH3ProgressiveLowSaveEXPT8.execute(boundary, "example/LOW").result
    assert kept is boundary
    restored, plan, text = nodes.MiniMaxH3ProgressiveLowLoadEXPT8.execute(path, digest).result
    assert plan == case["plan"] and not json.loads(text)["low_executed"]
    assert nodes.MiniMaxH3ProgressiveLowLoadEXPT8.fingerprint_inputs(path, digest) == digest
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("LOW ran in HIGH graph"))
    monkeypatch.setattr(stages.legacy, "sample_progressive_h3", lambda *a, **k: pytest.fail("Whole executor ran"))
    assert high_nodes(case, restored)[0]["samples"].is_nested


def test_missing_fingerprint_is_nan_without_creating_storage(tmp_path, monkeypatch):
    absent = tmp_path / "not-created"
    monkeypatch.setattr(nodes, "_store_root", lambda: absent)
    assert nodes.MiniMaxH3ProgressiveLowLoadEXPT8.fingerprint_inputs("x", "a" * 64) != nodes.MiniMaxH3ProgressiveLowLoadEXPT8.fingerprint_inputs("x", "a" * 64)
    assert not absent.exists()


def test_schema_separates_typed_states_and_has_no_high_dependency_in_low():
    schemas = [node.define_schema() for node in nodes.NODES]
    assert len({s.node_id for s in schemas}) == len(schemas)
    low = nodes.MiniMaxH3ProgressiveLowStageEXPT8.define_schema()
    assert {item.id for item in low.inputs} == {
        "model", "sampler", "noise", "plan", "low_source", "positive", "negative", "cfg", "reserve_vram_mib"}
    assert low.outputs[0].io_type == nodes.BOUNDARY
    high = nodes.MiniMaxH3ProgressiveHighStageEXPT8.define_schema()
    assert "low_model" not in {item.id for item in high.inputs}
    assert nodes.PLAN != nodes.RESTART != nodes.BOUNDARY
    for node in nodes.NODES:
        info = node.GET_NODE_INFO_V1()
        assert info["name"] == node.__name__


def test_noise_provider_cannot_mutate_source():
    class MutatingNoise:
        seed = 7
        def generate_noise(self, latent):
            latent["samples"].unbind()[0].add_(1)
            return latent["samples"]
    case = inputs()
    with pytest.raises(ValueError, match="NOISE provider changed"):
        nodes._generate_noise(MutatingNoise(), case["source"])


def test_individual_progress_bars_count_only_their_own_stage(monkeypatch, stub_lifter):
    import comfy.utils
    observed = []
    class Progress:
        def __init__(self, total):
            self.index = len(observed)
            observed.append({"total": total, "steps": []})
        def update_absolute(self, step, total):
            observed[self.index]["steps"].append((step, total))
    monkeypatch.setattr(comfy.utils, "ProgressBar", Progress)
    case = inputs()
    high_nodes(case, low_nodes(case)[0])
    assert observed == [{"total": 2, "steps": [(1, 2), (2, 2)]}] * 2


@pytest.mark.parametrize("noise", [object(), type("InvalidSeed", (), {"seed": -1, "generate_noise": lambda _: None})()])
def test_unknown_noise_contract_is_a_real_input_error(noise):
    with pytest.raises(ValueError, match="NOISE provider"):
        nodes._noise_seed(noise)
