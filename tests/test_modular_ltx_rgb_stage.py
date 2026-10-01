"""S27 RGB bridge: no hidden LTX sampler and no H3 audio conversion."""
from copy import deepcopy
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
from h3_audio_t8_pkg.modular_sampling.ltx_rgb_nodes import NODES
from h3_audio_t8_pkg.modular_sampling.ltx_rgb_stage import (
    bind_ltx_rgb_stage, audit_ltx_rgb_stage)
from tools.build_modular_ltx_rgb_workflows import sources, split_frontend
from test_sol_engine_h3_super_advanced import _FakePatcher


def _case(identity=False):
    frames = torch.linspace(0, 1, 17 * 32 * 48 * 3).reshape(17, 32, 48, 3)
    audio = {"waveform": torch.zeros(1, 2, 2000), "sample_rate": 32000}
    prepared, _, _, _, _, _, prep_report = sol.prepare_h3_draft_for_ltx_refiner(
        frames, 192, 128)
    lifted = {"samples": torch.zeros(1, 128, 3, 4, 6)}
    model = type("LTXModel", (), {"model_options": {}})()
    if identity:
        _, sigmas, _, setup = sol.setup_ltx_identity_preserve_refiner(model, enabled=False)
    else:
        _, sigmas, _, setup = sol.setup_ltx_stage2_refiner(model, enabled=False)
    noise = object()
    guider = comfy.samplers.CFGGuider(model)
    sampler = comfy.samplers.KSAMPLER(lambda *_args, **_kwargs: None)
    return (frames, audio, prepared, prep_report, lifted, model,
            noise, guider, sampler, sigmas, setup)


@pytest.mark.parametrize("identity", [False, True])
def test_ltx_rgb_boundary_preserves_controls_and_original_audio(identity):
    args = _case(identity)
    bound = bind_ltx_rgb_stage(*args)
    assert all(left is right for left, right in zip(bound[:5],
        (args[6], args[7], args[8], args[9], args[4])))
    assert json.loads(bound[6])["sampled"] is False
    candidate = {"samples": args[4]["samples"] + .125}
    audited = audit_ltx_rgb_stage(bound[5], *args, candidate)
    assert audited[0] is candidate and audited[1] is args[1]
    report = json.loads(audited[2])
    assert report["quality_acceptance"] is False
    assert report["portable_cache_reuse_authorized"] is False


def test_ltx_rgb_boundary_rejects_changed_source_setup_or_candidate():
    args = list(_case())
    boundary = bind_ltx_rgb_stage(*args)[5]
    changed = list(args)
    changed[0] = args[0].clone()
    changed[0][0, 0, 0, 0] = .9
    with pytest.raises(ValueError, match="source, Setup or sampling controls"):
        audit_ltx_rgb_stage(boundary, *changed, args[4])
    changed = list(args)
    changed[9] = args[9].clone()
    changed[9][0] = .75
    with pytest.raises(ValueError, match="SIGMAS"):
        bind_ltx_rgb_stage(*changed)
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][0, 0, 0, 0] = .9
    with pytest.raises(ValueError, match="source, Setup or sampling controls"):
        audit_ltx_rgb_stage(boundary, *changed, args[4])
    bad_candidate = {"samples": torch.zeros(1, 128, 2, 4, 6)}
    with pytest.raises(ValueError, match="candidate shape"):
        audit_ltx_rgb_stage(boundary, *args, bad_candidate)
    forged = deepcopy(boundary)
    forged["source_audio"]["sample_rate"] = 16000
    with pytest.raises(ValueError, match="SHA-256"):
        audit_ltx_rgb_stage(forged, *args, args[4])


def test_ltx_rgb_boundary_keeps_a_video_without_audio_valid():
    args = list(_case())
    args[1] = None
    bound = bind_ltx_rgb_stage(*args)
    candidate, audio, _ = audit_ltx_rgb_stage(bound[5], *args, args[4])
    assert candidate is args[4] and audio is None


@pytest.mark.parametrize("identity", [False, True])
def test_ltx_rgb_boundary_accepts_actual_enabled_setup_reports(identity):
    args = list(_case(identity))
    original = _FakePatcher()
    if identity:
        model, sigmas, _, report = sol.setup_ltx_identity_preserve_refiner(
            original, enabled=True, attention_backend="dense_reference")
    else:
        model, sigmas, _, report = sol.setup_ltx_stage2_refiner(
            original, enabled=True, attention_backend="dense_reference")
    assert json.loads(report)["status"] == "configured"
    args[5] = model
    args[7] = comfy.samplers.CFGGuider(model)
    args[9] = sigmas
    args[10] = report
    boundary = bind_ltx_rgb_stage(*args)[5]
    assert boundary["variant"] == ("identity_preserve" if identity else "official_stage2")
    assert audit_ltx_rgb_stage(boundary, *args, args[4])[1] is args[1]


def test_ltx_rgb_stage_nodes_are_append_only_and_typed():
    assert [node.define_schema().node_id for node in NODES] == [
        "MiniMaxH3LTXRGBStageBindEXPT8", "MiniMaxH3LTXRGBStageAuditEXPT8"]


@pytest.mark.parametrize("path", sources())
def test_ltx_rgb_candidate_keeps_external_sampler_and_audio_bypass(path):
    original_bytes = path.read_bytes()
    old = json.loads(original_bytes)
    graph = split_frontend(old)
    assert path.read_bytes() == original_bytes
    assert len(graph["nodes"]) == len(old["nodes"]) + 2
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {node["type"]: node for node in graph["nodes"]}
    sampler = by_type["SamplerCustomAdvanced"]
    bind = by_type["MiniMaxH3LTXRGBStageBindEXPT8"]
    audit = by_type["MiniMaxH3LTXRGBStageAuditEXPT8"]
    decode = by_type["MiniMaxH3SolEngineTAEHVDecodeT8Advanced"]
    trim = by_type["MiniMaxH3OutputTrimT8"]
    links = {link[0]: link for link in graph["links"]}

    def edge(target, name):
        item = next(item for item in target["inputs"] if item["name"] == name)
        return links[item["link"]][1:3]

    assert edge(sampler, "latent_image") == [bind["id"], 4]
    assert edge(sampler, "noise") == [bind["id"], 0]
    assert edge(sampler, "guider") == [bind["id"], 1]
    assert edge(sampler, "sampler") == [bind["id"], 2]
    assert edge(sampler, "sigmas") == [bind["id"], 3]
    assert edge(audit, "candidate_latent") == [sampler["id"], 0]
    assert edge(decode, "latent") == [audit["id"], 0]
    assert edge(trim, "audio") == [audit["id"], 1]
    assert nodes[edge(bind, "source_audio")[0]]["type"] == "GetVideoComponents"
    assert nodes[edge(bind, "ltx_latent")[0]]["type"] == "LTXVLatentUpsampler"
