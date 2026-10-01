"""One-character multi-face sampling is source-bound, not auto-accepted."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.face_refine_advanced import source_proxy_sha256
from h3_audio_t8_pkg.modular_sampling.face_stage import bind_multiface_face_stage, audit_multiface_face_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.face_nodes import NODES
from test_face_refine_parity_advanced import _plan, _locked_av, _rehash
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_face_multiface_workflows import TARGET, source_paths, split_api, split_frontend


def _job(*, pad=0):
    parent = torch.zeros((9 if pad == 0 else 19, 64, 96, 3))
    parent[:, 0, 0, 0] = torch.arange(parent.shape[0]) / parent.shape[0]
    start = 2 if pad == 0 else 0
    end = 6 if pad == 0 else 18
    source_window = parent[start:end + 1]
    source = torch.cat((source_window, source_window[-1:].expand(pad, -1, -1, -1)), dim=0)
    plan = _plan(source)[0]
    plan["multiface"] = {
        "parent_source_proxy_sha256": source_proxy_sha256(parent),
        "source_window_proxy_sha256": source_proxy_sha256(source_window),
        "character_id": "Alice", "track_key": "0:0",
        "window_start_absolute": start, "window_end_absolute": end,
        "source_window_frame_count": len(source_window),
        "model_window_frame_count": len(source),
        "alignment_context_pad_frames": pad,
        "sequential_generation_required": True,
    }
    _rehash(plan)
    return parent, source, plan


def test_multiface_stage_binds_parent_and_matches_old_core_sampler():
    parent, source, plan = _job()
    latent = _locked_av()
    prepared, _, _ = sampling.setup_dual_clock_sampling(model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    bound, selected, table, context, report = bind_multiface_face_stage(
        plan, source, parent, prepared, sampler, sigmas, latent)
    assert json.loads(report)["variant"] == "multiface"
    assert json.loads(context.profile)["face_refine"]["absolute_window"] == [2, 6]
    noise = RandomNoise.execute(319).result[0]
    torch.manual_seed(319)
    reference = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(prepared, conditioning()).result[0], sampler, sigmas, latent).result
    torch.manual_seed(319)
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          selected, table, latent, context)[2]
    for actual, expected in zip((result.output, result.denoised_output), reference):
        assert all(torch.equal(x, y) for x, y in zip(
            actual["samples"].unbind(), expected["samples"].unbind()))
    candidate, audit = audit_multiface_face_stage(result, plan, source, parent, latent)
    assert candidate is result.output
    assert json.loads(audit)["automatic_accept"] is False
    changed_parent = parent.clone()
    changed_parent[-1] = 1
    with pytest.raises(ValueError, match="parent|source frames"):
        audit_multiface_face_stage(result, plan, source, changed_parent, latent)
    assert [node.define_schema().node_id for node in NODES][-4:-2] == [
        "MiniMaxH3MultiFaceStageBindEXPT8", "MiniMaxH3MultiFaceStageAuditEXPT8"]


def test_multiface_rejects_wrong_parent_window_and_padding():
    parent, source, plan = _job(pad=3)
    latent = _locked_av(frame_count=22)
    prepared, _, _ = sampling.setup_dual_clock_sampling(model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    _, _, _, context, _ = bind_multiface_face_stage(
        plan, source, parent, prepared, sampler, sigmas, latent)
    assert json.loads(context.profile)["face_refine"]["absolute_window"] == [0, 18]
    changed = source.clone()
    changed[-1] = 1
    with pytest.raises(ValueError, match="source frames"):
        bind_multiface_face_stage(plan, changed, parent, prepared, sampler, sigmas, latent)
    stale = deepcopy(plan)
    stale["multiface"]["alignment_context_pad_frames"] = 2
    _rehash(stale)
    with pytest.raises(ValueError, match="padding"):
        bind_multiface_face_stage(stale, source, parent, prepared, sampler, sigmas, latent)
    with pytest.raises(ValueError, match="parent frames"):
        bind_multiface_face_stage(plan, source, None, prepared, sampler, sigmas, latent)


@pytest.mark.parametrize("people,nodes,links,api_nodes", [(2, 56, 116, 51), (3, 75, 169, 70)])
def test_private_multiface_graph_separates_each_character_and_keeps_old_examples(
        people, nodes, links, api_nodes):
    source, fixture = source_paths(people)
    original = json.loads(source.read_text(encoding="utf-8"))
    old_api = json.loads(fixture.read_text(encoding="utf-8"))
    graph = split_frontend(deepcopy(original))
    api = split_api(graph, old_api)
    stem = f"MultiFace_{people}Person_Separate_Stages_EXP"
    assert graph == json.loads((TARGET / f"{stem}.json").read_text(encoding="utf-8"))
    assert api == json.loads((TARGET / f"{stem}.api.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == nodes and len(graph["links"]) == links and len(api) == api_nodes
    assert sum(node["type"] == "SamplerCustomAdvanced" for node in original["nodes"]) == people
    assert sum(node["type"] == "MiniMaxH3StageSamplerEXPT8" for node in graph["nodes"]) == people
    assert sum(node["type"] == "MiniMaxH3MultiFaceStageBindEXPT8" for node in graph["nodes"]) == people
    assert sum(node["type"] == "MiniMaxH3MultiFaceStageAuditEXPT8" for node in graph["nodes"]) == people
    assert sum(node["type"] == "MiniMaxH3DualClockSamplerT8" for node in graph["nodes"]) == people
    for node_id, node in api.items():
        if node["class_type"] == "MiniMaxH3MultiFaceStageBindEXPT8":
            assert node["inputs"]["parent_frames"] == ["2", 0]
            assert node["inputs"]["source_frames"][1] == 1
        if node["class_type"] == "MiniMaxH3MultiFaceStageAuditEXPT8":
            assert node["inputs"]["parent_frames"] == ["2", 0]
        if node["class_type"] == "CreateVideo":
            assert node["inputs"]["audio"] == ["2", 1]
        if node["class_type"] == "MiniMaxH3MultiFaceCompositeT8Advanced":
            assert node["inputs"].get("previous_composite") == old_api[node_id]["inputs"].get("previous_composite")
    mapped = {node["id"]: node for node in graph["nodes"]}
    for link_id, source_id, output_slot, target_id, input_slot, dtype in graph["links"]:
        assert mapped[target_id]["inputs"][input_slot]["link"] == link_id
        assert link_id in mapped[source_id]["outputs"][output_slot]["links"]
        assert mapped[source_id]["outputs"][output_slot]["type"] == dtype
        assert mapped[target_id]["inputs"][input_slot]["type"] == dtype
