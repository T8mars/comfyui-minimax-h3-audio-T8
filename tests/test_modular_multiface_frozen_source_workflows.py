"""New explicit pairs preserve all old graphs and the original H3 math."""
from collections import Counter
from copy import deepcopy
import asyncio
import hashlib

import pytest

from tools import build_formal_multiface_frozen_source_workflows as builder
from tools import build_formal_face_refine_storage_workflows as storage


@pytest.mark.parametrize("variant,mode", builder.FILES)
def test_additive_pair_keeps_old_bytes_and_every_named_edge(variant, mode):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in storage.face.DESTINATION.glob("*.json")}
    assert len(before) == 84
    frontend, api = builder.graph_for(variant, mode)
    mapped = {n["id"]: n for n in frontend["nodes"]}
    for link_id, source, source_slot, target, target_slot, dtype in frontend["links"]:
        item = mapped[target]["inputs"][target_slot]
        assert item["link"] == link_id and link_id in mapped[source]["outputs"][source_slot]["links"]
        assert api[str(target)]["inputs"][item["name"]] == [str(source), source_slot]
        assert item["type"] == dtype == mapped[source]["outputs"][source_slot]["type"]
    old_frontend, old_api = storage.graph_for(variant, "combined", mode)
    counts = Counter(node["class_type"] for node in api.values())
    writer, = (n for n in api.values() if n["class_type"] == "MiniMaxH3SaveVideoIsolatedEXPT8")
    assert set(writer["inputs"]) == {"video", "filename_prefix", "timeout_seconds", "max_staging_gib", "min_free_disk_gib"}
    people = 2 if variant == "multiface2" else 3
    assert counts["MiniMaxH3MultiFaceStageAuditEXPT8"] == people
    create, = (n for n in api.values() if n["class_type"] == "CreateVideo")
    old_create, = (n for n in old_api.values() if n["class_type"] == "CreateVideo")
    assert create == old_create
    approval = {k: n for k, n in old_api.items() if n["class_type"] == "MiniMaxH3MultiFaceCompositeT8Advanced"}
    for key, old in approval.items():
        assert api[key]["inputs"]["accept_candidate"] == old["inputs"]["accept_candidate"]
        assert api[key]["inputs"]["overlap_policy"] == old["inputs"]["overlap_policy"]
        assert api[key]["inputs"].get("previous_composite") == old["inputs"].get("previous_composite")
    if mode == "full_save":
        assert counts[builder.SAVE] == counts["MiniMaxH3StageSamplerEXPT8"] == people
        assert counts["MiniMaxH3StageSaveEXPT8"] == people
        for key, node in old_api.items():
            if node["class_type"] != "SaveVideo":
                assert api[key] == node
        assert all(n["inputs"]["confirm_save"] is False for n in api.values() if n["class_type"] == builder.SAVE)
    else:
        assert counts[builder.LOAD] == counts["MiniMaxH3StageLoadEXPT8"] == people
        assert not counts["MiniMaxH3StageSamplerEXPT8"] and not counts["MiniMaxH3SAM31MultiPersonTrackT8Advanced"]
        assert not counts["MiniMaxH3MultiFaceRepairJobT8Advanced"] and not counts["MiniMaxH3PromptRelayConditioningT8Advanced"]
        assert not counts["UNETLoader"] and not counts["CLIPLoader"]
        for node in api.values():
            if node["class_type"] in (builder.LOAD, "MiniMaxH3StageLoadEXPT8"):
                assert node["inputs"]["artifact_path"] == node["inputs"]["artifact_sha256"] == ""
            if node["class_type"] == "MiniMaxH3MultiFaceStageAuditEXPT8":
                for name in ("face_plan", "source_frames", "av_latent"):
                    assert api[node["inputs"][name][0]]["class_type"] == builder.LOAD
    assert frontend["id"] != old_frontend.get("id")
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before


def test_non_multiface_or_unknown_modes_cannot_inherit_this_storage():
    with pytest.raises(ValueError):
        builder.graph_for("anime", "full_save")
    with pytest.raises(ValueError):
        builder.graph_for("multiface2", "unknown")


@pytest.fixture(scope="module")
def current_core():
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    load_live_info()
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise
    from comfy_extras.nodes_video import LoadVideo, GetVideoComponents, CreateVideo
    for cls in (BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise,
                LoadVideo, GetVideoComponents, CreateVideo):
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls


@pytest.mark.parametrize("variant,mode", builder.FILES)
def test_additive_pair_validates_on_actual_Core_and_not_just_edge_inventory(variant, mode, current_core):
    import execution
    _, api = builder.graph_for(variant, mode)
    for node in api.values():
        if node["class_type"] == "LoadVideo":
            node["inputs"]["file"] = "0.6.mp4"
        elif node["class_type"] == "LoadImage":
            node["inputs"]["image"] = "0 (1).png"
    valid, error, _outputs, failures = asyncio.run(
        execution.validate_prompt(f"frozen-multiface-{variant}-{mode}", deepcopy(api), None))
    assert valid and not failures, (error, failures)
