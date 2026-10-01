"""S26 existing external sampler controls remain exact across the stage guard."""
from copy import deepcopy
import json
from pathlib import Path

import comfy.samplers
import pytest

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg.modular_sampling.audio_refine_stage import (
    bind_audio_refine_stage, audit_audio_refine_stage,
    audit_audio_refine_tail_delivery)
from h3_audio_t8_pkg.learned_latent_upscale_advanced import audit_two_pass_h3_audio
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import NODES
from tools.build_modular_audio_refine_workflows import FAMILIES, sources, split_api, split_frontend
from test_audio_refine_advanced import (
    _allow_bundle, _abstain_bundle, _phase2_bundle, _full_video_sigmas,
    _runtime, _audit, _positive, _latent, FakeModelPatcher)


def _configured(variant):
    if variant == "dual_clock":
        model, positive, latent, plan = _allow_bundle()
        setup = refine.setup_audio_refine
        kwargs = {"model": model}
    elif variant == "dual_model":
        _, model, positive, latent, _, plan = _phase2_bundle()
        setup = refine.setup_audio_refine_dual_model
        kwargs = {"refine_model": model}
    else:
        model = FakeModelPatcher()
        positive, latent = _positive(), _latent()
        audit, _, _ = _audit(model=model, positive=positive, av_latent=latent)
        _, route, decision, _ = refine.route_audio_refine_compatibility(
            audit=audit, refine_model=model, positive=positive,
            generation_profile="turbo8", declared_first_pass_nfe=8)
        assert decision == "ALLOW"
        plan, decision, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 29)
        assert decision == "ALLOW"
        setup = refine.setup_audio_refine_compatibility
        kwargs = {"refine_model": model}

    def configured_sampling(connected_model, _latent_value, steps, *_args):
        return connected_model.clone(), comfy.samplers.KSAMPLER(lambda *_a, **_kw: None), _full_video_sigmas(steps)

    result = setup(plan=plan, positive=positive, av_latent=latent,
                   setup_sampling_fn=configured_sampling, runtime_snapshot_fn=_runtime, **kwargs)
    return plan, latent, result


@pytest.mark.parametrize("variant", ["dual_clock", "dual_model", "compatibility"])
def test_audio_refine_stage_guard_preserves_existing_controls_and_audits_candidate(variant):
    plan, original, setup = _configured(variant)
    bound = bind_audio_refine_stage(plan, original, setup.model, setup.noise, setup.guider,
                                    setup.sampler, setup.sigmas, setup.latent, setup.report_json,
                                    expected_variant=variant)
    assert all(left is right for left, right in zip(bound[:6],
        (setup.model, setup.noise, setup.guider, setup.sampler, setup.sigmas, setup.latent)))
    context = bound[6]
    assert context["bypassed"] is False
    assert json.loads(bound[7])["sampled"] is False
    candidate, report = audit_audio_refine_stage(
        context, plan, original, setup.latent, setup.report_json, setup.latent)
    assert candidate is setup.latent
    assert json.loads(report)["quality_acceptance"] is False
    changed = deepcopy(original)
    video, audio = changed["samples"].unbind()
    changed["samples"] = type(changed["samples"])((video, audio + 1))
    with pytest.raises(ValueError, match="source or Plan"):
        audit_audio_refine_stage(context, plan, changed, setup.latent, setup.report_json, setup.latent)
    wrong_sigmas = setup.sigmas.clone()
    wrong_sigmas[0] = .1
    with pytest.raises(ValueError, match="sample controls"):
        bind_audio_refine_stage(plan, original, setup.model, setup.noise, setup.guider,
                                setup.sampler, wrong_sigmas, setup.latent, setup.report_json)


def test_audio_refine_abstain_remains_empty_sigmas_and_exact_original_av():
    model, positive, latent, plan = _abstain_bundle()
    setup = refine.setup_audio_refine(plan=plan, model=model, positive=positive,
                                      av_latent=latent, runtime_snapshot_fn=_runtime)
    assert setup.sigmas.numel() == 0
    bound = bind_audio_refine_stage(plan, latent, setup.model, setup.noise, setup.guider,
                                    setup.sampler, setup.sigmas, setup.latent, setup.report_json,
                                    expected_variant="dual_clock")
    assert bound[6]["bypassed"] is True
    _, report = audit_audio_refine_stage(bound[6], plan, latent, setup.latent,
                                         setup.report_json, setup.latent)
    assert json.loads(report)["status"] == "abstain_passthrough"
    changed = deepcopy(setup.latent)
    video, audio = changed["samples"].unbind()
    changed["samples"] = type(changed["samples"])((video, audio + .01))
    with pytest.raises(ValueError, match="abstain path"):
        audit_audio_refine_stage(bound[6], plan, latent, setup.latent, setup.report_json, changed)


def test_abstain_safe_delivery_preserves_old_sampled_audit_and_rejects_false_passthrough():
    plan, original, setup = _configured("dual_clock")
    bound = bind_audio_refine_stage(plan, original, setup.model, setup.noise, setup.guider,
                                    setup.sampler, setup.sigmas, setup.latent, setup.report_json,
                                    expected_variant="dual_clock")
    candidate = deepcopy(setup.latent)
    video, audio = candidate["samples"].unbind()
    candidate["samples"] = type(candidate["samples"])((video, audio + .25))
    candidate, stage_report = audit_audio_refine_stage(
        bound[6], plan, original, setup.latent, setup.report_json, candidate)
    expected = audit_two_pass_h3_audio(setup.latent, candidate, 1.0, False, 0.0)
    assert audit_audio_refine_tail_delivery(
        bound[6], stage_report, setup.latent, candidate, 1.0, False, 0.0) == expected

    model, positive, original, plan = _abstain_bundle()
    setup = refine.setup_audio_refine(plan=plan, model=model, positive=positive,
                                      av_latent=original, runtime_snapshot_fn=_runtime)
    bound = bind_audio_refine_stage(plan, original, setup.model, setup.noise, setup.guider,
                                    setup.sampler, setup.sigmas, setup.latent, setup.report_json,
                                    expected_variant="dual_clock")
    candidate, stage_report = audit_audio_refine_stage(
        bound[6], plan, original, setup.latent, setup.report_json, original)
    output, report = audit_audio_refine_tail_delivery(
        bound[6], stage_report, setup.latent, candidate, 1.0, False, 0.0)
    assert output is original and json.loads(report)["status"] == "abstain_no_sample"
    with pytest.raises(ValueError, match="no noise_mask"):
        audit_two_pass_h3_audio(setup.latent, candidate, 1.0, False, 0.0)
    altered_report = json.loads(stage_report)
    altered_report["status"] = "candidate_received"
    with pytest.raises(ValueError, match="passthrough audit"):
        audit_audio_refine_tail_delivery(bound[6], json.dumps(altered_report),
                                         setup.latent, candidate, 1.0, False, 0.0)
    altered = deepcopy(original)
    video, audio = altered["samples"].unbind()
    altered["samples"] = type(altered["samples"])((video, audio + .01))
    with pytest.raises(ValueError, match="changed original AV"):
        audit_audio_refine_tail_delivery(bound[6], stage_report, setup.latent,
                                         altered, 1.0, False, 0.0)


def test_audio_refine_node_families_are_append_only_and_typed():
    assert [node.define_schema().node_id for node in NODES] == [
        "MiniMaxH3AudioRefineDualClockStageBindEXPT8",
        "MiniMaxH3AudioRefineDualClockStageAuditEXPT8",
        "MiniMaxH3AudioRefineDualModelStageBindEXPT8",
        "MiniMaxH3AudioRefineDualModelStageAuditEXPT8",
        "MiniMaxH3AudioRefineCompatStageBindEXPT8",
        "MiniMaxH3AudioRefineCompatStageAuditEXPT8",
        "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8"]


def test_all_existing_audio_refine_workflows_keep_the_external_sampler_and_gate():
    source_paths = sources()
    assert len(source_paths) == 10
    for path in source_paths:
        old = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = split_frontend(old)
        nodes = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        old_nodes = {node["id"]: node for node in old["nodes"]}
        setup = next(node for node in graph["nodes"] if node["type"] in FAMILIES)
        bind_type, audit_type = FAMILIES[setup["type"]]
        bind = next(node for node in graph["nodes"] if node["type"] == bind_type)
        audit = next(node for node in graph["nodes"] if node["type"] == audit_type)
        sampler = next(node for node in graph["nodes"] if node["type"] == "SamplerCustomAdvanced"
                       and any(item["name"] == "sigmas" and item.get("link") is not None
                               and links[item["link"]][1] == bind["id"]
                               for item in node["inputs"]))
        for field, slot in (("noise", 1), ("guider", 2), ("sampler", 3),
                            ("sigmas", 4), ("latent_image", 5)):
            item = next(item for item in sampler["inputs"] if item["name"] == field)
            assert links[item["link"]][1:3] == [bind["id"], slot]
        candidate_input = next(item for item in audit["inputs"]
                               if item["name"] == "candidate_av_latent")
        assert links[candidate_input["link"]][1:3] == [sampler["id"], 0]
        old_sampler = old_nodes[sampler["id"]]
        assert sampler["type"] == old_sampler["type"]
        assert sampler.get("widgets_values") == old_sampler.get("widgets_values")
        # Existing downstream Quality Gate stays the authority; only its decode
        # candidate is routed through the new guard.
        assert any(node["type"].endswith("AudioRefineQualityGateT8Advanced")
                   for node in graph["nodes"])
        assert len(graph["nodes"]) == len(old["nodes"]) + 2
        assert all(link[1] in nodes and link[3] in nodes for link in graph["links"])
        assert all(item["link"] in links for node in graph["nodes"]
                   for item in node.get("inputs", []) if item.get("link") is not None)


def test_all_ten_abstain_safe_candidates_keep_old_audit_controls_and_new_stage_proof():
    candidate_dir = (Path(__file__).resolve().parents[1] / "artifacts/development"
                     / "modular-sampling-m4-audio-refine-abstain-20260924/candidate-v3")
    if not candidate_dir.is_dir():
        pytest.skip("private abstain-safe S26 candidates are absent")
    assert len(sources()) == 10
    for path in sources():
        old = json.loads(path.read_text(encoding="utf-8"))
        expected = split_frontend(old, abstain_safe=True)
        saved = json.loads((candidate_dir / (path.stem + "_StageBound_EXP.json"))
                           .read_text(encoding="utf-8"))
        assert saved == expected
        old_audit = next(node for node in old["nodes"]
                         if node["type"] == "MiniMaxH3TwoPassAudioAuditT8Advanced")
        new_audit = next(node for node in saved["nodes"] if node["id"] == old_audit["id"])
        assert new_audit["type"] == "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8"
        assert new_audit["widgets_values"] == old_audit["widgets_values"]
        links = {link[0]: link for link in saved["links"]}
        for field, owner in (("stage_boundary", "StageBindEXPT8"),
                             ("stage_report_json", "StageAuditEXPT8")):
            item = next(item for item in new_audit["inputs"] if item["name"] == field)
            source = next(node for node in saved["nodes"]
                          if node["id"] == links[item["link"]][1])
            assert source["type"].endswith(owner)


def test_prompt_relay_audio_refine_has_independent_tail_plan_and_paired_inputs():
    path = next(path for path in sources()
                if "Prompt_Relay_Turbo8" in path.stem and "Long_Video" not in path.stem)
    original = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="ABSTAIN-safe"):
        split_frontend(original, external_refine_relay=True)
    graph = split_frontend(original, abstain_safe=True, external_refine_relay=True)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    plans = [node for node in graph["nodes"]
             if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"]
    conds = [node for node in graph["nodes"]
             if node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"]
    assert len(plans) == len(conds) == 2
    old_plan, new_plan = sorted(plans, key=lambda node: node["id"])
    old_cond, new_cond = sorted(conds, key=lambda node: node["id"])
    assert new_plan["widgets_values"] == old_plan["widgets_values"]
    assert new_plan["widgets_values"] is not old_plan["widgets_values"]

    def source(node, field):
        item = next(item for item in node["inputs"] if item["name"] == field)
        return links[item["link"]][1:3]

    assert source(new_cond, "prompt_relay_plan") == [new_plan["id"], 0]
    for field in ("model", "clip", "video_vae", "audio_vae"):
        assert source(new_cond, field) == source(old_cond, field)
    audit = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineAuditT8Advanced")
    route = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineCompatibilityRouteT8Advanced")
    setup = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced")
    for field, slot in (("model", 0), ("positive", 1), ("conditioned_prompt", 4),
                        ("media_map_json", 5), ("conditioning_report", 6)):
        assert source(audit, field) == [new_cond["id"], slot]
    for field, slot in (("refine_model", 0), ("positive", 1)):
        assert source(route, field) == [new_cond["id"], slot]
    assert source(setup, "positive") == [new_cond["id"], 1]
    assert source(audit, "av_latent") != [new_cond["id"], 2]
    first_sampler = nodes[10]
    assert source(first_sampler, "latent_image") == [old_cond["id"], 2]
    new_plan["widgets_values"][0] = "refine-only edit"
    assert old_plan["widgets_values"][0] != "refine-only edit"


def test_long_video_refine_relay_has_independent_segment_map_and_first_sampler():
    path = next(path for path in sources() if "Long_Video_Prompt_Relay_Turbo8" in path.stem)
    original = json.loads(path.read_text(encoding="utf-8"))
    graph = split_frontend(original, abstain_safe=True, external_refine_relay=True)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    old_links = {link[0]: link for link in original["links"]}

    def source(node, field, *, original_graph=False):
        item = next(item for item in node["inputs"] if item["name"] == field)
        edge = (old_links if original_graph else links)[item["link"]]
        return edge[1:3]

    def pair(kind):
        found = sorted((node for node in graph["nodes"] if node["type"] == kind),
                       key=lambda node: node["id"])
        assert len(found) == 2
        return found

    old_plan, new_plan = pair("MiniMaxH3PromptRelayPlanT8Advanced")
    old_segment, new_segment = pair("MiniMaxH3PromptRelayLongVideoPlanT8Advanced")
    old_cond, new_cond = pair("MiniMaxH3PromptRelayLongVideoConditioningT8Advanced")
    assert source(new_segment, "prompt_relay_plan") == [new_plan["id"], 0]
    for name in ("segment_index", "length", "context_frames",
                 "timeline_start_seconds", "timeline_end_seconds"):
        assert source(new_segment, name) == source(old_segment, name)
    assert source(new_cond, "prompt_relay_plan") == [new_segment["id"], 0]
    for name in ("model", "clip", "video_vae", "audio_vae", "context",
                 "segment_index", "context_frames", "length"):
        assert source(new_cond, name) == source(old_cond, name)
    assert source(old_segment, "prompt_relay_plan") == [old_plan["id"], 0]
    assert source(old_cond, "prompt_relay_plan") == [old_segment["id"], 0]
    first_sampler = nodes[14]
    for item in first_sampler["inputs"]:
        assert source(first_sampler, item["name"]) == source(
            next(node for node in original["nodes"] if node["id"] == 14),
            item["name"], original_graph=True)
    audit = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineAuditT8Advanced")
    route = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineCompatibilityRouteT8Advanced")
    setup = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced")
    for field, slot in (("model", 0), ("positive", 1), ("conditioned_prompt", 4),
                        ("media_map_json", 5), ("conditioning_report", 6)):
        assert source(audit, field) == [new_cond["id"], slot]
    for field, slot in (("refine_model", 0), ("positive", 1)):
        assert source(route, field) == [new_cond["id"], slot]
    assert source(setup, "positive") == [new_cond["id"], 1]
    assert source(audit, "av_latent") == [first_sampler["id"], 0]
    new_plan["widgets_values"][0] = "refine-only edit"
    assert old_plan["widgets_values"][0] != "refine-only edit"


def test_single_refine_relay_has_independent_checkpoint_and_disabled_tail_lora():
    source_path = next(path for path in sources()
                       if "Prompt_Relay_Turbo8" in path.stem and "Long_Video" not in path.stem)
    old = json.loads(source_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="separate Relay"):
        split_frontend(old, abstain_safe=True, independent_refine_model=True)
    long_path = next(path for path in sources() if "Long_Video_Prompt_Relay_Turbo8" in path.stem)
    with pytest.raises(ValueError, match="cold-stage route adapter"):
        split_frontend(json.loads(long_path.read_text(encoding="utf-8")),
                       abstain_safe=True, external_refine_relay=True,
                       independent_refine_model=True)
    graph = split_frontend(old, abstain_safe=True, external_refine_relay=True,
                           independent_refine_model=True)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    loaders = sorted((node for node in graph["nodes"] if node["type"] == "UNETLoader"),
                     key=lambda node: node["id"])
    loras = sorted((node for node in graph["nodes"]
                    if node["type"] == "LoraLoaderBypassModelOnly"),
                   key=lambda node: node["id"])
    conds = sorted((node for node in graph["nodes"]
                    if node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"),
                   key=lambda node: node["id"])
    assert len(loaders) == len(loras) == len(conds) == 2
    first_loader, tail_loader = loaders
    first_lora, tail_lora = loras
    first_cond, tail_cond = conds

    def source(node, field):
        item = next(item for item in node["inputs"] if item["name"] == field)
        return links[item["link"]][1:3]

    assert source(first_cond, "model") == [first_loader["id"], 0]
    assert source(tail_cond, "model") == [tail_loader["id"], 0]
    assert source(first_lora, "model") == [first_cond["id"], 0]
    assert source(tail_lora, "model") == [tail_cond["id"], 0]
    assert source(nodes[7], "model") == [first_lora["id"], 0]
    audit = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineAuditT8Advanced")
    route = next(node for node in graph["nodes"]
                 if node["type"] == "MiniMaxH3AudioRefineCompatibilityRouteT8Advanced")
    assert source(audit, "model") == [tail_lora["id"], 0]
    assert source(route, "refine_model") == [tail_lora["id"], 0]
    assert source(audit, "positive") == [tail_cond["id"], 1]
    assert tail_lora["widgets_values"][1] == 0.0
    assert first_lora["widgets_values"][1] == 1.0
    assert tail_loader["widgets_values"] == first_loader["widgets_values"]
    assert tail_loader["widgets_values"] is not first_loader["widgets_values"]
    tail_loader["widgets_values"][0] = "refine-only-model.safetensors"
    tail_lora["widgets_values"][0] = "refine-only-lora.safetensors"
    assert first_loader["widgets_values"][0] != "refine-only-model.safetensors"
    assert first_lora["widgets_values"][0] != "refine-only-lora.safetensors"


def test_audio_refine_api_conversion_preserves_legacy_widgets_and_new_core_defaults():
    frontend = {"nodes": [
        {"id": 1, "type": "CreateVideo", "widgets_values": [24.0, 8], "inputs": []},
        {"id": 2, "type": "SaveVideo", "widgets_values": ["segment", "mp4"], "inputs": []},
        {"id": 3, "type": "LoadImage", "widgets_values": ["source.png", "image"], "inputs": []},
        {"id": 4, "type": "RandomNoise", "widgets_values": [17, "fixed"], "inputs": []},
        {"id": 5, "type": "SaveVideo", "widgets_values": ["older_segment"], "inputs": []}],
        "links": []}
    schema = {
        "CreateVideo": {"required": {"fps": ["FLOAT", {}]}, "optional": {
            "bit_depth": ["COMBO", {}], "color_space": ["COMBO", {"default": "sRGB"}],
            "codec": ["COMBO", {"default": "none"}]}},
        "SaveVideo": {"required": {"filename_prefix": ["STRING", {}]}, "optional": {
            "format": ["COMFY_DYNAMICCOMBO_V3", {}], "codec": ["COMFY_DYNAMICCOMBO_V3", {}]}},
        "LoadImage": {"required": {"image": [["source.png"], {}]}},
        "RandomNoise": {"required": {"noise_seed": ["INT", {}]}}}
    api = split_api(frontend, schema)
    assert api["1"]["inputs"] == {"fps": 24.0, "bit_depth": 8,
                                     "color_space": "sRGB", "codec": "none"}
    assert api["2"]["inputs"] == {"filename_prefix": "segment", "format": "mp4"}
    assert api["5"]["inputs"] == {"filename_prefix": "older_segment", "format": "auto"}
    assert api["3"]["inputs"] == {"image": "source.png"}
    assert api["4"]["inputs"] == {"noise_seed": 17}
    frontend["nodes"][0]["widgets_values"] = [24.0, 8, "sRGB", "none", "unexpected"]
    with pytest.raises(ValueError, match="Widget contract changed"):
        split_api(frontend, schema)
    frontend["nodes"][0]["widgets_values"] = [24.0, 8]
    frontend["nodes"][1]["widgets_values"] = ["segment"]
    schema["SaveVideo"]["optional"].pop("format")
    with pytest.raises(ValueError, match="no format input"):
        split_api(frontend, schema)
