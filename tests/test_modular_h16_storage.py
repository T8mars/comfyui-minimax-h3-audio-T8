"""H16 exact frozen-window storage and subsequent-window resume contracts."""
from dataclasses import replace
import json

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.h16_nodes import (
    MiniMaxH3H16WindowLoadEXPT8, MiniMaxH3H16WindowSaveEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.h16_stages import build_h16_plan, sample_h16_pass2
from h3_audio_t8_pkg.modular_sampling.h16_storage import load_window, save_window
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise
from test_modular_h16_stages import _h16_source, _native_like_piece, _no_op_lift
from tools.build_modular_h16_storage_workflow import (
    TARGET, TARGET_V2, candidate_api, freeze_frontend, resume_frontend,
)
from tools.build_modular_h16_workflow import TEMPLATE, split_frontend
from tools.build_modular_h16_effect_storage_workflow import (
    TARGET as EFFECT_TARGET, candidate_pair,
)


def _prepare(source):
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    return plan, noise, context


def _window(source, plan, noise, context, index, previous=None,
            audio_output="preserve_first_pass"):
    segment, spec, _ = slice_chunked_source(source, plan, index)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    output, result, _core, _report = sample_h16_pass2(
        object(), [[torch.zeros(1), {}]], segment, lifted, spec,
        context, plan, noise, object(), torch.tensor([0.5, 0.0]), previous,
        audio_output=audio_output,
    )
    return segment, spec, output, result


@pytest.mark.parametrize("audio_output", ["preserve_first_pass", "refined_exp"])
def test_h16_exact_freeze_reloads_with_fresh_audio_object_and_resumes_next_window(
        monkeypatch, tmp_path, audio_output):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _m, _p, sampler: sampler)
    monkeypatch.setattr(legacy, "sample_piece", _native_like_piece)
    source = _h16_source()
    plan, noise, context = _prepare(source)
    segment, spec, _output, first = _window(source, plan, noise, context, 0,
                                            audio_output=audio_output)
    _s1, _p1, expected, _r1 = _window(source, plan, noise, context, 1, first,
                                       audio_output=audio_output)
    _output, _result, path, digest, manifest_json = save_window(
        first, segment, spec, context, plan, tmp_path,
    )
    manifest = json.loads(manifest_json)
    assert manifest["automatic_cache_reuse"] is False
    assert manifest["execution_identity_certified"] is False

    fresh_source = _h16_source()
    fresh_plan, fresh_noise, fresh_context = _prepare(fresh_source)
    assert fresh_context.original_audio is not context.original_audio
    fresh_segment, fresh_spec, _ = slice_chunked_source(fresh_source, fresh_plan, 0)
    loaded_output, loaded, report_json = load_window(
        fresh_segment, fresh_spec, fresh_context, fresh_plan, tmp_path, path, digest,
    )
    assert loaded.core_result.output_latent["samples"].tensors[1] is fresh_context.original_audio
    assert loaded_output["samples"].tensors[1] is fresh_context.original_audio
    assert json.loads(report_json)["automatic_cache_reuse"] is False
    _segment, _spec, actual, _result = _window(
        fresh_source, fresh_plan, fresh_noise, fresh_context, 1, loaded,
        audio_output=audio_output,
    )
    for got, want in zip(actual["samples"].unbind(), expected["samples"].unbind(), strict=True):
        assert torch.equal(got, want)


def test_h16_freeze_rejects_changed_content_wrong_window_and_corrupt_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _m, _p, sampler: sampler)
    monkeypatch.setattr(legacy, "sample_piece", _native_like_piece)
    source = _h16_source()
    plan, noise, context = _prepare(source)
    segment, spec, _output, first = _window(source, plan, noise, context, 0)
    _output, _result, path, digest, _manifest = save_window(
        first, segment, spec, context, plan, tmp_path,
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_window(segment, spec, context, plan, tmp_path, path, "0" * 64)
    with pytest.raises(ValueError, match="traversal|escapes|relative"):
        load_window(segment, spec, context, plan, tmp_path, "../manifest.json", digest)
    next_segment, next_spec, _ = slice_chunked_source(source, plan, 1)
    with pytest.raises(ValueError, match="does not match"):
        load_window(next_segment, next_spec, context, plan, tmp_path, path, digest)
    altered = _h16_source()
    altered["samples"].tensors[0][..., 0, 0] += 1
    altered_plan, _altered_noise, altered_context = _prepare(altered)
    altered_segment, altered_spec, _ = slice_chunked_source(altered, altered_plan, 0)
    with pytest.raises(ValueError, match="does not match"):
        load_window(altered_segment, altered_spec, altered_context, altered_plan,
                    tmp_path, path, digest)
    bad_result = replace(first, audio_output="refined_exp")
    with pytest.raises(ValueError, match="history"):
        save_window(bad_result, segment, spec, context, plan, tmp_path)
    state = tmp_path / path.replace("manifest.json", "state.safetensors")
    with state.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="size or SHA"):
        load_window(segment, spec, context, plan, tmp_path, path, digest)


def test_h16_freeze_final_refined_audio_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _m, _p, sampler: sampler)
    monkeypatch.setattr(legacy, "sample_piece", _native_like_piece)
    source = _h16_source()
    plan, noise, context = _prepare(source)
    _s0, _p0, _o0, first = _window(source, plan, noise, context, 0,
                                    audio_output="refined_exp")
    segment, spec, original, final = _window(source, plan, noise, context, 1, first,
                                              audio_output="refined_exp")
    assert original["samples"].tensors[1] is not context.original_audio
    _output, _result, path, digest, _manifest = save_window(
        final, segment, spec, context, plan, tmp_path,
    )
    fresh_source = _h16_source()
    fresh_plan, _noise, fresh_context = _prepare(fresh_source)
    fresh_segment, fresh_spec, _ = slice_chunked_source(fresh_source, fresh_plan, 1)
    loaded_output, loaded, _report = load_window(
        fresh_segment, fresh_spec, fresh_context, fresh_plan, tmp_path, path, digest,
    )
    assert loaded.core_result.output_latent["samples"].tensors[1] is fresh_context.original_audio
    assert loaded_output["samples"].tensors[1] is not fresh_context.original_audio
    for got, want in zip(loaded_output["samples"].unbind(), original["samples"].unbind(), strict=True):
        assert torch.equal(got, want)


def test_h16_freeze_public_node_schemas_are_additive():
    save = MiniMaxH3H16WindowSaveEXPT8.GET_NODE_INFO_V1()
    load = MiniMaxH3H16WindowLoadEXPT8.GET_NODE_INFO_V1()
    assert save["output_name"] == ["cumulative_av_latent", "window_result",
                                   "artifact_path", "artifact_sha256", "report_json"]
    assert load["output_name"] == ["cumulative_av_latent", "window_result", "report_json"]


def test_h16_fixed_freeze_resume_drafts_have_no_low_sampler_in_resume():
    freeze = json.loads((TARGET / "01_freeze_after_window_2.json").read_text(encoding="utf-8"))
    resume = json.loads((TARGET / "02_resume_windows_3_to_6_DRAFT.json").read_text(encoding="utf-8"))
    freeze_nodes = {node["id"]: node for node in freeze["nodes"]}
    resume_nodes = {node["id"]: node for node in resume["nodes"]}
    assert {node["type"] for node in freeze_nodes.values()} >= {
        "MiniMaxH3NativeLatentCheckpointSaveT8Advanced", "MiniMaxH3H16WindowSaveEXPT8"}
    assert not any(node["type"] in {"SamplerCustomAdvanced",
                                     "MiniMaxH3LearnedLatentUpscaleT8Advanced"}
                   for node in resume_nodes.values())
    windows = sorted(node["id"] for node in resume_nodes.values()
                     if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8")
    assert windows == [41, 44, 47, 50]
    links = {link[0]: link for link in resume["links"]}
    previous = next(item for item in resume_nodes[41]["inputs"]
                    if item["name"] == "previous_result")
    assert resume_nodes[links[previous["link"]][1]]["type"] == "MiniMaxH3H16WindowLoadEXPT8"
    assert (resume_nodes[51]["type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
            and resume_nodes[52]["widgets_values"] == ["", ""])
    freeze_api = candidate_api(freeze, freeze=True)
    resume_api = candidate_api(resume, freeze=False)
    assert freeze_api["51"]["inputs"]["confirm_save"] is False
    assert freeze_api["52"]["inputs"]["confirm_save"] is False
    assert resume_api["51"]["inputs"]["checkpoint_path"] == ""


def test_h16_saved_relay_eav_pair_has_only_later_windows_after_load():
    regenerated = candidate_pair()
    labels = ("01_freeze_after_window_2_relay_eav",
              "02_resume_windows_3_to_6_relay_eav_DRAFT")
    for label, expected in zip(labels, regenerated, strict=True):
        frontend = json.loads((EFFECT_TARGET / f"{label}.json").read_text(encoding="utf-8"))
        api = json.loads((EFFECT_TARGET / f"{label}.api.json").read_text(encoding="utf-8"))
        assert frontend == expected
        assert api == candidate_api(expected, freeze=label.startswith("01"))
        counts = [sum(node["type"] == kind for node in frontend["nodes"])
                  for kind in ("MiniMaxH3H16Pass2WindowEXPT8",
                               "MiniMaxH3H16RelayProjectEXPT8",
                               "MiniMaxH3H16EAVApplyEXPT8",
                               "MiniMaxH3H16EAVAuditEXPT8")]
        assert counts == ([3] * 4 if label.startswith("01") else [4] * 4)

    freeze, resume = regenerated
    freeze_types = {node["type"] for node in freeze["nodes"]}
    assert {"MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
            "MiniMaxH3H16WindowSaveEXPT8"} <= freeze_types
    nodes = {node["id"]: node for node in resume["nodes"]}
    links = {link[0]: link for link in resume["links"]}
    assert not {"SamplerCustomAdvanced", "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"} & {
                    node["type"] for node in nodes.values()}
    loaded = next(node for node in nodes.values()
                  if node["type"] == "MiniMaxH3H16WindowLoadEXPT8")
    native = next(node for node in nodes.values()
                  if node["type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced")
    bridge = next(node for node in nodes.values()
                  if node["type"] == "MiniMaxH3H16VerifiedNativeSourceEXPT8")
    learned = next(item for item in nodes[15]["inputs"]
                   if item["name"] == "learned_latent")
    assert tuple(links[learned["link"]][1:3]) == (bridge["id"], 0)
    for slot, name in enumerate(("av_latent", "resume_verified", "checkpoint_id",
                                 "content_sha256", "file_sha256", "manifest_json",
                                 "report_json")):
        item = next(item for item in bridge["inputs"] if item["name"] == name)
        source_slot = (0, 2, 3, 4, 5, 6, 7)[slot]
        assert tuple(links[item["link"]][1:3]) == (native["id"], source_slot)
    for node_id in (41, 56, 70):
        previous = next(item for item in nodes[node_id]["inputs"]
                        if item["name"] == "previous_result")
        assert tuple(links[previous["link"]][1:3]) == (loaded["id"], 1)
    relay = next(node for node in nodes.values()
                 if node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced")
    assert all(next(item for item in relay["inputs"] if item["name"] == name)["link"] is None
               for name in ("width", "height"))
    assert candidate_api(resume, freeze=False)[str(relay["id"])]["inputs"]["width"] > 0
    assert candidate_api(resume, freeze=False)[str(relay["id"])]["inputs"]["height"] > 0
    assert sorted(node["id"] for node in nodes.values()
                  if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8") == [41, 44, 47, 50]


def test_h16_plain_storage_v2_uses_verified_native_bridge_without_rewriting_v1():
    base = split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    generated = (freeze_frontend(base), resume_frontend(base))
    for label, expected in zip(("01_freeze_after_window_2",
                                "02_resume_windows_3_to_6_DRAFT"), generated, strict=True):
        saved = json.loads((TARGET_V2 / f"{label}.json").read_text(encoding="utf-8"))
        api = json.loads((TARGET_V2 / f"{label}.api.json").read_text(encoding="utf-8"))
        assert saved == expected
        assert api == candidate_api(expected, freeze=label.startswith("01"))
    resume = generated[1]
    nodes = {node["id"]: node for node in resume["nodes"]}
    links = {link[0]: link for link in resume["links"]}
    assert nodes[53]["type"] == "MiniMaxH3H16VerifiedNativeSourceEXPT8"
    learned = next(item for item in nodes[15]["inputs"]
                   if item["name"] == "learned_latent")
    assert tuple(links[learned["link"]][1:3]) == (53, 0)
    assert not any(node["type"] == "SamplerCustomAdvanced" for node in nodes.values())
    assert candidate_api(resume, freeze=False)["51"]["inputs"]["checkpoint_path"] == ""
