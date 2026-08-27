from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced as effects
from h3_audio_t8_pkg.nodes_long_video_in_node_loop_advanced import (
    LONG_VIDEO_IN_NODE_LOOP_ADVANCED_NODE_CLASSES,
    MiniMaxH3LongVideoInNodeLoopT8Advanced,
)
from h3_audio_t8_pkg.nodes_long_video_in_node_loop_effects_advanced import (
    LONG_VIDEO_IN_NODE_LOOP_EFFECTS_ADVANCED_NODE_CLASSES,
    MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from helpers import plugin_widget_map


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-08-27_H3_In_Node_Long_Video_Prompt_Relay_EAV_Stock20_Advanced_EXP.json"
)


def _relay_plan(length: int = 226) -> dict:
    return build_prompt_relay_plan(
        global_prompt="同一人物在连续镜头中完成动作，环境声连续",
        local_prompts="人物抬手并说话\n人物继续奔跑并停下",
        length=length,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )[0]


def _kwargs(plan: dict) -> dict:
    return {
        "chain_id": "effects-loop-test",
        "total_duration_seconds": 226 / 24,
        "render_window_frames": 124,
        "context_frames": 22,
        "global_prompt": "",
        "segment_prompts_json": "",
        "prompt_relay_mode": "apply_exp",
        "query_chunk_rows": 256,
        "eav_mode": "apply_exp",
        "eav_tau": 4.0,
        "eav_start_video_progress": 0.0,
        "eav_end_video_progress": 1.0,
        "eav_max_workspace_mib": 32,
        "eav_g_hard_limit": 1.5,
        "minimum_free_vram_mib": 512,
        "base_seed": 17,
        "seed_policy": "increment",
        "steps": 20,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
        "width": 736,
        "height": 416,
        "task_type": "auto",
        "context_audio": "video_and_audio",
        "audio_mode": "native",
        "audio_denoise_strength": 0.35,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "first_frame_reuse": "segment0_only",
        "persistent_identity_strategy": "single_reference",
        "persistent_identity_interval": 1,
        "resume_existing": True,
        "filename_prefix": "Effects_Loop_Test",
        "audio_seam_policy": "cosine_bridge",
        "bridge_ms": 5.0,
        "bit_depth": 8,
        "crf": 18,
        "model_id": "test-model",
        "prompt_relay_plan": plan,
    }


def test_new_node_is_append_only_and_old_node_schema_is_untouched():
    assert LONG_VIDEO_IN_NODE_LOOP_ADVANCED_NODE_CLASSES == [
        MiniMaxH3LongVideoInNodeLoopT8Advanced
    ]
    assert LONG_VIDEO_IN_NODE_LOOP_EFFECTS_ADVANCED_NODE_CLASSES == [
        MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced
    ]
    old_schema = MiniMaxH3LongVideoInNodeLoopT8Advanced.define_schema()
    new_schema = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
    assert old_schema.node_id == "MiniMaxH3LongVideoInNodeLoopT8Advanced"
    assert new_schema.node_id == "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
    assert new_schema.is_output_node is True
    assert new_schema.is_experimental is True
    inputs = {item.id: item for item in new_schema.inputs}
    assert inputs["steps"].default == 20
    assert inputs["prompt_relay_mode"].default == "disabled"
    assert inputs["eav_mode"].default == "disabled"
    assert inputs["minimum_free_vram_mib"].default == 512


def test_frontend_workflow_is_native_stock20_and_documents_combined_contract():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert all("class_type" not in node for node in nodes.values())

    effect_nodes = [
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
    ]
    assert len(effect_nodes) == 1
    values = plugin_widget_map(
        effect_nodes[0], MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced
    )
    assert values["prompt_relay_mode"] == "apply_exp"
    assert values["eav_mode"] == "apply_exp"
    assert values["steps"] == 20
    assert values["sampler_name"] == "dual_clock_euler"
    assert values["scheduler"] == "native_flow"
    assert values["global_prompt"] == ""
    assert values["segment_prompts_json"] == ""
    assert values["resume_existing"] is True
    assert values["minimum_free_vram_mib"] == 512

    loader = next(node for node in nodes.values() if node["type"] == "UNETLoader")
    assert "turbo" not in str(loader["widgets_values"][0]).lower()
    assert not any("LoraLoader" in node["type"] for node in nodes.values())
    plan = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
    )
    assert plan["widgets_values"][2] >= round(values["total_duration_seconds"] * 24)
    notes = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 4
    assert "Prompt Relay不会在每段重新开始" in notes
    assert "20步" in notes
    assert "effects_audit.json" in notes
    assert "不等于绝不会OOM" in notes


def test_effect_mode_contracts_fail_before_sampling():
    with pytest.raises(ValueError, match="requires a connected"):
        effects._validate_effect_modes(
            prompt_relay_mode="apply_exp",
            prompt_relay_plan=None,
            eav_mode="disabled",
            steps=20,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
            global_prompt="",
            segment_prompts_json="",
        )
    with pytest.raises(ValueError, match="clear segment_prompts_json"):
        effects._validate_effect_modes(
            prompt_relay_mode="apply_exp",
            prompt_relay_plan=_relay_plan(),
            eav_mode="disabled",
            steps=20,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
            global_prompt="",
            segment_prompts_json='["override"]',
        )
    with pytest.raises(ValueError, match="requires Stock20"):
        effects._validate_effect_modes(
            prompt_relay_mode="disabled",
            prompt_relay_plan=None,
            eav_mode="apply_exp",
            steps=8,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
            global_prompt="test",
            segment_prompts_json="",
        )


def test_effect_audit_sidecar_is_signed_and_bound_to_candidate(tmp_path):
    descriptor = tmp_path / "candidate.json"
    descriptor.write_text("{}", encoding="utf-8")
    effects._write_effects_audit(
        str(descriptor),
        {
            "contract_sha256": "contract",
            "segment_index": 2,
            "candidate_id": "candidate",
        },
    )
    loaded = effects._load_effects_audit(
        descriptor,
        contract_sha256="contract",
        segment_index=2,
        candidate_id="candidate",
    )
    assert loaded["format"] == effects.EFFECTS_AUDIT_FORMAT

    sidecar = descriptor.parent / effects.EFFECTS_AUDIT_NAME
    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["segment_index"] = 3
    sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        effects._load_effects_audit(
            descriptor,
            contract_sha256="contract",
            segment_index=2,
            candidate_id="candidate",
        )


def _install_fake_combined_runtime(monkeypatch, tmp_path: Path):
    manifest = {"revision": 0, "segments": []}
    candidates: dict[str, dict] = {}
    events: list[tuple] = []
    plans = (
        SimpleNamespace(
            context_frames=0,
            render_frames=124,
            final_frame_count=124,
            timeline_start_seconds=0.0,
            timeline_end_seconds=124 / 24,
            final_duration_seconds=124 / 24,
            trim_start_seconds=0.0,
            save_context=True,
            is_final_segment=False,
        ),
        SimpleNamespace(
            context_frames=22,
            render_frames=124,
            final_frame_count=102,
            timeline_start_seconds=124 / 24,
            timeline_end_seconds=226 / 24,
            final_duration_seconds=102 / 24,
            trim_start_seconds=22 / 24,
            save_context=False,
            is_final_segment=True,
        ),
    )
    segments = tuple(
        SimpleNamespace(index=i, prompt=f"prompt-{i}", seed=17 + i, plan=plan)
        for i, plan in enumerate(plans)
    )

    def resolve(*_args, **_kwargs):
        accepted = len(manifest["segments"])
        return (
            SimpleNamespace(
                chain_id="effects-loop-test",
                segments=segments,
                accepted_count=accepted,
                manifest_revision=manifest["revision"],
                complete=accepted == len(segments),
                sampling_summary="20-step dual_clock_euler/native_flow shift12/3",
                steps=20,
                shift_video=12.0,
                shift_audio=3.0,
                sampler_name="dual_clock_euler",
                scheduler="native_flow",
            ),
            manifest,
        )

    def load_manifest(_chain_id, allow_new=False):
        assert allow_new or manifest["segments"]
        return manifest, "primary" if manifest["segments"] else "new"

    def load_context(_chain_id, index):
        if index == 0:
            return {"empty": True}, False, "", manifest["revision"], "{}"
        parent = manifest["segments"][index - 1]
        return (
            {"empty": False, "source": index - 1},
            True,
            parent["candidate_id"],
            manifest["revision"],
            "{}",
        )

    def build_relay(**kwargs):
        index = int(kwargs["segment_index"])
        projected = kwargs["prompt_relay_plan"]
        events.append(("relay", index, projected["long_video_projection"]["render_start_frame"]))
        return (
            f"relay-model-{index}",
            index,
            {"samples": torch.zeros((1, 1, 1, 1))},
            None,
            projected["compiled_prompt"],
            "{}",
            json.dumps(
                {
                    "status": "applied_exp",
                    "long_video_report": {"segment_index": index},
                }
            ),
        )

    def setup(model, _latent, *_args):
        index = int(str(model).rsplit("-", 1)[-1])
        events.append(("sigmas", index))
        return model, object(), torch.linspace(1.0, 0.0, 21)

    def combine(model, _sigmas, **kwargs):
        index = int(kwargs["segment_index"])
        assert model == f"relay-model-{index}"
        events.append(("compose_owner", index))
        return model, {"segment_index": index}, json.dumps({"status": "ready"})

    def sample(_model, positive, latent, **_kwargs):
        events.append(("sample", int(positive)))
        return dict(latent)

    def finalize(sampled, runtime):
        events.append(("eav_audit", int(runtime["segment_index"])))
        return sampled, json.dumps({"status": "verified", "nfe": 20})

    def decode(_latent, _video_vae, _audio_vae):
        frames = torch.zeros((124, 4, 4, 3))
        audio = {"waveform": torch.zeros((1, 2, 170000)), "sample_rate": 32000}
        return frames, audio, {}, {}

    def trim(frames, start_seconds, duration_seconds, audio, _fps):
        start = round(start_seconds * 24)
        frame_count = round(duration_seconds * 24)
        sample_count = round(duration_seconds * 32000)
        return (
            frames[start : start + frame_count],
            {"waveform": audio["waveform"][..., :sample_count], "sample_rate": 32000},
            json.dumps({"frame_count": frame_count}),
        )

    def save(*args):
        index = int(args[4])
        candidate_id = str(args[9])
        candidate_dir = (
            tmp_path
            / "candidates"
            / f"segment_{index:05d}"
            / candidate_id
        )
        candidate_dir.mkdir(parents=True)
        path = candidate_dir / "candidate.json"
        candidate = {
            "index": index,
            "candidate_id": candidate_id,
            "parent_candidate_id": str(args[7]),
            "parent_manifest_revision": int(args[8]),
            "frame_count": int(args[0].shape[0]),
            "timeline_start_frame": round(float(args[5]) * 24),
            "timeline_end_frame": round(float(args[5]) * 24) + int(args[0].shape[0]),
            "is_final_segment": not bool(args[6]),
            "model_id": str(args[10]),
            "sampling_summary": str(args[11]),
            "prompt": str(args[12]),
            "seed": int(args[13]),
            "width": 736,
            "height": 416,
            "video_sha256": f"video-{index}",
        }
        path.write_text(json.dumps(candidate), encoding="utf-8")
        candidates[str(path)] = candidate
        events.append(("save", index))
        return str(path), str(candidate_dir / "candidate.mp4"), "{}"

    def load_candidate(path):
        candidate = candidates[str(Path(path))]
        return candidate, str(Path(path).with_suffix(".mp4"))

    def accept(path, accept_candidate, _replace_policy, _strict):
        assert accept_candidate is True
        candidate = candidates[str(Path(path))]
        index = int(candidate["index"])
        audit_path = Path(path).parent / effects.EFFECTS_AUDIT_NAME
        assert audit_path.is_file()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["enhance_a_video_audit"]["status"] == "verified"
        assert events[-2:] == [("eav_audit", index), ("save", index)]
        events.append(("accept", index))
        manifest["segments"].append(dict(candidate))
        manifest["revision"] += 1
        return "accepted.mp4", True, str(tmp_path / "manifest.json"), "{}"

    def compose(_chain_id, *_args):
        output = tmp_path / "final.mp4"
        output.write_bytes(b"effects-loop-final")
        return str(output), json.dumps(
            {"output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        )

    monkeypatch.setattr(effects, "resolve_long_video_orchestration", resolve)
    monkeypatch.setattr(effects, "long_video_chain_root", lambda _chain_id: tmp_path)
    monkeypatch.setattr(effects, "_exclusive_loop_lock", lambda _root: nullcontext())
    monkeypatch.setattr(effects, "load_delivery_manifest", load_manifest)
    monkeypatch.setattr(effects, "load_accepted_context", load_context)
    monkeypatch.setattr(effects, "build_prompt_relay_long_video_conditioning", build_relay)
    monkeypatch.setattr(effects, "patch_long_video_model", lambda model: model)
    monkeypatch.setattr(effects, "setup_dual_clock_sampling", setup)
    monkeypatch.setattr(effects, "build_eav_prompt_relay_long_video_model", combine)
    monkeypatch.setattr(effects, "_sample_prepared_segment", sample)
    monkeypatch.setattr(effects, "finalize_eav_runtime", finalize)
    monkeypatch.setattr(effects, "decode_av_latent", decode)
    monkeypatch.setattr(effects, "trim_av_output", trim)
    monkeypatch.setattr(effects, "save_long_video_candidate", save)
    monkeypatch.setattr(effects, "load_long_video_candidate_descriptor", load_candidate)
    monkeypatch.setattr(effects, "accept_long_video_candidate", accept)
    monkeypatch.setattr(effects, "compose_accepted_long_video", compose)
    monkeypatch.setattr(effects, "_resource_snapshot", lambda minimum: {"passed": True, "minimum": minimum})
    monkeypatch.setattr(effects, "_check_interrupted", lambda: None)
    monkeypatch.setattr(effects, "_release_segment_memory", lambda: None)
    return manifest, events


def test_two_segment_relay_eav_loop_audits_before_accept_and_resumes(monkeypatch, tmp_path):
    plan = _relay_plan()
    manifest, events = _install_fake_combined_runtime(monkeypatch, tmp_path)
    result = effects.run_long_video_in_node_loop_effects(
        object(), object(), object(), object(), **_kwargs(plan)
    )
    assert result[2:4] == (2, "complete")
    assert [item["index"] for item in manifest["segments"]] == [0, 1]
    assert ("relay", 0, 0) in events
    assert ("relay", 1, 102) in events
    assert events.count(("compose_owner", 0)) == 1
    assert events.count(("compose_owner", 1)) == 1
    report = json.loads(result[4])
    assert len(report["segment_audits"]) == 2
    assert all(
        audit["enhance_a_video_audit"]["status"] == "verified"
        for audit in report["segment_audits"]
    )

    events.clear()
    repeated = effects.run_long_video_in_node_loop_effects(
        object(), object(), object(), object(), **_kwargs(plan)
    )
    assert repeated[0] == result[0]
    assert repeated[3] == "complete"
    assert events == []


def test_short_global_relay_plan_is_rejected_before_first_segment(monkeypatch, tmp_path):
    plan = _relay_plan(124)
    manifest, events = _install_fake_combined_runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="shorter than the requested long video"):
        effects.run_long_video_in_node_loop_effects(
            object(), object(), object(), object(), **_kwargs(plan)
        )
    assert manifest["segments"] == []
    assert events == []


def test_candidate_saved_before_audit_is_preserved_and_retried(monkeypatch, tmp_path):
    plan = _relay_plan()
    manifest, _events = _install_fake_combined_runtime(monkeypatch, tmp_path)
    real_write_audit = effects._write_effects_audit
    fail_once = {"active": True}

    def write_audit_with_one_failure(*args, **kwargs):
        if fail_once["active"]:
            fail_once["active"] = False
            raise OSError("simulated audit publication failure")
        return real_write_audit(*args, **kwargs)

    monkeypatch.setattr(effects, "_write_effects_audit", write_audit_with_one_failure)
    with pytest.raises(OSError, match="simulated audit publication failure"):
        effects.run_long_video_in_node_loop_effects(
            object(), object(), object(), object(), **_kwargs(plan)
        )
    assert manifest["segments"] == []

    result = effects.run_long_video_in_node_loop_effects(
        object(), object(), object(), object(), **_kwargs(plan)
    )
    assert result[3] == "complete"
    assert [item["index"] for item in manifest["segments"]] == [0, 1]
    first_segment_dirs = sorted(
        path.name
        for path in (tmp_path / "candidates" / "segment_00000").iterdir()
        if path.is_dir()
    )
    assert len(first_segment_dirs) == 2
    assert first_segment_dirs[1].endswith("_retry0001")
    assert manifest["segments"][0]["candidate_id"] == first_segment_dirs[1]
