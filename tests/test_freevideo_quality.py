"""Quality clocks/typed completed stages/schema: CPU only, no model downloads."""
import asyncio
import json
from pathlib import Path

import pytest
import torch
import h3_audio_t8_pkg
from h3_audio_t8_pkg.freevideo_quality import profiles as p, runtime as q, assets
from h3_audio_t8_pkg.freevideo_quality.nodes import NODES
from h3_audio_t8_pkg.freevideo_exp import runtime as old
from h3_audio_t8_pkg.freevideo_exp.nodes import NODES as OLD_NODES


def stage(quality="light", role=None):
    selected = p.plan(quality, role)
    video, audio = torch.zeros(1, 24, 12, 16, 16), torch.ones(1, 32, 2, 65)
    roles = ["LOW", "HIGH"] if selected["role"] == "HIGH" else [selected["role"]]
    rows = [dict(identity=p.table_identity("3" * 64, quality, stage_role, "t2va"),
                 index=index, bytes=100, sha256="4" * 64) for stage_role in roles for index in range(50)]
    receipt = dict(schema="t8-freevideo-quality-stage-v2", plan=selected, profile=quality, role=selected["role"],
        freevideo_revision=p.FREEVIDEO_REVISION, vdn_revision=old.VDN_REVISION, model_revision=old.MODEL_REVISION,
        geometry=dict(width=256, height=256, frames=39), task="t2va", clock=p.clock(quality, role, "t2va"),
        completed_nfe=selected["nfe"], sample=dict(step_seconds=[.1] * selected["nfe"]),
        request_sha256="0" * 64, output_sha256="1" * 64, source_inventory_sha256="2" * 64,
        weight_identity="3" * 64, tables=rows, tables_sha256=assets.evidence_sha(rows))
    if selected["role"] == "HIGH":
        receipt.update(low_receipt_sha256="5" * 64, low_audio=old.tensor_record(audio))
    return q.make_stage(video, audio, receipt)


@pytest.mark.parametrize("quality", list(p.PROFILES))
@pytest.mark.parametrize("task", p.TASKS)
def test_full_clocks_all_profiles_actual_modality_rows(quality, task):
    selected, clock = p.plan(quality), p.clock(quality, task=task)
    n = selected["nfe"]
    grid = p.raw_grid(n)
    assert clock["video_sigmas"] == (12 * grid / (1 + 11 * grid)).tolist()
    assert clock["audio_sigmas"] == (3 * grid / (1 + 2 * grid)).tolist()
    assert len(clock["modulation_timesteps"]) == n
    for index, row in enumerate(clock["modulation_timesteps"]):
        expected = [clock["video_timesteps"][index], clock["audio_timesteps"][index]]
        if task in ("i2va", "l2va", "fl2va", "ref2va", "ref2va_av"):
            expected.append(float(torch.tensor(.999)))
        if task in ("ref2va_audio", "ref2va_av"):
            expected.append(0.)
        assert row == sorted(set(expected))
    assert selected["plan_complete"] is (quality != "light")


@pytest.mark.parametrize("task", p.TASKS)
def test_community3_not_legacy_tail3(task):
    clock = p.clock("light", "HIGH", task)
    assert clock["video_sigmas"] == torch.tensor([.9035, .6316, .3158, 0.]).tolist()
    assert len(clock["video_timesteps"]) == 3 and clock["leading_initialization_not_NFE"]
    assert clock["video_sigmas"] != p.clock("light", "LOW", task)["video_sigmas"][-4:]
    assert p.plan("light", "HIGH")["plan_complete"]


@pytest.mark.parametrize("quality,role", [(k, None) for k in p.PROFILES] + [("light", "HIGH")])
def test_completed_stage_cold_exact_create_only_and_legacy_separate(tmp_path, quality, role):
    original = stage(quality, role)
    path, sha = q.save_stage(original, tmp_path)
    second, _ = q.save_stage(original, tmp_path)
    loaded = q.load_stage(path, sha)
    assert loaded.receipt_sha256 == original.receipt_sha256 and path != second
    assert torch.equal(loaded.audio, original.audio)
    with pytest.raises(ValueError):
        old.validate_stage(original)
    with pytest.raises(ValueError):
        q.validate_stage(loaded, "LOW" if loaded.receipt_json.find('"role":"LOW"') < 0 else "HIGH")
    with pytest.raises(ValueError):
        q.load_stage(path, "f" * 64)


@pytest.mark.parametrize("change", [dict(role="MID"), dict(profile="medium"), dict(completed_nfe=7),
    dict(tables=[]), dict(tables_sha256="f" * 64), dict(freevideo_revision="old"), dict(clock={}),
    dict(sample={"step_seconds": [float("nan")] * 8})])
def test_forged_incomplete_contract_rejected(change):
    s = stage()
    receipt = dict(json.loads(s.receipt_json), **change)
    with pytest.raises(ValueError):
        q.make_stage(s.video, s.audio, receipt)


def test_high_audio_lineage_and_tensor_mutation_rejected():
    s = stage("light", "HIGH")
    receipt = json.loads(s.receipt_json)
    with pytest.raises(ValueError, match="audio"):
        q.make_stage(s.video, s.audio + 1, receipt)
    s.video.add_(1)
    with pytest.raises(ValueError, match="content"):
        q.validate_stage(s)


def test_sockets_and_original_schema_defaults_still_separate():
    old_schema = OLD_NODES[3].GET_SCHEMA().get_v1_info(OLD_NODES[3])
    high_schema = NODES[2].GET_SCHEMA().get_v1_info(NODES[2])
    assert old_schema.input["required"]["tail_steps"][1]["default"] == 2
    assert "tail_steps" not in high_schema.input["required"]
    assert old_schema.input["required"]["completed_low"][0] != high_schema.input["required"]["completed_low"][0]
    for node in NODES[5:]:
        schema = node.GET_SCHEMA().get_v1_info(node)
        assert schema.input["required"]["freevideo_model"][0] == "H3_T8_FREEVIDEO_QUALITY_MODEL"
        assert schema.output[0] == "H3_T8_FREEVIDEO_QUALITY_MODEL"


def test_new_effects_do_not_inherit_finalized_legacy_return_type_cache():
    # Core's execution validator uses RETURN_TYPES, not just GET_NODE_INFO_V1.
    # Warm legacy first, as normal registration does, before touching new nodes.
    for legacy, current in zip([OLD_NODES[1], *OLD_NODES[6:8]], NODES[5:8]):
        legacy.GET_SCHEMA()
        assert not issubclass(current, legacy)
        current.GET_SCHEMA()
        assert current.RETURN_TYPES[0] == "H3_T8_FREEVIDEO_QUALITY_MODEL"
        assert current.CATEGORY == "T8/MiniMax H3/FreeVideo/Quality v2"
    for current in NODES[5:8]:
        assert current.GET_NODE_INFO_V1()["output"][0] == current.RETURN_TYPES[0]
    assert OLD_NODES[6].GET_SCHEMA().get_v1_info(OLD_NODES[6]).input["required"]["start"][1]["default"] == .15


def test_append_order_all_nodes_and_features():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [cls.GET_SCHEMA().node_id for cls in classes]
    features = json.loads((Path(__file__).parents[1] / "features.json").read_text(encoding="utf8"))
    # The local paused H05 family is not shipped by this independent release.
    # Both exact deployment layouts remain checked, with no skipped assertions.
    qwen = ["MiniMaxH3ReferenceQwenViewEXPT8"] if "MiniMaxH3ReferenceQwenViewEXPT8" in ids else []
    assert ids == features["nodes"] and len(ids) == len(set(ids)) == 671 + len(qwen)
    assert ids[663:] == qwen + [node.__name__ for node in NODES]
    assert ids[-8:] == [node.__name__ for node in NODES]
    assert ids[620:632] == [node.__name__ for node in OLD_NODES]


def test_quality_descriptor_cannot_enter_legacy_or_wrong_config(tmp_path):
    with pytest.raises(ValueError):
        q.facade(old.FreeVideoModel("unused", "unused"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema": "t8-freevideo-runtime-v1"}))
    with pytest.raises(ValueError):
        q.load_model(path)
    with pytest.raises(ValueError):
        old.config_for(q.QualityModel(str(path), old.digest(path)))
    original = q.QualityModel("unused", "unused")
    changed = q.from_facade(old.with_lora(q.facade(original), "missing_disabled", 0))
    assert original.loras == () and len(changed.loras) == 1


def test_unknown_bank_pin_never_accepted():
    with pytest.raises(ValueError, match="Unpinned"):
        assets.banks(dict(optional_adaln=dict(prefix="unknown", revision="floating")))


def test_max20_published_clock_is_explicit_not_relaxed_comparison():
    actual = p.clock("max")
    assert p.plan("max")["clock_contract"] == "published_20_FP32_raw_index9_v1"
    assert p.raw_grid(20)[9].item() == float(torch.tensor(.55))
    assert actual["audio_timesteps"][9] == .21428561210632324
    changed = dict(actual, audio_timesteps=list(actual["audio_timesteps"]))
    changed["audio_timesteps"][9] = .2142857313156128
    with pytest.raises(ValueError, match="clock"):
        p.validate_clock(changed, "max", "SINGLE", "t2va")


@pytest.mark.parametrize("name", ["../evil", "/evil", "G:/evil", "x\\evil", "x/../evil", "x//evil", "./evil", ""])
def test_table_path_keeps_real_containment(tmp_path, name):
    with pytest.raises(ValueError):
        assets.table_path(tmp_path, name)


def test_table_path_new_and_existing_leaf_share_resolved_namespace(tmp_path):
    target = assets.table_path(tmp_path, "tables/00.safetensors")
    target.parent.mkdir()
    target.write_bytes(b"test")
    assert assets.table_path(tmp_path, "tables/00.safetensors") == target


def test_owned_worker_isolated_bootstrap_reaches_request_guard(tmp_path):
    import os
    import subprocess
    import sys
    worker = Path(__file__).parents[1] / "h3_t8/freevideo_quality/worker.py"
    (tmp_path / "launch-gate").write_text("CPU fixture only, no model or kernel", encoding="utf8")
    (tmp_path / "request.json").write_text(json.dumps(dict(schema="invalid_CPU_fixture",
        plan=p.plan("light"))), encoding="utf8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(worker), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode != 0
    assert "ValueError: Quality request profile/role mismatch" in result.stderr, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "output.safetensors").exists()
