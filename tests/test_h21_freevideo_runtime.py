"""v4 source/plan/owned transport contracts. Optional external runtime tests are explicit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg.freevideo_h21 import effects, execution, plans, runtime, source, stages


def temporary_source(tmp_path, monkeypatch, content="value = 1\n"):
    root = tmp_path / "source"
    (root / "freevideo_engine").mkdir(parents=True)
    data = content.encode()
    (root / "freevideo_engine/sample.py").write_bytes(data)
    row = dict(file="freevideo_engine/sample.py", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
               git_blob=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
    (root / "t8-source-proof.json").write_text(json.dumps(dict(commit=source.REVISION, files=[row])), encoding="utf8")
    content_identity = f"{row['file']}\t{row['bytes']}\t{row['sha256']}\t{row['git_blob']}\n"
    monkeypatch.setattr(source, "CONTENT_PIN", hashlib.sha256(content_identity.encode()).hexdigest())
    # A synthetic self-consistent source is only a loader/integrity fixture,
    # never presented as the installed official 439-blob source qualification.
    return root


def test_source_module_does_not_run_initializer_or_mutate_global_old_family(tmp_path, monkeypatch):
    root = temporary_source(tmp_path, monkeypatch)
    (root / "freevideo_engine/__init__.py").write_text("raise AssertionError('initializer executed')", encoding="utf8")
    monkeypatch.setenv("FREEVIDEO_STEP_CACHE", "user-host-value")
    before = set(sys.modules)
    actual = source.Source(root)
    assert actual.module("sample").value == 1
    assert "freevideo_engine" not in set(sys.modules) - before
    assert os.environ["FREEVIDEO_STEP_CACHE"] == "user-host-value"
    actual.verify()


@pytest.mark.parametrize("field", ["bytes", "sha256", "git_blob"])
def test_actual_source_bytes_size_and_git_blob_all_verified(tmp_path, monkeypatch, field):
    root = temporary_source(tmp_path, monkeypatch)
    proof_path = root / "t8-source-proof.json"
    proof = json.loads(proof_path.read_text())
    proof["files"][0][field] = 0 if field == "bytes" else "f" * 64
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="Git blob"):
        source.Source(root)


@pytest.mark.parametrize("name", ["../escape", "C:/foo", "dir\\file", "/absolute", "a//b"])
def test_source_resolved_containment_not_a_filename_class_whitelist(tmp_path, monkeypatch, name):
    root = temporary_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="relative path"):
        source.contained(root, name)


def test_source_changed_after_preparation_does_not_publish_identity(tmp_path, monkeypatch):
    root = temporary_source(tmp_path, monkeypatch)
    selected = source.Source(root)
    (root / "freevideo_engine/sample.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="changed"):
        selected.verify()


def test_self_consistent_forged_revision_label_does_not_become_official_source(tmp_path, monkeypatch):
    actual_pin = source.CONTENT_PIN
    root = temporary_source(tmp_path, monkeypatch)
    monkeypatch.setattr(source, "CONTENT_PIN", actual_pin)
    with pytest.raises(ValueError, match="fixed H21 Git content set"):
        source.Source(root)


def test_ordered_lora_selections_and_disabled_missing_path_preserved(tmp_path):
    from safetensors.torch import save_file
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    save_file({"real_lora.lora_A.weight": torch.ones(1, 2), "real_lora.lora_B.weight": torch.ones(2, 1)}, str(first))
    save_file({"real_lora.lora_A.weight": torch.ones(1, 2) * 2, "real_lora.lora_B.weight": torch.ones(2, 1)}, str(second))
    model = runtime.Model("explicit-unused-config-path", "0" * 64)
    selected = runtime.with_lora(runtime.with_lora(model, first, -.75), second, 1.25)
    selected = runtime.with_lora(selected, tmp_path / "disabled-missing.safetensors", 0.)
    rows = [json.loads(row) for row in selected.loras]
    assert [row["strength"] for row in rows] == [-.75, 1.25, 0.]
    assert rows[0]["sha256"] == source.digest(first) and rows[1]["sha256"] == source.digest(second)
    assert rows[2]["disabled"] and not (tmp_path / "disabled-missing.safetensors").exists()
    assert model.loras == ()


def test_eav_h21_value_bridge_preserves_new_model_type_and_ordered_slots():
    from h3_audio_t8_pkg.freevideo_exp.runtime import FreeVideoModel
    model = runtime.Model("unused", "0" * 64, ("one", "two"))
    selected = effects.with_eav(model, tau=2., mode="report_only")
    assert type(selected) is runtime.Model and not isinstance(selected, FreeVideoModel)
    assert selected.loras == model.loras and selected.config_sha256 == model.config_sha256
    assert json.loads(selected.effects_json)["eav"]["tau"] == 2.
    assert json.loads(model.effects_json) == {}
    assert effects.transport(selected, {}, dict(width=256, height=256, frames=39))["relay"] is None


@pytest.mark.parametrize("mutation", ["none", "unpaired", "embeds", "tags", "geometry", "self_hash"])
def test_h21_effects_transport_binds_real_native_tensors_not_old_runtime_identity(mutation):
    from h3_audio_t8_pkg.freevideo_exp import effects as common, runtime as common_io
    from dataclasses import replace
    embeds, tags = torch.zeros(1, 3, 5120), torch.ones(3, dtype=torch.int64)
    canvas = dict(width=256, height=256, frames=39)
    value = dict(schema="t8-freevideo-relay-v1", mode="apply_exp", query_route="video_only_paper", query_chunk_rows=32,
        geometry=canvas, events=[dict(text_key_start=1, text_key_end=3, midpoint=1., window=2., sigma=1.)],
        embeddings=common_io.tensor_record(embeds), tags_sha256=hashlib.sha256(tags.view(torch.uint8).numpy().tobytes()).hexdigest())
    condition = [[embeds, {"minimax_token_tags": tags}]]
    value["conditions_sha256"] = common.condition_identity(condition, canvas)
    value["sha256"] = hashlib.sha256(common_io.canonical(value).encode()).hexdigest()
    condition[0][1][common.RELAY_KEY] = value
    model = runtime.Model("unused", "0" * 64, effects_json=source.canonical(dict(relay=value)))
    if mutation == "none":
        assert effects.transport(model, condition, canvas)["relay"] == value
        assert type(model) is runtime.Model
        with pytest.raises(ValueError, match="paired native H21 Relay"):
            effects.transport(model, {}, canvas)
        return
    if mutation == "unpaired":
        condition[0][1].pop(common.RELAY_KEY)
    elif mutation == "embeds":
        embeds.add_(1)
    elif mutation == "tags":
        tags[0] = 0
    elif mutation == "geometry":
        canvas = dict(canvas, width=288)
    else:
        changed = dict(value, query_chunk_rows=64)
        model = replace(model, effects_json=source.canonical(dict(relay=changed)))
    with pytest.raises(ValueError):
        effects.transport(model, condition, canvas)


def test_eight_h21_node_schemas_are_distinct_and_do_not_change_original_effect_schema():
    from h3_audio_t8_pkg.freevideo_h21.nodes import NODES
    from h3_audio_t8_pkg.freevideo_exp.nodes import MiniMaxH3FreeVideoPromptRelayEXPT8
    schemas = [node.define_schema() for node in NODES]
    assert len(schemas) == 8 and len({row.node_id for row in schemas}) == 8
    relay = schemas[-1]
    assert relay.inputs[0].io_type == "H3_T8_FREEVIDEO_H21_MODEL"
    assert relay.outputs[0].io_type == "H3_T8_FREEVIDEO_H21_MODEL"
    assert MiniMaxH3FreeVideoPromptRelayEXPT8.define_schema().inputs[0].io_type == "H3_T8_FREEVIDEO_MODEL"


def test_actual_current_registered_features_and_original_quality_block_not_a_stale_global_tail():
    import asyncio
    import h3_audio_t8_pkg
    from h3_audio_t8_pkg.freevideo_quality.nodes import NODES
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.GET_SCHEMA().node_id for node in classes]
    features = json.loads((Path(__file__).parents[1] / "features.json").read_text(encoding="utf8"))
    assert ids == features["nodes"] and len(ids) == len(set(ids))
    assert ids[663:671] == [node.__name__ for node in NODES]
    if "MiniMaxH3ReferenceQwenViewEXPT8" in ids:
        assert ids[671] == "MiniMaxH3ReferenceQwenViewEXPT8"
    from h3_audio_t8_pkg.freevideo_exp.nodes import NODES as original
    assert ids[620:632] == [node.__name__ for node in original]


def test_actual_tensor_json_transport_and_opaque_owners_are_not_pickled():
    original = dict(prompt_embeds=torch.randn(3, 8), tags=torch.tensor([1, 0, 1]),
                    references=[dict(latent=torch.randn(1, 24, 1, 2, 2), metadata="用户对白\n♪")])
    text, tensors = runtime.pack_condition(original)
    restored = runtime.unpack_condition(json.loads(source.canonical(text)), tensors)
    assert torch.equal(restored["prompt_embeds"], original["prompt_embeds"])
    assert torch.equal(restored["tags"], original["tags"])
    assert restored["references"][0]["metadata"] == original["references"][0]["metadata"]
    with pytest.raises(TypeError, match="executable"):
        runtime.pack_condition(dict(owner=lambda value: value))
    with pytest.raises(ValueError, match="Reserved"):
        runtime.pack_condition({"$h21_tensor": "inject"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_actual_nonfinite_condition_still_errors(bad):
    with pytest.raises(ValueError, match="non-finite"):
        runtime.pack_condition(dict(scalar=bad))
    with pytest.raises(ValueError, match="finite"):
        runtime.pack_condition(dict(tensor=torch.tensor([bad])))


def test_host_step_cache_override_removed_only_within_owned_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FREEVIDEO_STEP_CACHE", "fbcache:9")
    config = dict(home=str(tmp_path), vdn_root=str(tmp_path / "vdn"))
    environment = runtime.worker_environment(config)
    assert "FREEVIDEO_STEP_CACHE" not in environment and os.environ["FREEVIDEO_STEP_CACHE"] == "fbcache:9"
    with pytest.raises(RuntimeError):
        with execution.controlled_cache_environment():
            assert "FREEVIDEO_STEP_CACHE" not in os.environ
            raise RuntimeError("actual interruption")
    assert os.environ["FREEVIDEO_STEP_CACHE"] == "fbcache:9"


def test_actual_block_observations_not_callback_minus_cache_hits():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
            self.cached = False
        def forward(self, value):
            if self.cached:
                return value
            for block in self.transformer_blocks:
                value = block(value)
            return value
    owner = Network()
    with execution.Observations(owner) as observed:
        owner(torch.ones(1, 2))
        owner.cached = True
        owner(torch.ones(1, 2))
        actual = observed.snapshot()
    assert actual["actual_transformer_calls"] == 2
    assert actual["actual_full_DiT_forwards"] == 1 and actual["actual_block_calls"] == [1, 1]
    assert not owner._forward_pre_hooks and all(not block._forward_pre_hooks for block in owner.transformer_blocks)


def receipt_fixture():
    # Serialization-only completion fixture; not asserted as model execution.
    canvas = dict(width=256, height=256, frames=39)
    selected = dict(source_revision=source.REVISION, role="LOW", expected_callbacks=8,
                    actual_canvas=canvas, sampling=dict(enabled=True, first=canvas, second=canvas,
                                                       base_steps=8, refine_steps=3, total_steps=11))
    selected["identity"] = hashlib.sha256(source.canonical(selected).encode()).hexdigest()
    actual = dict(completed_callbacks=8, plan_identity=selected["identity"],
        plan_sha256=hashlib.sha256(source.canonical(selected).encode()).hexdigest(),
        passes=[dict(role="LOW", completed_callbacks=8, sample=dict(step_seconds=[.1] * 8))])
    return dict(plan=selected, actual=actual, config_sha256="1" * 64, inventory_sha256="2" * 64,
                request_sha256="3" * 64, output_sha256="4" * 64)


@pytest.mark.parametrize("mutation", ["wrong_order", "omit", "duplicate"])
def test_original_registry_blocks_reject_bad_same_features_not_only_tail_length(mutation):
    from tests.test_freevideo_quality import assert_original_quality_registry_blocks
    features = json.loads((Path(__file__).parents[1] / "features.json").read_text(encoding="utf8"))
    ids = list(features["nodes"])
    assert_original_quality_registry_blocks(ids, list(ids))
    if mutation == "wrong_order":
        ids[663], ids[664] = ids[664], ids[663]
    elif mutation == "omit":
        ids.pop(663)
    else:
        ids.insert(664, ids[663])
    # Corrupt the declared list too: equality alone must not hide order,
    # omission or duplication within the original accepted blocks.
    with pytest.raises(AssertionError):
        assert_original_quality_registry_blocks(ids, list(ids))


@pytest.mark.parametrize("mutation", ["wrong_order", "omit", "duplicate"])
def test_original_exp_registry_blocks_reject_wrong_order_omission_and_duplicate(mutation):
    from tests.test_freevideo_exp import assert_original_exp_registry_blocks
    features = json.loads((Path(__file__).parents[1] / "features.json").read_text(encoding="utf8"))
    ids = list(features["nodes"])
    assert_original_exp_registry_blocks(ids, list(ids))
    if mutation == "wrong_order":
        ids[620], ids[621] = ids[621], ids[620]
    elif mutation == "omit":
        ids.pop(620)
    else:
        ids.insert(621, ids[620])
    with pytest.raises(AssertionError):
        assert_original_exp_registry_blocks(ids, list(ids))


@pytest.mark.parametrize("attention", ["dense", "cudnn/cudnn", "foreign_policy_preserved"])
def test_actual_hybrid_effect_installation_alias_count_and_close_API(tmp_path, official_source, attention):
    """Actual tiny HybridAttention + author installer, no cuDNN kernel execution."""
    python, vdn = os.environ.get("T8_H21_FREEVIDEO_PYTHON"), os.environ.get("T8_H21_FREEVIDEO_VDN")
    if not python or not vdn:
        pytest.skip("Select installed VDN Python; no implicit install or CUDA emulation")
    script = r'''
import json, pathlib, sys
from types import SimpleNamespace
repo, selected_source, vdn, mode, output = sys.argv[1:]
sys.path[:0] = [repo, selected_source, str(pathlib.Path(vdn) / 'diffusers/src'), vdn]
import torch
from h3_t8.freevideo_h21.source import Source
from h3_t8.freevideo_h21.plans import plan
from h3_t8.freevideo_h21.tiny_worker import TinyVDNCPU
from h3_t8.freevideo_h21.effects import install
from src.models.hybrid_transform import apply_hybrid_attention_transform, iter_hybrids
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
source = Source(selected_source)
owner = TinyVDNCPU(source, plan(source, 'light', 256, 256, 39))
apply_hybrid_attention_transform(owner.transformer, dict(softmax_attention=dict(radius=1, chunk=0),
    linear_attention=dict(delta_rule='vdn', linear_head_dim=24, bridge='alpha', a_fp32=True,
        short_conv=dict(targets=[]), enable_text_state=True), anchor_frames='none', enable_softmax_gate=True))
hybrids = list(iter_hybrids(owner.transformer))
assert len(hybrids) == 2
originals = [(attn._hybrid_forward, attn.linear_attention._readout_inference) for attn in hybrids]
calls = []
class ActualPolicyOwner:
    def __call__(self, value):
        calls.append(value)
        return value * 2
policy = ActualPolicyOwner()
if mode == 'foreign_policy_preserved':
    policy.global_backend = policy.window_backend = 'foreign-user-choice'
else:
    policy.global_backend, policy.window_backend = source.module('attention').split_backend(mode)
engine = SimpleNamespace(transformer=owner.transformer, attention=policy, config=dict(attention=mode,
    head_parallelism=1, inference_kernels=True, head_chunk=1, attention_cpu_outputs=False,
    projection_chunk=32, grouped_attention_outputs=False))
original_config = dict(engine.config)
effect = dict(eav=dict(mode='report_only', tau=4., start=.15, end=.9, workspace_mib=32, g_hard_limit=1.5), relay=None)
state = install(engine, effect, dict(width=256, height=256, frames=39))
report = state.snapshot()
assert engine.config == original_config and engine.attention is policy
assert report['actual_hybrid_blocks'] == 2 and not report['CUDA_kernel_qualification']
assert report['effects_consumed'] is False
if mode == 'foreign_policy_preserved':
    assert report['partial'] and report['warning'] and report['installed_blocks'] == 0
    x = torch.arange(4).float()
    assert torch.equal(engine.attention(x), x * 2) and len(calls) == 1
else:
    assert report['installed_blocks'] == 2 and not report['partial']
    assert report['actual_attention_backends'] == ['cudnn', 'cudnn']
    assert len(owner.transformer._forward_pre_hooks) == 1
state.close()
assert not owner.transformer._forward_pre_hooks
for attn, (attention, readout) in zip(hybrids, originals):
    assert attn._hybrid_forward == attention and attn.linear_attention._readout_inference == readout
    assert not attn.linear_attention._forward_hooks
assert not torch.cuda.is_initialized()
pathlib.Path(output).write_text(json.dumps(dict(actual=report, CUDA_initialized=False,
    qualification='actual_two_layer_HybridAttention_install_close_API_only_no_effect_GPU_kernel')), encoding='utf8')
'''
    output = tmp_path / "actual-effect-install-API.json"
    child = subprocess.run([python, "-I", "-B", "-X", "utf8", "-c", script, str(Path(__file__).parents[1]),
        str(official_source.root), vdn, attention, str(output)], capture_output=True, text=True, encoding="utf8",
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", FREEVIDEO_VDN_ROOT=vdn, OMP_NUM_THREADS="1"), timeout=180)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(output.read_text(encoding="utf8"))
    assert actual["CUDA_initialized"] is False and actual["actual"]["actual_hybrid_blocks"] == 2


def test_distinct_typed_stage_actual_bytes_saved_and_cold_reloaded(tmp_path):
    original = stages.make(torch.randn(1, 24, 12, 16, 16), torch.randn(1, 32, 2, 65), receipt_fixture())
    path, sha = stages.save(original, tmp_path)
    other_path, _ = stages.save(original, tmp_path)
    reloaded = stages.load(path, sha)
    assert path != other_path and reloaded.receipt_sha256 == original.receipt_sha256
    assert torch.equal(reloaded.video, original.video) and torch.equal(reloaded.audio, original.audio)
    reloaded.audio.add_(1)
    with pytest.raises(ValueError, match="content changed"):
        stages.validate(reloaded)
    with pytest.raises(ValueError, match="manifest SHA"):
        stages.load(path, "0" * 64)


@pytest.mark.parametrize("change", ["clock", "callbacks", "pass_order", "source", "shape"])
def test_partial_changed_clock_and_wrong_family_do_not_mint_completed_identity(change):
    receipt = receipt_fixture()
    video, audio = torch.zeros(1, 24, 12, 16, 16), torch.zeros(1, 32, 2, 65)
    if change == "clock":
        receipt["plan"]["actual_canvas"]["frames"] = 56
    elif change == "callbacks":
        receipt["actual"]["completed_callbacks"] = 7
    elif change == "pass_order":
        receipt["actual"]["passes"][0]["role"] = "HIGH"
    elif change == "source":
        receipt["plan"]["source_revision"] = "old"
    elif change == "shape":
        video = video[..., :-1]
    with pytest.raises(ValueError):
        stages.make(video, audio, receipt)


@pytest.fixture
def official_source():
    location = os.environ.get("T8_H21_FREEVIDEO_SOURCE")
    if not location:
        pytest.skip("External fixed FreeVideo source not selected; no implicit download")
    return source.Source(location)


@pytest.mark.parametrize("tier,nfe,two_pass,cache", [("light", 11, True, None), ("medium", 11, True, None),
                                                    ("high", 20, False, "easycache-av"), ("max", 20, False, None)])
def test_actual_installed_fixed_four_tiers_and_old_defaults_unchanged(official_source, tier, nfe, two_pass, cache):
    from h3_audio_t8_pkg.freevideo_quality import profiles as old
    selected = plans.plan(official_source, tier, 1344, 768, 124)
    assert selected["expected_callbacks"] == nfe and selected["sampling"]["enabled"] == two_pass
    assert selected["step_cache"] == cache
    assert old.PROFILES == dict(light=8, medium=12, high=16, max=20)
    if tier == "medium":
        assert selected["requested_upscale"] == 1.5 and selected["actual_learned_upscale"] == 1.5
        assert selected["sampling"]["first"] == dict(width=896, height=512, frames=124)
    official_source.verify()


@pytest.mark.parametrize("width,height", [(256, 256), (512, 288), (1024, 576)])
def test_actual_medium_short_side_and_small_canvas_fallback_reported(official_source, width, height):
    selected = plans.plan(official_source, "medium", width, height, 39)
    if min(width, height) < 512:
        assert selected["actual_learned_upscale"] != 1.5
    if width == height == 256:
        assert selected["actual_learned_upscale"] is None
        assert selected["sampling"]["first"] == selected["sampling"]["second"]
    assert selected["expected_callbacks"] == 11


@pytest.mark.parametrize("tier,audio_reference,cancel_retry", [("light", False, False), ("medium", False, False),
    ("high", False, False), ("max", False, False), ("light", True, False), ("high", False, True),
    ("high", True, False)])
def test_real_tiny_vdn_owned_CPU_worker_not_fake_Engine(official_source, tmp_path, tier, audio_reference, cancel_retry):
    python, vdn = os.environ.get("T8_H21_FREEVIDEO_PYTHON"), os.environ.get("T8_H21_FREEVIDEO_VDN")
    if not python or not vdn:
        pytest.skip("Select the existing independent VDN Python/dependency tree for the real CPU worker")
    output = tmp_path / (tier + "-actual.json")
    command = [python, "-I", "-B", "-X", "utf8", str(Path(runtime.__file__).with_name("tiny_worker.py")),
        "--source-root", str(official_source.root), "--vdn-root", vdn, "--output", str(output),
        "--tier", tier, "--zero-heads", "--allow-real-CPU-contract"]
    if audio_reference:
        command.append("--audio-reference")
    if cancel_retry:
        command.append("--retry-after-cancel")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    child = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf8", timeout=180)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text(encoding="utf8"))
    assert value["CUDA_initialized"] is False and value["Engine_qualification"] is False
    assert value["actual"]["completed_callbacks"] == (11 if tier in ("light", "medium") else 20)
    assert value["video"]["shape"] == [1, 24, 12, 16, 16]
    passes = value["actual"]["passes"]
    if tier in ("light", "medium"):
        assert [row["completed_callbacks"] for row in passes] == [8, 3]
        assert value["actual"]["high_audio_preserved"] is True
    if tier == "high":
        measured = passes[0]["measurements"]
        assert measured["actual_transformer_calls"] == 20
        assert 0 < measured["actual_full_DiT_forwards"] < 20
        assert passes[0]["sample"]["step_cache"]["cached_steps"]
    if tier == "max":
        assert passes[0]["measurements"]["actual_full_DiT_forwards"] == 20
    if audio_reference:
        for row in passes:
            for trace in row["sample"]["actual_clock_trace"]:
                ref_rows = trace["audio_indices"][:4]
                assert all(trace["timestep"][trace["timestep_indices"][index]] == 1. for index in ref_rows)
    if cancel_retry:
        retry = value["cancellation_retry"]
        assert retry["aborted_request"]["actual_transformer_calls"] == 2
        assert retry["retry_callbacks"] == retry["fresh_reference_callbacks"] == 20
        assert retry["model_bytes_preserved"] and retry["AV_bit_exact_with_actual_fresh_owner"]
        assert retry["cancelled_request_not_completed"]
    assert not torch.cuda.is_initialized()


def test_actual_different_LOW_canvas_pass_geometry_with_real_eight_step_sampler(official_source, tmp_path):
    python, vdn = os.environ.get("T8_H21_FREEVIDEO_PYTHON"), os.environ.get("T8_H21_FREEVIDEO_VDN")
    if not python or not vdn:
        pytest.skip("Select existing independent VDN dependency tree; no download/install")
    output = tmp_path / "actual-low-different-canvas.json"
    command = [python, "-I", "-B", "-X", "utf8", str(Path(runtime.__file__).with_name("tiny_worker.py")),
        "--source-root", str(official_source.root), "--vdn-root", vdn, "--output", str(output),
        "--tier", "light", "--role", "LOW", "--width", "1024", "--height", "576",
        "--zero-heads", "--allow-real-CPU-contract"]
    child = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True, text=True, encoding="utf8", timeout=180)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text(encoding="utf8"))
    assert value["plan"]["actual_canvas"] == dict(width=1024, height=576, frames=39)
    assert value["plan"]["sampling"]["first"] == dict(width=512, height=288, frames=39)
    assert value["video"]["shape"] == [1, 24, 12, 18, 32]
    assert value["actual"]["completed_callbacks"] == 8
    trace = value["actual_pass_canvas_trace"]
    assert len(trace) == 1 and trace[0]["role"] == "LOW"
    assert trace[0]["actual_canvas"] == dict(width=512, height=288, frames=39)
    assert trace[0]["actual_video_rows"] == [1728] * 8 and trace[0]["original_canvas_restored"]
    assert not value["CUDA_initialized"] and not value["effects_kernels_qualified"]
    assert not value["Engine_qualification"] and not value["upscaler_qualification"]


def test_effect_pass_canvas_scope_preserves_original_object_after_actual_error():
    from types import SimpleNamespace
    original = dict(width=1024, height=576, frames=39)
    owner = SimpleNamespace(canvas=original)
    actual = dict(width=512, height=288, frames=39)
    with pytest.raises(RuntimeError, match="actual forward error"):
        with effects.pass_canvas(owner, actual, "LOW"):
            assert owner.canvas == actual and owner.canvas is not actual
            raise RuntimeError("actual forward error")
    assert owner.canvas is original
