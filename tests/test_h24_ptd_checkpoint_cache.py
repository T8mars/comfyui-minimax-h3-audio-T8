"""Real Core cache + real tiny PTD loaders/consumers on CPU.

No trained assets, CLIP/VAE encoders, services, or GPU. Fixtures reuse existing
synthetic expert banks; no claim of trained model quality follows.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import comfy.ops
from safetensors.torch import save_file
from comfy_api.latest import io
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import nodes_ptd_advanced as public
from h3_audio_t8_pkg import nodes_ptd_continuous as stage_nodes
from h3_audio_t8_pkg import ptd_advanced as ptd
from h3_audio_t8_pkg import ptd_checkpoint as epfl
from h3_audio_t8_pkg import ptd_continuous as stage_runtime
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent
from test_h21_ptd_advanced import experts
from test_h24_ptd_compact import stacked


@pytest.fixture(autouse=True)
def actual_cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def cache_model():
    base = model()
    # A real refiner makes stale preprocessing observable as missing expert
    # calls, rather than testing only a zero-refiner structural fixture.
    base.model.diffusion_model.token_refiner = native.TokenRefiner(
        1, 24, 3, 128, 32, 1e-6, 1e-6, 1e-6, dtype=torch.float32,
        device="cpu", operations=comfy.ops.disable_weight_init)
    generator = torch.Generator().manual_seed(26102683)
    with torch.no_grad():
        for parameter in base.model.diffusion_model.token_refiner.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .03)
    return base


def replace_checkpoint(path, kind, generation):
    previous = path.stat() if path.exists() else None
    base = cache_model()
    if kind.startswith("compact"):
        values, metadata = stacked(base, hops=int(kind.removeprefix("compact")))
    else:
        values, metadata = experts(base, kind), epfl.metadata_for(kind)
        # The older generic fixture uses rank 2, which intentionally emits an
        # unpublished-rank warning. Zero-extend to the published rank so these
        # complete-consumer assertions do not suppress a legitimate warning.
        for key, value in list(values.items()):
            _, _, factor, _, _, dense, _ = epfl.map_factor(key)
            if not dense:
                shape = (128, value.shape[1]) if factor == "A" else (value.shape[0], 128)
                expanded = value.new_zeros(shape)
                expanded[:value.shape[0], :value.shape[1]] = value
                values[key] = expanded
    # Change actual expert bytes while preserving the complete parser contract.
    key = next(key for key in values if ".lora_" in key)
    values[key] = values[key] + generation * .001
    save_file(values, str(path), metadata=metadata)
    if previous is not None:
        # No sleeps or filesystem timestamp-granularity assumptions.
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PTDCacheTinyInputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[],
            outputs=[io.Model.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(cache_model(), conditioning(), latent())


class PTDCacheResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
            inputs=[io.Model.Input("model"), io.Latent.Input("after_sampling"),
                    stage_nodes.COMPLETE.Input("completion", optional=True)],
            outputs=[io.String.Output()])

    @classmethod
    def execute(cls, model, after_sampling, completion=None):
        report = ptd.get_controller(model).report()
        if completion is None:
            assert report["status"] == "observed_complete", report
        else:
            # The raw controller records only the final range. The real typed
            # completion is the complete HEAD/TAIL authority, not raw status.
            combined = stage_runtime.inspect_stage(completion)
            assert combined["whole_recipe_completed"] and not combined["partial"], combined
            assert combined["all_global_hops"] == list(range(len(report["grid"]) - 1))
            report["continuous_completion"] = combined
        assert all(bool(torch.isfinite(part).all()) for part in after_sampling["samples"].unbind())
        value = json.dumps(report, sort_keys=True, allow_nan=False)
        return io.NodeOutput(value, ui={"text": [value]})


class PTDCacheOffResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True, inputs=[], outputs=[io.String.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput("PTD branch absent")


def graph(path, continuous=False):
    value = {
        "1": {"class_type": "PTDCacheTinyInputs", "inputs": {}},
        "2": {"class_type": "MiniMaxH3PTDSetupEXPT8", "inputs": {"model": ["1", 0], "checkpoint": str(path)}},
        "3": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["1", 1]}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 26102682}},
    }
    if continuous:
        value["5"] = {"class_type": "MiniMaxH3PTDContinuousHEAD_EXPT8", "inputs": {
            "noise": ["4", 0], "guider": ["3", 0], "av_latent": ["1", 2]}}
        value["6"] = {"class_type": "MiniMaxH3PTDContinuousTAIL_EXPT8", "inputs": {"guider": ["3", 0], "head": ["5", 0]}}
        output = ["6", 0]
    else:
        value["5"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["4", 0], "guider": ["3", 0], "sampler": ["2", 1], "sigmas": ["2", 2], "latent_image": ["1", 2]}}
        output = ["5", 0]
    value["9"] = {"class_type": "PTDCacheResult", "inputs": {"model": ["2", 0], "after_sampling": output}}
    if continuous:
        value["9"]["inputs"]["completion"] = ["6", 1]
    return value


def executor(monkeypatch):
    import execution
    import nodes
    for cls in (*public.NODES, *stage_nodes.NODES, PTDCacheTinyInputs, PTDCacheResult, PTDCacheOffResult,
                BasicGuider, RandomNoise, SamplerCustomAdvanced):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **k: None)
    return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False))


def run(worker, value, name, *, success=True):
    worker.execute(deepcopy(value), name, execute_outputs=["9"])
    assert worker.success is success, worker.status_messages
    if success and value["9"]["class_type"] == "PTDCacheResult":
        return json.loads(worker.history_result["outputs"]["9"]["text"][0])


@pytest.mark.parametrize("kind,continuous", [
    ("compact4", False), ("compact8", False),
    ("h3_4step", False), ("h3_6step", False),
    ("h3_4step", True), ("h3_6step", True),
])
def test_real_Core_reload_only_on_checkpoint_identity_change(tmp_path, monkeypatch, kind, continuous):
    path = tmp_path / "selected.safetensors"
    sha1 = replace_checkpoint(path, kind, 1)
    calls = []
    original = public.load_ptd_setup
    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result[3]["file"]["sha256"])
        return result
    monkeypatch.setattr(public, "load_ptd_setup", counted)
    worker, value = executor(monkeypatch), graph(path, continuous)
    report1 = run(worker, value, "initial")
    assert calls == [sha1] and report1["file"]["sha256"] == sha1
    assert any(":token_refiner.blocks." in key for key in report1["last_request"]["expert_calls"])
    if kind.startswith("compact"):
        assert len(report1["grid"]) == int(kind.removeprefix("compact")) + 1
    else:
        assert report1["grid"] == list(ptd.ptd_sigmas(kind).tolist())
    run(worker, value, "unchanged")
    assert calls == [sha1]
    value["4"]["inputs"]["noise_seed"] += 1
    report2 = run(worker, value, "seed-only")
    assert calls == [sha1] and report2["file"]["sha256"] == sha1
    assert report2["last_request"]["request_number"] > report1["last_request"]["request_number"]
    sha2 = replace_checkpoint(path, kind, 2)
    assert sha2 != sha1
    # Keep graph, model, seed and path unchanged: only the file changes.
    report3 = run(worker, value, "same-path-replaced")
    assert calls == [sha1, sha2] and report3["file"]["sha256"] == sha2
    assert report3["expert_content_sha256"] != report1["expert_content_sha256"]


def test_real_Core_missing_then_created_is_not_stuck(tmp_path, monkeypatch):
    path = tmp_path / "arrives-later.safetensors"
    missing = public.MiniMaxH3PTDSetupEXPT8.fingerprint_inputs(checkpoint=str(path))
    worker, value = executor(monkeypatch), graph(path)
    run(worker, value, "missing", success=False)
    sha = replace_checkpoint(path, "compact4", 1)
    assert public.MiniMaxH3PTDSetupEXPT8.fingerprint_inputs(checkpoint=str(path)) != missing
    assert run(worker, value, "now-present")["file"]["sha256"] == sha
    path.unlink()
    run(worker, value, "now-removed", success=False)


def test_fingerprint_only_stats_and_uses_the_same_path_resolver(tmp_path, monkeypatch):
    import hashlib as hash_module
    monkeypatch.setattr(public.folder_paths, "models_dir", str(tmp_path))
    path = tmp_path / "ptd/one.safetensors"
    path.parent.mkdir()
    path.write_bytes(b"identity probe; contents deliberately not parsed")
    target = path.resolve()
    original_open, original_digest = Path.open, hash_module.file_digest
    def guarded_open(candidate, *args, **kwargs):
        if candidate.resolve() == target:
            pytest.fail("A cache fingerprint read checkpoint contents")
        return original_open(candidate, *args, **kwargs)
    def guarded_digest(stream, *args, **kwargs):
        name = getattr(stream, "name", None)
        if isinstance(name, (str, os.PathLike)) and Path(name).resolve() == target:
            pytest.fail("A cache fingerprint hashed checkpoint contents")
        return original_digest(stream, *args, **kwargs)
    # Restore before pytest reports the result, even when an assertion fails.
    # Non-target files remain accessible throughout the probe.
    with monkeypatch.context() as local:
        local.setattr(Path, "open", guarded_open)
        local.setattr(hash_module, "file_digest", guarded_digest)
        relative = public.MiniMaxH3PTDSetupEXPT8.fingerprint_inputs(checkpoint="one.safetensors")
        absolute = public.MiniMaxH3PTDSetupEXPT8.fingerprint_inputs(checkpoint=str(path))
        assert relative == absolute
        assert relative.startswith("ptd-checkpoint-stat-v1:")
        with pytest.raises(ValueError, match="within models/ptd"):
            public.MiniMaxH3PTDSetupEXPT8.fingerprint_inputs(checkpoint="../outside.safetensors")


def test_actual_submitted_off_graph_never_checks_checkpoint(tmp_path, monkeypatch):
    # Setup has no enabled widget. A UI-muted/bypassed/pruned branch is absent
    # from the submitted API graph; merely disconnected nodes can still be signed.
    worker = executor(monkeypatch)
    def forbidden(*_args, **_kwargs):
        pytest.fail("An absent PTD branch tried to resolve or read a checkpoint")
    monkeypatch.setattr(public, "checkpoint_path", forbidden)
    monkeypatch.setattr(public, "load_ptd_setup", forbidden)
    value = {"9": {"class_type": "PTDCacheOffResult", "inputs": {}}}
    run(worker, value, "ptd-off")


def test_disconnected_but_submitted_Setup_is_signed_without_loading(tmp_path, monkeypatch):
    path = tmp_path / "not-a-checkpoint.safetensors"
    path.write_bytes(b"A disconnected node must not parse this payload")
    worker = executor(monkeypatch)
    checked = []
    original = public.checkpoint_path
    def counted(name):
        checked.append(name)
        return original(name)
    def forbidden(*_args, **_kwargs):
        pytest.fail("A disconnected Setup loaded checkpoint contents")
    monkeypatch.setattr(public, "checkpoint_path", counted)
    monkeypatch.setattr(public, "load_ptd_setup", forbidden)
    value = {
        "1": {"class_type": "PTDCacheTinyInputs", "inputs": {}},
        "2": {"class_type": "MiniMaxH3PTDSetupEXPT8", "inputs": {
            "model": ["1", 0], "checkpoint": str(path)}},
        "9": {"class_type": "PTDCacheOffResult", "inputs": {}},
    }
    run(worker, value, "disconnected-present-in-api")
    assert checked and all(name == str(path) for name in checked)


@pytest.mark.parametrize("kind", ["compact4", "h3_4step"])
def test_selected_user_preprocess_object_patch_is_not_replaced(tmp_path, kind):
    path = tmp_path / "user-selected.safetensors"
    replace_checkpoint(path, kind, 1)
    base, calls = cache_model(), []
    def user_preprocess(value):
        calls.append(tuple(value.shape))
        return value  # Explicit user-owned raw-text behavior.
    key = "diffusion_model.preprocess_text_embeds"
    base.add_object_patch(key, user_preprocess)
    selected, sampler, grid, _ = public.MiniMaxH3PTDSetupEXPT8.execute(base, str(path)).result
    assert selected.object_patches[key] is user_preprocess
    result = SamplerCustomAdvanced.execute(RandomNoise.execute(26102684).result[0],
        BasicGuider.execute(selected, conditioning()).result[0], sampler, grid, latent()).result[0]
    assert calls and all(bool(torch.isfinite(part).all()) for part in result["samples"].unbind())
    report = ptd.get_controller(selected).report()
    assert report["status"] == "observed_partial"
    assert any("Unknown text preprocess" in warning for warning in report["warnings"])
    assert selected.object_patches[key] is user_preprocess
