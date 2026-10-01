"""Data-only cold RGB/LTX handoff, with no model loading or fake completion."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.modular_sampling import ltx_rgb_source_storage as store
from h3_audio_t8_pkg.modular_sampling import ltx_rgb_source_nodes as nodes
from h3_audio_t8_pkg.modular_sampling.ltx_rgb_stage import bind_ltx_rgb_stage, audit_ltx_rgb_stage
from tests.test_modular_ltx_rgb_stage import _case

ROOT = Path(__file__).resolve().parents[1]


def source(identity=False, audio=True):
    args = _case(identity)
    values = dict(zip(("source_frames", "source_audio", "prepared_frames", "prep_report_json", "ltx_latent"), args[:5]))
    values["bit_depth"] = 10
    if not audio:
        values["source_audio"] = None
    values["ltx_latent"]["metadata"] = {"tuple": ("中文", 4, .5), "bytes": b"exact", "mask": torch.ones(1)}
    return values, args


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("audio", [False, True])
def test_source_roundtrip_rebuilds_current_stage_with_original_audio(tmp_path, identity, audio):
    values, args = source(identity, audio)
    before = store.validate_source(values)
    path, digest, report = store.save_source(values, tmp_path)
    loaded, read_report = store.load_source(tmp_path, path, digest)
    assert store.validate_source(loaded) == before == store.validate_source(values)
    assert loaded["source_frames"].data_ptr() != values["source_frames"].data_ptr()
    assert loaded["bit_depth"] == 10
    for raw in (report, read_report):
        assert json.loads(raw)["sampler_completion_verified"] is False
        assert json.loads(raw)["automatic_cache_reuse"] is False
    current = (*[loaded[k] for k in ("source_frames", "source_audio", "prepared_frames", "prep_report_json", "ltx_latent")], *args[5:])
    bound = bind_ltx_rgb_stage(*current)
    result = audit_ltx_rgb_stage(bound[5], *current, loaded["ltx_latent"])
    assert result[1] is loaded["source_audio"]


def test_cross_process_load_has_no_model_or_sampler_and_exact_identity(tmp_path):
    values, _ = source()
    path, digest, _ = store.save_source(values, tmp_path)
    code = '''
import json, os, runpy, sys
from pathlib import Path
root=Path.cwd(); sys.path[:0]=[str(root),str(root.parents[1])]
import comfy.cli_args
comfy.cli_args.args.cpu=True
runpy.run_path('tests/conftest.py')
import torch
from h3_audio_t8_pkg.modular_sampling import ltx_rgb_source_storage as s
args=json.load(sys.stdin)
value,report=s.load_source(**args)
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps({'identity':s.validate_source(value),'report':json.loads(report),'pid':os.getpid()}))
'''
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
        input=json.dumps(dict(storage_root=str(tmp_path), artifact_path=path, artifact_sha256=digest)), timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    import os
    assert result["pid"] != os.getpid()
    assert result["identity"] == store.validate_source(values)
    assert result["report"]["sampling_executed"] is False


@pytest.mark.parametrize("failure", ["nan", "shape", "audio_rate", "audio_nan", "depth", "report", "callable", "metadata_nan"])
def test_invalid_or_executable_source_is_not_promoted(tmp_path, failure):
    values, _ = source()
    if failure == "nan":
        values["source_frames"][0, 0, 0, 0] = float("nan")
    elif failure == "shape":
        values["ltx_latent"]["samples"] = torch.zeros(1, 128, 2, 4, 6)
    elif failure == "audio_rate":
        values["source_audio"]["sample_rate"] = True
    elif failure == "audio_nan":
        values["source_audio"]["waveform"][0, 0, 0] = float("inf")
    elif failure == "depth":
        values["bit_depth"] = "10"
    elif failure == "report":
        data = json.loads(values["prep_report_json"])
        data["dropped_tail_frames"] += 1
        values["prep_report_json"] = json.dumps(data)
    elif failure == "callable":
        values["ltx_latent"]["hook"] = lambda: None
    else:
        values["ltx_latent"]["metadata"]["mask"][0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        store.save_source(values, tmp_path)
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize("failure", ["sha", "tensor", "manifest", "partial", "traversal", "missing"])
def test_broken_artifact_never_loads_or_regenerates(tmp_path, failure):
    values, _ = source()
    path, digest, _ = store.save_source(values, tmp_path)
    manifest_path = tmp_path / path
    if failure == "sha":
        digest = "f" * 64
    elif failure == "tensor":
        (manifest_path.parent / "state.safetensors").write_bytes(b"corrupt")
    elif failure == "manifest":
        data = json.loads(manifest_path.read_text())
        data["data_only"] = False
        manifest_path.write_text(json.dumps(data))
        digest = store.file_sha(manifest_path)
    elif failure == "partial":
        path = str(Path(path).with_name("state.safetensors.partial"))
    elif failure == "traversal":
        path = "../" + path
    else:
        manifest_path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        store.load_source(tmp_path, path, digest)


@pytest.mark.parametrize("when", [2, 3])
def test_cancellation_preserves_orphans_without_completed_manifest(tmp_path, when):
    values, _ = source()
    calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls == when:
            raise RuntimeError("cancel source save")
    with pytest.raises(RuntimeError, match="cancel source save"):
        store.save_source(values, tmp_path, interrupt=interrupt)
    assert not list(tmp_path.rglob("manifest.json"))
    assert list(tmp_path.rglob("*.partial"))


def test_source_mutation_during_write_cannot_commit(tmp_path, monkeypatch):
    values, _ = source()
    real = store.save_file
    def mutate(*args, **kwargs):
        real(*args, **kwargs)
        values["source_frames"][0, 0, 0, 0] += .1
    monkeypatch.setattr(store, "save_file", mutate)
    with pytest.raises(ValueError, match="changed during persistence"):
        store.save_source(values, tmp_path)
    assert not list(tmp_path.rglob("manifest.json"))


def test_nodes_disabled_save_no_files_and_registration_is_append_only(tmp_path, monkeypatch):
    values, _ = source()
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    result = nodes.MiniMaxH3LTXRGBSourceSaveEXPT8.execute(**values, confirm_save=False).result
    assert result[0] is values["source_frames"] and result[4] is values["ltx_latent"]
    assert result[7] == 10 and result[8:10] == ("", "")
    assert not list(tmp_path.iterdir())
    with pytest.raises(FileNotFoundError):
        nodes.MiniMaxH3LTXRGBSourceLoadEXPT8.fingerprint_inputs("missing/manifest.json", "a" * 64)
    before = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
              *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
              *h3_audio_t8_pkg._audio_refine_effect_node_classes]
    actual = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert actual[:len(before) + len(nodes.NODES)] == before + nodes.NODES
    assert len(actual) == len({cls.define_schema().node_id for cls in actual})


def test_save_twice_never_overwrites_and_busy_load_rejects(tmp_path):
    values, _ = source()
    first = store.save_source(values, tmp_path)
    second = store.save_source(values, tmp_path)
    assert first[0] != second[0]
    assert store.file_sha(tmp_path / first[0]) == first[1]
    with store._lease((tmp_path / first[0]).parent):
        with pytest.raises(RuntimeError, match="busy"):
            store.load_source(tmp_path, *first[:2])


def test_linked_store_root_and_artifact_are_rejected(tmp_path):
    values, _ = source()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="link or junction"):
        store.save_source(values, link)
    path, digest, _ = store.save_source(values, real)
    linked_case = real / "linked-case"
    linked_case.symlink_to((real / path).parent, target_is_directory=True)
    with pytest.raises(ValueError, match="link or junction"):
        store.load_source(real, "linked-case/manifest.json", digest)


def test_post_hash_file_mutation_is_detected(tmp_path):
    values, _ = source()
    path, digest, _ = store.save_source(values, tmp_path)
    calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:
            with (tmp_path / path).open("a") as handle:
                handle.write(" ")
    with pytest.raises(ValueError, match="changed while reading"):
        store.load_source(tmp_path, path, digest, interrupt=interrupt)


def test_real_core_load_cache_never_hides_changed_or_deleted_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from comfy_api.latest import io
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(ROOT.parents[1]))
        import nodes as core_nodes
        import execution
    assert Path(core_nodes.__file__).resolve() == ROOT.parents[1] / "nodes.py"
    values, _ = source()
    observed = []
    class Source(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="RGBSourceFixture", outputs=nodes._outputs())
        @classmethod
        def execute(cls):
            return io.NodeOutput(*nodes._values(values))
    class Capture(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="RGBSourceCapture", is_output_node=True,
                inputs=[io.Image.Input("frames"), io.Audio.Input("audio"), io.Latent.Input("latent")])
        @classmethod
        def execute(cls, frames, audio, latent):
            observed.append(store._input_identity({"frames": frames, "audio": audio, "latent": latent}))
            return io.NodeOutput()
    for cls in (Source, Capture, *nodes.NODES):
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    executor = execution.PromptExecutor(SimpleNamespace(client_id=None, last_node_id=None,
        sockets_metadata={}, send_sync=lambda *_a, **_k: None), cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False))
    save = {"source": {"class_type": "RGBSourceFixture", "inputs": {}},
            "save": {"class_type": nodes.NODES[0].__name__, "inputs": {
                **{name: ["source", slot] for slot, name in enumerate(
                    ("source_frames", "source_audio", "prepared_frames", "prep_report_json", "ltx_latent"))},
                "bit_depth": ["source", 7], "filename_prefix": "core-input", "confirm_save": True}}}
    executor.execute(save, "save-source", execute_outputs=["save"])
    assert executor.success, executor.status_messages
    committed, = nodes._root().rglob("manifest.json")
    graph = {"load": {"class_type": nodes.NODES[1].__name__, "inputs": {
        "artifact_path": committed.relative_to(nodes._root()).as_posix(),
        "artifact_sha256": store.file_sha(committed)}}, "capture": {"class_type": "RGBSourceCapture",
        "inputs": {"frames": ["load", 0], "audio": ["load", 1], "latent": ["load", 4]}}}
    executor.execute(deepcopy(graph), "cold-data-only", execute_outputs=["capture"])
    assert executor.success, executor.status_messages
    assert observed == [store._input_identity({"frames": values["source_frames"],
        "audio": values["source_audio"], "latent": values["ltx_latent"]})]
    executor.execute(deepcopy(graph), "unchanged-data", execute_outputs=["capture"])
    assert executor.success and len(observed) == 1
    # Core annotates each prompt with is_changed. Each actual HTTP submission
    # gets a fresh JSON payload, not the previously annotated dict instance.
    # Same executor and identical input prompt: real fingerprint invalidates
    # cached data and failure prevents downstream use, rather than regeneration.
    state = committed.parent / "state.safetensors"
    with state.open("ab") as handle:
        handle.write(b"tampered")
    executor.execute(deepcopy(graph), "tampered-data", execute_outputs=["capture"])
    assert not executor.success and len(observed) == 1
    committed.unlink()
    executor.execute(deepcopy(graph), "deleted-data", execute_outputs=["capture"])
    assert not executor.success and len(observed) == 1
    assert not torch.cuda.is_initialized()
