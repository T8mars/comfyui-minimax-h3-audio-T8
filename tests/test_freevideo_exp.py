"""Small affected CPU scope; these are not GPU or media-quality certification."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.freevideo_exp import runtime as fv
from h3_audio_t8_pkg.freevideo_exp.nodes import NODES


def stage(role="LOW", frames=39):
    return fv.make_stage(torch.zeros(1, 24, (frames-5)//17*5+2, 16, 16),
                         torch.zeros(1, 32, 2, round(frames/24*40)),
                         dict(schema="t8-freevideo-stage-v1", role=role, completed_nfe=8 if role=="LOW" else 2,
                              tail_steps=None if role=="LOW" else 2, model_revision=fv.MODEL_REVISION,
                              freevideo_revision=fv.FREEVIDEO_REVISION, vdn_revision=fv.VDN_REVISION,
                              sample={"step_seconds":[0.1]*(8 if role=="LOW" else 2)},
                              request_sha256="0"*64, output_sha256="1"*64, source_inventory_sha256="2"*64,
                              geometry=dict(width=256, height=256, frames=frames)))


def conditioning(extra=None):
    return [[torch.zeros(1, 3, 5120), {"minimax_token_tags": torch.ones(3, dtype=torch.int64), **(extra or {})}]]


def test_append_only_registration_and_lazy_engine():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [c.define_schema().node_id for c in classes]
    from h3_audio_t8_pkg.nodes_temporal_dialogue import NODES as scope_nodes
    from h3_audio_t8_pkg.nodes_postprocess import NODES as postprocess_nodes
    from h3_audio_t8_pkg.nodes_reference_package import NODES as reference_nodes
    qwen_nodes = []
    if "MiniMaxH3ReferenceQwenViewEXPT8" in ids:
        from h3_audio_t8_pkg.nodes_qwen_reference_view import NODES as qwen_nodes
        assert [c.__name__ for c in qwen_nodes] == ["MiniMaxH3ReferenceQwenViewEXPT8"]
    from h3_audio_t8_pkg.freevideo_quality.nodes import NODES as quality_nodes
    appended = [*scope_nodes, *postprocess_nodes, *reference_nodes, *qwen_nodes, *quality_nodes]
    assert len(ids) == 632 + len(appended) and len(set(ids)) == len(ids)
    assert ids[620:632] == [c.__name__ for c in NODES]
    assert ids[632:] == [c.__name__ for c in appended]
    features = json.loads((Path(__file__).parents[1] / "features.json").read_text(encoding="utf8"))
    assert ids[:len(features["nodes"])] == features["nodes"]
    assert not any(k == "freevideo_engine" or k.startswith("freevideo_engine.") for k in sys.modules)
    assert ids[619] == "MiniMaxH3KijaiPDD8StepSetupEXPT8"


def test_shape_roundtrip_and_create_only_saved_stage(tmp_path):
    original = stage()
    path, sha = fv.save_stage(original, tmp_path)
    saved = fv.load_stage(path, sha)
    assert saved.receipt_sha256 == original.receipt_sha256
    assert torch.equal(saved.audio, original.audio)
    second, _ = fv.save_stage(original, tmp_path)
    assert path != second
    assert fv.digest(path) == sha
    av = fv.av_output(original)
    v, a = av["samples"].unbind()
    assert torch.equal(a, original.audio)
    v.add_(1)
    fv.validate_stage(original)


def test_saved_tampering_and_wrong_role_rejected(tmp_path):
    original = stage()
    path, sha = fv.save_stage(original, tmp_path)
    with pytest.raises(ValueError, match="HIGH"):
        fv.load_stage(path, sha, "HIGH")
    payload = Path(path).parent / "latents.safetensors"
    with payload.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    with pytest.raises(ValueError, match="content"):
        fv.load_stage(path, sha)


def test_in_memory_tensor_mutation_rejected():
    original = stage()
    original.audio.add_(1)
    with pytest.raises(ValueError, match="content"):
        fv.validate_stage(original)


@pytest.mark.parametrize("change", [{"completed_nfe":None,"tail_steps":None}, {"sample":{"step_seconds":[0.1]}},
                                  {"vdn_revision":"foreign"}, {"request_sha256":""}])
def test_forged_incomplete_receipt_rejected_even_with_matching_self_sha(change):
    original = stage("HIGH")
    receipt = dict(json.loads(original.receipt_json), **change)
    text = fv.canonical(receipt)
    forged = fv.FreeVideoStage(original.video, original.audio, text, hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError):
        fv.validate_stage(forged)


def test_wrong_audio_length_and_partial_nfe_rejected():
    original = stage(frames=124)
    assert original.audio.shape[-1] == 207
    receipt = json.loads(original.receipt_json)
    with pytest.raises(ValueError, match="audio shape"):
        fv.make_stage(original.video, original.audio[..., :-1], receipt)
    with pytest.raises(ValueError, match="complete"):
        fv.make_stage(original.video, original.audio, dict(receipt, completed_nfe=7))


@pytest.mark.parametrize("shape", [(256,256,128), (255,256,124), (256,256,22)])
def test_unaligned_geometry_rejected(shape):
    with pytest.raises(ValueError):
        fv.geometry(*shape)


def test_zero_lora_skips_missing_file_and_nonzero_requires_file(tmp_path):
    model = fv.FreeVideoModel("unused", "unused")
    bypass = fv.with_lora(model, tmp_path / "missing.safetensors", 0)
    assert json.loads(bypass.loras[0])["disabled"]
    with pytest.raises(FileNotFoundError):
        fv.with_lora(model, tmp_path / "missing.safetensors", 1)


def test_file_modified_at_same_size_and_mtime_rehashed(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"one")
    row = dict(path=str(path), bytes=3, sha256=fv.digest(path))
    fv.verify_files([row])
    stamp = path.stat()
    path.write_bytes(b"two")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="content"):
        fv.verify_files([row])


def test_native_conditions_bridge_is_preserved_not_applied_twice():
    cond = conditioning()
    value = cond[0][0]
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    bridge = {"output_sha256": digest.hexdigest(), "schema": "t8_semantic_bridge_v1"}
    cond[0][1]["t8_semantic_bridge"] = bridge
    tensors, meta = fv.conditioning_transport(cond, fv.geometry(256,256,39))
    assert meta["semantic_bridge"] == bridge
    assert torch.equal(tensors["embeds"], value[0])
    with pytest.raises(ValueError, match="modifiers"):
        fv.conditioning_transport(conditioning({"minimax_payload": {}}), fv.geometry(256,256,39))


def test_bad_audio_references_and_clock_change_fail_before_worker():
    with pytest.raises(ValueError, match="reference audio shape"):
        fv.conditioning_transport(conditioning({"minimax_refs":[{"kind":"audio"}]}), fv.geometry(256,256,39))
    with pytest.raises(ValueError, match="frame count"):
        fv.conditioning_transport(conditioning({"minimax_frame_count":124}), fv.geometry(256,256,39))


def test_cold_node_has_no_low_producer_socket():
    schema = NODES[5].define_schema()
    info = schema.get_v1_info(NODES[5]).input
    names = set(info.get("required", {})) | set(info.get("optional", {}))
    assert names == {"manifest_path", "manifest_sha256", "role"}


def test_explicit_clip_release_never_unloads_all(monkeypatch):
    from types import SimpleNamespace
    import comfy.model_management as mm
    from h3_audio_t8_pkg.freevideo_exp.nodes import offload_supplied_clip
    calls = []
    monkeypatch.setattr(mm, "unload_model_and_clones", lambda patcher, **kw: calls.append((patcher, kw)))
    patcher = object()
    offload_supplied_clip(None)
    assert not calls
    offload_supplied_clip(SimpleNamespace(patcher=patcher))
    assert calls == [(patcher, {"unload_additional_models":False})]


def test_owned_process_interrupt_stops_only_owned_child(tmp_path):
    import subprocess
    from h3_audio_t8_pkg.freevideo_exp.supervision import owned_process
    command = [sys.executable, "-c", "import time;time.sleep(60)"]
    peer = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        with (tmp_path / "worker.log").open("w") as log:
            with pytest.raises(RuntimeError, match="cancel"), owned_process(command, dict(os.environ), tmp_path, log) as child:
                assert child.poll() is None
                raise RuntimeError("cancel")
        assert child.poll() is not None
        assert peer.poll() is None
    finally:
        peer.terminate()
        peer.wait(timeout=5)
