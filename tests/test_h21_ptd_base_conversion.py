"""Actual small safetensors/plain-Core contracts, never full trained weights."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg import ptd_base_conversion as api

ROOT = Path(__file__).resolve().parents[1]


def official():
    path = ROOT / "tests/fixtures/h21_ptd_base_reference.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "87502acb79bc95665f9bc211a253f226acb64b01712882cd2911dbfa3e0b596f"
    spec = importlib.util.spec_from_file_location("_h21_pinned_official_base_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg():
    return dict(hidden_size=8, num_layers=1, num_refiner_layers=1,
                num_attention_heads=1, attention_head_dim=8, ffn_dim=8,
                in_channels=3, audio_in_channels=4, text_dim=6, freq_dim=4,
                time_embed_hidden_dim=8, time_embed_dim=4, rope_freq_dim=1,
                patch_size=[1, 2, 2], rope_theta=10000., norm_eps=.001,
                qk_norm_eps=.0001, final_norm_eps=.01)


def refresh(case):
    case["manifest"] = {"repository": "MiniMaxAI/MiniMax-H3", "revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
                        "files": [{"file": str(p.relative_to(case["source"])).replace("\\", "/"),
                                   "bytes": p.stat().st_size, "sha256": api.file_sha(p)}
                                  for p in sorted(case["source"].rglob("*")) if p.is_file()]}


def write_source(case):
    source = case["source"] / "transformer"
    source.mkdir(parents=True, exist_ok=True)
    names = list(case["state"])
    groups = [{name: case["state"][name] for i, name in enumerate(names) if i % 3 == k} for k in range(3)]
    index = {}
    for i, group in enumerate(groups):
        name = f"source-{i}.safetensors"
        save_file(group, source / name)
        index.update({key: name for key in group})
    (source / "config.json").write_text(json.dumps(case["config"]), encoding="utf-8")
    (source / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": index}), encoding="utf-8")
    refresh(case)


@pytest.fixture
def case(tmp_path):
    config = cfg()
    generator = torch.Generator().manual_seed(20261021)
    state = {name: torch.randn(shape, generator=generator).to(torch.bfloat16)
             for name, shape in official()._expected_source_shapes(config).items()}
    for name in state:
        if name.startswith(("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")):
            state[name] = state[name].float()
    value = {"config": config, "state": state, "source": tmp_path / "source",
             "job": tmp_path / "job", "output": tmp_path / "output/base.safetensors"}
    write_source(value)
    return value


def convert(case, **options):
    return api.run_conversion(case["source"], case["job"], case["output"], case["manifest"], **options)


def test_full_fixed_architecture_mapping_consumes638_to535_without_allocating_weights():
    reference = official()._expected_source_shapes(api.DEFAULTS)
    _, rules = api._rules(api.DEFAULTS)
    actual = {name: shape for rule in rules for name, shape in zip(rule.sources, rule.shapes, strict=True)}
    assert actual == reference and len(actual) == 638 and len(rules) + 1 == 535
    assert sum(r.operation == "concat_qkv" for r in rules) == 52
    assert sum(r.operation == "swap_swiglu" for r in rules) == 52


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("chunk_bytes", [3, 512])
def test_real_cross_shard_streaming_is_bit_exact_to_fixed_official_function(case, dtype, chunk_bytes):
    case["state"] = {name: value.to(dtype) for name, value in case["state"].items()}
    write_source(case)
    before = {name: value.clone() for name, value in case["state"].items()}
    manifest = deepcopy(case["manifest"])
    expected = official().convert_base_state_dict(case["state"], case["config"])
    receipt = convert(case, chunk_bytes=chunk_bytes)
    actual = load_file(case["output"])
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype and torch.equal(actual[name], expected[name]), name
    assert all(torch.equal(before[name], value) for name, value in case["state"].items())
    assert manifest == case["manifest"]
    assert receipt["plain_dtype_preserved"] and not receipt["experts_or_DMD_merged"] and not receipt["quantized"]
    assert receipt["sha256"] == api.file_sha(case["output"])
    assert not receipt["GPU_validated"] and not torch.cuda.is_initialized()


def test_mixed_high_precision_islands_remain_their_actual_dtypes(case):
    convert(case)
    actual = load_file(case["output"])
    expected = official().convert_base_state_dict(case["state"], case["config"])
    assert all(torch.equal(actual[name], value) and actual[name].dtype == value.dtype for name, value in expected.items())
    assert actual["video_patch_proj.weight"].dtype == torch.float32
    assert actual["blocks.0.attn.qkv_proj.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("change,match", [("nan", "Nonfinite"), ("inf", "Nonfinite"), ("shape", "shape"),
                                          ("qkv_dtype", "dtypes"), ("extra", "keys"), ("missing", "keys"),
                                          ("truncated", "truncated"), ("duplicate", "Duplicate"), ("integer", "dtype")])
def test_actual_invalid_tensor_and_container_inputs_are_not_published(case, change, match):
    first = next(iter(case["state"]))
    if change in ("nan", "inf"):
        case["state"][first].flatten()[0] = float(change)
    elif change == "shape":
        case["state"][first] = case["state"][first][:1]
    elif change == "qkv_dtype":
        key = "transformer_blocks.0.attn.to_k.weight"
        case["state"][key] = case["state"][key].double()
    elif change == "extra":
        case["state"]["unknown.weight"] = torch.ones(1)
    elif change == "missing":
        del case["state"][first]
    elif change == "integer":
        case["state"][first] = case["state"][first].int()
    write_source(case)
    path = case["source"] / "transformer/source-0.safetensors"
    if change == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
        refresh(case)
    elif change == "duplicate":
        raw = b'{"duplicate":0,"duplicate":1}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"actual payload")
        refresh(case)
    with pytest.raises(ValueError, match=match):
        convert(case, chunk_bytes=512)
    assert not case["output"].exists()


def test_bad_manifest_SHA_missing_download_and_path_escape_are_real_errors(case):
    wanted = case["manifest"]["files"][0]["sha256"]
    case["manifest"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        convert(case)
    case["manifest"]["files"][0]["sha256"] = wanted
    # Failed jobs deliberately retain their identity; changed inputs need a new job.
    case["job"] = case["job"].with_name("missing-job")
    path = case["source"] / "transformer/source-0.safetensors"
    held = path.with_name("held")
    path.rename(held)
    with pytest.raises(api.SourcePending):
        convert(case)
    held.rename(path)
    manifest = deepcopy(case["manifest"])
    manifest["files"][0]["file"] = "../outside.safetensors"
    with pytest.raises(ValueError, match="canonical"):
        api.prepare(case["source"], manifest)
    assert not case["output"].exists()


def test_change_during_copy_then_restore_cannot_bypass_full_SHA_proof(case, monkeypatch):
    original = api._read_verified
    changed = []
    def intercepted(path, offset, length, ledger, chunk_bytes, cancelled):
        if offset > 8 and not changed and path.suffix == ".safetensors":
            raw = path.read_bytes()
            replacement = bytearray(raw)
            replacement[offset] ^= 1
            changed.append(path)
            path.write_bytes(replacement)
            try:
                yield from original(path, offset, length, ledger, chunk_bytes, cancelled)
            finally:
                path.write_bytes(raw)
        else:
            yield from original(path, offset, length, ledger, chunk_bytes, cancelled)
    monkeypatch.setattr(api, "_read_verified", intercepted)
    with pytest.raises(ValueError, match="block changed"):
        convert(case, chunk_bytes=512)
    assert changed and not case["output"].exists()
    assert all(api.file_sha(case["source"] / row["file"]) == row["sha256"] for row in case["manifest"]["files"])
    convert(case, chunk_bytes=512)


def test_cancellation_preserves_verified_parts_and_resume_does_not_repeat_them(case):
    stop, initial = [], {}
    def progress(row):
        if row["stage"] == "completed_tensor" and row["index"] == 2:
            initial.update(api._json((case["job"] / "state.json").read_bytes())["completed"])
            stop.append(True)
    with pytest.raises(InterruptedError):
        convert(case, cancelled=lambda: bool(stop), progress=progress)
    assert len(initial) == 3 and not case["output"].exists()
    rows = []
    convert(case, progress=rows.append)
    final = api._json((case["job"] / "state.json").read_bytes())["completed"]
    assert all(final[name] == item for name, item in initial.items())
    assert {row["target"] for row in rows if row["stage"] == "reused_tensor"} == set(initial)


@pytest.mark.parametrize("when", ["before_publish", "published"])
def test_actual_publication_crashes_recover_without_overwriting_user_file(case, when):
    def crash(row):
        if row["stage"] == when:
            raise RuntimeError("Actual owned publication interruption")
    with pytest.raises(RuntimeError, match="publication interruption"):
        convert(case, progress=crash)
    existed = case["output"].exists()
    before = case["output"].stat().st_mtime_ns if existed else None
    receipt = convert(case)
    assert api.file_sha(case["output"]) == receipt["sha256"]
    if existed:
        assert case["output"].stat().st_mtime_ns == before
    again = convert(case)
    assert receipt == again


def test_saved_part_corruption_and_unrelated_existing_output_are_not_adopted(case):
    stop = []
    def progress(row):
        if row["stage"] == "completed_tensor":
            stop.append(True)
    with pytest.raises(InterruptedError):
        convert(case, cancelled=lambda: bool(stop), progress=progress)
    item = next(iter(api._json((case["job"] / "state.json").read_bytes())["completed"].values()))
    part = case["job"] / item["file"]
    raw = bytearray(part.read_bytes())
    raw[-1] ^= 1
    part.write_bytes(raw)
    with pytest.raises(ValueError, match="part SHA"):
        convert(case)
    case["job"] = case["job"].with_name("unrelated-job")
    case["output"].parent.mkdir()
    case["output"].write_bytes(b"Existing user output must remain untouched")
    with pytest.raises(FileExistsError):
        convert(case)
    assert case["output"].read_bytes() == b"Existing user output must remain untouched"


def test_prepare_and_real_CLI_subprocess_do_not_import_Torch_or_payloads(case, tmp_path):
    path, report = tmp_path / "manifest.json", tmp_path / "prepared.json"
    path.write_text(json.dumps(case["manifest"]), encoding="utf-8")
    command = [sys.executable, "-B", str(ROOT / "tools/convert_h21_ptd_base.py"), "--source", str(case["source"]),
               "--manifest", str(path), "--report", str(report)]
    result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    prepared = json.loads(report.read_text(encoding="utf-8"))
    assert not prepared["Torch_imported"] and not prepared["CUDA_initialized"]
    assert not prepared["source_complete_SHA_verified"] and prepared["native_tensors"] == 37
    again = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert again.returncode != 0 and "Report already exists" in again.stderr


def test_real_tiny_plain_model_strictly_loads_through_current_Core(case):
    import comfy.model_detection
    import comfy.ops
    import comfy.sd
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model
    convert(case)
    state, metadata = comfy.utils.load_torch_file(str(case["output"]), return_metadata=True)
    config = comfy.model_detection.detect_unet_config(state, "", metadata=metadata)
    native = MiniMaxH3Model(**config, dtype=torch.bfloat16, device="cpu", operations=comfy.ops.disable_weight_init)
    loaded = native.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert native.blocks[0].norm1.eps == case["config"]["norm_eps"]
    assert native.blocks[0].attn.q_norm.eps == case["config"]["qk_norm_eps"]
    assert native.final_layer.norm.eps == case["config"]["final_norm_eps"]
    patcher = comfy.sd.load_diffusion_model(str(case["output"]), model_options={"dtype": torch.bfloat16})
    assert isinstance(patcher.model.diffusion_model, MiniMaxH3Model)
    assert not any("comfy_quant" in name or "weight_scale" in name for name in state)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("target", ["video_patch_proj.weight", "blocks.0.attn.qkv_proj.weight",
                                    "blocks.0.mlp.fc1.weight", "rope.inv_freq"])
def test_real_interrupted_resume_rejects_part_and_state_SHA_changed_together(case, target):
    case["state"] = {name: torch.zeros_like(value) for name, value in case["state"].items()}
    write_source(case)
    stopped = []
    def progress(row):
        if row["stage"] == "completed_tensor" and row["target"] == target:
            stopped.append(True)
    with pytest.raises(InterruptedError):
        convert(case, cancelled=lambda: bool(stopped), progress=progress)
    state_file = case["job"] / "state.json"
    state = api._json(state_file.read_bytes())
    item = state["completed"][target]
    part = case["job"] / item["file"]
    header = api._header(part)
    # Only keep dtype: a live mmap-backed tensor prevents Windows truncation.
    dtype = load_file(part)[target].dtype
    replacement = torch.tensor([123.], dtype=dtype).view(torch.uint8).numpy().tobytes()
    raw = bytearray(part.read_bytes())
    raw[header["start"]:header["start"] + len(replacement)] = replacement
    part.write_bytes(raw)
    item["sha256"] = api.file_sha(part)
    api.atomic_json(state_file, state, replace_existing=True)
    unchanged_part, unchanged_state = bytes(raw), state_file.read_bytes()
    with pytest.raises(ValueError, match="source|derived|payload"):
        convert(case)
    assert not case["output"].exists()
    assert part.read_bytes() == unchanged_part and state_file.read_bytes() == unchanged_state
    assert all(api.file_sha(case["source"] / row["file"]) == row["sha256"] for row in case["manifest"]["files"])


def test_resume_source_proof_is_bound_to_the_same_part_bytes_as_saved_SHA(case, monkeypatch):
    stop = []
    def interrupt(row):
        if row["stage"] == "completed_tensor":
            stop.append(True)
    with pytest.raises(InterruptedError):
        convert(case, cancelled=lambda: bool(stop), progress=interrupt)
    state_file = case["job"] / "state.json"
    state = api._json(state_file.read_bytes())
    name, item = next(iter(state["completed"].items()))
    part = case["job"] / item["file"]
    original = part.read_bytes()
    modified = bytearray(original)
    modified[-1] ^= 1
    part.write_bytes(modified)
    item["sha256"] = api.file_sha(part)
    api.atomic_json(state_file, state, replace_existing=True)
    verify, restored = api._verify_part, []
    def verify_then_restore(job, saved, target, identity, cancelled):
        result = verify(job, saved, target, identity, cancelled)
        if target["name"] == name and not restored:
            part.write_bytes(original)
            restored.append(True)
        return result
    monkeypatch.setattr(api, "_verify_part", verify_then_restore)
    def change_back_after_proof(row):
        if row["stage"] == "reused_tensor" and row["target"] == name:
            part.write_bytes(modified)
    with pytest.raises(ValueError, match="source payload proof"):
        convert(case, progress=change_back_after_proof)
    assert restored and not case["output"].exists()
    assert all(api.file_sha(case["source"] / row["file"]) == row["sha256"] for row in case["manifest"]["files"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_matching_NaN_source_and_forged_completed_part_still_rejects(case, dtype):
    case["state"] = {name: torch.zeros_like(value).to(dtype) for name, value in case["state"].items()}
    case["state"]["proj_in.weight"][0, 0] = float("nan")
    write_source(case)
    target = api.prepare(case["source"], case["manifest"])["targets"][0]
    files = api.manifest_files(case["manifest"])
    implementation = {p.name: api.file_sha(p) for p in (
        ROOT / "h3_t8/ptd_base_conversion.py", ROOT / "h3_t8/meridian_conversion.py",
        ROOT / "h3_t8/meridian_checkpoint_io.py")}
    identity_data = {"schema": "t8.ptd_plain_base_job.v1", "source": str(case["source"].resolve()),
                     "output": str(case["output"].resolve()), "files": files, "implementation": implementation,
                     "chunk_bytes": 3, "repository": case["manifest"]["repository"],
                     "revision": case["manifest"]["revision"]}
    identity = api.canonical_identity(identity_data)
    case["job"].mkdir()
    api.atomic_json(case["job"] / "job.json", identity_data)
    raw = case["state"]["proj_in.weight"].contiguous().view(torch.uint8).numpy().tobytes()
    part = case["job"] / "forged-matching-nan.safetensors"
    prefix = api._encoded_header({target["name"]: {
        "dtype": target["dtype"], "shape": target["shape"], "data_offsets": [0, len(raw)]}},
        {"t8_identity": identity, "target": target["name"]})
    part.write_bytes(prefix + raw)
    api.atomic_json(case["job"] / "state.json", {"identity": identity, "completed": {
        target["name"]: {"file": part.name, "sha256": api.file_sha(part)}}})
    before = part.read_bytes()
    with pytest.raises(ValueError, match="Nonfinite actual source"):
        convert(case, chunk_bytes=3)
    assert part.read_bytes() == before and not case["output"].exists()
    assert all(api.file_sha(case["source"] / row["file"]) == row["sha256"] for row in case["manifest"]["files"])


def test_post_publish_interruption_cannot_adopt_output_and_journal_changed_together(case):
    def crash_after_link(row):
        if row["stage"] == "published":
            raise RuntimeError("Actual interruption after link before final result")
    with pytest.raises(RuntimeError, match="after link"):
        convert(case, progress=crash_after_link)
    assert case["output"].is_file() and not (case["job"] / "result.json").exists()
    header = api._header(case["output"])
    raw = bytearray(case["output"].read_bytes())
    offset = header["start"] + header["tensors"]["video_patch_proj.weight"]["data_offsets"][0]
    raw[offset:offset + 4] = struct.pack("<f", 123.)
    case["output"].write_bytes(raw)
    publication = case["job"] / "publication.json"
    journal = api._json(publication.read_bytes())
    journal["output"]["sha256"] = api.file_sha(case["output"])
    api.atomic_json(publication, journal, replace_existing=True)
    before_output, before_publication = bytes(raw), publication.read_bytes()
    with pytest.raises(ValueError, match="[Pp]ublished.*(?:source|payload)"):
        convert(case)
    assert case["output"].read_bytes() == before_output
    assert publication.read_bytes() == before_publication
    assert not (case["job"] / "result.json").exists()
    assert all(api.file_sha(case["source"] / row["file"]) == row["sha256"] for row in case["manifest"]["files"])
