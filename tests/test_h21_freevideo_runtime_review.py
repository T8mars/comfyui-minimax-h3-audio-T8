"""Independent v4 integrity counters; real CPU APIs, never CUDA Engine qualification."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.freevideo_h21 import assets, plans, source, stages


@pytest.fixture
def official_source():
    location = os.environ.get("T8_H21_FREEVIDEO_SOURCE")
    if not location:
        pytest.skip("Select the installed pinned source; never download a substitute")
    return source.Source(location)


def _projection_cache(tmp_path):
    root = tmp_path / "cache"
    (root / "adaln").mkdir(parents=True)
    save_file({"time_embedder.linear_1.weight": torch.eye(2)}, str(root / "root.safetensors"))
    save_file({"transformer_blocks.0.adaln_proj.linear.weight": torch.arange(72).float().reshape(36, 2)},
              str(root / "adaln/00.safetensors"))
    groups = []
    for name in ("root", "adaln/00"):
        path = root / (name + ".safetensors")
        groups.append(dict(group=name, file=name + ".safetensors", bytes=path.stat().st_size,
                           sha256=source.digest(path)))
    manifest = dict(groups=groups, precision="bf16", source_id="independent_real_CPU_tensor_fixture")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return root, manifest


def test_owned_incomplete_projection_publication_does_not_poison_real_retry(tmp_path, monkeypatch, official_source):
    cache, full = _projection_cache(tmp_path)
    slim = dict(full, groups=full["groups"][:1], format="freevideo-fp8-slim-v1",
                adaln_sources=dict(groups=full["groups"][1:]))
    original = {str(path): path.read_bytes() for path in cache.rglob("*") if path.is_file()}
    actual_link = os.link
    calls = []
    def interrupted_link(src, dst, *args, **kwargs):
        calls.append((str(src), str(dst)))
        if len(calls) == 2:
            raise OSError("independent_actual_second_link_fault")
        return actual_link(src, dst, *args, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(assets.os, "link", interrupted_link)
        with pytest.raises(OSError, match="independent_actual_second_link_fault"):
            assets.dynamic_projection_view(cache, slim, tmp_path / "owned", official_source.module("adaln_assets"))
    assert len(calls) == 2 and Path(calls[0][1]).is_file()
    # Keep the failed producer's actual partial files. Recovery must not erase
    # unknown directories or overwrite a previously published valid view.
    partial_bytes = Path(calls[0][1]).read_bytes()
    view, actual = assets.dynamic_projection_view(cache, slim, tmp_path / "owned", official_source.module("adaln_assets"))
    assert actual["groups"] == full["groups"]
    assert os.path.samefile(view / "adaln/00.safetensors", cache / "adaln/00.safetensors")
    assert Path(calls[0][1]).read_bytes() == partial_bytes
    assert all(Path(path).read_bytes() == data for path, data in original.items())
    assert assets.dynamic_projection_view(cache, slim, tmp_path / "owned", official_source.module("adaln_assets")) == (view, actual)


def _prepare_tool():
    path = Path(__file__).parents[1] / "tools/prepare_h21_freevideo_runtime.py"
    spec = importlib.util.spec_from_file_location("h21_independent_prepare_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owned_empty_home_after_real_config_publication_fault_can_retry(tmp_path, monkeypatch, official_source):
    vdn = os.environ.get("T8_H21_FREEVIDEO_VDN")
    if not vdn:
        pytest.skip("Select an existing VDN API source, no implicit installation")
    vdn = Path(vdn).resolve(strict=True)
    api = vdn / "src/inference/render.py"
    assert api.is_file()
    cache, _ = _projection_cache(tmp_path)
    base, checkpoint, sampling = tmp_path / "base", tmp_path / "checkpoint", tmp_path / "sampling"
    (base / "transformer").mkdir(parents=True)
    checkpoint.mkdir()
    sampling.mkdir()
    (base / "transformer/config.json").write_text(json.dumps(dict(hidden_size=6, num_layers=1, in_channels=24, text_dim=8)), encoding="utf8")
    weight = cache / "root.safetensors"
    config = dict(python=str(Path(sys.executable).resolve()), vdn_root=str(vdn), vdn_revision="actual_API_file_inventory",
                  cache=str(cache), base=str(base), checkpoint=str(checkpoint), files=[
        dict(path=str(weight), bytes=weight.stat().st_size, sha256=source.digest(weight), kind="weight"),
        dict(path=str(api), bytes=api.stat().st_size, sha256=source.digest(api), kind="source")])
    base_path, output, home = tmp_path / "base-config.json", tmp_path / "v4.json", tmp_path / "owned-home"
    base_path.write_text(json.dumps(config), encoding="utf8")
    original = base_path.read_bytes()
    tool = _prepare_tool()
    def publication_fault(*args, **kwargs):
        raise OSError("independent_actual_config_publication_fault")
    with monkeypatch.context() as fault:
        fault.setattr(tool, "write_json_new", publication_fault)
        with pytest.raises(OSError, match="independent_actual_config_publication_fault"):
            tool.prepare(base_path, official_source.root, home, sampling, output, tiers=["max"], tasks=["t2va"],
                         engine_options=dict(adaln_cache=False, adaln_disk_cache=False))
    assert not output.exists() and (not home.exists() or not list(home.iterdir()))
    result = tool.prepare(base_path, official_source.root, home, sampling, output, tiers=["max"], tasks=["t2va"],
                          engine_options=dict(adaln_cache=False, adaln_disk_cache=False))
    assert home.is_dir() and output.is_file() and result["GPU_started"] is False
    assert base_path.read_bytes() == original
    published = output.read_bytes()
    with pytest.raises(FileExistsError):
        tool.prepare(base_path, official_source.root, home, sampling, output, tiers=["max"], tasks=["t2va"],
                     engine_options=dict(adaln_cache=False, adaln_disk_cache=False))
    assert output.read_bytes() == published


def _receipt(selected, counts):
    roles = ["LOW", "HIGH"] if selected["role"] == "FULL" else [selected["role"]]
    return dict(plan=selected, actual=dict(completed_callbacks=sum(counts), plan_identity=selected["identity"],
        plan_sha256=hashlib.sha256(source.canonical(selected).encode()).hexdigest(), passes=[
            dict(role=role, completed_callbacks=count, sample=dict(step_seconds=[.01] * count))
            for role, count in zip(roles, counts)]), config_sha256="1" * 64, inventory_sha256="2" * 64,
        request_sha256="3" * 64, output_sha256="4" * 64)


def test_real_full_plan_requires_eight_plus_three_not_seven_plus_four_total(official_source):
    # Serialization integrity only, not asserted as a real model completion.
    selected = plans.plan(official_source, "light", 256, 256, 39, role="FULL")
    video, audio = torch.zeros(1, 24, 12, 16, 16), torch.zeros(1, 32, 2, 65)
    assert stages.validate(stages.make(video, audio, _receipt(selected, [8, 3])))["actual"]["completed_callbacks"] == 11
    with pytest.raises(ValueError, match="pass|callback|coverage"):
        stages.make(video, audio, _receipt(selected, [7, 4]))


def test_real_high_plan_rejects_truthy_non_sha_low_lineage(official_source):
    # Actual output content records remain correct; only the predecessor SHA is invalid.
    selected = plans.plan(official_source, "light", 256, 256, 39, role="HIGH")
    video, audio = torch.zeros(1, 24, 12, 16, 16), torch.zeros(1, 32, 2, 65)
    receipt = dict(_receipt(selected, [3]), low_audio=stages.tensor_record(audio), low_receipt_sha256="a" * 64)
    assert stages.validate(stages.make(video, audio, receipt))["low_receipt_sha256"] == "a" * 64
    receipt["low_receipt_sha256"] = "a"
    with pytest.raises(ValueError, match="LOW|lineage|sha|SHA"):
        stages.make(video, audio, receipt)


def test_actual_cpu_eight_plus_three_path_reads_cannot_publish_changed_owned_condition(tmp_path, official_source):
    python, vdn = os.environ.get("T8_H21_FREEVIDEO_PYTHON"), os.environ.get("T8_H21_FREEVIDEO_VDN")
    if not python or not vdn:
        pytest.skip("Select the independent existing VDN Python; never emulate CUDA Engine")
    condition_path = tmp_path / "owned-conditioning.pt"
    report_path = tmp_path / "actual-tiny-condition-seal.json"
    script = textwrap.dedent(r'''
        import json, pathlib, sys
        repo, source_root, vdn, condition_path, report_path = sys.argv[1:]
        sys.path[:0] = [repo, str(pathlib.Path(vdn) / "diffusers/src"), vdn]
        import torch
        from h3_t8.freevideo_h21.source import Source, digest
        from h3_t8.freevideo_h21.plans import plan
        from h3_t8.freevideo_h21.execution import execute
        from h3_t8.freevideo_h21.tiny_worker import TinyVDNCPU
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        selected_source = Source(source_root)
        selected = plan(selected_source, "light", 256, 256, 39)
        condition = dict(prompt_embeds=torch.arange(24).float().reshape(3, 8) / 20,
                         text_token_tags=torch.ones(3, dtype=torch.long))
        torch.save(condition, condition_path)
        initial_sha = digest(condition_path)
        reads, progress_rows = [], []
        class PathReadingTinyCPU(TinyVDNCPU):
            def sample(self, path, *args, **kwargs):
                reads.append(digest(path))
                data = torch.load(path, weights_only=True, map_location="cpu")
                return super().sample(data, *args, **kwargs)
        owner = PathReadingTinyCPU(selected_source, selected)
        before = dict(owner.config)
        def mutate_last_low(completed, total, seconds):
            progress_rows.append(completed)
            if completed == 8:
                changed = dict(condition, prompt_embeds=condition["prompt_embeds"] + 1)
                torch.save(changed, condition_path)
        error, completed = None, None
        try:
            _, _, actual = execute(owner, pathlib.Path(condition_path), 42, selected, progress=mutate_last_low)
            completed = actual["completed_callbacks"]
        except ValueError as exc:
            error = str(exc)
        value = dict(reads=reads, initial_sha=initial_sha, final_sha=digest(condition_path),
            actual_callback_indices=progress_rows, completed_callbacks=completed, caught_error=error,
            CUDA_initialized=torch.cuda.is_initialized(), Engine_qualification=False,
            model="real random-weight two-block Diffusers CPU + stock author samplers, not CUDA Engine")
        pathlib.Path(report_path).write_text(json.dumps(value), encoding="utf8")
        assert reads and reads[0] == initial_sha
        assert value["final_sha"] != initial_sha
        assert owner.config == before and not torch.cuda.is_initialized()
        assert error is not None, "changed owned conditioning.pt published completion: " + json.dumps(value)
        assert any(word in error.lower() for word in ("conditioning", "condition", "content", "sha")), error
        assert completed is None
    ''')
    command = [python, "-I", "-B", "-X", "utf8", "-c", script, str(Path(__file__).parents[1]),
               str(official_source.root), vdn, str(condition_path), str(report_path)]
    child = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                           capture_output=True, text=True, encoding="utf8", timeout=180)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(report_path.read_text(encoding="utf8"))
    assert actual["Engine_qualification"] is False and actual["CUDA_initialized"] is False
    assert actual["caught_error"] and actual["completed_callbacks"] is None
    assert not torch.cuda.is_initialized()
