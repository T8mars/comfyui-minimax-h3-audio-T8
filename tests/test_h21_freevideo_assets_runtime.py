"""Real local tensor assets and declared-table integrity, not trained model quality."""
from __future__ import annotations

import copy
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg.freevideo_h21 import assets, source


@pytest.fixture
def official():
    root = os.environ.get("T8_H21_FREEVIDEO_SOURCE")
    if not root:
        pytest.skip("External fixed source not selected; never implicitly download")
    return source.Source(root).module("adaln_assets")


def projection_cache(tmp_path):
    root = tmp_path / "cache"
    (root / "adaln").mkdir(parents=True)
    save_file({"time_embedder.linear_1.weight": torch.eye(2)}, str(root / "root.safetensors"))
    tensor = torch.arange(72, dtype=torch.float32).reshape(36, 2) / 100
    save_file({"transformer_blocks.0.adaln_proj.linear.weight": tensor}, str(root / "adaln/00.safetensors"))
    def row(group):
        path = root / (group + ".safetensors")
        return dict(group=group, file=group + ".safetensors", bytes=path.stat().st_size, sha256=source.digest(path))
    manifest = dict(groups=[row("root"), row("adaln/00")], precision="bf16", source_id="actual_CPU_tensor_fixture")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return root, manifest, tensor


def test_actual_model_config_hidden_width_not_a_fixed_5376_default(tmp_path):
    (tmp_path / "transformer").mkdir()
    actual = dict(hidden_size=24, num_layers=2, in_channels=24, text_dim=8, other="kept outside contract")
    (tmp_path / "transformer/config.json").write_text(json.dumps(actual), encoding="utf8")
    assert assets.model_structure(tmp_path) == {key: actual[key] for key in ("hidden_size", "num_layers", "in_channels", "text_dim")}
    actual["hidden_size"] = True
    (tmp_path / "transformer/config.json").write_text(json.dumps(actual), encoding="utf8")
    with pytest.raises(ValueError, match="positive integer hidden_size"):
        assets.model_structure(tmp_path)


def test_actual_selected_metadata_paths_and_bytes_bound_even_if_old_inventory_labels_omit_them(tmp_path):
    from h3_audio_t8_pkg.freevideo_exp.runtime import verify_files
    cache, _, _ = projection_cache(tmp_path)
    base, checkpoint = tmp_path / "base", tmp_path / "checkpoint"
    (base / "transformer").mkdir(parents=True)
    checkpoint.mkdir()
    config = base / "transformer/config.json"
    config.write_text(json.dumps(dict(hidden_size=24)), encoding="utf8")
    (checkpoint / "hybrid_config.json").write_text("{}", encoding="utf8")
    rows = assets.model_metadata(base, checkpoint, cache)
    assert {row["path"] for row in rows} == {str(path.resolve()) for path in
        (cache / "manifest.json", config, checkpoint / "hybrid_config.json")}
    assert all(row["kind"] == "metadata" for row in rows)
    verify_files(rows)
    config.write_text(json.dumps(dict(hidden_size=48)), encoding="utf8")
    with pytest.raises(ValueError, match="content changed"):
        verify_files(rows)


def test_actual_full_projection_no_table_route_keeps_real_tensor_and_cpu_forward(tmp_path, official):
    root, manifest, tensor = projection_cache(tmp_path)
    # The Source-only pure API is sufficient here: explicit no-cache/full storage
    # does not import a scheduler or initialize the real CUDA-only Engine.
    selected_source = source.Source(os.environ["T8_H21_FREEVIDEO_SOURCE"])
    before = {str(path): source.digest(path) for path in root.rglob("*") if path.is_file()}
    selected, result, options, tables, report = assets.modulation_strategy(selected_source, root, manifest, {},
        dict(sampling=dict(enabled=False)), dict(adaln_cache=False, adaln_disk_cache=False), dict(hidden_size=2),
        sampling_root=tmp_path, home=tmp_path / "owned")
    assert selected == root and result == manifest and not tables
    assert options == dict(adaln_cache=False, adaln_disk_cache=False)
    assert not report["optimization_enabled"] and not report["portable_optimization"]
    actual = load_file(str(selected / "adaln/00.safetensors"))["transformer_blocks.0.adaln_proj.linear.weight"]
    assert torch.equal(actual, tensor)
    layer = torch.nn.Linear(2, 36, bias=False)
    layer.weight = torch.nn.Parameter(actual, requires_grad=False)
    x = torch.tensor([[.25, -.5]])
    assert torch.equal(layer(x), torch.nn.functional.linear(x, tensor))
    assert all(source.digest(path) == sha for path, sha in before.items())
    assert not torch.cuda.is_initialized()


def test_missing_real_projection_reports_actual_required_bytes_not_model_name(tmp_path, official):
    root, manifest, _ = projection_cache(tmp_path)
    missing = root / "adaln/00.safetensors"
    required = missing.stat().st_size
    missing.unlink()
    with pytest.raises(FileNotFoundError, match=str(required) + " bytes in 1 missing projection files"):
        assets.local_projections(root, manifest, official)
    assert not missing.exists()


@pytest.mark.parametrize("change", ["bytes", "sha256", "escape"])
def test_actual_projection_corruption_and_escape_not_treated_as_optional_cache_miss(tmp_path, official, change):
    root, manifest, _ = projection_cache(tmp_path)
    modified = copy.deepcopy(manifest)
    row = modified["groups"][1]
    if change == "bytes":
        row["bytes"] += 1
    elif change == "sha256":
        row["sha256"] = "0" * 64
    else:
        row["file"] = "../outside.safetensors"
    with pytest.raises(ValueError):
        assets.local_projections(root, modified, official)


def test_complete_slim_view_has_actual_hardlinked_bytes_and_rechecks_them(tmp_path, official):
    root, full_manifest, _ = projection_cache(tmp_path)
    slim = dict(full_manifest, groups=full_manifest["groups"][:1], format="freevideo-fp8-slim-v1",
                adaln_sources=dict(groups=full_manifest["groups"][1:]), adaln_tables=[dict(fake="not reused")])
    original = (root / "manifest.json").read_bytes()
    view, actual = assets.dynamic_projection_view(root, slim, tmp_path / "owned", official)
    assert "format" not in actual and "adaln_tables" not in actual
    assert actual["groups"] == full_manifest["groups"]
    assert os.path.samefile(view / "adaln/00.safetensors", root / "adaln/00.safetensors")
    assert assets.dynamic_projection_view(root, slim, tmp_path / "owned", official) == (view, actual)
    assert (root / "manifest.json").read_bytes() == original
    # Editing the new *metadata* must not be mistaken for a valid old view.
    changed = dict(actual, source_id="different")
    (view / "manifest.json").write_text(json.dumps(changed), encoding="utf8")
    with pytest.raises(ValueError, match="metadata changed"):
        assets.dynamic_projection_view(root, slim, tmp_path / "owned", official)
