"""User strengths survive every entry point; finite and real errors still fail."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from safetensors.torch import save_file

from h3_audio_t8_pkg import h3_lora_compat_advanced as compat
from h3_audio_t8_pkg.director_project import ProjectStore, generation_settings, new_project
from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced as Loader
from test_h24_lora_file_alpha import check_forward, direct_pair
from test_progressive_sampling_runtime import tiny_model


@pytest.mark.parametrize("strength", (3., -3., 12., -12., 101., -101., 1e6, -1e6))
@pytest.mark.parametrize("mode", ("single", "two_pass", "hyperflow"))
def test_saved_sampling_has_no_arbitrary_strength_cap(mode, strength):
    row = {"id": "selected", "name": "user.safetensors", "strength": strength, "enabled": True}
    raw = {"mode": mode, "lora_mode": "manual", "loras": [row],
           "low_loras": [row], "high_loras": [], "hyperflow_file": "hyperflow/selected.safetensors"}
    original = deepcopy(raw)
    result = normalize_sampling(raw)
    assert result["loras" if mode == "single" else "low_loras"][0]["strength"] == strength
    assert raw == original


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("strength", (3., -3., 101., -101., 1e6))
def test_generation_project_roundtrip_keeps_selected_strength(tmp_path, legacy, strength):
    project = new_project()
    settings = project["doc"]["generation"]
    settings.update(lora_mode="manual")
    if legacy:
        settings.pop("loras", None)
        settings.update(lora="user.safetensors", lora_strength=strength)
    else:
        settings["loras"] = [{"name": "user.safetensors", "strength": strength, "enabled": True}]
    before = deepcopy(project)
    assert generation_settings(project["doc"])["loras"][0]["strength"] == strength
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    saved = store.save(project, 0)
    reopened = store.load(saved["id"])
    assert generation_settings(reopened["doc"])["loras"][0]["strength"] == strength
    assert project == before


@pytest.mark.parametrize("strength", (3., -3., 12., -12., 101., -101., 1e4))
def test_actual_core_lora_forward_uses_exact_user_strength(tmp_path, strength):
    state, delta = direct_pair()
    path = tmp_path / "user.safetensors"
    save_file(state, str(path))
    base = tiny_model()
    patched, report = compat.load_minimax_h3_lora_model(base, path, strength)
    assert json.loads(report)["selected_strength"] == strength
    check_forward(patched, base, delta, strength)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf"), True, "3", None))
def test_director_still_rejects_nonfinite_or_nonnumeric_strength(value):
    row = {"name": "user.safetensors", "strength": value}
    with pytest.raises(ValueError, match="强度"):
        generation_settings({"generation": {"lora_mode": "manual", "loras": [row]}})
    with pytest.raises(ValueError, match="强度"):
        normalize_sampling({"mode": "single", "loras": [row]})


@pytest.mark.parametrize("strength", (float("nan"), float("inf"), -float("inf")))
def test_loader_rejects_nonfinite_strength_before_file_access(tmp_path, strength):
    with pytest.raises(ValueError, match="finite"):
        compat.load_minimax_h3_lora_model(tiny_model(), tmp_path / "absent.safetensors", strength)


def test_node_schema_has_no_different_strength_cap():
    field = next(item for item in Loader.define_schema().inputs if item.id == "strength_model")
    assert field.min is None and field.max is None
    assert field.default == 1. and field.step == .01


def test_actual_frontend_dialog_accepts_finite_user_strengths():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    result = subprocess.run([node, "--test", str(Path(__file__).with_suffix(".mjs"))],
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
