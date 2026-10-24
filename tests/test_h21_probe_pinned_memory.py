"""Pure metadata/real argparse checks, not a Core/model/GPU preparation test.

Run with --noconftest and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. No Torch import,
server, weight loading, launcher mutation or sampling is performed here.
"""
from copy import deepcopy
import builtins
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(os.environ.get("T8_H21_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location(
    "h21_pinned_memory_controller", ROOT / "tools/h21_control_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
raw = probe._raw_api_helper()
MISSING = object()
BAD = [None, 0, 1, "false", "true", [], {}]


def bare_plan(tmp_path, value=MISSING):
    value_plan = dict(run_root=str(tmp_path / "owned"), core_root=str(tmp_path / "core"),
                      python=sys.executable, port=49157, package_name="only_owned_package",
                      identity_extension="only_owned_identity", reserve_vram_gb=5)
    if value is not MISSING:
        value_plan["disable_pinned_memory"] = value
    return value_plan


def original_default_command(plan, cpu=False):
    root = Path(plan["run_root"])
    command = [plan["python"], "-B", "-X", "utf8", str(root / "runtime_tools/h21_control_probe.py"),
               "--_serve", str(root / "manifest.json"), "--", "--listen", "127.0.0.1", "--port",
               str(plan["port"]), "--disable-auto-launch", "--preview-method", "none", "--cache-none",
               "--disable-comfy-compiler", "--base-directory", str(root), "--models-directory", str(root / "models"),
               "--disable-all-custom-nodes", "--whitelist-custom-nodes", plan["package_name"],
               plan["identity_extension"], "--extra-model-paths-config", str(root / "paths.json"),
               "--input-directory", str(root / "input"), "--output-directory", str(root / "output"),
               "--temp-directory", str(root / "temp"), "--user-directory", str(root / "user"),
               "--database-url", "sqlite:///:memory:"]
    if cpu:
        command.append("--cpu")
    if plan.get("reserve_vram_gb") is not None:
        command.extend(["--reserve-vram", str(plan["reserve_vram_gb"])])
    return command


def raw_config(value=MISSING):
    value_config = dict(mode="raw_api", schema="t8.h21.raw_api_config.v1", assets={},
        raw_api_cases=[dict(name="same_original_graph", api="owned.api.json", api_sha256="a" * 64)],
        models=[dict(node="10", input="unet_name", folder="diffusion_models", name="selected.safetensors")],
        model_directories={})
    if value is not MISSING:
        value_config["disable_pinned_memory"] = value
    return value_config


def metadata_manifest(tmp_path, plan_value=MISSING, config_value=MISSING):
    """Minimal actual file/hash fixture; no claim that a Core graph was prepared."""
    plan = bare_plan(tmp_path, plan_value)
    root = Path(plan["run_root"])
    root.mkdir()
    for name in ("runtime_nodes", "runtime_tools"):
        (root / name).mkdir()
    declared = {} if config_value is MISSING else {"disable_pinned_memory": config_value}
    probe.write_json(root / "config.json", declared)
    plan.update(schema=probe.SCHEMA, resources=dict(models={}), core_files={},
                python_identity=probe.file_identity(sys.executable),
                sealed_files={"config.json": probe.file_identity(root / "config.json")})
    plan["server_argv"] = probe.server_command(plan)
    path = root / "manifest.json"
    probe.write_json(path, plan)
    return path, plan


@pytest.mark.parametrize("value", [MISSING, False])
@pytest.mark.parametrize("cpu", [False, True])
def test_missing_and_false_keep_every_original_default_argument(tmp_path, value, cpu):
    plan = bare_plan(tmp_path, value)
    original = deepcopy(plan)
    assert probe.server_command(plan, cpu=cpu) == original_default_command(plan, cpu=cpu)
    assert plan == original


@pytest.mark.parametrize("cpu", [False, True])
def test_explicit_true_appends_exactly_one_owned_Core_flag(tmp_path, cpu):
    plan = bare_plan(tmp_path, True)
    original = deepcopy(plan)
    command = probe.server_command(plan, cpu=cpu)
    assert command == original_default_command(plan, cpu=cpu) + ["--disable-pinned-memory"]
    assert command.count("--disable-pinned-memory") == 1 and plan == original


@pytest.mark.parametrize("bad", BAD)
def test_plan_boolean_is_strict_not_truthiness(tmp_path, bad):
    with pytest.raises(ValueError, match="disable_pinned_memory.*bool"):
        probe.server_command(bare_plan(tmp_path, bad))


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_raw_config_accepts_only_the_new_explicit_boolean_without_dropping_it(value):
    config = raw_config(value)
    before = deepcopy(config)
    raw.validate_config(config)
    assert config == before


@pytest.mark.parametrize("bad", BAD)
def test_raw_config_bad_boolean_reports_the_actual_field(bad):
    with pytest.raises(ValueError, match="disable_pinned_memory.*bool"):
        raw.validate_config(raw_config(bad))


@pytest.mark.parametrize("mode", ["director", "raw_api"])
def test_prepare_rejects_bad_boolean_before_socket_freeze_or_publication(tmp_path, monkeypatch, mode):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(mode=mode, disable_pinned_memory="false")), encoding="utf8")
    core = tmp_path / "core"
    core.mkdir()
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("invalid boolean reached socket/resource preparation")
    monkeypatch.setattr(probe.socket, "socket", forbidden_socket)
    with pytest.raises(ValueError, match="disable_pinned_memory.*bool"):
        probe.prepare(config, tmp_path / "must-not-exist", core)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_same_real_config_and_plan_value_keeps_frozen_command(tmp_path, value):
    manifest, plan = metadata_manifest(tmp_path, value, value)
    assert probe.verify_prepared(manifest) == plan
    if value is MISSING:
        assert "disable_pinned_memory" not in plan
    else:
        assert plan["disable_pinned_memory"] is value


@pytest.mark.parametrize("plan_value,config_value", [
    (True, False), (False, True), (MISSING, True), (True, MISSING),
    (MISSING, False), (False, MISSING),
])
def test_plan_and_sealed_config_presence_and_value_cannot_disagree(tmp_path, plan_value, config_value):
    manifest, _ = metadata_manifest(tmp_path, plan_value, config_value)
    with pytest.raises(RuntimeError, match="disable_pinned_memory.*config|config.*disable_pinned_memory"):
        probe.verify_prepared(manifest)


@pytest.mark.parametrize("tamper", ["remove", "duplicate"])
def test_actual_frozen_argv_cannot_remove_or_duplicate_the_requested_flag(tmp_path, tamper):
    manifest, plan = metadata_manifest(tmp_path, True, True)
    if tamper == "remove":
        plan["server_argv"] = [item for item in plan["server_argv"] if item != "--disable-pinned-memory"]
    else:
        plan["server_argv"].append("--disable-pinned-memory")
    manifest.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(RuntimeError, match="command"):
        probe.verify_prepared(manifest)


@pytest.mark.parametrize("requested,actual_count", [(True, 0), (True, 2), (False, 1), (MISSING, 1)])
def test_serve_checks_actual_flag_count_before_any_Core_import(tmp_path, monkeypatch, requested, actual_count):
    plan = bare_plan(tmp_path, requested)
    monkeypatch.setattr(probe, "verify_prepared", lambda path: plan)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    original_import = builtins.__import__
    touched = []
    def watched_import(name, *args, **kwargs):
        if name == "comfy" or name.startswith(("comfy.", "utils.", "folder_paths")):
            touched.append(name)
            raise AssertionError("Core imports happened before actual owned argument validation")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", watched_import)
    with pytest.raises(RuntimeError, match="disable_pinned_memory"):
        probe._serve("metadata-only.json", ["--disable-pinned-memory"] * actual_count)
    assert touched == []


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_actual_Core_argparse_accepts_only_the_selected_flag_without_Torch(tmp_path, value):
    core = Path(os.environ.get("T8_H21_CORE") or next(p for p in ROOT.parents if (p / 'comfy/cli_args.py').is_file()))
    assert (core / "comfy/cli_args.py").is_file(), "actual Core cli_args is required for this narrow parser contract"
    plan = bare_plan(tmp_path, value)
    owned = Path(plan["run_root"])
    for name in ("models", "input", "output", "temp", "user"):
        (owned / name).mkdir(parents=True)
    (owned / "paths.json").write_text("{}", encoding="utf8")
    command = probe.server_command(plan)
    arguments = command[command.index("--") + 1:]
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from comfy.cli_args import parser
actual = parser.parse_args(json.loads(sys.argv[2]))
assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
print(json.dumps(dict(disable_pinned_memory=actual.disable_pinned_memory, Torch_imported=False)))
"""
    child = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-c", script, str(core), json.dumps(arguments)],
                           capture_output=True, text=True, encoding="utf8", check=False, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(child.stdout)
    assert actual == dict(disable_pinned_memory=value is True, Torch_imported=False)


@pytest.mark.parametrize("value", [MISSING, False, True])
def test_actual_prepare_persists_only_explicit_choice_with_synthetic_build_result(tmp_path, monkeypatch, value):
    """Exercise real prepare I/O, not a claim of Core/graph/model validation."""
    source, core, owned = (tmp_path / name for name in ("source", "core", "owned"))
    source.mkdir()
    core.mkdir()
    (source / "__init__.py").write_text("# metadata-only fixture, never imported\n", encoding="utf8")
    (source / "tools").mkdir()
    for name in probe.TOOLS:
        shutil.copyfile(ROOT / "tools" / name, source / "tools" / name)
    project = tmp_path / "project.json"
    project.write_text(json.dumps(dict(assets=[], doc=dict(generation={
        key: "explicit-graph-only-selection" for key in ("unet", "clip", "video_vae", "audio_vae")
    }))), encoding="utf8")
    config = dict(project=str(project), assets={}, cases=[dict(name="metadata_only")],
                  model_directories={name: [] for name in probe.MODEL_FOLDERS})
    if value is not MISSING:
        config["disable_pinned_memory"] = value
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf8")
    monkeypatch.setattr(probe, "runtime_sources", lambda selected: [source / "__init__.py"])
    monkeypatch.setattr(probe, "core_identity", lambda selected: {})
    seen = []
    def synthetic_build(command, **kwargs):
        assert command[-2] == "--_build" and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        captured = probe.read_json(owned / "prepare-plan.json")
        seen.append(captured)
        probe.write_json(owned / "build-result.json", dict(resources=dict(models={}), cases=[]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe.subprocess, "run", synthetic_build)
    result = probe.prepare(config_path, owned, core, source_root=source)
    manifest = probe.read_json(owned / "manifest.json")
    assert result == manifest and len(seen) == 1
    for plan in (seen[0], manifest):
        if value is MISSING:
            assert "disable_pinned_memory" not in plan
        else:
            assert plan["disable_pinned_memory"] is value
    assert probe.read_json(owned / "config.json") == config
    assert manifest["server_argv"] == original_default_command(manifest) + (
        ["--disable-pinned-memory"] if value is True else []
    )


@pytest.mark.parametrize("side", ["plan", "config"])
@pytest.mark.parametrize("bad", BAD)
def test_resealed_nonboolean_choice_still_cannot_authenticate(tmp_path, side, bad):
    manifest, plan = metadata_manifest(tmp_path, True, True)
    if side == "plan":
        plan["disable_pinned_memory"] = bad
    else:
        config_path = Path(plan["run_root"]) / "config.json"
        config_path.write_text(json.dumps(dict(disable_pinned_memory=bad)), encoding="utf8")
        plan["sealed_files"]["config.json"] = probe.file_identity(config_path)
    manifest.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(RuntimeError, match="disable_pinned_memory.*config"):
        probe.verify_prepared(manifest)


@pytest.mark.parametrize("damage", ["unseal", "remove"])
def test_explicit_option_requires_its_actual_sealed_config_file(tmp_path, damage):
    manifest, plan = metadata_manifest(tmp_path, True, True)
    if damage == "unseal":
        plan["sealed_files"].pop("config.json")
    else:
        (Path(plan["run_root"]) / "config.json").unlink()
        plan["sealed_files"].pop("config.json")
    manifest.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(RuntimeError, match="disable_pinned_memory.*config"):
        probe.verify_prepared(manifest)


@pytest.mark.parametrize("value,arguments", [
    (MISSING, ["--disable-pinned-mem"]),
    (False, ["--disable-pinned"]),
    (True, ["--disable-pinned-memory", "--disable-pinned-mem"]),
])
def test_actual_serve_abbreviations_cannot_bypass_real_sealed_metadata(tmp_path, monkeypatch, value, arguments):
    manifest, _ = metadata_manifest(tmp_path, value, value)
    original_import = builtins.__import__
    touched = []
    def forbid_Core(name, *args, **kwargs):
        if name == "comfy" or name.startswith(("comfy.", "utils.", "folder_paths")):
            touched.append(name)
            raise AssertionError("abbreviated flag passed owned option binding")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbid_Core)
    with pytest.raises(RuntimeError, match="disable_pinned_memory"):
        probe._serve(manifest, arguments)
    assert touched == []


@pytest.mark.parametrize("arguments,count", [
    (["--", "--disable-pinned-memory"], 0),
    (["--disable-pinned-mem", "--", "--disable-pinned"], 1),
    (["--disable-pinned-memory=True"], 0),
    (["--"], 0),
])
def test_option_terminator_and_invalid_equals_are_not_silently_reinterpreted(arguments, count):
    assert probe._pinned_memory_flag_count(arguments) == count


@pytest.mark.parametrize("arguments,exit_code", [
    (["--disable-pinned-mem"], 0),
    (["--disable-pinned"], 0),
    (["--disable-pinned-memory", "--disable-pinned-mem"], 0),
    (["--disable-pinned-memory=True"], 2),
])
def test_actual_Core_parser_abbreviations_and_invalid_equals_without_Torch(arguments, exit_code):
    core = Path(os.environ.get("T8_H21_CORE") or next(p for p in ROOT.parents if (p / 'comfy/cli_args.py').is_file()))
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from comfy.cli_args import parser
try:
    actual = parser.parse_args(json.loads(sys.argv[2]))
except SystemExit:
    assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
    raise
assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
print(json.dumps(dict(disable_pinned_memory=actual.disable_pinned_memory, Torch_imported=False)))
"""
    child = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-c", script, str(core), json.dumps(arguments)],
                           capture_output=True, text=True, encoding="utf8", check=False, timeout=30)
    assert child.returncode == exit_code, child.stdout + child.stderr
    if exit_code == 0:
        assert json.loads(child.stdout) == dict(disable_pinned_memory=True, Torch_imported=False)
    else:
        assert "ignored explicit argument" in child.stderr
