from types import SimpleNamespace

import pytest

from tools import run_community_update_real_validation as validation


def test_installed_repo_finds_actual_core_without_fixed_parent_depth(tmp_path, monkeypatch):
    core = tmp_path / "ComfyUI"
    (core / "comfy").mkdir(parents=True)
    (core / "main.py").write_text("", "utf8")
    tool = core / "custom_nodes" / "nested" / "repo" / "tools" / "probe.py"
    tool.parent.mkdir(parents=True)
    monkeypatch.setattr(validation, "__file__", str(tool))
    assert validation._default_comfy_root() == core


def test_standalone_checkout_uses_resolved_core_module_without_importing_it(tmp_path, monkeypatch):
    core = tmp_path / "core"
    (core / "comfy").mkdir(parents=True)
    (core / "main.py").write_text("", "utf8")
    monkeypatch.setattr(validation, "__file__", str(tmp_path / "repo" / "tools" / "probe.py"))
    monkeypatch.setattr(validation.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(core / "folder_paths.py")))
    assert validation._default_comfy_root() == core


def test_unconfigured_checkout_parses_help_and_explicit_root_but_execution_explains_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "__file__", str(tmp_path / "repo" / "tools" / "probe.py"))
    monkeypatch.setattr(validation.importlib.util, "find_spec", lambda _: None)
    assert validation._parser().parse_args(["--mode", "chunked_two_pass_global_noise"]).comfy_root is None
    assert validation._parser().parse_args(["--mode", "chunked_two_pass_global_noise", "--comfy-root", str(tmp_path)]).comfy_root == tmp_path
    with pytest.raises(SystemExit) as error:
        validation.main(["--mode", "chunked_two_pass_global_noise"])
    assert error.value.code == 2
