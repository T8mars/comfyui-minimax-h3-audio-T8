from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import prepare_release_version as release_version_tool
from tools.prepare_release_version import (
    apply_version_update,
    bumped_version,
    prepare_version_update,
    read_version_state,
)


def _project(tmp_path: Path, version: str = "1.45.0") -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "demo"\nversion = "{version}"\n', encoding="utf-8"
    )
    (root / "meta.json").write_text(
        json.dumps({"name": "demo", "version": version}, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"# Demo\n\n当前版本：**{version}**\n\n历史版本：1.2.3\n",
        encoding="utf-8",
    )
    (root / "README_EN.md").write_text(
        f"# Demo\n\nCurrent version: **{version}**\n\nPrevious version: 1.2.3\n",
        encoding="utf-8",
    )
    return root


def test_current_project_release_metadata_is_consistent():
    root = Path(__file__).resolve().parents[1]
    state = read_version_state(root)
    assert len(state["version"].split(".")) == 3
    assert all(part.isdigit() for part in state["version"].split("."))
    assert len(set(state["versions"].values())) == 1


def test_registry_action_is_gated_by_the_version_file_and_main_branch():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "publish_action.yml").read_text(
        encoding="utf-8"
    )
    assert '      - "pyproject.toml"' in workflow
    assert "      - main" in workflow
    version_check = "python3 tools/prepare_release_version.py --check"
    assert version_check in workflow
    assert "Comfy-Org/publish-node-action@" in workflow
    assert "secrets.REGISTRY_ACCESS_TOKEN" in workflow
    assert workflow.index(version_check) < workflow.index("Comfy-Org/publish-node-action@")


@pytest.mark.parametrize(
    ("level", "expected"),
    [("patch", "1.45.1"), ("minor", "1.46.0"), ("major", "2.0.0")],
)
def test_semver_bumps_reset_lower_components(level, expected):
    assert bumped_version("1.45.0", level) == expected


def test_dry_run_prepares_only_the_four_current_release_markers(tmp_path):
    root = _project(tmp_path)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    plan = prepare_version_update(root, bump="minor")

    assert plan["current_version"] == "1.45.0"
    assert plan["target_version"] == "1.46.0"
    assert set(plan["files"]) == {
        "pyproject.toml",
        "meta.json",
        "README.md",
        "README_EN.md",
    }
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert "历史版本：1.2.3" in plan["updated_texts"]["README.md"]


def test_apply_atomically_synchronizes_metadata_without_git_side_effects(tmp_path):
    root = _project(tmp_path)
    result = apply_version_update(prepare_version_update(root, bump="patch"))

    assert read_version_state(root)["version"] == "1.45.1"
    assert result["previous_version"] == "1.45.0"
    assert result["version"] == "1.45.1"
    assert result["git_commit_created"] is False
    assert result["git_push_started"] is False
    assert result["registry_publish_started"] is False
    assert set(result["sha256"]) == {
        "pyproject.toml",
        "meta.json",
        "README.md",
        "README_EN.md",
    }
    assert "历史版本：1.2.3" in (root / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("target", ["1.45.0", "1.44.9", "v1.46.0", "1.46"])
def test_same_downgrade_and_non_strict_versions_are_rejected(tmp_path, target):
    root = _project(tmp_path)
    with pytest.raises(ValueError):
        prepare_version_update(root, set_version=target)


def test_inconsistent_metadata_fails_closed_before_writing(tmp_path):
    root = _project(tmp_path)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    meta["version"] = "1.45.1"
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(ValueError, match="versions disagree"):
        prepare_version_update(root, bump="patch")

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_partial_write_failure_restores_all_original_metadata(monkeypatch, tmp_path):
    root = _project(tmp_path)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    real_write = release_version_tool._atomic_write_text
    calls = 0

    def fail_second_write(path, text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated metadata write failure")
        return real_write(path, text)

    monkeypatch.setattr(release_version_tool, "_atomic_write_text", fail_second_write)

    with pytest.raises(RuntimeError, match="original files were restored"):
        apply_version_update(prepare_version_update(root, bump="patch"))

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_readme_current_marker_must_be_unique(tmp_path):
    root = _project(tmp_path)
    readme = (root / "README.md").read_text(encoding="utf-8")
    (root / "README.md").write_text(
        readme + "\n当前版本：**1.45.0**\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one"):
        read_version_state(root)


def test_english_readme_current_marker_must_be_unique(tmp_path):
    root = _project(tmp_path)
    readme = (root / "README_EN.md").read_text(encoding="utf-8")
    (root / "README_EN.md").write_text(
        readme + "\nCurrent version: **1.45.0**\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one"):
        read_version_state(root)
