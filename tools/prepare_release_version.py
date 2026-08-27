from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PYPROJECT_LINE_RE = re.compile(r'(?m)^(version\s*=\s*")(\d+\.\d+\.\d+)("\s*)$')
META_LINE_RE = re.compile(
    r'(?m)^(\s*"version"\s*:\s*")(\d+\.\d+\.\d+)("\s*,?\s*)$'
)
README_CURRENT_RE = re.compile(r"(当前版本：\*\*)(\d+\.\d+\.\d+)(\*\*)")
README_EN_CURRENT_RE = re.compile(r"(Current version: \*\*)(\d+\.\d+\.\d+)(\*\*)")


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = VERSION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"version must be strict SemVer X.Y.Z, got {value!r}")
    return tuple(int(part) for part in match.groups())


def _read_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read UTF-8 release metadata: {path}") from error


def read_version_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    pyproject_path = root / "pyproject.toml"
    meta_path = root / "meta.json"
    readme_path = root / "README.md"
    readme_en_path = root / "README_EN.md"
    pyproject_text = _read_text(pyproject_path)
    meta_text = _read_text(meta_path)
    readme_text = _read_text(readme_path)
    readme_en_text = _read_text(readme_en_path)

    try:
        pyproject_data = tomllib.loads(pyproject_text)
        pyproject_version = str(pyproject_data["project"]["version"])
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("pyproject.toml has no valid [project].version") from error
    try:
        meta_version = str(json.loads(meta_text)["version"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("meta.json has no valid version") from error
    readme_match = README_CURRENT_RE.search(readme_text)
    if readme_match is None:
        raise ValueError("README.md has no unique current-version marker")
    if len(README_CURRENT_RE.findall(readme_text)) != 1:
        raise ValueError("README.md must contain exactly one current-version marker")
    readme_version = readme_match.group(2)
    readme_en_match = README_EN_CURRENT_RE.search(readme_en_text)
    if readme_en_match is None:
        raise ValueError("README_EN.md has no unique current-version marker")
    if len(README_EN_CURRENT_RE.findall(readme_en_text)) != 1:
        raise ValueError("README_EN.md must contain exactly one current-version marker")
    readme_en_version = readme_en_match.group(2)

    versions = {
        "pyproject.toml": pyproject_version,
        "meta.json": meta_version,
        "README.md": readme_version,
        "README_EN.md": readme_en_version,
    }
    for version in versions.values():
        _version_tuple(version)
    unique = set(versions.values())
    if len(unique) != 1:
        raise ValueError(f"release metadata versions disagree: {versions}")
    return {
        "project_root": str(root),
        "version": pyproject_version,
        "versions": versions,
        "texts": {
            "pyproject.toml": pyproject_text,
            "meta.json": meta_text,
            "README.md": readme_text,
            "README_EN.md": readme_en_text,
        },
    }


def bumped_version(current: str, level: str) -> str:
    major, minor, patch = _version_tuple(current)
    if level == "patch":
        patch += 1
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError("bump level must be patch, minor or major")
    return f"{major}.{minor}.{patch}"


def _replace_one(pattern: re.Pattern[str], text: str, target: str, label: str) -> str:
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{target}{match.group(3)}", text)
    if count != 1:
        raise ValueError(f"{label} version marker count must be exactly one, got {count}")
    return updated


def prepare_version_update(
    project_root: Path,
    *,
    bump: str | None = None,
    set_version: str | None = None,
) -> dict[str, Any]:
    if (bump is None) == (set_version is None):
        raise ValueError("choose exactly one of bump or set_version")
    state = read_version_state(project_root)
    current = state["version"]
    target = bumped_version(current, bump) if bump is not None else str(set_version)
    _version_tuple(target)
    if _version_tuple(target) <= _version_tuple(current):
        raise ValueError(f"target version {target} must be newer than current version {current}")

    texts = state["texts"]
    updated = {
        "pyproject.toml": _replace_one(
            PYPROJECT_LINE_RE, texts["pyproject.toml"], target, "pyproject.toml"
        ),
        "meta.json": _replace_one(META_LINE_RE, texts["meta.json"], target, "meta.json"),
        "README.md": _replace_one(
            README_CURRENT_RE, texts["README.md"], target, "README.md"
        ),
        "README_EN.md": _replace_one(
            README_EN_CURRENT_RE, texts["README_EN.md"], target, "README_EN.md"
        ),
    }
    return {
        "schema": "t8.minimax_h3.release_version_plan.v1",
        "project_root": state["project_root"],
        "current_version": current,
        "target_version": target,
        "bump": bump,
        "files": list(updated),
        "updated_texts": updated,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_version_update(plan: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(plan["project_root"])).resolve()
    target = str(plan["target_version"])
    updated_texts = plan.get("updated_texts")
    if not isinstance(updated_texts, dict) or set(updated_texts) != {
        "pyproject.toml",
        "meta.json",
        "README.md",
        "README_EN.md",
    }:
        raise ValueError("release version plan has an invalid file set")
    order = ("pyproject.toml", "meta.json", "README.md", "README_EN.md")
    originals = {relative: _read_text(root / relative) for relative in order}
    try:
        for relative in order:
            _atomic_write_text(root / relative, str(updated_texts[relative]))
        verified = read_version_state(root)
        if verified["version"] != target:
            raise RuntimeError("release version verification failed after atomic update")
    except Exception as error:
        rollback_errors = []
        for relative in order:
            try:
                _atomic_write_text(root / relative, originals[relative])
            except Exception as rollback_error:
                rollback_errors.append(f"{relative}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "release metadata update failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise RuntimeError("release metadata update failed; original files were restored") from error
    hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in updated_texts
    }
    return {
        "schema": "t8.minimax_h3.release_version_result.v1",
        "status": "updated",
        "previous_version": plan["current_version"],
        "version": target,
        "files": list(updated_texts),
        "sha256": hashes,
        "git_commit_created": False,
        "git_push_started": False,
        "registry_publish_started": False,
    }


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "updated_texts"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or atomically synchronize pyproject.toml, meta.json, README.md "
            "and README_EN.md before a GitHub push. This tool never commits, pushes "
            "or publishes."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--bump", choices=("patch", "minor", "major"))
    action.add_argument("--set-version")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the four metadata files atomically; otherwise print a dry-run plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if args.apply:
            raise ValueError("--apply cannot be combined with --check")
        state = read_version_state(args.project_root)
        output = {
            "schema": "t8.minimax_h3.release_version_check.v1",
            "status": "consistent",
            "project_root": state["project_root"],
            "version": state["version"],
            "versions": state["versions"],
        }
    else:
        plan = prepare_version_update(
            args.project_root,
            bump=args.bump,
            set_version=args.set_version,
        )
        output = apply_version_update(plan) if args.apply else {
            **_public_plan(plan),
            "status": "dry_run",
            "git_commit_created": False,
            "git_push_started": False,
            "registry_publish_started": False,
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
