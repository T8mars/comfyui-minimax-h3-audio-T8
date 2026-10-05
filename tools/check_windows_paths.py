"""Reject Windows-hostile repository paths without requiring long-path settings.

Check the Git index in CI; --working-tree includes nonignored development files
and excludes deleted pre-rename paths without changing the user's index.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_REPO_PATH = 140
MAX_WORKFLOW_FILENAME = 96
RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
FORBIDDEN = re.compile(r'[<>:"|?*\x00-\x1f]')


def windows_length(value: str) -> int:
    """Win32 measures UTF-16 code units, not UTF-8 bytes or Python characters."""
    return len(value.encode("utf-16-le")) // 2


def validate_paths(paths) -> list[str]:
    errors = []
    seen = {}
    for raw in paths:
        name = str(raw).replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or not name or any(part in {"", ".", ".."} for part in name.split("/")):
            errors.append(f"Unsafe repository-relative path: {name}")
            continue
        length = windows_length(name)
        if length > MAX_REPO_PATH:
            errors.append(f"Path {length} > {MAX_REPO_PATH}: {name}")
        if name.startswith("examples/workflows/") and path.suffix.lower() == ".json":
            length = windows_length(path.name)
            if length > MAX_WORKFLOW_FILENAME:
                errors.append(f"Workflow filename {length} > {MAX_WORKFLOW_FILENAME}: {name}")
        for part in path.parts:
            if FORBIDDEN.search(part) or RESERVED.match(part) or part.endswith((".", " ")):
                errors.append(f"Windows-invalid component: {name}")
                break
        folded = name.casefold()
        if folded in seen and seen[folded] != name:
            errors.append(f"Windows case collision: {seen[folded]} / {name}")
        seen[folded] = name
    return errors


def git_paths(root: Path, *, working_tree: bool = False) -> list[str]:
    command = ["git", "-C", str(root), "ls-files", "-z"]
    if working_tree:
        command.extend(["--cached", "--others", "--exclude-standard"])
    result = subprocess.run(command, capture_output=True, check=True)
    paths = result.stdout.decode("utf-8").split("\0")
    return sorted({path for path in paths if path and (not working_tree or (root / path).exists())})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--working-tree", action="store_true")
    parser.add_argument("--zip", type=Path, help="Check actual archive member names, without extracting")
    options = parser.parse_args(argv)
    if options.zip:
        with zipfile.ZipFile(options.zip) as archive:
            paths = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
    else:
        paths = git_paths(options.root, working_tree=options.working_tree)
    errors = validate_paths(paths)
    if errors:
        print("\n".join(errors))
        return 1
    longest = max((windows_length(path) for path in paths), default=0)
    print(f"Windows paths: {len(paths)} files; longest {longest}/{MAX_REPO_PATH}; workflow filenames <= {MAX_WORKFLOW_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
