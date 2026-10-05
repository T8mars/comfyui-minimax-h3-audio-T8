"""Small dependency-free path regression suite; no Core, models or CUDA."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_windows_paths import git_paths, validate_paths, windows_length  # noqa: E402
from tools.workflow_paths import legacy_workflow_path, public_workflow_path, rename_rows, workflow_sha256  # noqa: E402


class WindowsPathsTests(unittest.TestCase):
    def test_repository_path_boundary(self):
        self.assertEqual(validate_paths(["a" * 136 + ".txt"]), [])
        self.assertTrue(validate_paths(["a" * 137 + ".txt"]))

    def test_workflow_filename_boundary_even_in_a_short_directory(self):
        prefix = "examples/workflows/"
        self.assertEqual(validate_paths([prefix + "a" * 91 + ".json"]), [])
        self.assertTrue(validate_paths([prefix + "a" * 92 + ".json"]))

    def test_budget_is_not_utf8_bytes_or_codepoint_count(self):
        self.assertEqual(windows_length("中"), 1)
        self.assertEqual(windows_length("😀"), 2)
        self.assertTrue(validate_paths(["😀" * 69 + ".txt"]))

    def test_windows_invalid_names_and_collisions(self):
        for name in ("docs/CON.txt", "docs/aux.md", "a/../b", "/abs", "a.", "a ", "a/b:c"):
            with self.subTest(name=name):
                self.assertTrue(validate_paths([name]))
        self.assertTrue(validate_paths(["docs/Name.md", "docs/name.md"]))

    def test_current_working_tree_path_budget(self):
        self.assertEqual(validate_paths(git_paths(ROOT, working_tree=True)), [])

    def test_all_public_workflow_names_are_short(self):
        paths = [path.relative_to(ROOT).as_posix() for path in (ROOT / "examples/workflows").rglob("*.json")]
        self.assertEqual(validate_paths(paths), [])

    def test_renamed_examples_are_byte_exact_and_old_aliases_are_not_shipped(self):
        rows = rename_rows()
        self.assertEqual(len(rows), 37)
        self.assertEqual(validate_paths([row["new_path"] for row in rows]), [])
        for row in rows:
            with self.subTest(path=row["new_path"]):
                old, new = ROOT / row["old_path"], ROOT / row["new_path"]
                self.assertFalse(old.exists())
                self.assertEqual(workflow_sha256(new.read_bytes()), row["sha256"])
                self.assertEqual(public_workflow_path(old), new)
                self.assertEqual(legacy_workflow_path(new), old)
                self.assertEqual(public_workflow_path(new), new)
                self.assertEqual(public_workflow_path(row["old_path"]), Path(row["new_path"]))

    def test_unknown_or_another_project_path_is_not_silently_rewritten(self):
        unknown = Path("examples/workflows/new_" + "x" * 160 + ".json")
        self.assertEqual(public_workflow_path(unknown), unknown)
        with tempfile.TemporaryDirectory(prefix="t8-unowned-") as directory:
            foreign = Path(directory) / rename_rows()[0]["old_path"]
            self.assertEqual(public_workflow_path(foreign), foreign)

    @unittest.skipUnless(sys.platform == "win32", "Actual classic-limit Git behavior requires Windows")
    def test_windows_git_old_path_probe_and_short_path_checkout_update(self):
        def git(where, *args, check=True, longpaths=False):
            result = subprocess.run(
                ["git", "-c", f"core.longpaths={str(longpaths).lower()}", "-c", "core.autocrlf=false", "-c", "user.name=T8 Path Test",
                 "-c", "user.email=t8-path-test@example.invalid", "-C", str(where), *args],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if check and (result.returncode or "filename too long" in result.stderr.lower()
                          or "error: unable to create file" in result.stderr.lower()):
                self.fail(result.stdout + result.stderr)
            return result

        with tempfile.TemporaryDirectory(prefix="t8-wpath-") as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            git(source, "init", "-q")
            # Genuine old public path, rather than an unrelated invented failure.
            old_row = max(rename_rows(), key=lambda row: len(row["old_path"]))
            self.assertTrue(validate_paths([old_row["old_path"]]))
            old = source / old_row["old_path"]
            old.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / old_row["new_path"], old)
            git(source, "add", ".", longpaths=True)
            git(source, "commit", "-qm", "before short filenames", longpaths=True)
            before = git(source, "rev-parse", "HEAD").stdout.strip()
            old.unlink()
            for row in rename_rows():
                target = source / row["new_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / row["new_path"], target)
            git(source, "add", "-A", longpaths=True)
            git(source, "commit", "-qm", "short filenames", longpaths=True)

            def deep(label):
                prefix = base / label
                prefix.mkdir()
                count = 110 - windows_length(str(prefix.resolve())) - 1
                self.assertGreater(count, 0, "Temporary base unexpectedly longer than tested install root")
                return prefix / ("p" * count)

            legacy = deep("old")
            git(base, "clone", "-q", "--no-checkout", str(source), str(legacy))
            result = git(legacy, "checkout", before, check=False)
            if "filename too long" in result.stderr.lower():
                # Git 2.45 can report this error while returning zero. Inspect
                # actual files/errors instead of declaring success from exit code.
                self.assertFalse((legacy / old_row["old_path"]).exists())
                print("Reproduced old public path: Filename too long (including zero-exit checkout errors).")
            elif result.returncode:
                self.assertIn("filename too long", result.stderr.lower())
            else:
                # Current Windows/Git builds may support long paths already.
                # Do not alter system settings just to force an obsolete limit.
                self.assertEqual(workflow_sha256((legacy / old_row["old_path"]).read_bytes()), old_row["sha256"])
                print("Old over-budget path supported by this Windows/Git; not a legacy-failure reproduction.")

            fixed = deep("new")
            git(base, "clone", "-q", str(source), str(fixed))
            for row in rename_rows():
                target = fixed / row["new_path"]
                self.assertEqual(workflow_sha256(target.read_bytes()), row["sha256"])
            # A real normal fast-forward update, without global settings or long-path opt-in.
            (source / "README.md").write_text("path update test\n", encoding="utf-8")
            git(source, "add", "README.md")
            git(source, "commit", "-qm", "ordinary next update")
            git(fixed, "pull", "--ff-only")
            self.assertEqual((fixed / "README.md").read_text(encoding="utf-8"), "path update test\n")


if __name__ == "__main__":
    unittest.main()
