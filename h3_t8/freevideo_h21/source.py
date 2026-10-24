"""Read-only fixed-source binding, without executing the engine initializer."""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path, PurePosixPath
import sys
import types

REVISION = "f2c25e6e4f82d5ad9ed966998423762503d98e5c"
CONTENT_PIN = "d2377a3744b92f98914b97fcb1ebdb4a476c1f8be774f252834c7d0b5c0f14fc"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or PurePosixPath(relative).is_absolute() or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("Invalid fixed-source relative path")
    root = Path(root).resolve(strict=True)
    result = root.joinpath(*relative.split("/")).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("Fixed source escapes its selected root")
    return result


class Source:
    """Exact Git blob inventory; no checkout, installer, download or default mutation.

    A unique package name bypasses upstream __init__'s process/environment setup.
    Only the owned worker imports the real freevideo_engine initializer.
    """
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.proof_path = self.root / "t8-source-proof.json"
        self.proof_sha256 = digest(self.proof_path)
        proof = json.loads(self.proof_path.read_text(encoding="utf8"))
        if proof.get("commit") != REVISION or not isinstance(proof.get("files"), list) or not proof["files"]:
            raise ValueError("H21 requires its explicit fixed-source proof, not an older producer")
        self.files = []
        seen = set()
        for row in proof["files"]:
            path = contained(self.root, row["file"])
            if str(path) in seen:
                raise ValueError("Duplicate fixed-source inventory path")
            seen.add(str(path))
            data = path.read_bytes()
            if (len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]
                    or hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != row["git_blob"]):
                raise ValueError("Fixed source differs from its actual Git blob: " + row["file"])
            self.files.append(dict(path=str(path), bytes=len(data), sha256=row["sha256"], kind="source"))
        # Portable exact content pin excludes the preparer's absolute paths.
        # Merely writing a self-consistent JSON with a commit label is not
        # evidence that executable bytes came from that fixed Git tree.
        content = "".join(f"{r['file']}\t{r['bytes']}\t{r['sha256']}\t{r['git_blob']}\n" for r in proof["files"])
        if hashlib.sha256(content.encode()).hexdigest() != CONTENT_PIN:
            raise ValueError("Source inventory is not the fixed H21 Git content set")
        self.identity = hashlib.sha256(canonical(self.files).encode()).hexdigest()
        self.package_name = "_t8_h21_freevideo_" + self.identity

    def verify(self):
        if digest(self.proof_path) != self.proof_sha256:
            raise ValueError("Fixed-source proof changed")
        for row in self.files:
            if Path(row["path"]).stat().st_size != row["bytes"] or digest(row["path"]) != row["sha256"]:
                raise ValueError("Fixed-source file changed: " + row["path"])

    def module(self, name):
        # Relative imports resolve only within this proven source. No global
        # freevideo_engine registration and no upstream initializer side effects.
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Expected an engine module name, not an import expression")
        module_path = contained(self.root, "freevideo_engine/" + name + ".py")
        expected = next((r for r in self.files if r["path"] == str(module_path)), None)
        if expected is None or digest(module_path) != expected["sha256"]:
            raise ValueError("Engine module is not in its fixed Git inventory")
        if self.package_name not in sys.modules:
            package = types.ModuleType(self.package_name)
            package.__path__ = [str(self.root / "freevideo_engine")]
            package.__package__ = self.package_name
            package.__spec__ = importlib.machinery.ModuleSpec(self.package_name, loader=None, is_package=True)
            sys.modules[self.package_name] = package
        value = importlib.import_module(self.package_name + "." + name)
        if Path(value.__file__).resolve() != module_path:
            raise ValueError("A foreign module replaced the fixed-source binding")
        return value
