"""Explicit download of the pinned official source; never execute upstream setup."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from freevideo_quality.profiles import FREEVIDEO_REVISION
from freevideo_exp.runtime import canonical, write_json_new


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "T8-FreeVideo-Quality-Explicit-Preparation"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(128 * 1024**2 + 1)
    if len(data) > 128 * 1024**2:
        raise ValueError("Source response exceeds the explicit 128 MiB budget")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError("Use a new dedicated source directory; existing bytes are never replaced")
    base = "https://api.github.com/repos/FlashML-org/FreeVideo/git/"
    commit = json.loads(fetch(base + "commits/" + FREEVIDEO_REVISION))
    if commit.get("sha") != FREEVIDEO_REVISION:
        raise ValueError("Official source commit mismatch")
    tree = json.loads(fetch(base + "trees/" + commit["tree"]["sha"] + "?recursive=1"))
    if tree.get("truncated") or tree.get("sha") != commit["tree"]["sha"]:
        raise ValueError("Incomplete official source tree")
    entries = [row for row in tree["tree"] if row["type"] == "blob" and
               (row["path"].startswith("freevideo_engine/") or row["path"].startswith(("LICENSE", "NOTICE", "THIRD_PARTY")))]
    archive = zipfile.ZipFile(io.BytesIO(fetch("https://codeload.github.com/FlashML-org/FreeVideo/zip/" + FREEVIDEO_REVISION)))
    prefix = "FreeVideo-" + FREEVIDEO_REVISION + "/"
    checked, archive_exports = [], []
    for row in entries:
        name = row["path"]
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or row["mode"] not in ("100644", "100755"):
            raise ValueError("Unsupported official source path/type")
        data = archive.read(prefix + name)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != row["sha"]:
            # Git archives may apply .gitattributes export transformations.
            # Read the exact immutable blob, never accept arbitrary normalized code.
            original = json.loads(fetch(base + "blobs/" + row["sha"]))
            if original.get("sha") != row["sha"] or original.get("encoding") != "base64":
                raise ValueError("Official source blob lookup mismatch: " + name)
            data = base64.b64decode(original["content"])
            if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != row["sha"]:
                raise ValueError("Official blob payload mismatch: " + name)
            archive_exports.append(name)
        checked.append((row, data))
    target.mkdir(parents=True, exist_ok=False)
    inventory = []
    for row, data in checked:
        path = target / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        inventory.append(dict(path=str(path), file=row["path"], bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), kind="source", git_blob=row["sha"]))
    proof = dict(schema="t8-freevideo-quality-source-v1", commit=FREEVIDEO_REVISION,
                 official_tree=tree["sha"], files=inventory, source_archive="codeload.github.com",
                 archive_export_paths_replaced_with_exact_Git_blobs=archive_exports, executed_setup=False)
    write_json_new(target / "t8-source-proof.json", proof)
    print(canonical(dict(source_root=str(target), commit=FREEVIDEO_REVISION, files=len(inventory), executed=False)))


if __name__ == "__main__":
    main()
