"""Explicit pinned official video-VAE download for the independent decoder EXP.

Never installs software or changes an existing FreeVideo runtime configuration.
Partial downloads remain evidence and are not automatically overwritten/resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request

REVISION = '5d9b308a59ab12e67147f191e184baf704185bd1'
REPOSITORY = 'MiniMaxAI/MiniMax-H3'
SHARDS = {
    'diffusion_pytorch_model-00001-of-00003.safetensors': (5061033024, '72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374'),
    'diffusion_pytorch_model-00002-of-00003.safetensors': (4955986528, '2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198'),
    'diffusion_pytorch_model-00003-of-00003.safetensors': (398539336, 'c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39'),
}
NAMES = {*SHARDS, 'config.json', 'diffusion_pytorch_model.safetensors.index.json'}


def fetch(url, limit):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'T8-FreeVideo-Decoder-Explicit-Assets'}), timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Bounded metadata response is too large')
    return data


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_listing(rows):
    expected = {'vae/' + name for name in NAMES}
    if not isinstance(rows, list) or {row.get('path') for row in rows} != expected:
        raise ValueError('Unexpected official video-VAE file listing')
    for row in rows:
        name = PurePosixPath(row['path']).name
        if row.get('type') != 'file' or type(row.get('size')) is not int:
            raise ValueError('Invalid official file descriptor')
        if name in SHARDS:
            size, sha = SHARDS[name]
            if row['size'] != size or row.get('lfs', {}).get('oid') != sha:
                raise ValueError('Official immutable shard identity mismatch')
        elif not 0 < row['size'] <= 1024**2 or not isinstance(row.get('oid'), str) or len(row['oid']) != 40:
            raise ValueError('Invalid bounded Git JSON descriptor')
    return rows


def json_file(row, path):
    data = fetch(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{row["path"]}', 1024**2)
    git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if len(data) != row['size'] or git_sha != row['oid']:
        raise ValueError('Official Git JSON identity mismatch')
    json.loads(data)
    with path.open('xb') as stream:
        stream.write(data)


def shard(row, path):
    size, expected_sha = SHARDS[path.name]
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError('Inspect the previous partial download explicitly: ' + str(temporary))
    result, total, tick = hashlib.sha256(), 0, time.monotonic()
    url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{row["path"]}'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'T8-FreeVideo-Decoder-Explicit-Assets'}), timeout=60) as response, temporary.open('xb') as stream:
        while chunk := response.read(8 * 1024**2):
            total += len(chunk)
            if total > size:
                raise ValueError('Download exceeds the pinned shard size')
            stream.write(chunk)
            result.update(chunk)
            if time.monotonic() - tick >= 20:
                print(json.dumps(dict(file=path.name, downloaded_bytes=total, total_bytes=size)), flush=True)
                tick = time.monotonic()
    if total != size or result.hexdigest() != expected_sha:
        raise ValueError('Downloaded shard bytes/SHA mismatch; partial retained')
    temporary.rename(path)  # Existing destination is never replaced.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Dedicated directory; no runtime or model files replaced')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / 'vae'
    directory.mkdir(exist_ok=True)
    rows = validate_listing(json.loads(fetch(
        f'https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}/vae?expand=true', 2 * 1024**2)))
    needed = sum(row['size'] for row in rows if not (directory / PurePosixPath(row['path']).name).exists())
    if shutil.disk_usage(root).free < needed + 2 * 1024**3:
        raise ValueError('Insufficient disk space for immutable original weights and margin')
    inventory = []
    for row in rows:
        path = directory / PurePosixPath(row['path']).name
        if path.is_symlink():
            raise ValueError('Use a dedicated original-file directory, not ambiguous symlink assets')
        if not path.exists():
            json_file(row, path) if path.suffix == '.json' else shard(row, path)
        if path.stat().st_size != row['size']:
            raise ValueError('Existing asset has the wrong pinned size: ' + str(path))
        if path.name in SHARDS:
            if digest(path) != SHARDS[path.name][1]:
                raise ValueError('Existing shard SHA mismatch')
        else:
            data = path.read_bytes()
            if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != row['oid']:
                raise ValueError('Existing Git JSON identity mismatch')
        inventory.append(dict(path=str(path), bytes=row['size'], sha256=digest(path), source=row['path']))
        print(json.dumps(dict(file=path.name, verified=True, bytes=row['size'])), flush=True)
    proof = dict(schema='t8-freevideo-decoder-assets-v1', repository=REPOSITORY,
        model_revision=REVISION, license='MiniMax-H3-Community-License-Agreement',
        files=inventory, originals_not_converted=True, installed_software=False,
        existing_runtime_or_workflow_modified=False, decoder_GPU_qualified=False)
    proof_path = root / 'decoder-assets.json'
    with proof_path.open('x', encoding='utf8') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status='original_assets_verified_not_GPU_qualified', proof=str(proof_path))), flush=True)


if __name__ == '__main__':
    main()
