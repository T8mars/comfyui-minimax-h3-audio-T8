"""Build an identity-bound local prepared bundle; no GPU, download or adoption.

Inputs are explicit generation/decode path manifests, not workflow JSON files.
This prepares a reusable generation entry, not Base10/text/adapter computation.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types


def source_pin(path):
    root = subprocess.check_output(['git', '-C', str(path), 'rev-parse', '--show-toplevel'],
        text=True, stderr=subprocess.PIPE, timeout=15).strip()
    revision = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    if subprocess.check_output(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'], text=True, timeout=15).strip():
        raise ValueError('Prepared source has tracked modifications')
    return {'path': str(Path(root)), 'revision': revision}


def _ignored_import_runtime(path):
    """An ignored, wholly untracked dependency directory needs file inventory.

    Never inherit an enclosing project's Git pin for its ignored runtime.
    Actual tracked source directories still pass through the unchanged pin gate.
    """
    try:
        root = Path(subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', '--show-toplevel'],
            text=True, stderr=subprocess.PIPE, timeout=15).strip()).resolve()
        relative = Path(path).resolve(strict=True).relative_to(root).as_posix()
        tracked = subprocess.check_output(
            ['git', '-C', str(root), 'ls-files', '-z', '--', relative], timeout=15)
        if tracked:
            return False
        return subprocess.run(
            ['git', '-C', str(root), 'check-ignore', '--quiet', '--', relative],
            capture_output=True, timeout=15, check=False).returncode == 0
    except subprocess.CalledProcessError:
        return False


def prepare(kind, generation, decode, *, audio_seed=None, geometry=None, prompt=None, progress=lambda *args: None):
    # A local namespace avoids importing Comfy or loading any optional node.
    package_name = '_t8_prepared_bundle_tools'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'h3_t8')]
        sys.modules[package_name] = package
    from _t8_prepared_bundle_tools.prepared_generation_contract import validate_bundle
    from _t8_prepared_bundle_tools.prepared_identity import inventory, absolute_path
    from _t8_prepared_bundle_tools.prepared_backend.resource_guard import file_identity
    gen, dec = deepcopy(generation), deepcopy(decode)
    if kind not in ('tao5s', 'tao_stream', 'ltx_refine'):
        raise ValueError('Unknown prepared kind')
    for request in (gen, dec):
        for controller_field in ('gpu_uuid', 'identities', 'model_identities', 'seed', 'video_seed', 'inputs_sha256', 'text_cache_sha256'):
            request.pop(controller_field, None)
    dec.pop('latent', None)
    pins = {}
    def pin(path):
        value = source_pin(path)
        pins[value['path']] = value
        return value
    pin(dec['core'])
    directories = []
    if kind in ('tao5s', 'tao_stream'):
        if kind == 'tao5s':
            if audio_seed is None:
                raise ValueError('Explicit prepared teacher audio seed is required')
            gen.update(schema='t8-taomate-prepared-first-request-v1', audio_seed=audio_seed)
        else:
            if audio_seed is not None:
                raise ValueError('Tao stream uses per-request teacher audio seeds; do not pass audio_seed')
            from _t8_prepared_bundle_tools.prepared_stream_contract import validate_stream_requests
            validate_stream_requests(gen.get('stream_requests'))
            gen['schema'] = 't8-taomate-prepared-stream-v1'
            dec.update(schema='t8-taomate-stream-decode-v1', request_count=len(gen['stream_requests']))
        revision = pin(gen['source'])
        if gen['source_revision'] != revision['revision']:
            raise ValueError('Tao request source revision differs from current pinned source')
        directories.extend([str(Path(gen['base']) / 'FL2VA/transformer'), gen['adapter'], gen['teacher']])
    else:
        if geometry is None or prompt is None:
            raise ValueError('Explicit LTX geometry and cache-matching prompt required')
        gen.update(schema='t8-ltx-prepared-refinement-v1', geometry=geometry, prompt=prompt,
            normalization='normalized_ltx_av', reference_prefix_frames=0)
        dec['geometry'] = deepcopy(geometry)
        for path in gen['isolated_paths']:
            if _ignored_import_runtime(path):
                directories.append(path)
                continue
            try:
                pin(path)
            except subprocess.CalledProcessError:
                directories.append(path)
    paths = set()
    trees = []
    for directory in dict.fromkeys(directories):
        directory = absolute_path(directory)
        files = inventory(directory)
        trees.append({'path': directory, 'files': files})
        paths.update(files)
    file_fields = {'text_features', 'milestones', 'download_receipt', 'cpu_receipt',
        'inputs', 'text_cache', 'vae', 'audio', 'original_video', 'lora', 'video_vae', 'audio_vae'}
    if kind == 'ltx_refine':
        file_fields.add('base')
    for request in (gen, dec):
        paths.update(absolute_path(value) for key, value in request.items() if key in file_fields)
    if kind == 'tao_stream':
        for item in gen['stream_requests']:
            paths.update(absolute_path(item[key]) for key in ('text_features', 'milestones'))
    assets = []
    for index, path in enumerate(sorted(paths)):
        progress(index, len(paths), Path(path).name)
        assets.append(file_identity(path))
    bundle = {'schema': 't8_prepared_generation_bundle_v1', 'kind': kind, 'generation': gen,
        'decode': dec, 'assets': assets, 'trees': trees, 'source_revisions': list(pins.values()),
        'description': 'Prepared generation inputs only; no checkpoint adopted and no model inference performed.'}
    validate_bundle(bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['tao5s', 'tao_stream', 'ltx_refine'], required=True)
    parser.add_argument('--generation-request', type=Path, required=True)
    parser.add_argument('--decode-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-seed', type=int)
    parser.add_argument('--prompt')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new local bundle file; do not overwrite a previous identity')
    geometry = {'frames': args.frames, 'width': args.width, 'height': args.height, 'fps': 24} if args.kind == 'ltx_refine' else None
    def progress(index, count, filename):
        if index % 250 == 0 or filename.endswith('.safetensors'):
            print(json.dumps({'stage': 'hash_prepared_assets', 'index': index, 'count': count, 'file': filename}), flush=True)
    bundle = prepare(args.kind, json.loads(args.generation_request.read_text(encoding='utf8')),
        json.loads(args.decode_request.read_text(encoding='utf8')), audio_seed=args.audio_seed,
        geometry=geometry, prompt=args.prompt, progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'prepared_bundle_written_no_inference', 'path': str(args.output.resolve()),
        'assets': len(bundle['assets']), 'directories': len(bundle['trees']), 'source_pins': len(bundle['source_revisions'])}))


if __name__ == '__main__':
    main()
