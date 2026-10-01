"""Prepared-job manifest and cache identities. No model/framework imports."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

from .prepared_backend.backend_files import sha
from .prepared_identity import MAX_FILES, absolute_path

BACKEND = Path(__file__).resolve().parent / 'prepared_backend'
KINDS = {'tao5s', 'tao_stream', 'ltx_refine'}


def environment_identity():
    """Cache provenance without importing Torch or creating a CUDA context.

    Installed version metadata is not an adversarial integrity attestation.
    Isolated LTX import-directory file contents are separately inventoried.
    """
    packages = {}
    for name in ('torch', 'torchaudio', 'safetensors', 'numpy', 'psutil', 'nvidia-ml-py',
            'einops', 'av', 'pillow', 'scipy', 'comfy-kitchen', 'transformers',
            'huggingface-hub', 'tokenizers', 'triton', 'triton-windows', 'imageio-ffmpeg'):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = {'version': distribution.version,
                'location': str(Path(distribution.locate_file('')).resolve())}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
        'platform': platform.platform(), 'machine': platform.machine(), 'packages': packages}


def engine_sources():
    paths = sorted(BACKEND.glob('*.py')) + [Path(__file__), Path(__file__).with_name('prepared_generation_runtime.py'),
        Path(__file__).with_name('prepared_process.py'), Path(__file__).with_name('prepared_identity.py'),
        Path(__file__).with_name('prepared_checkpoint.py'),
        Path(__file__).with_name('prepared_stream_contract.py'),
        Path(__file__).parent / 'dlss_fi_backend/process.py']
    return {str(p.resolve()): sha(p) for p in paths}


def read_bundle(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024**2:
        raise ValueError('Prepared bundle must be a bounded local JSON file')
    bundle = json.loads(path.read_text(encoding='utf8'))
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get('schema') != 't8_prepared_generation_bundle_v1' or bundle.get('kind') not in KINDS:
        raise ValueError('Expected a qualified Tao/LTX prepared generation bundle')
    required = {'schema', 'kind', 'generation', 'decode', 'assets', 'source_revisions', 'trees'}
    if not required <= set(bundle) or set(bundle) - required - {'checkpoint', 'description'}:
        raise ValueError('Unknown or missing prepared bundle fields')
    gen, decode = bundle['generation'], bundle['decode']
    if not isinstance(gen, dict) or not isinstance(decode, dict):
        raise ValueError('Prepared generation and decode requests are required')
    schemas = {
        'tao5s': ('t8-taomate-prepared-first-request-v1', 't8-taomate-video-decode-v1'),
        'tao_stream': ('t8-taomate-prepared-stream-v1', 't8-taomate-stream-decode-v1'),
        'ltx_refine': ('t8-ltx-prepared-refinement-v1', 't8-ltx-refined-decode-v1'),
    }[bundle['kind']]
    if (gen.get('schema'), decode.get('schema')) != schemas:
        raise ValueError('Prepared worker schema does not match selected route')
    gen_fields = ({'schema', 'source', 'source_revision', 'base', 'adapter', 'teacher', 'text_features',
        'milestones', 'download_receipt', 'cpu_receipt', 'audio_seed'} if bundle['kind'] == 'tao5s' else
        {'schema', 'base', 'lora', 'inputs', 'text_cache', 'isolated_paths', 'prompt', 'geometry',
         'normalization', 'reference_prefix_frames'})
    decode_fields = ({'schema', 'core', 'source', 'milestones', 'audio', 'vae'} if bundle['kind'] == 'tao5s'
        else {'schema', 'core', 'vae', 'original_video', 'geometry'})
    if bundle['kind'] == 'tao_stream':
        gen_fields = {'schema', 'source', 'source_revision', 'base', 'adapter', 'teacher',
            'download_receipt', 'stream_requests'}
        decode_fields = {'schema', 'core', 'source', 'video_vae', 'audio_vae', 'request_count'}
    optional_gen = {'effects'} if bundle['kind'] == 'ltx_refine' else set()
    if not gen_fields <= set(gen) or set(gen) - gen_fields - optional_gen or set(decode) != decode_fields:
        raise ValueError('Unknown or missing prepared request fields; workers/seeds/identities are controller bound')
    if bundle['kind'] == 'ltx_refine':
        if gen.get('geometry') != decode.get('geometry') or gen.get('normalization') != 'normalized_ltx_av' or gen.get('reference_prefix_frames') != 0:
            raise ValueError('LTX generation/decode geometry and normalization must agree')
        validate_geometry(gen['geometry'])
        if type(gen['reference_prefix_frames']) is not int or not isinstance(gen['prompt'], str) or not 0 < len(gen['prompt'].strip()) <= 16384:
            raise ValueError('LTX needs a matching nonempty prompt and exact zero reference prefix')
    elif gen.get('source') != decode.get('source') or gen.get('milestones') != decode.get('milestones'):
        raise ValueError('Tao generation/decode teacher or source differs')
    if bundle['kind'] == 'tao_stream':
        from .prepared_stream_contract import validate_stream_requests
        validate_stream_requests(gen['stream_requests'])
        if type(decode['request_count']) is not int or decode['request_count'] != len(gen['stream_requests']):
            raise ValueError('Tao stream generation/decode request counts differ')
    if bundle['kind'] == 'tao5s' and (type(gen['audio_seed']) is not int or not 0 <= gen['audio_seed'] < 2**64):
        raise ValueError('Prepared teacher audio_seed must be unsigned64')
    assets = bundle['assets']
    if not isinstance(assets, list) or not assets or len(assets) > MAX_FILES:
        raise ValueError('A nonempty bounded asset identity list is required')
    paths = set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {'path', 'bytes', 'mtime_ns', 'sha256'}:
            raise ValueError('Assets need absolute file paths and complete identities')
        key = absolute_path(asset['path'])
        if key in paths or not isinstance(asset['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', asset['sha256']):
            raise ValueError('Duplicate asset or invalid SHA256')
        if type(asset['bytes']) is not int or asset['bytes'] < 0 or type(asset['mtime_ns']) is not int or asset['mtime_ns'] < 0:
            raise ValueError('Invalid asset stat identity')
        paths.add(key)
    if 'effects' in gen:
        from .prepared_backend.ltx_effects_contract import validate_effects
        validate_effects(gen['effects'], gen['geometry'], gen['prompt'], paths)
        indexed = {absolute_path(asset['path']): asset for asset in assets}
        relay = gen['effects']['relay']
        if relay is not None and any(indexed[cache['path']]['sha256'] != cache['sha256']
                                     for cache in relay['caches']):
            raise ValueError('Prepared Relay cache SHA differs from the bundle asset identity')
    # Explicit path-bearing fields cannot escape the identity list.
    file_fields = ('text_features', 'milestones', 'download_receipt', 'cpu_receipt', 'inputs', 'text_cache', 'vae', 'audio', 'original_video', 'lora', 'video_vae', 'audio_vae')
    for request in (gen, decode):
        for field in file_fields:
            if field in request and absolute_path(request[field]) not in paths:
                raise ValueError(f'Prepared {field} is missing an asset identity')
    if bundle['kind'] == 'tao_stream':
        validate_stream_requests(gen['stream_requests'], asset_paths=paths)
    if not isinstance(bundle['source_revisions'], list) or not bundle['source_revisions']:
        raise ValueError('Explicit upstream source pins are required')
    source_pins = {}
    for source in bundle['source_revisions']:
        if not isinstance(source, dict) or set(source) != {'path', 'revision'} or not isinstance(source['revision'], str) or not re.fullmatch('[0-9a-f]{40}', source['revision']):
            raise ValueError('Invalid upstream source pin')
        key = absolute_path(source['path'])
        if key in source_pins:
            raise ValueError('Duplicate upstream source pin')
        source_pins[key] = source['revision']
    if absolute_path(decode['core']) not in source_pins:
        raise ValueError('The actual decoding Core directory needs its source pin')
    if not isinstance(bundle['trees'], list):
        raise ValueError('Prepared directory inventories are required')
    trees = {}
    for tree in bundle['trees']:
        if not isinstance(tree, dict) or set(tree) != {'path', 'files'}:
            raise ValueError('Invalid prepared directory inventory')
        key = absolute_path(tree['path'])
        files = tree['files']
        if key in trees or not isinstance(files, list) or not files or len(files) > MAX_FILES:
            raise ValueError('Duplicate, empty or excessive directory inventory')
        members = [absolute_path(p) for p in files]
        if len(set(members)) != len(members) or not set(members) <= paths:
            raise ValueError('Directory members need unique asset identities')
        if any(not Path(p).is_relative_to(Path(key)) or Path(p) == Path(key) for p in members):
            raise ValueError('Directory inventory member leaves its declared directory')
        trees[key] = members
    if bundle['kind'] in ('tao5s', 'tao_stream'):
        if source_pins.get(absolute_path(gen['source'])) != gen['source_revision']:
            raise ValueError('The actual Tao source/revision must match the source pin')
        for path in (Path(absolute_path(gen['base'])) / 'FL2VA/transformer', Path(absolute_path(gen['adapter'])), Path(absolute_path(gen['teacher']))):
            if str(path) not in trees:
                raise ValueError('Tao base/adapter/teacher require full directory inventories')
    else:
        if absolute_path(gen['base']) not in paths:
            raise ValueError('LTX base needs its file identity')
        imports = gen['isolated_paths']
        if not isinstance(imports, list) or not 1 <= len(imports) <= 16:
            raise ValueError('Bounded explicit LTX import paths are required')
        normalized = [absolute_path(p) for p in imports]
        if len(set(normalized)) != len(normalized):
            raise ValueError('Duplicate isolated import path')
        for path in normalized:
            if path not in trees and not any(Path(path).is_relative_to(Path(root)) for root in source_pins):
                raise ValueError('Isolated imports need a pinned source or complete inventory')
    return bundle


def validate_geometry(values):
    """Early import-free envelope; native LTX types validate actual tensors again."""
    if not isinstance(values, dict) or set(values) != {'frames', 'width', 'height', 'fps'} or any(type(v) is not int for v in values.values()):
        raise ValueError('Geometry must contain exact integer frames/width/height/fps')
    frames, width, height, fps = (values[k] for k in ('frames', 'width', 'height', 'fps'))
    if fps != 24 or frames < 9 or (frames - 1) % 8:
        raise ValueError('Use native8n+1 frames at24fps, at least9 frames without implicit crop')
    if min(width, height) < 32 or width % 32 or height % 32 or ((frames - 1) // 8 + 1) * (width // 32) * (height // 32) > 20480:
        raise ValueError('LTX spatial/token envelope exceeded')
    return values


def fingerprint(bundle, seed, sources):
    validate_bundle(bundle)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Noise seed must be an unsigned64 integer')
    canonical = {key: value for key, value in bundle.items() if key not in ('checkpoint', 'description')}
    payload = {'bundle': canonical, 'noise_seed': seed, 'engine_sources': sources,
        'environment': environment_identity()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generation_request(bundle, seed, gpu_uuid, sources):
    request = deepcopy(bundle['generation'])
    request['video_seed' if bundle['kind'] in ('tao5s', 'tao_stream') else 'seed'] = seed
    if bundle['kind'] == 'tao_stream':
        from .prepared_stream_contract import bind_stream_video_seeds
        request['stream_requests'] = bind_stream_video_seeds(request['stream_requests'], seed)
    request['gpu_uuid'] = gpu_uuid
    request['identities'] = {asset['path']: asset['sha256'] for asset in bundle['assets']}
    request['identities'].update(sources)
    if bundle['kind'] == 'ltx_refine':
        indexed = {absolute_path(asset['path']): asset for asset in bundle['assets']}
        request['model_identities'] = [deepcopy(indexed[absolute_path(request[key])]) for key in ('base', 'lora')]
        for key in ('inputs', 'text_cache'):
            request[key + '_sha256'] = indexed[absolute_path(request[key])]['sha256']
    return request


def decode_request(bundle, latent, gpu_uuid, sources):
    request = deepcopy(bundle['decode'])
    request.update(latent=str(Path(latent).resolve(strict=True)), gpu_uuid=gpu_uuid)
    request['identities'] = {asset['path']: asset['sha256'] for asset in bundle['assets']}
    request['identities'].update(sources)
    request['identities'][request['latent']] = sha(latent)
    return request


def job_directory(output, chain_id):
    if not isinstance(chain_id, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,95}', chain_id):
        raise ValueError('chain_id must contain only letters/digits/underscore/hyphen, up to96characters')
    output = Path(output).resolve(strict=True)
    root = output / 'MiniMaxH3-Prepared' / chain_id
    if not root.resolve().is_relative_to(output) or root.is_symlink() or root.parent.is_symlink():
        raise ValueError('Prepared output leaves the selected Comfy output directory')
    return root
