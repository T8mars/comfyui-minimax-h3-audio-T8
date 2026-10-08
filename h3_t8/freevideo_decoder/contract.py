"""Content-bound decoder configuration and normalized BCTHW input contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..freevideo_exp.runtime import canonical, digest, tensor_record, verify_files
from ..freevideo_exp.runtime import VDN_REVISION
from ..freevideo_quality.profiles import FREEVIDEO_REVISION

VAE_REVISION = '5d9b308a59ab12e67147f191e184baf704185bd1'
SCHEMA = 't8-freevideo-video-decoder-v1'
NORMALIZATION = 'MiniMaxH3_normalized_denoiser_BCTHW_24_native_mean_std_once_v1'
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
MAX_JSON = 16 * 1024**2
ASSETS = {
    'config.json': (2011, '78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745'),
    'diffusion_pytorch_model.safetensors.index.json': (74228, '15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca'),
    'diffusion_pytorch_model-00001-of-00003.safetensors': (5061033024, '72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374'),
    'diffusion_pytorch_model-00002-of-00003.safetensors': (4955986528, '2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198'),
    'diffusion_pytorch_model-00003-of-00003.safetensors': (398539336, 'c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39'),
}


def read_json(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError('Decoder metadata exceeds 16 MiB')
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    result = json.loads(data, parse_constant=invalid)
    if type(result) is not dict:
        raise ValueError('Decoder metadata must be an object')
    return result


def validate_config(value):
    names = {'schema', 'freevideo_revision', 'vdn_revision', 'vae_revision', 'normalization',
        'python', 'source_root', 'vdn_root', 'base', 'home', 'files', 'inventory_sha256',
        'vae_config_sha256', 'assets_proof_sha256', 'pixel_mean', 'pixel_std', 'parameter_policy'}
    if type(value) is not dict or set(value) != names:
        raise ValueError('Use the independent decoder configuration, not a sampler runtime')
    if (value['schema'] != SCHEMA or value['freevideo_revision'] != FREEVIDEO_REVISION
            or value['vdn_revision'] != VDN_REVISION or value['vae_revision'] != VAE_REVISION
            or value['normalization'] != NORMALIZATION
            or value['parameter_policy'] != 'original_FP32_streamed_no_Linear_compute_cache'
            or value['pixel_mean'] != list(PIXEL_MEAN) or value['pixel_std'] != list(PIXEL_STD)):
        raise ValueError('Unsupported decoder source, normalization or precision contract')
    for name in ('python', 'source_root', 'vdn_root', 'base', 'home'):
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute():
            raise ValueError('Decoder requires an explicit absolute ' + name)
    rows = value['files']
    if (not isinstance(rows, list) or not rows or len(rows) > 10000
            or any(type(row) is not dict for row in rows)
            or len({row.get('path') for row in rows}) != len(rows)
            or value['inventory_sha256'] != hashlib.sha256(canonical(rows).encode()).hexdigest()):
        raise ValueError('Decoder inventory is missing or inconsistent')
    for row in rows:
        if (set(row) != {'path', 'bytes', 'sha256', 'kind'}
                or not isinstance(row['path'], str) or not Path(row['path']).is_absolute()
                or type(row['bytes']) is not int or row['bytes'] < 0
                or row['kind'] not in ('borrowed_source_dependency', 'original_video_vae', 'assets_proof', 'decoder_adapter')
                or not isinstance(row['sha256'], str) or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('Malformed immutable decoder file row')
    for name in ('vae_config_sha256', 'assets_proof_sha256'):
        if not isinstance(value[name], str) or len(value[name]) != 64 or any(c not in '0123456789abcdef' for c in value[name]):
            raise ValueError('Decoder is missing actual asset identity')
    original = [row for row in rows if row['kind'] == 'original_video_vae']
    expected = {str(Path(value['base']) / 'vae' / name): pair for name, pair in ASSETS.items()}
    if (len(original) != len(expected)
            or {row['path']: (row['bytes'], row['sha256']) for row in original} != expected
            or value['vae_config_sha256'] != ASSETS['config.json'][1]):
        raise ValueError('Decoder requires all five immutable official VAE assets')
    proof = [row for row in rows if row['kind'] == 'assets_proof']
    if (len(proof) != 1 or proof[0]['path'] != str(Path(value['base']) / 'decoder-assets.json')
            or proof[0]['sha256'] != value['assets_proof_sha256']):
        raise ValueError('Decoder asset proof identity mismatch')
    adapters = {row['path']: row for row in rows if row['kind'] == 'decoder_adapter'}
    package = Path(__file__).resolve().parent
    if set(adapters) != {str(path) for path in package.glob('*.py')}:
        raise ValueError('Decoder adapter inventory must include the actual complete package')
    return value


def config_for(path, full=False):
    value = validate_config(read_json(path))
    if full:
        verify_files(value['files'])
        if digest(Path(value['base']) / 'vae/config.json') != value['vae_config_sha256']:
            raise ValueError('Decoder VAE config changed')
    return value


def stage_input(stage, family):
    if family == 'quality':
        from ..freevideo_quality.runtime import validate_stage
        receipt = validate_stage(stage)
        if receipt['role'] not in ('HIGH', 'SINGLE') or not receipt['plan']['plan_complete']:
            raise ValueError('Decode a completed Quality HIGH or SINGLE, not pending Light LOW')
    elif family == 'legacy':
        from ..freevideo_exp.runtime import validate_stage
        receipt = validate_stage(stage, 'HIGH')
    else:
        raise ValueError('Select the actual typed FreeVideo stage family')
    import torch
    if stage.video.dtype != torch.float32 or tuple(stage.video.shape[:2]) != (1, 24):
        raise ValueError('Completed FreeVideo decoder input must be normalized float32 BCTHW')
    return stage.video, dict(geometry=receipt['geometry'], stage_receipt_sha256=stage.receipt_sha256,
        family=family, video=tensor_record(stage.video), audio=tensor_record(stage.audio),
        sampling_NFE_added=0, normalization=NORMALIZATION)


def denormalize(video, config):
    """Only the new decoder path uses this exactly-once native transformation."""
    import torch
    mean, std = config.get('latents_mean'), config.get('latents_std')
    if (config.get('_class_name') != 'AutoencoderKLMiniMaxH3' or config.get('latent_channels') != 24
            or not isinstance(mean, list) or not isinstance(std, list) or len(mean) != len(std) or len(mean) != 24
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*mean, *std))
            or any(v <= 0 for v in std)):
        raise ValueError('Decoder requires the actual native video VAE mean/std')
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24) or video.dtype != torch.float32 or not torch.isfinite(video).all():
        raise ValueError('Expected finite normalized float32 [1,24,T,H,W]')
    means = torch.tensor(mean, dtype=torch.float32, device=video.device).view(1, 24, 1, 1, 1)
    scales = torch.tensor(std, dtype=torch.float32, device=video.device).view(1, 24, 1, 1, 1)
    return video * scales + means


def pixel_frames(video):
    """Author's FP32 ImageNet postprocess and uint8 rounding, in bounded chunks."""
    import torch
    if (video.ndim != 5 or tuple(video.shape[:2]) != (1, 3)
            or not video.is_floating_point() or not bool(torch.isfinite(video).all())):
        raise ValueError('Expected finite decoded [1,3,F,H,W]')
    video = video.detach().cpu()
    frames = torch.empty((*video.shape[2:], 3), dtype=torch.uint8)
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
    for start in range(0, video.shape[2], 8):
        chunk = (video[:, :, start:start + 8].float() * std + mean).clamp_(0, 1)
        frames[start:start + chunk.shape[2]] = (chunk[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8)
    return frames
