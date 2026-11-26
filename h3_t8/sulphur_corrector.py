"""Optional post-CFG corrector, with Core-owned loading and untouched AV audio."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
import logging
import math
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from .sulphur_corrector_network import CorrectorConfig, H3Corrector, bound_residual

log = logging.getLogger(__name__)


def register_corrector_folder():
    # Preserve extra_model_paths and another extension's registered search paths.
    default = str(Path(folder_paths.models_dir) / 'h3_correctors')
    folder_paths.add_model_folder_path('h3_correctors', default)
    paths, extensions = folder_paths.folder_names_and_paths['h3_correctors']
    folder_paths.folder_names_and_paths['h3_correctors'] = (paths, set(extensions) | {'.safetensors'})


def corrector_names():
    register_corrector_folder()
    names = set(folder_paths.get_filename_list('h3_correctors'))
    for filename in folder_paths.get_filename_list('loras'):
        if filename in names or Path(filename).suffix.lower() != '.safetensors':
            continue
        path = folder_paths.get_full_path('loras', filename)
        if path is None:
            continue
        try:
            stat = Path(path).stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if _has_corrector_metadata(path, identity):
                names.add(filename)
        except (OSError, SafetensorError):
            # An unreadable unrelated LoRA must not break the dropdown.
            continue
    return sorted(names)


@lru_cache(maxsize=1024)
def _has_corrector_metadata(path, file_identity):
    """Use the file stat in the cache key so replaced models are rescanned."""
    with safe_open(path, framework='pt', device='cpu') as handle:
        return 'h3_corrector' in (handle.metadata() or {})


def load_corrector(corrector):
    register_corrector_folder()
    path = folder_paths.get_full_path('h3_correctors', corrector)
    if path is None and Path(corrector).suffix.lower() == '.safetensors':
        path = folder_paths.get_full_path('loras', corrector)
    if path is None:
        raise FileNotFoundError(f"Corrector '{corrector}' not found in h3_correctors or loras.")
    with safe_open(path, framework='pt', device='cpu') as handle:
        config = CorrectorConfig.from_metadata(handle.metadata())
    # Avoid initializing an 8 GB throwaway random network before loading it.
    with torch.device('meta'):
        network = H3Corrector(config)
    state = load_file(path, device='cpu')
    state = {key: value.to(dtype=torch.float32) for key, value in state.items()}
    network.load_state_dict(state, strict=True, assign=True)
    network.eval().requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(
        network, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())
    selected = {'patcher': patcher, 'network': network, 'name': Path(corrector).stem}
    report = {'schema': 't8.h3_corrector.load.v1', 'status': 'loaded_not_sampled',
              'selected_file': corrector, 'config': asdict(config), 'weight_dtype': 'float32',
              'parameters': sum(p.numel() for p in network.parameters()),
              'weights_bytes': sum(p.numel() * p.element_size() for p in network.parameters()),
              'trained_quality_verified': False}
    return selected, json.dumps(report, ensure_ascii=False)


def _schedule(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < 2:
        raise ValueError('Corrector sigmas must be a 1D schedule with at least two values')
    if not torch.isfinite(value).all() or (value < 0).any() or (value[1:] > value[:-1]).any():
        raise ValueError('Corrector sigmas must be finite, nonnegative and descending')
    return value.detach().to(device='cpu', dtype=torch.float64)


def _step_interval(sigma, sigmas):
    values = _schedule(sigmas)
    index = int((values[:-1] >= sigma - 1e-6).sum()) - 1
    index = max(0, min(index, values.numel() - 2))
    return index, float(values[index + 1])


class CorrectorPostCFG:
    """Per-selection callable; no global schedule, tensor cache or mutable counter."""
    def __init__(self, corrector, strength, budget_fraction, velocity_floor,
                 start_step, end_step, fallback_sigmas):
        self.network, self.name = corrector['network'], corrector['name']
        self.strength, self.budget_fraction, self.velocity_floor = strength, budget_fraction, velocity_floor
        self.start_step, self.end_step = start_step, end_step
        self.fallback_sigmas = fallback_sigmas

    def __call__(self, args):
        denoised = args['denoised']
        sigma = args['sigma'].detach().flatten()
        if not sigma.numel() or not torch.isfinite(sigma).all() or not torch.all(sigma == sigma[0]):
            raise ValueError('Corrector requires a finite shared sigma for the sampled batch')
        s = float(sigma[0])
        if s == 0:
            return denoised
        if s < 0:
            raise ValueError('Corrector sigma must be nonnegative')
        options = args.get('model_options', {}).get('transformer_options', {})
        schedule = options.get('sample_sigmas')
        if schedule is None:
            schedule = self.fallback_sigmas
        if schedule is None:
            log.warning('H3 Corrector skipped: sampler did not provide sample_sigmas; connect optional sigmas')
            return denoised
        index, following = _step_interval(s, schedule)
        if index < self.start_step or (self.end_step >= 0 and index > self.end_step):
            return denoised
        shapes = getattr(args['model'], 'latent_shapes', None)
        if not shapes or len(shapes) < 2:
            raise ValueError('H3 Corrector requires packed audio+video latent_shapes from the sampler')
        video = comfy.utils.unpack_latents(args['input'], shapes)[0]
        prediction = comfy.utils.unpack_latents(denoised, shapes)[0]
        caption = next((c['cross_attn'] for c in (args.get('cond') or [])
                        if c.get('cross_attn') is not None), None)
        if not isinstance(caption, torch.Tensor):
            raise ValueError('H3 Corrector requires raw positive caption cross_attn states')
        if caption.ndim == 3 and caption.shape[0] == 1 and video.shape[0] > 1:
            caption = caption.expand(video.shape[0], -1, -1)
        device = next(self.network.parameters()).device
        if device.type == 'meta':
            raise RuntimeError('Corrector weights have not been loaded from disk')
        z = video.to(device=device, dtype=torch.float32)
        velocity = (prediction.to(device=device, dtype=torch.float32) - z) / s
        # Match the original precision contract even inside a caller's CPU
        # autocast scope: FP32 CPU, FP32 weights with BF16 CUDA operations.
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
        with torch.no_grad(), autocast:
            raw = self.network(z, velocity, s, following, caption.to(device=device, dtype=torch.float32))
        if raw.shape != video.shape or not torch.isfinite(raw).all():
            raise ValueError('Corrector returned a mismatched shape or nonfinite residual')
        applied, ratio = bound_residual(raw, velocity, self.budget_fraction, self.velocity_floor)
        corrected = prediction.to(device=device, dtype=torch.float32) + s * self.strength * applied
        flat = corrected.to(device=denoised.device, dtype=denoised.dtype).reshape(denoised.shape[0], 1, -1)
        if not torch.isfinite(flat).all():
            raise ValueError('Corrector prediction overflowed; check strength, input values and output dtype')
        log.info('H3 Corrector %s step %d sigma %.5f -> %.5f RMS ratio %.4f',
                 self.name, index, s, following, float(ratio.mean()))
        # Copy the actual packed audio (and any following streams) byte-for-byte.
        return torch.cat([flat, denoised[:, :, flat.shape[-1]:]], dim=-1)


def apply_corrector(model, corrector, strength=1., budget_fraction=.5, velocity_floor=.001,
                    start_step=0, end_step=-1, sigmas=None):
    for name, value in [('strength', strength), ('budget_fraction', budget_fraction), ('velocity_floor', velocity_floor)]:
        if not math.isfinite(value) or value < 0 or (name != 'strength' and value == 0):
            raise ValueError(f'Corrector {name} must be finite and ' + ('nonnegative' if name == 'strength' else 'positive'))
    if type(start_step) is not int or type(end_step) is not int or start_step < 0 or end_step < -1:
        raise ValueError('Corrector start_step must be >= 0; end_step must be -1 or >= 0')
    if end_step >= 0 and end_step < start_step:
        raise ValueError('Corrector end_step must be >= start_step or -1')
    fallback = _schedule(sigmas).clone() if sigmas is not None else None
    patched = model.clone()
    if strength:
        hook = CorrectorPostCFG(corrector, strength, budget_fraction, velocity_floor, start_step, end_step, fallback)
        # Append without replacing other extensions or another corrector's owner.
        count = 0
        while patched.get_additional_models_with_key(f't8_h3_corrector_{count}'):
            count += 1
        key = f't8_h3_corrector_{count}'
        patched.set_additional_models(key, [corrector['patcher']])
        patched.set_model_sampler_post_cfg_function(hook)
    report = {'schema': 't8.h3_corrector.apply.v1', 'status': 'prepared_not_sampled' if strength else 'disabled',
              'corrector': corrector['name'], 'strength': strength, 'budget_fraction': budget_fraction,
              'velocity_floor': velocity_floor, 'start_step': start_step, 'end_step_inclusive': end_step,
              'schedule': 'sampler sample_sigmas; optional fallback only if absent',
              'audio': 'post-CFG audio slice unchanged; later steps may couple audio/video normally',
              'patches': 'existing LoRAs, post-CFG hooks and additional models preserved in order',
              'trained_quality_verified': False}
    return patched, json.dumps(report, ensure_ascii=False)
