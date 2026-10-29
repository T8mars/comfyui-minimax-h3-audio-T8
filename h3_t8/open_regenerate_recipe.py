"""Open-H3 merged linear adapters and independent LMS Sharp curve branches.

The author bundle uses scale-one merged factors and a separately selected
Ref2VA timestep table. Native H3 forward, attention and sampling are delegated.
"""
from contextvars import ContextVar
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import uuid

import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension
import comfy.utils
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .h3_lora_compat_advanced import build_minimax_h3_lora_key_map, convert_fastvideo_h3_adapter
from .h3_weight_diagnostics import lora_mapping_diagnostics
from .patch_stack_policy import warn_patch_stack


KEY = 't8_open_regenerate_recipe_v1'
ACTIVE = ContextVar(KEY, default={})


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def read_curve_table(path):
    with safe_open(str(path), framework='pt', device='cpu') as source:
        keys = [key for key in source.keys() if key == 'adaln_t_table' or key.endswith('.adaln_t_table')]
        if len(keys) != 1:
            raise ValueError('Select a source checkpoint containing one adaln_t_table tensor')
        table = source.get_tensor(keys[0]).float().contiguous()
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 1 or not torch.isfinite(table).all():
        raise ValueError('Sharp curve table must be a finite [grid>=2,coordinates>=1] matrix')
    return table


def interpolate_curve(table, times):
    table = table.to(device=times.device, dtype=torch.float32)
    position = times.float().clamp(0, 1) * (len(table) - 1)
    lower = position.floor().long().clamp(max=len(table) - 2)
    return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(-1))


def logical_object(model, name):
    """Resolve descendants of selected or backed-up parent object patches."""
    parts = name.split('.')
    for patches in (getattr(model, 'object_patches', {}), getattr(model, 'object_patches_backup', {})):
        for count in range(len(parts), 0, -1):
            parent = '.'.join(parts[:count])
            if parent in patches:
                return comfy.utils.get_attr(patches[parent], '.'.join(parts[count:])) if count < len(parts) else patches[parent]
    return model.get_model_object(name)


def native_clock_values(network, x, timestep, context, options, payload=None,
                        denoise_mask=None, audio_denoise_mask=None):
    """Use native layout/mask helpers and H3's actual packed clock ordering.

    No sampler grid is inferred here. This receives the native DIFFUSION_MODEL
    wrapper's inputs, after model_base has applied its sampling protocol.
    """
    from comfy.ldm.minimax import model as native
    video, audio = x[0], x[1]
    payload = payload or {}
    latent_t, height, width = video.shape[2:]
    ph, pw = network.patch_size[-2:]
    height, width = math.ceil(height / ph) * ph, math.ceil(width / pw) * pw
    signature = (context.shape[1], latent_t, height, width, audio.shape[-1])
    layout = payload.get('layout')
    if layout is None or layout.signature != signature:
        layout = native.PackedLayout(*signature, keyframes=payload.get('keyframes'), refs=payload.get('refs'))
    sigma = (timestep.flatten()[0] / 1000.).float().clamp(min=1e-6)
    video_time = float(1. - sigma)
    shift_v = float(options.get('minimax_h3_sigma_shift_video', network.sigma_shift_video))
    shift_a = float(options.get('minimax_h3_sigma_shift_audio', network.sigma_shift_audio))
    audio_time = float(1. - native.time_shift_sigma(sigma, shift_v, shift_a))
    visual_pin = max(video_time, float(payload.get('visual_cond_noise_aug', native.VISUAL_COND_TIMESTEP)))
    audio_pin = max(audio_time, float(payload.get('audio_cond_noise_aug', native.AUDIO_COND_TIMESTEP)))
    by_kind = {'text': video_time, 'video': video_time, 'audio': audio_time,
        'cond': visual_pin, 'ref_img': visual_pin, 'cond_audio': audio_pin, 'ref_audio': audio_pin}
    extras = set()
    if denoise_mask is not None:
        mask = native.mask_row_values(denoise_mask[0, 0].float(), latent_t, height, width)
        if mask is not None:
            times = (1. - mask * sigma.to(mask.device)).clamp(max=max(video_time, native.VISUAL_COND_TIMESTEP))
            if times.unique().numel() == 1:
                by_kind['video'] = float(times[0])
            else:
                extras.update(times.unique().tolist())
    if audio_denoise_mask is not None:
        mask = audio_denoise_mask[0, 0].float().reshape(-1)
        if not bool((mask >= 1. - 1e-3).all()):
            times = (1. - mask * (1. - audio_time)).clamp(max=max(audio_time, native.AUDIO_COND_TIMESTEP))
            if times.unique().numel() == 1:
                by_kind['audio'] = float(times[0])
            else:
                extras.update(times.unique().tolist())
    values = sorted({video_time, audio_time} | {by_kind[kind] for _, _, kind in layout.segments} | extras)
    return torch.tensor(values, dtype=torch.float32, device=video.device)


@dataclass
class CurveRuntime:
    table: torch.Tensor
    branches: dict
    strength: float
    declaration: dict
    owner: str = field(default_factory=lambda: uuid.uuid4().hex)
    completed_forwards: int = 0
    failed_forwards: int = 0
    calls: dict = field(default_factory=dict)
    forwards: list = field(default_factory=list)

    def scope(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = {} if transformer_options is None else transformer_options
        times = native_clock_values(executor.class_obj, x, timestep, context, options,
            kwargs.get('minimax_payload'), kwargs.get('denoise_mask'), kwargs.get('audio_denoise_mask'))
        native_table = getattr(executor.class_obj, 'adaln_t_table', None) if getattr(executor.class_obj, 'use_adaln_curves', False) else None
        frame = {'times': times, 'coordinates': interpolate_curve(self.table, times), 'seen': set(),
            'clock_source': 'packed_clock_inference', 'clock_verified': native_table is not None,
            'native_embedding': interpolate_curve(native_table, times) if native_table is not None else None}
        token = ACTIVE.set({**ACTIVE.get(), self.owner: frame})
        try:
            # Resolve the physically active modules after ModelPatcher has
            # applied the user's entire object-patch tree. Descendant forward
            # object patches can otherwise override a selected parent module
            # and also confuse Core's parent/child restoration order.
            with ExitStack() as cleanup:
                network = executor.class_obj
                embedder = getattr(network, 'time_embedder', None)
                if native_table is None and callable(getattr(embedder, 'register_forward_pre_hook', None)):
                    def observe_time(_module, inputs):
                        times = inputs[0]
                        frame.update(times=times, coordinates=interpolate_curve(self.table, times),
                            clock_source='actual_native_time_embedder_input', clock_verified=True)
                    cleanup.callback(embedder.register_forward_pre_hook(observe_time).remove)
                for name in self.branches:
                    layer = comfy.utils.get_attr(network, name)

                    def observe_linear(_module, inputs, output, name=name):
                        return self.apply_delta(inputs[0], output, name)

                    cleanup.callback(layer.register_forward_hook(observe_linear).remove)
                result = executor(x, timestep, context, options, **kwargs)
            self.completed_forwards += 1
            self.forwards.append({'times': frame['times'].detach().cpu().tolist(), 'targets': sorted(frame['seen']),
                'clock_source': frame['clock_source'], 'clock_alignment_verified': frame['clock_verified']})
            if frame['seen'] != set(self.branches) and self.strength != 0:
                warn_patch_stack('Some selected Sharp branches were not called by this MODEL forward; runtime coverage is incomplete')
            return result
        except BaseException:
            self.failed_forwards += 1
            raise
        finally:
            ACTIVE.reset(token)

    def apply_delta(self, value, output, name):
        a, b = self.branches[name]
        if self.strength == 0:
            return output
        frame = ACTIVE.get().get(self.owner)
        if frame is None:
            raise ValueError('Sharp branch needs its paired native diffusion wrapper to supply actual timestep coordinates')
        coordinates = frame['coordinates']
        native_embedding = frame['native_embedding']
        if native_embedding is not None:
            matches = native_embedding.shape == value.shape and torch.equal(native_embedding.to(value), value)
            frame['clock_verified'] = frame['clock_verified'] and bool(matches)
            frame['clock_source'] = 'packed_clock_checked_against_actual_native_curve_embedding'
            if not matches:
                warn_patch_stack('Actual pruned AdaLN embedding differs from inferred packed clocks; selected Sharp branch still runs, clock alignment is unverified')
        if coordinates.shape[0] != value.shape[0]:
            raise ValueError(f'Sharp curve row count {coordinates.shape[0]} differs from actual AdaLN rows {value.shape[0]}')
        with torch.autocast(device_type=value.device.type, enabled=False):
            delta = F.linear(F.linear(coordinates.to(value.device), a.to(value.device)), b.to(value.device))
        if delta.shape != output.shape:
            raise ValueError(f'Sharp branch output {tuple(delta.shape)} differs from actual linear output {tuple(output.shape)}')
        result = output + delta.to(output.dtype) * self.strength
        frame['seen'].add(name)
        self.calls[name] = self.calls.get(name, 0) + 1
        return result

    def report(self):
        covered = bool(self.forwards) and all(set(item['targets']) == set(self.branches) for item in self.forwards)
        return {**self.declaration, 'status': 'sampling_observed' if self.completed_forwards else 'prepared_sampling_unobserved',
            'observation_scope': 'Cumulative native calls since this MODEL was prepared; not a per-LATENT or CUDA-graph-replay certificate',
            'completed_model_forwards': self.completed_forwards, 'failed_model_forwards': self.failed_forwards,
            'curve_strength': self.strength, 'curve_target_count': len(self.branches),
            'execution_backend': 'Scoped active-module hooks; removed after every forward, including failure',
            'curve_target_call_counts': dict(self.calls), 'all_selected_curve_targets_observed': covered,
            'forward_timesteps': self.forwards, 'human_quality_accepted': False,
            'official_end_to_end_quality_or_speed_verified': False}


def attach_curves(model, table, branches, strength=1., declaration=None):
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError('Sharp strength must be finite')
    runtime = CurveRuntime(table, branches, strength, declaration or {})
    selected = model.clone()
    for name, (a, b) in branches.items():
        layer = logical_object(selected, 'diffusion_model.' + name)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[1] != table.shape[1]:
            raise ValueError(f'Invalid selected Sharp A/B pair or curve coordinate width: {name}')
        if b.shape[0] != layer.weight.shape[0]:
            raise ValueError(f'Sharp projection rows do not match the selected MODEL layer: {name}')
    selected.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, runtime.owner, runtime.scope)
    prior = tuple(selected.get_attachment(KEY) or ())
    selected.set_attachments(KEY, (*prior, runtime))
    return selected, runtime


def load_recipe(model, linear_path, curve_path, table_path, linear_strength=1., curve_strength=1.):
    linear_strength = float(linear_strength)
    if not math.isfinite(linear_strength):
        raise ValueError('Merged LoRA strength must be finite')
    raw = comfy.utils.load_torch_file(str(linear_path), safe_load=True)
    normalized = {}
    for name, tensor in raw.items():
        key = name[len('transformer.'):] if name.startswith('transformer.') else name
        if key in normalized:
            raise ValueError('Merged LoRA namespace normalization collision: ' + key)
        normalized[key] = tensor
    converted, mapping = convert_fastvideo_h3_adapter(normalized)
    converted = comfy.lora_convert.convert_lora(converted)
    key_map, _ = build_minimax_h3_lora_key_map(model.model)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)
    selected = model.clone()
    applied = selected.add_patches(patches, linear_strength)
    consumption = lora_mapping_diagnostics(converted, key_map, patches, set(applied))
    if len(applied) != len(patches) or not patches:
        warn_patch_stack('Merged LoRA was not fully mapped by this selected MODEL; selected patches are retained,full recipe consumption is not established')
    if consumption['converted_unmapped_tensor_count']:
        warn_patch_stack('Some selected merged LoRA tensor keys were not consumed; see mapping diagnostics')
    curves = comfy.utils.load_torch_file(str(curve_path), safe_load=True)
    branches, consumed = {}, set()
    suffix = '.curve_A.weight'
    for name, a in curves.items():
        if not name.endswith(suffix):
            continue
        b_name = name[:-len(suffix)] + '.curve_B.weight'
        if b_name not in curves:
            raise ValueError('Sharp curve A has no B pair: ' + name)
        stem = name[:-len(suffix)]
        stem = stem[len('transformer.'):] if stem.startswith('transformer.') else stem
        if stem.startswith('transformer_blocks.'):
            stem = 'blocks.' + stem[len('transformer_blocks.'):]
        if stem in branches:
            raise ValueError('Duplicate Sharp curve target: ' + stem)
        a, b = a.float().contiguous(), curves[b_name].float().contiguous()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Sharp curve factors contain non-finite values: ' + stem)
        branches[stem] = (a, b)
        consumed.update([name, b_name])
    if not branches or consumed != set(curves):
        raise ValueError('Sharp curve file must contain complete curve_A/curve_B pairs; unexpected tensors cannot be discarded')
    table = read_curve_table(table_path)
    declaration = {'schema': 't8.open_regenerate.recipe.v1', 'linear_strength': linear_strength,
        'linear_source_tensors': len(raw), 'linear_converted_tensors': len(converted),
        'linear_registered_targets': len(applied), 'linear_parsed_targets': len(patches),
        'linear_mapping_diagnostics': consumption,
        'structural_mapping': mapping, 'curve_table_shape': list(table.shape),
        'curve_table_source_name': Path(table_path).name,
        'curve_table_sha256': tensor_sha(table),
        'files': {kind: {'name': Path(path).name, 'sha256': file_sha(path)} for kind, path in
                  [('merged_lora', linear_path), ('sharp_curves', curve_path)]},
        'contract': 'Scale-one merged factors already include Turbo alpha/rank; independent Sharp branch is base(x)+strength*B(A(source_curve(t))). Existing MODEL patches and native forward are retained.'}
    selected, runtime = attach_curves(selected, table, branches, curve_strength, declaration)
    return selected, json.dumps(runtime.report(), ensure_ascii=False, indent=2)


def audit_recipe(model, after_sampling):
    runtimes = model.get_attachment(KEY) or ()
    return after_sampling, json.dumps({'schema': 't8.open_regenerate.recipe.audit.v1',
        'selected_owners': [runtime.report() for runtime in runtimes],
        'status': 'observed' if runtimes and all(runtime.completed_forwards for runtime in runtimes) else 'unobserved',
        'human_quality_accepted': False}, ensure_ascii=False, indent=2)
