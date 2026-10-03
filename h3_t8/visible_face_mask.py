"""Opt-in visible-pixel restriction AFTER an existing face composite.

No detector, crop, sampler, original blending rule or audio modification.
White permits the already blended candidate; black restores exact source RGB.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

TYPE_NAME = 'T8_VISIBLE_FACE_COMPOSITE_MASK'
SCHEMA = 't8.visible-face-composite-mask.v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError('Expected a finite RGB or MASK tensor')
    data = value.detach().cpu().contiguous()
    return {'shape': list(data.shape), 'dtype': str(data.dtype),
            'sha256': hashlib.sha256(memoryview(data.view(torch.uint8).numpy())).hexdigest()}


def rgb(value):
    identity(value)
    if (value.ndim != 4 or value.shape[-1] != 3 or min(value.shape[:3]) < 1
            or not value.is_floating_point() or bool(((value < 0) | (value > 1)).any())):
        raise ValueError('RGB requires finite floating [frames,height,width,3] within 0..1; no clamp/conversion')
    return tuple(value.shape[:3])


def mask(value, shape, *, broadcast=False):
    identity(value)
    if value.ndim != 3 or (not value.is_floating_point() and value.dtype != torch.bool):
        raise ValueError('MASK requires floating/bool [frames,height,width], no implicit geometry conversion')
    if tuple(value.shape) == shape:
        result = value
    elif broadcast and tuple(value.shape) == (1, *shape[1:]):
        result = value.expand(shape)
    else:
        raise ValueError('MASK must match complete RGB clock/geometry; one frame needs explicit broadcast')
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError('MASK must be within 0..1; no silent clamp')
    return result


def _implementation():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ('visible_face_mask.py', 'nodes_visible_face_mask.py')}


def _descriptor(source, visible, request):
    if set(request) != {'fps', 'source_start_frame', 'broadcast_single_mask'}:
        raise ValueError('Exact full-source clock and broadcast request required')
    shape = rgb(source)
    offset, broadcast = request['source_start_frame'], request['broadcast_single_mask']
    if type(offset) is not int or not 0 <= offset <= 5_000_000 or type(broadcast) is not bool:
        raise ValueError('Explicit integer frame offset and boolean broadcast policy required')
    if type(request['fps']) not in (int, float) or request['fps'] != 24:
        raise ValueError('This H3 post-composite route requires declared 24fps; no retiming')
    mask(visible, shape, broadcast=broadcast)
    return {'schema': SCHEMA, 'source': identity(source), 'visible_mask': identity(visible),
            'request': request, 'implementation': _implementation(),
            'frame_interval': [offset, offset + shape[0]], 'fps': {'num': 24, 'den': 1},
            'clock_scope': 'caller_declared_whole_RGB_buffer_not_measured_media_PTS',
            'geometry': 'full_source_same_frame_coordinates_no_crop_resize_or_retime',
            'semantics': 'user_supplied_visible_face_editability_not_verified_occlusion',
            'white': 'allow_existing_face_change', 'black': 'exact_original_RGB',
            'automatic_accept': False, 'sampling_invalidated': False,
            'invalidates': 'post_composite_delivery_only'}


@dataclass(frozen=True)
class VisibleFaceMask:
    source: torch.Tensor
    visible: torch.Tensor
    request_json: str
    contract_json: str
    contract_sha256: str

    def verify(self):
        try:
            request = json.loads(self.request_json)
            current = canonical(_descriptor(self.source, self.visible, request))
            if (current != self.contract_json or
                    hashlib.sha256(current.encode()).hexdigest() != self.contract_sha256):
                raise ValueError('binding changed')
            return json.loads(current)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise ValueError('Original RGB, visible MASK, request, receipt or implementation changed; bind again') from exc


def capture(source, visible, *, fps=24., source_start_frame=0, broadcast_single_mask=False):
    request = {'fps': fps, 'source_start_frame': source_start_frame,
               'broadcast_single_mask': broadcast_single_mask}
    contract = canonical(_descriptor(source, visible, request))
    return VisibleFaceMask(source, visible, canonical(request), contract,
                           hashlib.sha256(contract.encode()).hexdigest())


@torch.no_grad()
def composite(binding, candidate, changed_alpha, audio=None):
    if type(binding) is not VisibleFaceMask:
        raise ValueError('Connect the explicit full-source visible-face MASK binding')
    contract = binding.verify()
    base = binding.source
    shape = rgb(base)
    if rgb(candidate) != shape or candidate.dtype != base.dtype:
        raise ValueError('Already blended candidate must match original full RGB shape and dtype')
    candidate_input = candidate
    original_candidate, original_alpha = identity(candidate), identity(changed_alpha)
    alpha = mask(changed_alpha, shape).to(base.device)
    visibility = mask(binding.visible, shape,
                      broadcast=contract['request']['broadcast_single_mask']).to(base.device)
    candidate = candidate.to(base.device)
    if not torch.equal(candidate[alpha == 0], base[alpha == 0]):
        raise ValueError('Candidate changes RGB outside declared changed-alpha; no false source proof')
    # Candidate ALREADY includes the old alpha. Never multiply its delta by that
    # alpha again: doing so would square the original feather/blend strength.
    work_dtype = torch.float64 if torch.float64 in (alpha.dtype, visibility.dtype, base.dtype) else torch.float32
    visibility_work = visibility.to(work_dtype)
    effective = alpha.to(work_dtype) * visibility_work
    output = (base.to(work_dtype) +
              (candidate.to(work_dtype) - base.to(work_dtype)) * visibility_work[..., None]).to(base.dtype)
    # Explicit endpoints preserve source/candidate bits even for floating RGB.
    output = torch.where((visibility == 1)[..., None], candidate, output)
    output = torch.where((effective > 0)[..., None], output, base)
    rgb(output)
    if not torch.equal(output[effective == 0], base[effective == 0]):
        raise RuntimeError('Outside-mask exact source audit failed')
    binding.verify()
    if identity(candidate_input) != original_candidate or identity(changed_alpha) != original_alpha:
        raise ValueError('Candidate or changed-alpha changed during composite; retry explicit inputs')
    report = {'schema': SCHEMA + '.composite', 'source_mask_binding': contract,
              'candidate': original_candidate, 'changed_alpha': original_alpha,
              'effective_alpha': identity(effective), 'output': identity(output),
              'outside_effective_mask_bit_exact': True, 'candidate_already_blended': True,
              'original_alpha_not_applied_twice': True, 'interior_math_dtype': str(work_dtype),
              'original_changed_pixels': int((alpha > 0).sum().item()),
              'effective_changed_pixels': int((effective > 0).sum().item()),
              'protected_original_changed_pixels': int(((alpha > 0) & (effective == 0)).sum().item()),
              'audio_same_input_object': True, 'sampling_nfe': 0, 'automatic_accept': False,
              'occlusion_semantic_quality_verified': False,
              'invalidates': 'post_composite_delivery_only'}
    return output, audio, effective, canonical(report)


def multiface_support(binding, candidate, state):
    """Read the unchanged Multi-Face final full-source support, not a window mask."""
    from .multiface_refine_advanced import COMPOSITE_SCHEMA
    from .face_refine_advanced import source_proxy_sha256
    if type(binding) is not VisibleFaceMask:
        raise ValueError('Explicit original full RGB binding required')
    binding.verify()
    if (type(state) is not dict or state.get('schema') != COMPOSITE_SCHEMA
            or state.get('automatic_accept') is not False
            or state.get('source_proxy_sha256') != source_proxy_sha256(binding.source)
            or identity(state.get('frames')) != identity(candidate)):
        raise ValueError('Multi-Face state must describe this exact full-source final candidate')
    support = state.get('applied_mask')
    mask(support, rgb(binding.source))
    if support.dtype != torch.bool:
        raise ValueError('Multi-Face final applied_mask is boolean support, not a second blend alpha')
    return support
