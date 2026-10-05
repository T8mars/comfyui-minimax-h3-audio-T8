"""Content-bound native conditioning producers for progressive checkpoints.

No filename/UUID identity and no model loading. This describes actual loaded
components; it does not certify pretrained weight quality or a whole chain.
"""

import hashlib
import dataclasses
import inspect
import json
import marshal
from pathlib import Path
from types import FunctionType, MethodType

import torch

from .long_video_dual_identity import content_identity
from .nodes_long_video_dual_model import _component_identity
from .patch_stack_policy import UnverifiedModelStack, nonportable_component_identity, model_identity_matches


def _source(obj):
    path = inspect.getsourcefile(obj)
    if path is None:
        raise UnverifiedModelStack('Producer implementation source is unavailable')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configuration(value, references, active=None):
    if id(value) in references:
        return {'reference': references[id(value)]}
    active = set() if active is None else active
    if id(value) in active:
        raise UnverifiedModelStack('Cyclic producer configuration has no portable identity')
    active = active | {id(value)}
    if isinstance(value, FunctionType):
        return {'function': value.__module__ + '.' + value.__qualname__,
                'source': _source(value),
                'code': hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                'defaults': _configuration(value.__defaults__, references, active),
                'kwdefaults': _configuration(value.__kwdefaults__, references, active),
                'closure': {name: _configuration(cell.cell_contents, references, active)
                            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}}
    if isinstance(value, MethodType):
        return {'method': _configuration(value.__func__, references, active),
                'owner': _configuration(value.__self__, references, active)}
    if isinstance(value, type):
        return {'class': value.__module__ + '.' + value.__qualname__, 'source': _source(value)}
    if dataclasses.is_dataclass(value) and type(value).__module__.startswith('comfy.text_encoders.'):
        config = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        config.update(vars(value))
        return {'dataclass': type(value).__module__ + '.' + type(value).__qualname__,
                'source': _source(type(value)), 'fields': _configuration(config, references, active)}
    if isinstance(value, (set, frozenset)):
        items = [_configuration(v, references, active) for v in value]
        return {'type': type(value).__name__,
                'items': sorted(items, key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [_configuration(v, references, active) for v in value]}
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: _configuration(v, references, active) for k, v in sorted(value.items())}
    return content_identity(value)


def _dynamic_runtime_fields(patcher, module, name):
    """Authenticate Core's per-module residency aliases, not arbitrary objects.

    Actual weights/buffers and selected cast policy remain content-bound below.
    HostBuffer/virtual addresses track residency, not a new model or setting.
    """
    from comfy.model_patcher import ModelPatcherDynamic
    if type(patcher) is not ModelPatcherDynamic:
        return set()
    fields = vars(module)
    if '_pin_state' in fields:
        expected = patcher.model.dynamic_pins.get(patcher.load_device)
        if expected is None or fields['_pin_state'] is not expected:
            raise ValueError('Producer module has a foreign dynamic pin-state alias')
    if 'seed_key' in fields and fields['seed_key'] != name:
        raise ValueError('Producer dynamic seed key differs from its module path')
    if 'comfy_cast_weights' in fields and fields['comfy_cast_weights'] is not True:
        raise ValueError('Producer dynamic cast policy was replaced')
    for key in ('weight_function', 'bias_function'):
        if key in fields and fields[key] != []:
            raise UnverifiedModelStack('Producer dynamic weight function needs a portable identity adapter')
    for key in ('weight_lowvram_function', 'bias_lowvram_function'):
        if fields.get(key) is not None:
            raise UnverifiedModelStack('Producer dynamic LoRA function needs a portable identity adapter')
    for key in ('_v', '_v_block'):
        value = fields.get(key)
        if value is not None and (not isinstance(value, tuple) or len(value) != 3
                or value[0] is not patcher.model.dynamic_vbars.get(patcher.load_device)
                or any(type(v) is not int or v < 0 for v in value[1:])):
            raise ValueError('Producer virtual allocation is outside its Core residency owner')
    for key in ('weight', 'bias'):
        if '_v_' + key not in fields:
            continue
        cached, original = fields['_v_' + key], getattr(module, key, None)
        if cached is None and original is None:
            continue
        allocation = fields.get('_v')
        if not isinstance(cached, torch.Tensor) or original is None or allocation is None:
            raise ValueError('Producer resident view is not backed by its Core allocation')
        # QuantizedTensor.data_ptr delegates to its packed data, not its logical
        # floating shape. Do not dereference a possibly evicted virtual view.
        from comfy_kitchen.tensor import QuantizedTensor
        packed = cached._qdata if isinstance(cached, QuantizedTensor) else cached
        pointer = packed.data_ptr()
        if (cached.shape != original.shape or packed.device != patcher.load_device
                or not packed.is_contiguous() or pointer < allocation[1]
                or pointer + packed.numel() * packed.element_size() > allocation[1] + allocation[2]):
            raise ValueError('Producer resident view escaped its Core allocation')
    return {'_pin_state', '_pins', '_v', '_v_block', '_v_signature', '_prefetch', '_v_weight', '_v_bias',
            'seed_key', 'comfy_cast_weights', 'weight_function', 'bias_function',
            'weight_lowvram_function', 'bias_lowvram_function'}


def _legacy_runtime_fields(patcher, module):
    """Canonicalize authenticated Core no-op lists and selected cast aliases.

    ModelPatcher.load/wipe_lowvram_weight materializes these no-op class
    defaults during residency changes. Its cast backup must equal the declared
    Core class default, with the selected force-cast policy bound on patcher.
    Nonempty/replaced functions and foreign cast values remain content-bound;
    this does not omit weights, selected casts, hooks or arbitrary fields.
    """
    from comfy.model_patcher import ModelPatcher
    from comfy.ops import CastWeightBiasOp
    if type(patcher) is not ModelPatcher or not isinstance(module, CastWeightBiasOp):
        return set()
    ignored = set()
    for key in ('weight_function', 'bias_function'):
        default, effective = getattr(type(module), key, None), getattr(module, key, None)
        if type(default) is list and not default and type(effective) is list and not effective:
            ignored.add(key)
    fields = vars(module)
    default_cast = getattr(type(module), 'comfy_cast_weights', None)
    current_cast = getattr(module, 'comfy_cast_weights', None)
    if type(default_cast) is bool and type(current_cast) is bool:
        if 'prev_comfy_cast_weights' not in fields and current_cast == default_cast:
            ignored.add('comfy_cast_weights')
        elif (type(fields.get('prev_comfy_cast_weights')) is bool
              and fields['prev_comfy_cast_weights'] == default_cast and current_cast is True
              and (patcher.force_cast_weights or getattr(patcher.model, 'model_lowvram', False))):
            ignored.update({'comfy_cast_weights', 'prev_comfy_cast_weights'})
    return ignored


def _native_producer_description(component, role):
    import comfy.sd
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    from comfy.text_encoders.minimax import MiniMaxH3TEModel, MiniMaxH3ClipModel, MiniMaxQwen3VL
    expected = {'clip': comfy.sd.CLIP, 'video_vae': comfy.sd.VAE, 'audio_vae': comfy.sd.VAE}
    if role not in expected or type(component) is not expected[role]:
        raise ValueError('Progressive producers require native CLIP/video VAE/audio VAE components')
    network = component.cond_stage_model if role == 'clip' else component.first_stage_model
    network_type = {'clip': MiniMaxH3TEModel, 'video_vae': MiniMaxH3VideoVAE, 'audio_vae': MiniMaxH3AudioVAE}[role]
    if not isinstance(network, network_type) or component.patcher.model is not network:
        raise ValueError('Progressive producer has the wrong native H3 network or patcher')
    # Hash methods actually selected by the public native wrapper. Instance
    # replacements need an explicit adapter, even if their filename looks right.
    methods = ('tokenize', 'encode_from_tokens', 'encode_from_tokens_scheduled') if role == 'clip' else (
        'encode', 'decode', 'encode_tiled', 'decode_tiled', 'vae_encode_crop_pixels')
    for name in methods:
        if name in vars(component):
            raise UnverifiedModelStack('Producer execution method was replaced on the instance')
    modules = list(network.named_modules())
    references = {id(component): 'component', **{id(module): 'network.' + name for name, module in modules}}
    classes, configs = {}, {}
    module_internals = set(vars(torch.nn.Module())) - {'training'}
    for name, module in modules:
        if module._forward_pre_hooks or module._forward_hooks or 'forward' in vars(module):
            raise UnverifiedModelStack('Producer network contains live hooks or replaced forward')
        cls = type(module)
        label = cls.__module__ + '.' + cls.__qualname__
        if label not in classes:
            classes[label] = {'source': _source(cls),
                             'methods': {method: _configuration(getattr(cls, method), references)
                                         for method in ('forward', 'encode', 'decode')
                                         if isinstance(getattr(cls, method, None), FunctionType)}}
        ignored = set(module_internals)
        ignored.update(_dynamic_runtime_fields(component.patcher, module, name))
        ignored.update(_legacy_runtime_fields(component.patcher, module))
        # Core writes these for each loaded module. The selected force-cast
        # policy is bound on the patcher; patched_weights is residency status.
        ignored.update({'comfy_force_cast_weights', 'comfy_patched_weights'})
        if module is network:
            # Core loader bookkeeping is not model configuration. Actual
            # weights/buffers, selected casts and placement are bound below.
            ignored.update({'device', 'model_loaded_weight_memory', 'lowvram_patch_counter',
                'model_lowvram', 'current_weight_patches_uuid', 'model_offload_buffer_memory',
                'current_patcher', 'dynamic_vbars', 'dynamic_pins', 'dynamic_patchers'})
            if 'manual_cast_dtype' in component.patcher.object_patches:
                ignored.add('manual_cast_dtype')
        if type(module) is MiniMaxQwen3VL:
            # Produced by this forward, returned in conditioning and hashed as
            # an actual input. It is not a persistent encoder setting.
            ignored.add('last_token_tags')
        if type(module) is MiniMaxH3ClipModel:
            # Core resets these before every encode, then sets layer/device from
            # the bound CLIP wrapper/patcher. Keep options_default itself bound.
            ignored.update({'execution_device', 'layer', 'layer_idx', 'return_projected_pooled'})
        configs[name] = {'class': label, 'settings': _configuration({
            k: v for k, v in vars(module).items() if k not in ignored}, references)}
    excluded = {'patcher', 'first_stage_model', 'cond_stage_model', 'tokenizer', 'size'}
    settings = _configuration({k: v for k, v in vars(component).items() if k not in excluded}, references)
    component_identity = _component_identity(component)
    if component_identity.get('portable_cache_reuse') is False:
        raise UnverifiedModelStack('Progressive producer uses an unverified component stack')
    description = {'role': role, 'component': component_identity, 'settings': settings,
                   'network': configs, 'classes': classes,
                   # state_dict omits native H3 pixel_mean/pixel_std. They still
                   # change encode/decode and must invalidate cached work.
                   'buffers': content_identity(dict(network.named_buffers())),
                   'methods': {name: _configuration(getattr(type(component), name), references) for name in methods},
                   'placement': content_identity({'load': component.patcher.load_device,
                                                  'offload': component.patcher.offload_device,
                                                  'force_cast_weights': component.patcher.force_cast_weights}),
                   'model_options': _configuration(component.patcher.model_options, references),
                   'implementation': _source(native_producer_identity)}
    return description


def native_producer_identity(component, role):
    try:
        description = _native_producer_description(component, role)
    except UnverifiedModelStack as error:
        result = nonportable_component_identity(component, str(error), schema='progressive_producer_user_stack_v1')
        result['role'] = role
        return result
    encoded = json.dumps(description, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'sha256': hashlib.sha256(encoded.encode()).hexdigest(), 'role': role,
            'scope': 'actual_native_weights_tokenizer_configuration_and_implementation',
            'module_count': len(description['network']), 'class_count': len(description['classes'])}


class NativeProgressiveProducers:
    def __init__(self, *, clip, video_vae, audio_vae):
        self.components = dict(clip=clip, video_vae=video_vae, audio_vae=audio_vae)
        self._identity_json = json.dumps(self._capture(), sort_keys=True, allow_nan=False)

    @property
    def identity(self):
        return json.loads(self._identity_json)

    def _capture(self):
        return {role: native_producer_identity(component, role) for role, component in self.components.items()}

    def verify(self, **components):
        if components and (set(components) != set(self.components) or any(
                components[key] is not self.components[key] for key in components)):
            raise ValueError('Progressive producer objects differ from the bound components')
        if not model_identity_matches(self.identity, self._capture()):
            raise ValueError('Progressive conditioning producer changed')
        return json.loads(json.dumps(self.identity))


def verify_producers(producers, **components):
    if type(producers) is not NativeProgressiveProducers:
        raise ValueError('Use a native progressive producer binding, not a supplied identity label')
    return producers.verify(**components)
