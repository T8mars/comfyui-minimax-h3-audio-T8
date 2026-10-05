"""External Relay rebuilt from this window's actual native text and AV layout.

Uses the released finite-bias Relay machinery. This is not hard speech-time
isolation, and zero/single-event windows explicitly bypass the attention patch.
"""
from dataclasses import dataclass
import json
import math

import comfy.patcher_extension as extension
import node_helpers

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import enhance_a_video_advanced as feta
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..temporal_dialogue_bank import select_window_conditioning
from . import chunked_v5_relay as v5_relay
from .chunked_v5 import _window_input
from .results import _input_identity, canonical


KEY = 't8_temporal_dialogue_v5_relay'
CONFIG_TYPE = 'T8_TEMPORAL_DIALOGUE_RELAY_CONFIG'
RUNTIME_TYPE = 'T8_TEMPORAL_DIALOGUE_RELAY_RUNTIME'


@dataclass(frozen=True)
class ScopedRelayConfig:
    epsilon: float = 0.1
    query_route: str = 'video_only_paper'
    query_chunk_rows: int = 256

    def __post_init__(self):
        if type(self.epsilon) not in (float, int) or not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError('Scoped Relay epsilon must be finite and between 0 and 1')
        if self.query_route not in relay.PROMPT_RELAY_QUERY_ROUTES:
            raise ValueError('Unknown Scoped Relay query route')
        if type(self.query_chunk_rows) is not int or not 16 <= self.query_chunk_rows <= 4096:
            raise ValueError('Scoped Relay query rows must be an integer in [16,4096]')


@dataclass(frozen=True)
class ScopedRelayRuntime:
    bank_sha256: str
    window_index: int
    status: str
    runtime: object


def _local_plan(encoded, config):
    events = []
    for index, event in enumerate(encoded.report['prepared_dialogue_events'], 1):
        start, end = event['local_start_frame'], event['local_end_frame']
        events.append(dict(event_index=index, event_id=event['event_id'],
                           start_frame=start, end_frame_exclusive=end,
                           prompt_char_start=event['prompt_char_start'],
                           prompt_char_end=event['prompt_char_end'],
                           **relay._paper_parameters(start, end, config.epsilon)))
    # Already local. Do NOT subtract the global window start again.
    window = encoded.compiled.window
    plan = dict(type=relay.PROMPT_RELAY_PLAN_TYPE, schema=relay.PROMPT_RELAY_PLAN_SCHEMA,
                compiled_prompt=encoded.prepared_prompt, events=events,
                frame_count=window.end_frame-window.start_frame,
                query_route=config.query_route, policy=encoded.compiled.policy,
                text_sha256=encoded.compiled.sha256, epsilon=config.epsilon)
    plan['plan_hash'] = relay._sha256_json(plan)
    return plan


def bind_scoped_v5_relay(model, clip, source, lifted, prepared, plan, sigmas,
                          bank, index, config, previous=None):
    from .temporal_chunked_v5 import ScopedWindowResult
    if type(config) is not ScopedRelayConfig:
        raise ValueError('Connect the external Temporal Dialogue Relay Config')
    encoded = select_window_conditioning(bank, source, plan, index)
    if bank.executor_contract != 'v5_joint_refine4':
        raise ValueError('Scoped v5 Relay requires the v5 joint refine4 bank')
    if previous is not None and (type(previous) is not ScopedWindowResult or previous.bank_sha256 != bank.sha256):
        raise ValueError('Scoped Relay previous window belongs to another bank')
    prior = previous.base if previous is not None else None
    window = _window_input(source, lifted, prepared, plan, index, prior)
    sigma_values = v5_relay._validate_sigmas(sigmas, plan)
    if model.get_attachment(KEY) is not None or model.get_attachment(v5_relay.KEY) is not None:
        raise ValueError('Bind each scoped Relay window from the raw HIGH MODEL branch')
    if model.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY):
        raise ValueError('A full-clip Relay MODEL cannot be reused on new scoped text')
    if len(encoded.compiled.dialogue) <= 1:
        status = 'passthrough_single_event' if encoded.compiled.dialogue else 'passthrough_no_events'
        runtime = ScopedRelayRuntime(bank.sha256, index, status, None)
        return model, encoded.positive, runtime, canonical(dict(
            status=status, event_count=len(encoded.compiled.dialogue), sampled=False,
            hard_time_isolation=False, quality_accepted=False))
    if clip is None:
        raise ValueError('Scoped Relay needs the native CLIP tokenizer for actual encoded spans')
    local_plan = _local_plan(encoded, config)
    binding = relay.build_prompt_relay_binding(clip, local_plan, encoded.prepared_prompt,
                                               encoded.positive, encoded.tokens)
    local_positive = legacy.reanchor_conditioning(encoded.positive, window.start_frame,
                                                  window.end_frame, tuple(window.high_video.shape[-2:]))
    meta = local_positive[0][1]
    video, audio = window.piece['samples'].unbind()
    keyframes, refs = meta.get('minimax_keyframes', []), meta.get('minimax_refs', [])
    layout = build_packed_layout(binding['text_len'], *video.shape[2:], audio.shape[-1],
                                  keyframes=keyframes, refs=refs, frame_count=meta.get('minimax_frame_count'))
    task = feta._classify_visual_task(keyframes, latent_frames=video.shape[2], refs=refs)
    binding = relay._bind_layout_contract(binding, layout, resolved_task=task,
                                           keyframes=keyframes, refs=refs)
    # Return global media coordinates. The unchanged sampler reanchors once.
    paired = node_helpers.conditioning_set_values(encoded.positive, {relay.PROMPT_RELAY_BINDING_KEY: binding})
    paired = relay._attach_binding_model_cond(paired, binding['binding_hash'])
    selected, _hashes = relay.patch_prompt_relay_model(model, binding, config.query_chunk_rows)
    owner = v5_relay.V5RelayOwner(prepared.lift.plan_sha256, prepared.lift.source_identity,
        prepared.lift.lifted_identity, _input_identity(window.piece), index,
        binding['binding_hash'], sigma_values, model.get_model_object('model_sampling'))
    selected = selected.clone()
    selected.set_attachments(v5_relay.KEY, owner)
    base_runtime = v5_relay.V5RelayRuntime(selected, owner,
                                         len(selected.get_model_object('diffusion_model').blocks))
    selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, v5_relay.KEY, base_runtime.prepare)
    selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, v5_relay.KEY, base_runtime.cleanup)
    runtime = ScopedRelayRuntime(bank.sha256, index, 'scoped_relay_bound', base_runtime)
    selected.set_attachments(KEY, runtime)
    return selected, paired, runtime, canonical(dict(status=runtime.status, sampled=False,
        event_ids=[event.event_id for event in encoded.compiled.dialogue],
        binding_hash=binding['binding_hash'], local_plan_hash=local_plan['plan_hash'],
        clock_projection_count=1, query_route=config.query_route, hard_time_isolation=False,
        quality_accepted=False))


def assert_scoped_relay(model, bank, index):
    runtime = model.get_attachment(KEY)
    if model.get_attachment(v5_relay.KEY) is not None and runtime is None:
        raise ValueError('Scoped text requires its own newly paired Relay, not a full-text window binding')
    if runtime is not None and (type(runtime) is not ScopedRelayRuntime
            or runtime.bank_sha256 != bank.sha256 or runtime.window_index != index):
        raise ValueError('Scoped Relay MODEL belongs to different native text/bank/window')


def audit_scoped_relay(result, source, plan, prepared, bank, runtime):
    from .temporal_chunked_v5 import ScopedWindowResult
    if type(result) is not ScopedWindowResult or result.bank_sha256 != bank.sha256:
        raise ValueError('Scoped Relay audit requires the matching actual window result')
    encoded = select_window_conditioning(bank, source, plan, result.base.index)
    if (type(runtime) is not ScopedRelayRuntime or runtime.bank_sha256 != bank.sha256
            or runtime.window_index != result.base.index):
        raise ValueError('Scoped Relay audit runtime differs')
    if runtime.runtime is None:
        if len(encoded.compiled.dialogue) > 1 or not runtime.status.startswith('passthrough_'):
            raise ValueError('Scoped Relay bypass cannot certify multi-event application')
        return result.base.output_latent, canonical(dict(status=runtime.status,
            attention_patch_applied=False, hard_time_isolation=False, quality_accepted=False))
    output, report = v5_relay.audit_v5_relay(result.base, prepared, runtime.runtime)
    return output, canonical({**json.loads(report), 'scope_bank_sha256': bank.sha256,
                               'hard_time_isolation': False})
