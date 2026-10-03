"""Append-only external motion Relay/EAV sockets, separate from native parents."""
from comfy_api.latest import io

from . import external_continuation as bridge, external_continuation_effects as effects
from .nodes_external_continuation import CATEGORY
from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .nodes_prompt_relay_long_video_advanced import MiniMaxH3PromptRelayLongVideoConditioningT8Advanced
from .modular_sampling.eav import CONFIG_TYPE, RUNTIME_TYPE, apply_stage_eav


class MiniMaxH3ExternalRelayWindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 External · Relay Window (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Project an explicit global Relay Plan against this reencoded external context. '
                'generated_start_frame is your independent global Relay clock after the context prefix, '
                'not inferred from the external movie/trim or a native accepted ancestor. No adoption or sampling.',
            inputs=[io.Custom(bridge.CONTEXT_TYPE).Input('external_context'),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input('global_plan'),
                    io.Int.Input('length', default=124, min=22, step=17),
                    io.Int.Input('generated_start_frame', default=22, min=0, max=10000000)],
            outputs=[io.Custom(effects.RELAY_TYPE).Output('projected_relay'),
                     io.String.Output('compiled_prompt'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, external_context, global_plan, length=124, generated_start_frame=22):
        projected = effects.project_relay(external_context, global_plan, length, generated_start_frame)
        return io.NodeOutput(projected, projected.projected['compiled_prompt'], projected.contract_json)

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3ExternalRelayConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3PromptRelayLongVideoConditioningT8Advanced.define_schema()
        removed = {'context', 'prompt_relay_plan', 'segment_index', 'context_frames', 'context_audio', 'width', 'height'}
        inputs = [item for item in original.inputs if item.id not in removed]
        for item in inputs:
            if item.id == 'length':
                item.force_input = False
        return io.Schema(node_id=cls.__name__, display_name='H3 External · Relay Conditions / Model (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Original native Long Video Relay conditioning with independently bound external RGB/PCM '
                'motion guides. LOW/HIGH use separate contexts/plans/models. report_only keeps numerical Relay '
                'disabled but still supplies the native motion MODEL patch. EAV remains a separate stage node. '
                'Includes context frames; trim explicitly. No hidden second sampler, append, approval or audio mux.',
            inputs=[io.Custom(bridge.CONTEXT_TYPE).Input('external_context'),
                    io.Custom(effects.RELAY_TYPE).Input('projected_relay'), *inputs], outputs=original.outputs)

    @classmethod
    def execute(cls, external_context, projected_relay, **inputs):
        return io.NodeOutput(*effects.relay_condition(external_context, projected_relay, **inputs))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3ExternalMotionEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 External · Motion Effects Bind (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Bind the actual external source/context, native motion guides, positive and exact stage '
                'AV before stage setup/EAV. Works with external ONE or Relay conditions. For HIGH, bind its own '
                'reencoded HIGH context and actual reconciled HIGH AV. Never an accepted native parent/P7 phase. '
                'Does not alter LoRA, conditions, masks, audio, sampling or QualityGate; unknown user encoders '
                'remain usable/nonportable.',
            inputs=[io.Custom(bridge.CONTEXT_TYPE).Input('external_context'), io.Model.Input('model'),
                    io.Conditioning.Input('positive'), io.Latent.Input('av_latent')],
            outputs=[io.Model.Output('model'), io.Custom(effects.SCOPE_TYPE).Output('effect_scope'),
                     io.String.Output('report_json')])

    @classmethod
    def execute(cls, external_context, model, positive, av_latent):
        return io.NodeOutput(*effects.bind_scope(external_context, model, positive, av_latent))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3ExternalStageEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 External · Stage EAV Apply (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='External Stage EAV after the original stage setup. Authenticated external RGB/PCM scope '
                'and actual native motion payload are rechecked, without fake native ancestry or a forced20-step '
                'schedule. Existing Relay delegate is composed; actual completed Stage EAV audit reports coverage. '
                'Sampling, SIGMAS, masks and audio remain controlled by your separate original nodes.',
            inputs=[io.Model.Input('model'), io.Sigmas.Input('sigmas'), io.Latent.Input('av_latent'),
                    io.Custom('T8_STAGE_CONTEXT').Input('stage_context'),
                    io.Custom(CONFIG_TYPE).Input('eav_config'),
                    io.Custom(effects.SCOPE_TYPE).Input('effect_scope')],
            outputs=[io.Model.Output('model'), io.Custom(RUNTIME_TYPE).Output('runtime'),
                     io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, sigmas, av_latent, stage_context, eav_config, effect_scope):
        return io.NodeOutput(*apply_stage_eav(model, sigmas, av_latent, stage_context,
                                            eav_config, external_scope=effect_scope))


NODES = [MiniMaxH3ExternalRelayWindowEXPT8, MiniMaxH3ExternalRelayConditioningEXPT8,
         MiniMaxH3ExternalMotionEffectsBindEXPT8, MiniMaxH3ExternalStageEAVApplyEXPT8]
