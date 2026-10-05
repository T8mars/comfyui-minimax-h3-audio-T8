"""Append-only opt-in Temporal Chunk dialogue scopes and literal window caches."""
from dataclasses import asdict
from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .conditioning import build_conditioning
from .core import align_frame_count
from .nodes import MiniMaxH3AudioConditioningT8
from .temporal_dialogue_scope import build_dialogue_plan, compile_window_text, WindowDescriptor
from .temporal_dialogue_bank import prepare_window_bank, select_window_conditioning
from .modular_sampling.chunked_v5_nodes import PLAN, PREPARED, RESULT
from .modular_sampling.eav import CONFIG_TYPE as EAV_CONFIG_TYPE
from .modular_sampling.results import canonical
from .modular_sampling.temporal_chunked_v5 import sample_scoped_v5_window
from .modular_sampling.temporal_chunked_storage import (
    save_scoped_window, load_scoped_window, verify_scoped_window, fingerprint_scoped_window,
)
from .modular_sampling.temporal_chunked_driver import run_scoped_v5
from .modular_sampling.temporal_chunked_relay import (
    ScopedRelayConfig, CONFIG_TYPE as RELAY_CONFIG_TYPE, RUNTIME_TYPE as RELAY_RUNTIME_TYPE,
    bind_scoped_v5_relay, audit_scoped_relay,
)


CATEGORY = 'T8/MiniMax H3/Temporal Dialogue Experimental'
DIALOGUE = 'T8_TEMPORAL_DIALOGUE_PLAN'
RECIPE = 'T8_TEMPORAL_NATIVE_TEXT_RECIPE'
BANK = 'T8_TEMPORAL_DIALOGUE_WINDOW_BANK'
SCOPED = 'T8_TEMPORAL_DIALOGUE_V5_WINDOW_RESULT'


def _schema(cls, name, inputs, outputs, description='', output=False):
    return io.Schema(node_id=cls.__name__, display_name='H3 Temporal Dialogue · '+name+' (T8 EXP)',
        category=CATEGORY, is_experimental=True, is_output_node=output,
        inputs=inputs, outputs=outputs, description=description)


def _source_inputs():
    return [io.Latent.Input('partial4_denoised_output'), io.Latent.Input('lifted_full_av'),
            io.Custom(PREPARED).Input('prepared'), io.Custom(PLAN).Input('plan'), io.Custom(BANK).Input('bank')]


def _window_outputs():
    return [io.Latent.Output('cumulative_av_latent'), io.Custom(RESULT).Output('base_window_result'),
            io.Custom(SCOPED).Output('scoped_window_result')]


def _store_root():
    return Path(folder_paths.get_output_directory()) / 'MiniMaxH3' / 'temporal_dialogue_window_artifacts'


class MiniMaxH3TemporalDialoguePlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Global + Speech + Performance Plan', [
            io.String.Input('global_prompt', multiline=True, default='One continuous scene.'),
            io.String.Input('dialogue_events_json', multiline=True, default='[]'),
            io.String.Input('performance_events_json', multiline=True, default='[]'),
            io.Int.Input('length', default=294, min=5, max=None, step=17)],
            [io.Custom(DIALOGUE).Output('dialogue_plan'), io.String.Output('full_prompt_for_LOW'),
             io.String.Output('report_json')],
            'Explicit dialogue events: event_id, speaker, utterance, start_seconds, end_seconds. '
            'Performance events: event_id, cue, start_seconds, end_seconds. Global/Performance must '
            'not contain repeated speech commands. No guessing or automatic deletion of free-text dialogue.')

    @classmethod
    def execute(cls, global_prompt, dialogue_events_json, performance_events_json, length):
        plan = build_dialogue_plan(global_prompt, dialogue_events_json, align_frame_count(length),
                                   performance_events=performance_events_json)
        audio_length = round(plan.total_frames*5/3)
        full = WindowDescriptor(0, 1, 0, plan.total_frames, 0, 0, audio_length, 0,
                                plan.total_frames, audio_length)
        text = compile_window_text(plan, full).prompt
        return io.NodeOutput(plan, text, canonical({**asdict(plan),
            'sampling_executed': False, 'hard_time_isolation': False, 'quality_accepted': False}))


class MiniMaxH3TemporalNativeRecipeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = MiniMaxH3AudioConditioningT8.define_schema()  # new instance, never mutate the old schema
        schema.node_id, schema.display_name = cls.__name__, 'H3 Temporal Dialogue · Native HIGH Media Recipe (T8 EXP)'
        schema.category, schema.is_experimental = CATEGORY, True
        schema.description = ('The original native media builder plus an explicit live text recipe. '
            'Use the actual HIGH dimensions and full source length. Window Bank re-encodes text before '
            'HIGH; media VAEs are not rerun per window. Old Conditioning is unchanged.')
        schema.outputs.append(io.Custom(RECIPE).Output('native_text_recipe'))
        return schema

    @classmethod
    def execute(cls, **inputs):
        result = build_conditioning(**inputs, return_text_recipe=True)
        return io.NodeOutput(*result[:6], result[-1]['text_recipe'])


class MiniMaxH3TemporalWindowBankEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Precompile ALL Native Window Text', [
            io.Custom(RECIPE).Input('native_text_recipe'), io.Custom(DIALOGUE).Input('dialogue_plan'),
            io.Latent.Input('source_av_latent'), io.Custom(PLAN).Input('plan')],
            [io.Custom(BANK).Output('bank'), io.String.Output('report_json')],
            'Select speech on real writable intervals from window 0; retain visual performance on '
            'render intervals. All CLIP encodes happen here, not between HIGH windows.')

    @classmethod
    def execute(cls, native_text_recipe, dialogue_plan, source_av_latent, plan):
        bank, report = prepare_window_bank(native_text_recipe, dialogue_plan, source_av_latent, plan)
        return io.NodeOutput(bank, canonical(report))


class MiniMaxH3TemporalWindowConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Select One Scoped CONDITIONING', [io.Custom(BANK).Input('bank'),
            io.Latent.Input('source_av_latent'), io.Custom(PLAN).Input('plan'),
            io.Int.Input('window_index', default=0, min=0, max=9999)],
            [io.Conditioning.Output('positive'), io.String.Output('prepared_prompt'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, bank, source_av_latent, plan, window_index):
        encoded = select_window_conditioning(bank, source_av_latent, plan, window_index)
        return io.NodeOutput(encoded.positive, encoded.prepared_prompt, canonical(encoded.report))


class MiniMaxH3TemporalV5WindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'One Joint AV Window', [io.Model.Input('model'), *_source_inputs(),
            io.Noise.Input('noise'), io.Sampler.Input('sampler'), io.Sigmas.Input('sigmas'),
            io.Int.Input('window_index', default=0, min=0, max=9999),
            io.Custom(SCOPED).Input('previous_result', optional=True),
            io.Conditioning.Input('positive', optional=True), io.Conditioning.Input('negative', optional=True),
            io.Float.Input('cfg', default=1., min=0., max=100., step=0.1)],
            [*_window_outputs(), io.String.Output('report_json')],
            'Partial4 audio remains editable. The actual accepted audio/video prefix is protected. '
            'Optional positive must be this native scoped condition, optionally paired by Scoped Relay. '
            'base_window_result connects to the unchanged external v5 EAV/Audit nodes.')

    @classmethod
    def execute(cls, model, partial4_denoised_output, lifted_full_av, prepared, plan, bank,
                noise, sampler, sigmas, window_index, previous_result=None, positive=None, negative=None, cfg=1.):
        return io.NodeOutput(*sample_scoped_v5_window(model, partial4_denoised_output, lifted_full_av,
            prepared, plan, noise, sampler, sigmas, bank, window_index, previous_result, positive, negative, cfg))


class MiniMaxH3TemporalV5WindowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Freeze Scoped Window + Audio Receipt', [
            io.Custom(SCOPED).Input('scoped_window_result'), *_source_inputs(),
            io.Boolean.Input('confirm_save', default=False)],
            [*_window_outputs(), io.String.Output('artifact_path'), io.String.Output('artifact_sha256'),
             io.String.Output('report_json')],
            'Independent sidecar; old v5 cache reader/source pins remain intact. Copy both explicit '
            'scope.json path and SHA. Not automatic reuse or provider/quality certification.', output=True)

    @classmethod
    def execute(cls, scoped_window_result, partial4_denoised_output, lifted_full_av, prepared, plan, bank,
                confirm_save=False):
        args = (scoped_window_result, partial4_denoised_output, lifted_full_av, prepared, plan, bank)
        if not confirm_save:
            verify_scoped_window(*args)
            result = scoped_window_result
            report = canonical(dict(status='not_saved_confirm_save_false', automatic_cache_reuse=False))
            return io.NodeOutput(result.base.output_latent, result.base, result, '', '', report,
                                 ui={'text': [report]})
        output = save_scoped_window(*args, _store_root())
        return io.NodeOutput(*output, ui={'text': [f'artifact_path: {output[3]}',
                                                f'artifact_sha256: {output[4]}', output[5]]})


class MiniMaxH3TemporalV5WindowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Load Explicit Scoped Window', [*_source_inputs(),
            io.Int.Input('expected_window_index', default=0, min=0, max=9999),
            io.String.Input('artifact_path', default=''), io.String.Input('artifact_sha256', default='')],
            [*_window_outputs(), io.String.Output('report_json')],
            'Reconstruct current native bank; exact source/lift/noise/policy/content checks. '
            'Old unscoped caches cannot serve as new accepted-dialogue receipts.')

    @classmethod
    def execute(cls, partial4_denoised_output, lifted_full_av, prepared, plan, bank,
                expected_window_index, artifact_path, artifact_sha256):
        return io.NodeOutput(*load_scoped_window(partial4_denoised_output, lifted_full_av, prepared,
            plan, bank, _store_root(), artifact_path, artifact_sha256, expected_window_index))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_inputs):
        try:
            return fingerprint_scoped_window(_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float('nan')


class MiniMaxH3TemporalRelayConfigEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'External Relay Config', [
            io.Float.Input('epsilon', default=0.1, min=0.001, max=0.999, step=0.001),
            io.Combo.Input('query_route', options=['video_only_paper', 'joint_av_exp'], default='video_only_paper'),
            io.Int.Input('query_chunk_rows', default=256, min=16, max=4096, step=16)],
            [io.Custom(RELAY_CONFIG_TYPE).Output('relay_config')],
            'Optional finite timing bias, not hard audio isolation; zero/single-event windows bypass.')

    @classmethod
    def execute(cls, epsilon, query_route, query_chunk_rows):
        return io.NodeOutput(ScopedRelayConfig(epsilon, query_route, query_chunk_rows))


class MiniMaxH3TemporalV5RelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'External Native Scoped Relay Pair', [io.Model.Input('model'), io.Clip.Input('clip'),
            *_source_inputs(), io.Sigmas.Input('sigmas'), io.Int.Input('window_index', default=0, min=0, max=9999),
            io.Custom(RELAY_CONFIG_TYPE).Input('relay_config'), io.Custom(SCOPED).Input('previous_result', optional=True)],
            [io.Model.Output('model'), io.Conditioning.Output('positive'),
             io.Custom(RELAY_RUNTIME_TYPE).Output('runtime'), io.String.Output('report_json')],
            'Fresh native prepared char/token spans and actual local AV layout; clock projected once. '
            'Connect the paired MODEL+positive to the matching scoped Window; never reuse full-text Relay.')

    @classmethod
    def execute(cls, model, clip, partial4_denoised_output, lifted_full_av, prepared, plan, bank,
                sigmas, window_index, relay_config, previous_result=None):
        return io.NodeOutput(*bind_scoped_v5_relay(model, clip, partial4_denoised_output, lifted_full_av,
            prepared, plan, sigmas, bank, window_index, relay_config, previous_result))


class MiniMaxH3TemporalV5RelayAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Audit Scoped Relay / Explicit Bypass', [
            io.Custom(SCOPED).Input('scoped_window_result'), io.Latent.Input('source_av_latent'),
            io.Custom(PLAN).Input('plan'), io.Custom(PREPARED).Input('prepared'),
            io.Custom(BANK).Input('bank'), io.Custom(RELAY_RUNTIME_TYPE).Input('runtime')],
            [io.Latent.Output('cumulative_av_latent'), io.String.Output('report_json')], output=True)

    @classmethod
    def execute(cls, scoped_window_result, source_av_latent, plan, prepared, bank, runtime):
        output, report = audit_scoped_relay(scoped_window_result, source_av_latent, plan, prepared, bank, runtime)
        return io.NodeOutput(output, report, ui={'text': [report]})


class MiniMaxH3TemporalV5JointPass2EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Integrated Joint PASS2 (same separated windows)', [
            io.Model.Input('model'), io.Latent.Input('partial4_denoised_output'), io.Custom(PLAN).Input('plan'),
            io.Noise.Input('noise'), io.Sampler.Input('sampler'), io.Sigmas.Input('sigmas'),
            io.Custom(BANK).Input('bank'), io.Custom(EAV_CONFIG_TYPE).Input('eav_config', optional=True),
            io.Custom(RELAY_CONFIG_TYPE).Input('relay_config', optional=True), io.Clip.Input('clip', optional=True),
            io.Conditioning.Input('negative', optional=True), io.Float.Input('cfg', default=1., min=0., max=100., step=0.1)],
            [*_window_outputs(), io.Latent.Output('lifted_full_av'), io.Custom(PREPARED).Output('prepared'),
             io.String.Output('report_json')],
            'Global learned lift/noise once, the same scoped joint windows in sequence. External EAV '
            'and Relay remain optional config inputs. Does not change the old integrated node/default.')

    @classmethod
    def execute(cls, model, partial4_denoised_output, plan, noise, sampler, sigmas, bank,
                eav_config=None, relay_config=None, clip=None, negative=None, cfg=1.):
        return io.NodeOutput(*run_scoped_v5(model, partial4_denoised_output, plan, noise, sampler,
            sigmas, bank, eav_config=eav_config, relay_config=relay_config, clip=clip, negative=negative, cfg=cfg))


NODES = [MiniMaxH3TemporalDialoguePlanEXPT8, MiniMaxH3TemporalNativeRecipeEXPT8,
         MiniMaxH3TemporalWindowBankEXPT8, MiniMaxH3TemporalWindowConditioningEXPT8,
         MiniMaxH3TemporalV5WindowEXPT8, MiniMaxH3TemporalV5WindowSaveEXPT8,
         MiniMaxH3TemporalV5WindowLoadEXPT8, MiniMaxH3TemporalRelayConfigEXPT8,
         MiniMaxH3TemporalV5RelayApplyEXPT8, MiniMaxH3TemporalV5RelayAuditEXPT8,
         MiniMaxH3TemporalV5JointPass2EXPT8]

from .nodes_temporal_h16 import NODES as _h16_nodes  # noqa: E402
NODES.extend(_h16_nodes)
from .nodes_temporal_resume import NODES as _resume_nodes  # noqa: E402
NODES.extend(_resume_nodes)
from .nodes_temporal_h16_resume import NODES as _h16_resume_nodes  # noqa: E402
NODES.extend(_h16_resume_nodes)
