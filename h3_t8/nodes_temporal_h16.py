"""Opt-in H16 scoped windows/effects/storage, independent of released H16."""
from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .nodes_temporal_dialogue import BANK, PLAN, _schema
from .modular_sampling.chunked_stage_nodes import SPEC, CONTEXT
from .modular_sampling.eav import CONFIG_TYPE as EAV_CONFIG
from .modular_sampling.temporal_chunked_relay import CONFIG_TYPE as RELAY_CONFIG
from .modular_sampling.temporal_h16 import sample_scoped_h16
from .modular_sampling.temporal_h16_effects import (
    RUNTIME_TYPE, bind_h16_scoped_effects, audit_h16_scoped_effects,
)
from .modular_sampling.temporal_h16_storage import (
    save_h16_scoped_window, load_h16_scoped_window, verify_h16_scoped_window, fingerprint_h16_scoped_window,
)
from .modular_sampling.results import canonical


RESULT = 'T8_TEMPORAL_H16_WINDOW_RESULT'


def _inputs():
    return [io.Latent.Input('source_segment'), io.Latent.Input('lifted_segment'),
            io.Custom(SPEC).Input('segment_spec'), io.Custom(CONTEXT).Input('pass2_context'),
            io.Custom(PLAN).Input('plan'), io.Custom(BANK).Input('bank')]


def _outputs():
    return [io.Latent.Output('cumulative_av_latent'), io.Custom(RESULT).Output('scoped_window_result')]


def _root():
    return Path(folder_paths.get_output_directory()) / 'MiniMaxH3' / 'temporal_dialogue_h16_artifacts'


class MiniMaxH3TemporalH16WindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 One Refined AV Window / Real Prefix', [io.Model.Input('model'),
            *_inputs(), io.Noise.Input('noise'), io.Sampler.Input('sampler'), io.Sigmas.Input('sigmas'),
            io.Custom(RESULT).Input('previous_result', optional=True), io.Conditioning.Input('positive', optional=True),
            io.Conditioning.Input('negative', optional=True), io.Float.Input('cfg', default=1., min=0., max=100., step=0.1)],
            [*_outputs(), io.String.Output('report_json')],
            'Explicit new H16 EXP: accepted AV prefix read-only, exact append, original unsampled audio '
            'padding retained. No old post-loop crossfade/energy gate. Old H16 defaults remain unchanged.')

    @classmethod
    def execute(cls, model, source_segment, lifted_segment, segment_spec, pass2_context, plan, bank,
                noise, sampler, sigmas, previous_result=None, positive=None, negative=None, cfg=1.):
        return io.NodeOutput(*sample_scoped_h16(model, source_segment, lifted_segment, segment_spec, pass2_context,
            plan, noise, sampler, sigmas, bank, previous_result, positive, negative, cfg))


class MiniMaxH3TemporalH16EffectsApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 External Scoped EAV / Relay Pair', [io.Model.Input('model'), *_inputs(),
            io.Sigmas.Input('sigmas'), io.Custom(RESULT).Input('previous_result', optional=True),
            io.Custom(EAV_CONFIG).Input('eav_config', optional=True),
            io.Custom(RELAY_CONFIG).Input('relay_config', optional=True), io.Clip.Input('clip', optional=True)],
            [io.Model.Output('model'), io.Conditioning.Output('positive'), io.Custom(RUNTIME_TYPE).Output('runtime'),
             io.String.Output('report_json')],
            'Independent effect configs bound to actual scoped H16 audio/video prefix. Relay native spans '
            'and local layout rebuilt; zero/single event bypass, finite bias is not hard isolation.')

    @classmethod
    def execute(cls, model, source_segment, lifted_segment, segment_spec, pass2_context, plan, bank, sigmas,
                previous_result=None, eav_config=None, relay_config=None, clip=None):
        return io.NodeOutput(*bind_h16_scoped_effects(model, source_segment, lifted_segment, segment_spec,
            pass2_context, plan, sigmas, bank, previous_result, clip=clip, relay_config=relay_config, eav_config=eav_config))


class MiniMaxH3TemporalH16EffectsAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 Audit Actual Scoped Effects', [io.Custom(RESULT).Input('scoped_window_result'),
            io.Custom(RUNTIME_TYPE).Input('runtime')], [io.Latent.Output('cumulative_av_latent'), io.String.Output('report_json')],
            output=True)

    @classmethod
    def execute(cls, scoped_window_result, runtime):
        output, report = audit_h16_scoped_effects(scoped_window_result, runtime)
        return io.NodeOutput(output, report, ui={'text': [report]})


class MiniMaxH3TemporalH16WindowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 Freeze Exact Scoped AV Window', [io.Custom(RESULT).Input('scoped_window_result'),
            *_inputs(), io.Boolean.Input('confirm_save', default=False)],
            [*_outputs(), io.String.Output('artifact_path'), io.String.Output('artifact_sha256'), io.String.Output('report_json')],
            'Separate new format binds real accepted audio, native bank, source, lift, global noise, policy '
            'and implementation. Old H16 refined crossfade caches cannot become new prefix receipts.', output=True)

    @classmethod
    def execute(cls, scoped_window_result, source_segment, lifted_segment, segment_spec, pass2_context,
                plan, bank, confirm_save=False):
        args = scoped_window_result, source_segment, lifted_segment, segment_spec, pass2_context, plan, bank
        if not confirm_save:
            verify_h16_scoped_window(*args)
            report = canonical(dict(status='not_saved_confirm_save_false', automatic_cache_reuse=False))
            return io.NodeOutput(scoped_window_result.output_latent, scoped_window_result, '', '', report, ui={'text': [report]})
        output = save_h16_scoped_window(*args, _root())
        return io.NodeOutput(*output, ui={'text': [f'artifact_path: {output[2]}', f'artifact_sha256: {output[3]}', output[4]]})


class MiniMaxH3TemporalH16WindowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 Load Explicit Scoped AV Window', [*_inputs(), io.String.Input('artifact_path', default=''),
            io.String.Input('artifact_sha256', default='')], [*_outputs(), io.String.Output('report_json')],
            'Literal selected prefix, no earlier sampling. Exact new-contract manifest SHA and current '
            'source/noise/lift/native conditions required. Not an automatic MODEL/provider-equivalence claim.')

    @classmethod
    def execute(cls, source_segment, lifted_segment, segment_spec, pass2_context, plan, bank, artifact_path, artifact_sha256):
        return io.NodeOutput(*load_h16_scoped_window(source_segment, lifted_segment, segment_spec, pass2_context,
            plan, bank, _root(), artifact_path, artifact_sha256))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_inputs):
        try:
            return fingerprint_h16_scoped_window(_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float('nan')


class MiniMaxH3TemporalH16JointPass2EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'H16 Integrated Scoped PASS2', [io.Model.Input('model'), io.Latent.Input('source_av_latent'),
            io.Custom(PLAN).Input('plan'), io.Custom(BANK).Input('bank'), io.Noise.Input('noise'),
            io.Sampler.Input('sampler'), io.Sigmas.Input('sigmas'), io.Custom(EAV_CONFIG).Input('eav_config', optional=True),
            io.Custom(RELAY_CONFIG).Input('relay_config', optional=True), io.Clip.Input('clip', optional=True),
            io.Conditioning.Input('negative', optional=True), io.Float.Input('cfg', default=1., min=0., max=100., step=0.1)],
            [*_outputs(), io.String.Output('report_json')],
            'Convenience loop over the same separated scoped H16 windows and external configs. '
            'One global AV noise preparation, original per-segment learned lifts. Old H16 remains unchanged.')

    @classmethod
    def execute(cls, model, source_av_latent, plan, bank, noise, sampler, sigmas, eav_config=None,
                relay_config=None, clip=None, negative=None, cfg=1.):
        from .temporal_dialogue_bank import validate_window_bank
        from .modular_sampling.chunked_source import slice_chunked_source
        from .modular_sampling.chunked_stages import prepare_chunked_pass2, lift_chunked_segment
        validate_window_bank(bank, source_av_latent, plan)
        context, _ = prepare_chunked_pass2(source_av_latent, plan, noise)
        previous, reports = None, []
        for index in range(len(bank.encoded)):
            segment, spec, _ = slice_chunked_source(source_av_latent, plan, index)
            lifted, _ = lift_chunked_segment(segment, spec, context, plan)
            selected, positive, runtime = model, None, None
            if eav_config is not None or relay_config is not None:
                selected, positive, runtime, _ = bind_h16_scoped_effects(model, segment, lifted, spec,
                    context, plan, sigmas, bank, previous, clip=clip, eav_config=eav_config, relay_config=relay_config)
            output, current, report = sample_scoped_h16(selected, segment, lifted, spec, context, plan,
                noise, sampler, sigmas, bank, previous, positive, negative, cfg)
            if runtime is not None:
                report = canonical(dict(window_report=report, effect_report=audit_h16_scoped_effects(current, runtime)[1]))
            reports.append(report)
            previous = current
        return io.NodeOutput(output, previous, canonical(dict(status='scoped_h16_chain_completed_quality_unverified',
            windows=reports, bank_sha256=bank.sha256, hard_time_isolation=False, quality_accepted=False)))


NODES = [MiniMaxH3TemporalH16WindowEXPT8, MiniMaxH3TemporalH16EffectsApplyEXPT8,
         MiniMaxH3TemporalH16EffectsAuditEXPT8, MiniMaxH3TemporalH16WindowSaveEXPT8,
         MiniMaxH3TemporalH16WindowLoadEXPT8, MiniMaxH3TemporalH16JointPass2EXPT8]
