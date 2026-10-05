"""Explicit H16 literal Full/Cold bundle, appended after the original nodes."""
from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .nodes_temporal_dialogue import BANK,PLAN,_schema
from .modular_sampling.chunked_stage_nodes import CONTEXT,SPEC
from .nodes_temporal_h16 import RESULT
from .modular_sampling.temporal_h16_resume import (
    prepare_bundle,select_bundle_window,save_h16_resume,load_h16_resume,verify_bundle,
)
from .modular_sampling.storage import fingerprint_stage
from .modular_sampling.results import canonical

BUNDLE = 'T8_TEMPORAL_H16_LITERAL_BUNDLE'


def _root():
    return Path(folder_paths.get_output_directory())/'MiniMaxH3'/'temporal_dialogue_h16_resume_artifacts'


def _piece_outputs():
    return [io.Latent.Output('source_segment'),io.Latent.Output('lifted_segment'),io.Custom(SPEC).Output('segment_spec'),
        io.Custom(CONTEXT).Output('pass2_context'),io.Custom(PLAN).Output('plan'),io.Custom(BANK).Output('bank'),
        io.Latent.Output('full_source_av_latent')]


class MiniMaxH3TemporalH16BundlePrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,'H16 Prepare Literal Bundle / ALL Piece Lifts',[
            io.Custom(CONTEXT).Input('pass2_context'),io.Custom(PLAN).Input('plan'),io.Custom(BANK).Input('bank')],
            [io.Custom(BUNDLE).Output('bundle'),io.String.Output('report_json')],
            'Optional literal restart preparation. Original per-piece learned lift, native values once. '
            'No HIGH sampling. Foreign live-only metadata uses ordinary Full instead.')

    @classmethod
    def execute(cls,pass2_context,plan,bank):
        bundle = prepare_bundle(pass2_context,plan,bank)
        return io.NodeOutput(bundle,canonical(dict(status='H16_literal_bundle_prepared',pieces=len(bundle.pieces),
            quality_accepted=False,sampling_executed=False)))


class MiniMaxH3TemporalH16BundleWindowEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,'H16 Select Literal Piece / NO Earlier Work',[
            io.Custom(BUNDLE).Input('bundle'),io.Int.Input('window_index',default=0,min=0,max=9999)],_piece_outputs(),
            'Exact piece, stored original lift, global AV noise context and native bank. No sampling/encoding/lift.')

    @classmethod
    def execute(cls,bundle,window_index):
        return io.NodeOutput(*select_bundle_window(bundle,window_index),bundle.context.source_latent)


class MiniMaxH3TemporalH16ResumeSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,'H16 Freeze Whole Literal Restart Capsule',[
            io.Custom(BUNDLE).Input('bundle'),io.Custom(RESULT).Input('scoped_window_result'),
            io.Boolean.Input('confirm_save',default=False)],
            [io.Custom(BUNDLE).Output('bundle'),io.Custom(RESULT).Output('scoped_window_result'),
             io.String.Output('artifact_path'),io.String.Output('artifact_sha256'),io.String.Output('report_json')],
            'Independent H16 format, explicit saved prefix plus all native source/conditions/noise/piece lifts. '
            'Not old crossfade cache, MODEL identity certification or quality approval.',output=True)

    @classmethod
    def execute(cls,bundle,scoped_window_result,confirm_save=False):
        verify_bundle(bundle)
        from .modular_sampling.temporal_h16_storage import verify_h16_scoped_window
        verify_h16_scoped_window(scoped_window_result,*select_bundle_window(bundle,scoped_window_result.index))
        saved = save_h16_resume(bundle,scoped_window_result,_root()) if confirm_save else (
            '','',canonical(dict(status='not_saved_confirm_save_false')))
        return io.NodeOutput(bundle,scoped_window_result,*saved,
            ui={'text':[f'artifact_path: {saved[0]}',f'artifact_sha256: {saved[1]}',saved[2]]})


class MiniMaxH3TemporalH16ResumeLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,'H16 Load Whole Capsule / ONLY Remaining Windows',[
            io.String.Input('artifact_path',default=''),io.String.Input('artifact_sha256',default=''),
            io.Int.Input('expected_window_index',default=0,min=0,max=9999)],
            [io.Custom(BUNDLE).Output('bundle'),io.Custom(RESULT).Output('scoped_window_result'),io.String.Output('report_json')],
            'Literal CPU restore. Select index+1 with Bundle Window; no LOW/CLIP/media encoding/learned '
            'lift/noise regeneration/earlier HIGH. Reconnect original HIGH MODEL/sampler/seed/SIGMAS.')

    @classmethod
    def execute(cls,artifact_path,artifact_sha256,expected_window_index):
        return io.NodeOutput(*load_h16_resume(_root(),artifact_path,artifact_sha256,expected_window_index))

    @classmethod
    def fingerprint_inputs(cls,artifact_path,artifact_sha256,**_inputs):
        try:
            return fingerprint_stage(_root(),artifact_path)
        except (ValueError,OSError,RuntimeError):
            return float('nan')


NODES = [MiniMaxH3TemporalH16BundlePrepareEXPT8,MiniMaxH3TemporalH16BundleWindowEXPT8,
         MiniMaxH3TemporalH16ResumeSaveEXPT8,MiniMaxH3TemporalH16ResumeLoadEXPT8]
