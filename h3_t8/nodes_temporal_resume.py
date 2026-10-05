"""Literal restart capsules and checked completed-audio delivery, append-only."""
from pathlib import Path

import comfy.nested_tensor
from comfy_api.latest import io
import folder_paths

from .nodes_temporal_dialogue import _schema, _source_inputs, BANK, SCOPED, PLAN, PREPARED
from .modular_sampling.temporal_resume import save_resume, load_resume, fingerprint_resume
from .modular_sampling.temporal_chunked_storage import verify_scoped_window
from .modular_sampling.results import StageResult, _input_identity, canonical
from .modular_sampling.contracts import StageContext


def _root():
    return Path(folder_paths.get_output_directory())/'MiniMaxH3'/'temporal_dialogue_resume_artifacts'


class MiniMaxH3TemporalV5ResumeSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Freeze Literal V5 Restart Capsule', [io.Custom(SCOPED).Input('scoped_window_result'),
            *_source_inputs(), io.Boolean.Input('confirm_save', default=False)],
            [io.Latent.Output('cumulative_av_latent'), io.Custom(SCOPED).Output('scoped_window_result'),
             io.String.Output('artifact_path'), io.String.Output('artifact_sha256'), io.String.Output('report_json')],
            'Explicit whole restart capsule: actual partial4, learned lift, global noise, all native window '
            'conditions and accepted AV prefix. No MODEL/CLIP/callback pickle or provider certification.', output=True)

    @classmethod
    def execute(cls, scoped_window_result, partial4_denoised_output, lifted_full_av, prepared, plan, bank, confirm_save=False):
        args = scoped_window_result, partial4_denoised_output, lifted_full_av, prepared, plan, bank
        verify_scoped_window(*args)
        output = ('','',canonical(dict(status='not_saved_confirm_save_false'))) if not confirm_save else save_resume(*args, _root())
        return io.NodeOutput(scoped_window_result.base.output_latent, scoped_window_result, *output,
            ui={'text':[f'artifact_path: {output[0]}',f'artifact_sha256: {output[1]}',output[2]]})


class MiniMaxH3TemporalV5ResumeLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Load Literal V5 Capsule / NO Earlier Work', [
            io.String.Input('artifact_path', default=''), io.String.Input('artifact_sha256', default=''),
            io.Int.Input('expected_window_index', default=0, min=0, max=9999)],
            [io.Latent.Output('partial4_denoised_output'), io.Latent.Output('lifted_full_av'),
             io.Custom(PREPARED).Output('prepared'), io.Custom(PLAN).Output('plan'), io.Custom(BANK).Output('bank'),
             io.Custom(SCOPED).Output('scoped_window_result'), io.String.Output('report_json')],
            'Loads exact selected inert values on CPU. No LOW sampling, CLIP encoding, VAE media encode, '
            'learned lift or noise regeneration. Connect the original HIGH noise seed/sampler/refine sigmas '
            'and matching MODEL, then only window index+1. Not automatic reuse or quality approval.')

    @classmethod
    def execute(cls, artifact_path, artifact_sha256, expected_window_index):
        return io.NodeOutput(*load_resume(_root(), artifact_path, artifact_sha256, expected_window_index))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_inputs):
        try:
            return fingerprint_resume(_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float('nan')


def preserve_completed_audio(video_result, completed_source):
    if type(completed_source) is not StageResult:
        raise ValueError('Audio passthrough requires an actual completed Stage Result, not partial x0 or a label')
    receipt = completed_source.verify()
    context = StageContext.from_dict(receipt['request']['stage_context'])
    execution = receipt['execution']
    if (receipt.get('verified_recipe_completion') is not True
            or context.end != len(context.trajectory_sigmas)-1
            or context.trajectory_sigmas[-1] != 0.
            or execution['callbacks'] != list(range(context.steps))
            or execution['denoiser_evaluations'] != context.steps):
        raise ValueError('Source audio is not an observed terminal-zero completed trajectory; partial4 stays joint')
    video, _ = video_result['samples'].unbind()
    source_video, audio = completed_source.output['samples'].unbind()
    if tuple(video.shape[:3]) != tuple(source_video.shape[:3]):
        raise ValueError('Completed audio and HIGH video timelines differ')
    result = {'samples':comfy.nested_tensor.NestedTensor((video, audio))}
    return result, canonical(dict(status='verified_terminal_audio_exact_passthrough',
        source_receipt_sha256=receipt['receipt_sha256'], audio_identity=_input_identity(audio),
        sampling_executed=False, quality_accepted=False,
        boundary='For existing video-only/preserve_first_pass HIGH routes. Does not freeze partial4 or repair lip-sync.'))


class MiniMaxH3TemporalCompletedAudioEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, 'Verified Completed LOW Audio Passthrough', [io.Latent.Input('high_video_result'),
            io.Custom('T8_STAGE_RESULT').Input('completed_source')],
            [io.Latent.Output('av_latent'), io.String.Output('report_json')],
            'Explicit opt-in delivery for already completed native audio and existing preserve/video-only '
            'HIGH. Actual completion receipt + terminal zero required; partial4 is rejected, not frozen. '
            'Does not sample HIGH, certify voice quality or fix mouth motion.')

    @classmethod
    def execute(cls, high_video_result, completed_source):
        return io.NodeOutput(*preserve_completed_audio(high_video_result, completed_source))


NODES = [MiniMaxH3TemporalV5ResumeSaveEXPT8, MiniMaxH3TemporalV5ResumeLoadEXPT8, MiniMaxH3TemporalCompletedAudioEXPT8]
