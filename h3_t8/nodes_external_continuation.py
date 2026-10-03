"""Three append-only operations; external takes do not enter native-parent chains."""
from comfy_api.latest import io
from . import external_continuation as bridge
from .director_project import sha
from .director_routes import get_store
from .nodes_long_video_exp import MiniMaxH3LongVideoConditioningT8
from .long_video import patch_long_video_model

CATEGORY = 'T8/MiniMax H3/External Continuation Experimental'


class MiniMaxH3ExternalContinuationSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 External · Saved Adopted Take (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Capture the SAVED manually adopted external take, complete file SHA, actual24fps '
                'zero-origin clocks and saved integer trim. Not a generated StageResult/native latent ancestor. '
                'No save/adopt/queue/convert/mux. Unsaved Director draft is not used.',
            inputs=[io.String.Input('project_id', default=''), io.String.Input('shot_id', default=''),
                    io.String.Input('take_id', default=''),
                    io.Combo.Input('context_frames', options=[5, 22, 39], default=22),
                    io.Combo.Input('audio_policy', options=sorted(bridge.AUDIO_POLICIES), default='video_only')],
            outputs=[io.Custom(bridge.SOURCE_TYPE).Output('external_source'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, **request):
        import folder_paths
        source = bridge.capture_source(get_store(), folder_paths.get_output_directory(), **request)
        return io.NodeOutput(source, source.contract_json)

    @classmethod
    def fingerprint_inputs(cls, **request):
        import folder_paths
        try:
            return sha(bridge.current_binding(get_store(), folder_paths.get_output_directory(), request))
        except (ValueError, OSError, KeyError):
            return float('nan')


class MiniMaxH3ExternalContextEncodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 External · RGB / PCM Context Encode (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Decode the actual adopted integer tail, existing RGB resize then current H3 video VAE. '
                'Audio is explicitly unused or same-rate/same-channel stereo PCM reencoded with connected audio VAE. '
                'Reencoding is NOT reconstruction of original sampled latent state. Use independent LOW/HIGH instances. '
                'Zero diffusion evaluations; no original media modification or mux. Unknown VAE wrappers execute nonportable.',
            inputs=[io.Custom(bridge.SOURCE_TYPE).Input('external_source'), io.Vae.Input('video_vae'),
                    io.Int.Input('width', default=448, min=32, max=16384, step=32),
                    io.Int.Input('height', default=256, min=32, max=16384, step=32),
                    io.Vae.Input('audio_vae', optional=True)],
            outputs=[io.Custom(bridge.CONTEXT_TYPE).Output('external_context'),
                     io.Image.Output('actual_last_frame'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, external_source, video_vae, width=448, height=256, audio_vae=None):
        context = bridge.prepare_context(external_source, video_vae, width, height, audio_vae)
        return io.NodeOutput(context, context.last_frame, context.contract_json)

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        # No opaque encoder output cache, saved native-state substitution or
        # new persistent VAE cache. Source binding is checked on every execution.
        return float('nan')


class MiniMaxH3ExternalContinuationConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3LongVideoConditioningT8.define_schema()
        removed = {'context', 'segment_index', 'context_frames', 'context_audio', 'width', 'height'}
        inherited = [item for item in original.inputs if item.id not in removed]
        for item in inherited:
            if item.id == 'length':
                item.force_input = False
        return io.Schema(node_id=cls.__name__, display_name='H3 External · ONE Conditions / Model (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='ONE independently editable native conditioning node for the explicit external RGB/PCM '
                'context. Original long-video motion builder and payload patch, not fake accepted native-parent '
                'ancestry. Connect ordinary or separated sampler nodes; external EAV/Relay/Bridge remain separate. '
                'Returned render includes context frames: explicitly trim that prefix for delivery; no automatic '
                'append, adoption or exact-source seam guarantee.',
            inputs=[io.Custom(bridge.CONTEXT_TYPE).Input('external_context'), *inherited], outputs=original.outputs)

    @classmethod
    def execute(cls, external_context, model, clip, video_vae, audio_vae, prompt, length=124, **options):
        outputs = bridge.condition(external_context, clip=clip, video_vae=video_vae, audio_vae=audio_vae,
                                   prompt=prompt, length=length, **options)
        return io.NodeOutput(patch_long_video_model(model), *outputs)


NODES = [MiniMaxH3ExternalContinuationSourceEXPT8, MiniMaxH3ExternalContextEncodeEXPT8,
         MiniMaxH3ExternalContinuationConditioningEXPT8]
