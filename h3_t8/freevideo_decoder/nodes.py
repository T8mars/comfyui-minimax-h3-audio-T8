"""Append-only typed decoder routes; legacy Core VAE paths remain untouched."""
from comfy_api.latest import io

from . import runtime

CATEGORY = 'T8/MiniMax H3/FreeVideo/Decoder EXP'


def schema(cls, family):
    socket = 'H3_T8_FREEVIDEO_QUALITY_STAGE' if family == 'quality' else 'H3_T8_FREEVIDEO_STAGE'
    label = 'Quality HIGH／单采' if family == 'quality' else '原8+2／真4+4 HIGH'
    return io.Schema(node_id=cls.__name__, display_name='FreeVideo · 独立视频解码 · ' + label,
        category=CATEGORY, is_experimental=True,
        description='只接完整终态；原音频latent原样传出，另接正常音频VAE。独立原FP32 VAE分片与v0.2.3配置；默认eager，compile需显式选择，不自动回退，不改变旧解码或采样。',
        inputs=[io.Custom(socket).Input('completed_stage'), io.String.Input('decoder_config', default=''),
            io.Combo.Input('decode_mode', options=['eager', 'compile'], default='eager')],
        outputs=[io.Image.Output('images'), io.Latent.Output('original_audio_latent'), io.String.Output('report_json')])


def execute(stage, path, mode, family):
    import comfy.model_management
    import comfy.utils
    bar = comfy.utils.ProgressBar(2)
    images, audio, report = runtime.decode(stage, path, family, mode,
        interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
        progress=lambda done, total: bar.update_absolute(done, total))
    return io.NodeOutput(images, audio, report)


class MiniMaxH3FreeVideoQualityVideoDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, 'quality')

    @classmethod
    def execute(cls, completed_stage, decoder_config='', decode_mode='eager'):
        return execute(completed_stage, decoder_config, decode_mode, 'quality')


class MiniMaxH3FreeVideoVideoDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, 'legacy')

    @classmethod
    def execute(cls, completed_stage, decoder_config='', decode_mode='eager'):
        return execute(completed_stage, decoder_config, decode_mode, 'legacy')


NODES = [MiniMaxH3FreeVideoQualityVideoDecodeEXPT8, MiniMaxH3FreeVideoVideoDecodeEXPT8]
