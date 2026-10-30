import folder_paths
from comfy_api.latest import io

from .pika_two_pass import sample_pika_two_pass


class MiniMaxH3PikaTwoPassSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name='H3 Pika Flex · 双采4+2 / 3+3 (EXP/T8)',
            category='T8/MiniMax H3/Sampling/Experimental', is_experimental=True,
            description='Pika完整6步：低尺寸采样→学习型3D潜空间放大→高尺寸继续。'
                        'LOW/HIGH分别连接对应尺寸的条件和AV潜空间，时间网格保持相同。'
                        '底模可共用；各阶段可串联多个LoRA，独立设置顺序和强度。'
                        '不接model_high时沿用LOW模型。不会下载或替换LoRA。'
                        '音频继续联合采样；LOW输出是中间预测，不是完成的视频。',
            inputs=[io.Model.Input('model'), io.Conditioning.Input('positive_low'),
                    io.Latent.Input('av_latent_low'), io.Conditioning.Input('positive_high'),
                    io.Latent.Input('av_latent_high'),
                    io.Combo.Input('upscaler_model', options=folder_paths.get_filename_list('latent_upscale_models')),
                    io.Int.Input('seed', default=301030, min=0, max=2**64-1, control_after_generate=True),
                    io.Int.Input('low_evaluations', default=4, min=1, max=5,
                                 tooltip='4表示LOW4步+HIGH2步；3表示LOW3步+HIGH3步，总共6步。'),
                    io.Combo.Input('precision', options=['fp16', 'bf16', 'fp32'], default='fp16', advanced=True),
                    io.Model.Input('model_high', optional=True,
                                   tooltip='从共用原始底模另开LoRA分支，可独立叠加多个LoRA。')],
            outputs=[io.Latent.Output('av_latent'), io.Latent.Output('low_clean_prediction'),
                     io.String.Output('report_json')])

    @classmethod
    def execute(cls, **kwargs):
        import comfy.utils
        bar = comfy.utils.ProgressBar(6)

        def notify(index, prediction, state, total):
            bar.update_absolute(index + 1, total)

        return io.NodeOutput(*sample_pika_two_pass(**kwargs, callback=notify))


NODES = [MiniMaxH3PikaTwoPassSamplerEXPT8]
