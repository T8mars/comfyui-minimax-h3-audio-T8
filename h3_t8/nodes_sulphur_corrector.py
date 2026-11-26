"""Append-only T8 entrypoints; H3_CORRECTOR interoperates with the original nodes."""
from comfy_api.latest import io

from .sulphur_corrector import apply_corrector, corrector_names, load_corrector

CORRECTOR = io.Custom('H3_CORRECTOR')


class MiniMaxH3CorrectorLoaderT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Sulphur · Corrector加载 / Load (T8)',
                         category='T8/MiniMax H3/Corrector',
                         description='从 models/loras 或 models/h3_correctors 加载 Corrector，保留FP32权重。'
                                     '它不是LoRA，不加载或更换H3底模；原版s3_corrector约需8GB权重内存。',
                         inputs=[io.Combo.Input('corrector', options=corrector_names())],
                         outputs=[CORRECTOR.Output('corrector'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, corrector):
        return io.NodeOutput(*load_corrector(corrector))


class MiniMaxH3CorrectorApplyT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Sulphur · Corrector应用 / Apply (T8)',
                         category='T8/MiniMax H3/Corrector',
                         description='在CFG后修正视频预测，原音频切片不变。可接普通或T8采样器；'
                                     '双采两路MODEL分别应用，步号按各自采样表从0计。保留已有LoRA与补丁。'
                                     '原版训练于S3，其它H3模型允许尝试，效果需自验；0强度关闭。',
                         inputs=[io.Model.Input('model'), CORRECTOR.Input('corrector'),
                                 io.Float.Input('strength', default=1., min=0., max=None, step=.05),
                                 io.Float.Input('budget_fraction', default=.5, min=.000001, max=None, step=.05),
                                 io.Float.Input('velocity_floor', default=.001, min=.000001, max=None, step=.0001),
                                 io.Int.Input('start_step', default=0, min=0, max=None),
                                 io.Int.Input('end_step', default=-1, min=-1, max=None),
                                 io.Sigmas.Input('sigmas', optional=True)],
                         outputs=[io.Model.Output('model'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, corrector, strength=1., budget_fraction=.5, velocity_floor=.001,
                start_step=0, end_step=-1, sigmas=None):
        return io.NodeOutput(*apply_corrector(model, corrector, strength, budget_fraction,
                                              velocity_floor, start_step, end_step, sigmas))


NODES = [MiniMaxH3CorrectorLoaderT8, MiniMaxH3CorrectorApplyT8]
