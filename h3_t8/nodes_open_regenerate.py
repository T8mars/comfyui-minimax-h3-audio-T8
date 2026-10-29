from comfy_api.latest import io
import folder_paths

from .open_regenerate import SCHEDULE_OPTIONS, prepare_refine
from . import open_regenerate_recipe as recipe


class MiniMaxH3OpenRegenerateSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Open Regenerate · 细化采样准备 (EXP/T8)",
            category="T8/MiniMax H3/Refinement/Experimental", is_experimental=True,
            description="输入已编码/已放大的H3视频潜空间。输出Euler原始sigma和锁定的VAE编码静音，不二次套12/3时钟。用BasicGuider；原声音轨最后单独合并。不加载/叠加LoRA，不把Kijai纯refiner当成官方Turbo+Sharp合并配方。",
            inputs=[io.Model.Input("model"), io.Latent.Input("video_latent"), io.Vae.Input("audio_vae"),
                    io.Combo.Input("schedule", options=list(SCHEDULE_OPTIONS), default="published_3step"),
                    io.Int.Input("custom_steps", default=3, min=1, max=None, advanced=True),
                    io.Float.Input("start_sigma", default=0.909375, min=0.000001, max=1.0, step=0.000001, advanced=True)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*prepare_refine(**kwargs))


def _preferred(folder, preferred):
    # Comfy lists nested model files with the host separator. Portable graphs
    # and the matching combo options must use the same spelling on Windows.
    names = [name.replace('\\', '/') for name in folder_paths.get_filename_list(folder)]
    return sorted(names, key=lambda name: (name != preferred, name))


class MiniMaxH3OpenRegenerateRecipeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Open Regenerate · 完整配方加载 (EXP/T8)',
            category='T8/MiniMax H3/Refinement/Experimental', is_experimental=True,
            description='加载作者已合并的Refiner+Turbo+Sharp普通权重，另加独立Sharp时间曲线分支。'
                        '曲线来源另选作者使用的Ref2VA pruned底模；不会用当前底模的另一张曲线替换它。'
                        '保留上游MODEL/LoRA。加载不代表采样或画质通过，采样后连接实际运行报告。',
            inputs=[io.Model.Input('model'),
                io.Combo.Input('merged_lora_name', options=_preferred('loras', 'OpenRegenerateAuthor/open_h3_regenerate_2k_lora.safetensors')),
                io.Combo.Input('curve_lora_name', options=_preferred('loras', 'OpenRegenerateAuthor/sharp_curve_branches.safetensors')),
                io.Combo.Input('curve_source_name', options=_preferred('diffusion_models', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')),
                io.Float.Input('linear_strength', default=1., min=None, max=None, step=.01),
                io.Float.Input('curve_strength', default=1., min=None, max=None, step=.01)],
            outputs=[io.Model.Output('model'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, merged_lora_name, curve_lora_name, curve_source_name,
                linear_strength=1., curve_strength=1.):
        selected, report = recipe.load_recipe(model,
            folder_paths.get_full_path_or_raise('loras', merged_lora_name),
            folder_paths.get_full_path_or_raise('loras', curve_lora_name),
            folder_paths.get_full_path_or_raise('diffusion_models', curve_source_name),
            linear_strength, curve_strength)
        return io.NodeOutput(selected, report)

    @classmethod
    def fingerprint_inputs(cls, merged_lora_name, curve_lora_name, curve_source_name, **kwargs):
        return (recipe.file_sha(folder_paths.get_full_path_or_raise('loras', merged_lora_name)),
            recipe.file_sha(folder_paths.get_full_path_or_raise('loras', curve_lora_name)),
            recipe.tensor_sha(recipe.read_curve_table(folder_paths.get_full_path_or_raise('diffusion_models', curve_source_name))))


class MiniMaxH3OpenRegenerateAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Open Regenerate · 实际运行报告 (EXP/T8)',
            category='T8/MiniMax H3/Refinement/Experimental', is_experimental=True, is_output_node=True,
            description='after_sampling连接实际采样输出，报告已观察前向及Sharp分支调用，原样返回LATENT。'
                        '没有观察到采样时保持未验证，不把报告当成人工画质接受。',
            inputs=[io.Model.Input('model'), io.Latent.Input('after_sampling')],
            outputs=[io.Latent.Output('samples'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, model, after_sampling):
        latent, report = recipe.audit_recipe(model, after_sampling)
        return io.NodeOutput(latent, report, ui={'text': [report]})

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


NODES = [MiniMaxH3OpenRegenerateSetupEXPT8, MiniMaxH3OpenRegenerateRecipeEXPT8, MiniMaxH3OpenRegenerateAuditEXPT8]
