import json

from comfy_api.latest import io
import folder_paths

from .lora_windows import attach_windows, create_window_hook


class MiniMaxH3LoRAWindowT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LoRAWindowT8",
                         display_name="H3 LoRA Window / LoRA采样生效区间 (EXP/T8)",
                         category="T8/MiniMax H3/LoRA/Experimental", is_experimental=True,
                         description="Create native weight hooks and attach them to conditioning. "
                                     "Progress uses the selected model's actual percent-to-sigma mapping. "
                                     "Quantized/offloaded weight switching may substantially increase host RAM and runtime. "
                                     "Chain previous_hooks for multiple LoRAs; don't also load the same LoRA statically.",
                         inputs=[io.Model.Input("model"),
                                 io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                                 io.Float.Input("strength", default=1.0, min=None, max=None, step=0.01),
                                 io.Float.Input("start_percent", default=0.0, min=0., max=1., step=0.01),
                                 io.Float.Input("end_percent", default=1.0, min=0., max=1., step=0.01),
                                 io.Custom("HOOKS").Input("previous_hooks", optional=True)],
                         outputs=[io.Custom("HOOKS").Output("hooks"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, lora_name, strength, start_percent, end_percent, previous_hooks=None):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        hooks, report = create_window_hook(model, path, strength, start_percent, end_percent, previous_hooks)
        return io.NodeOutput(hooks, json.dumps(report))


class MiniMaxH3AttachLoRAWindowsT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3AttachLoRAWindowsT8",
                         display_name="H3 Multi LoRA Windows / 多LoRA区间条件 (EXP/T8)",
                         category="T8/MiniMax H3/LoRA/Experimental", is_experimental=True,
                         inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                                 io.String.Input("windows_json", multiline=True, default="[]")],
                         outputs=[io.Conditioning.Output("positive"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, positive, windows_json):
        return io.NodeOutput(*attach_windows(model, positive, json.loads(windows_json)))


NODES = [MiniMaxH3LoRAWindowT8, MiniMaxH3AttachLoRAWindowsT8]
