"""Append-only DMD8 4+4 nodes; old completed LOW/HIGH contracts unchanged."""
from pathlib import Path
import folder_paths
from comfy_api.latest import io
from .runtime import av_output
from .split_runtime import load_mid, mid_av, sample_split, save_mid, validate_mid

MID = io.Custom("H3_T8_FREEVIDEO_MID")
MODEL = io.Custom("H3_T8_FREEVIDEO_MODEL")
STAGE = io.Custom("H3_T8_FREEVIDEO_STAGE")
CATEGORY = "T8/MiniMax H3/FreeVideo/Experimental"


class MiniMaxH3FreeVideoSplitLOWEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 分段LOW 前4步 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="原8步时序前4步；partial x0接外置learned放大。MID音频仍含噪声，必须接分段HIGH后4步，不能当成片解码。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"),
                io.Int.Input("width", default=448, min=256, max=4096, step=32),
                io.Int.Input("height", default=256, min=256, max=4096, step=32),
                io.Int.Input("frames", default=124, min=39, max=2000, step=17),
                io.Int.Input("seed", default=171, min=0, max=2**63-1, control_after_generate=True),
                io.Clip.Input("clip_to_offload", optional=True)],
            outputs=[io.Latent.Output("partial_x0_av"), MID.Output("mid_state"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, width=448, height=256, frames=124, seed=171, clip_to_offload=None):
        from .nodes import callbacks, offload_supplied_clip
        interrupt, progress = callbacks()
        offload_supplied_clip(clip_to_offload)
        state = sample_split(freevideo_model, conditioning, width, height, frames, seed, interrupt=interrupt, progress=progress)
        return io.NodeOutput(mid_av(state), state, state.receipt_json)


class MiniMaxH3FreeVideoSplitHIGHEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 分段HIGH 后4步 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="原8步时序后4步：放大的partial x0按独立seed重新加噪到原index4；真实MID音频继续联合去噪，不冻结。兼容分段路线，非作者8+2配方。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"), MID.Input("mid_state"),
                io.Latent.Input("lifted_partial_x0_av"), io.Int.Input("seed", default=172, min=0, max=2**63-1, control_after_generate=True),
                io.Clip.Input("clip_to_offload", optional=True)],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, mid_state, lifted_partial_x0_av, seed=172, clip_to_offload=None):
        from .nodes import callbacks, offload_supplied_clip
        from ..core import nested_av_parts
        receipt = validate_mid(mid_state)
        video, _ = nested_av_parts(lifted_partial_x0_av)
        interrupt, progress = callbacks()
        offload_supplied_clip(clip_to_offload)
        state = sample_split(freevideo_model, conditioning, video.shape[-1] * 16, video.shape[-2] * 16,
            receipt["geometry"]["frames"], seed, mid=mid_state, lifted=lifted_partial_x0_av, interrupt=interrupt, progress=progress)
        return io.NodeOutput(av_output(state), state, state.receipt_json)


class MiniMaxH3FreeVideoMIDSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 保存4步MID (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True, inputs=[MID.Input("mid_state")],
            outputs=[MID.Output("mid_state"), io.String.Output("manifest_path"), io.String.Output("manifest_sha256")])

    @classmethod
    def execute(cls, mid_state):
        path, sha = save_mid(mid_state, Path(folder_paths.get_output_directory()) / "T8-FreeVideo/split-mid")
        return io.NodeOutput(mid_state, path, sha, ui={"text": [path, sha]})


class MiniMaxH3FreeVideoMIDLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 冷加载4步MID (T8 EXP)", category=CATEGORY,
            is_experimental=True, inputs=[io.String.Input("manifest_path", default=""), io.String.Input("manifest_sha256", default="")],
            outputs=[io.Latent.Output("partial_x0_av"), MID.Output("mid_state"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, manifest_path, manifest_sha256):
        state = load_mid(manifest_path, manifest_sha256)
        return io.NodeOutput(mid_av(state), state, state.receipt_json)


NODES = [MiniMaxH3FreeVideoSplitLOWEXPT8, MiniMaxH3FreeVideoSplitHIGHEXPT8,
         MiniMaxH3FreeVideoMIDSaveEXPT8, MiniMaxH3FreeVideoMIDLoadEXPT8]
