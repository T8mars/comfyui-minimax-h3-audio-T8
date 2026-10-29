import json
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .h3_av_delivery import save_h3_av_safe


class MiniMaxH3SafeAVSaveT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SafeAVSaveT8Advanced",
                         display_name="MiniMax H3 Safe AV Save / 安全音视频保存 (Advanced EXP/T8)",
                         description="Encode raw even-size IMAGE and AUDIO at 24fps through isolated single-thread H264/AAC. Strict decode before publication, no overwrite. Requires FFmpeg on PATH. Returns reusable VIDEO without a second SaveVideo.",
                         category="T8/MiniMax H3/Output/Advanced", is_experimental=True, is_output_node=True,
                         inputs=[io.Image.Input("images"), io.Audio.Input("audio"),
                                 io.String.Input("filename_prefix", default="MiniMaxH3/VDN_Refine"),
                                 io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                                 io.Boolean.Input("preserve_audio_length", default=False, optional=True, advanced=True,
                                                  tooltip="Preserve an already trimmed audio interval instead of padding it to the video endpoint.")],
                         outputs=[io.Video.Output("video"), io.String.Output("saved_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, images, audio, filename_prefix, crf=18, preserve_audio_length=False):
        if getattr(images, "ndim", None) != 4:
            raise ValueError("IMAGE must be [frames,height,width,channels]")
        root = Path(folder_paths.get_output_directory()).resolve()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(root), images.shape[2], images.shape[1])
        name = f"{filename}_{counter:05}_.mp4"
        output = (Path(folder) / name).resolve()
        if not output.is_relative_to(root):
            raise ValueError("H3 safe output must stay inside ComfyUI output")
        report = save_h3_av_safe(images, audio, output, crf=crf, preserve_audio_length=preserve_audio_length)
        return io.NodeOutput(InputImpl.VideoFromFile(str(output)), str(output), json.dumps(report, ensure_ascii=False, indent=2),
                             ui=ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]))
