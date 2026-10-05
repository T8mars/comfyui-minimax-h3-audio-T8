"""Append-only pixel delivery adapter; existing saved master is a real dependency."""
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui
from comfy_execution.graph_utils import ExecutionBlocker

from .postprocess_delivery import finalize_postprocess


class MiniMaxH3PostprocessSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PostprocessSaveEXPT8",
            display_name="H3 Postprocess Save / 后处理保存·原片保留 (EXP/T8)",
            category="T8/MiniMax H3/Output/Advanced", is_experimental=True, is_output_node=True,
            description="Full-length final RGB crop/pad/resize, against an already saved untrimmed zero-origin24fps SDR master. Copy original audio packets and verify PCM/PTS. Failure blocks only the postprocessed output, preserves master, and writes postprocess_failed. Never reruns sampling or automatically accepts quality.",
            inputs=[io.Video.Input("master_video"), io.Image.Input("processed_frames"),
                    io.String.Input("filename_prefix", default="MiniMaxH3/Postprocess/final"),
                    io.Boolean.Input("confirm_postprocess", default=False),
                    io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                    io.String.Input("expected_master_sha256", default="", advanced=True,
                                    tooltip="Optional exact SHA256. Blank binds the actual selected saved master for this job.")],
            outputs=[io.Video.Output("postprocessed_video"), io.Video.Output("master_video"),
                     io.String.Output("status"), io.String.Output("saved_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, master_video, processed_frames, filename_prefix="MiniMaxH3/Postprocess/final",
                confirm_postprocess=False, crf=18, expected_master_sha256=""):
        if confirm_postprocess is not True:
            report = {"status": "not_started", "output_published": False, "automatic_accept": False}
            return io.NodeOutput(ExecutionBlocker("后处理未确认，未写文件"), master_video,
                "not_started", "", json.dumps(report), ui=ui.PreviewText("未开始后处理。原片不变；确认后才写新文件。"))
        root = Path(folder_paths.get_output_directory()).resolve()
        width, height = map(int, master_video.get_dimensions())
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(root), width, height)
        name = f"{filename}_{counter:05}_.mp4"
        target = (Path(folder) / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("postprocess output must stay inside ComfyUI output")
        result = finalize_postprocess(master_video, processed_frames, target, crf=crf,
                                     expected_master_sha256=expected_master_sha256)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        display = ui.PreviewText(text).as_dict()
        if result["state"] == "postprocess_complete":
            display.update(ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]).as_dict())
            output, path = InputImpl.VideoFromFile(str(target)), str(target)
        else:
            # Do not put master into the enhanced/cropped output or a success
            # player. It has a separate output and an explicit recovery path.
            output = ExecutionBlocker("后处理失败；原片在 master_video 独立出口，请看状态回执")
            path = ""
        return io.NodeOutput(output, master_video, result["state"], path, text, ui=display)


NODES = [MiniMaxH3PostprocessSaveEXPT8]
