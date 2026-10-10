"""Optional separate sidecar output; never changes or republishes the movie."""
from pathlib import Path
import uuid

import folder_paths
from comfy_api.latest import io

from .h07_delivery import delivery_documents
from .h07_reports import canonical
from .skin_finish_p1 import _file_video_source_path


class MiniMaxH3TakeSidecarEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 作品溯源侧车·私有／分享副本",
            category="T8/MiniMax H3/Output/Experimental", is_output_node=True, is_experimental=True,
            description="绑定真实成片／原生工作流SHA与显式原声／生成声／后期声。分享副本不含本机路径、文件名或原始metadata。不重新编码、不替换音轨、不自动上传或人审；确认后create-only保存两个独立JSON。",
            inputs=[io.Video.Input("video"), io.String.Input("video_sha256", default=""),
                    io.String.Input("workflow_path", default=""), io.String.Input("workflow_sha256", default=""),
                    io.String.Input("project_id", default="project"), io.String.Input("shot_id", default="shot"),
                    io.String.Input("take_id", default="take"), io.String.Input("recipe_id", default="recipe"),
                    io.String.Input("audio_sources_json", default="[]", multiline=True),
                    io.Boolean.Input("confirm_sidecar_write", default=False),
                    io.Custom("H3_T8_FREEVIDEO_QUALITY_STAGE").Input("completed_quality_stage", optional=True),
                    io.Custom("T8_STAGE_RESULT").Input("completed_modular_stage", optional=True)],
            outputs=[io.String.Output("private_json"), io.String.Output("share_json"),
                     io.String.Output("private_path"), io.String.Output("share_path")])

    @classmethod
    def execute(cls, video, confirm_sidecar_write=False, **kwargs):
        private, public = delivery_documents(video_path=_file_video_source_path(video), **kwargs)
        texts = canonical(private), canonical(public)
        paths = "", ""
        if confirm_sidecar_write:
            root = Path(folder_paths.get_output_directory()).resolve() / "T8/TakeSidecars" / uuid.uuid4().hex
            root.mkdir(parents=True, exist_ok=False)
            paths = str(root / "private.json"), str(root / "share.json")
            for path, text in zip(paths, texts):
                with Path(path).open("x", encoding="utf8") as stream:
                    stream.write(text)
        return io.NodeOutput(*texts, *paths, ui={"text": [texts[1], *paths]})


NODES = [MiniMaxH3TakeSidecarEXPT8]
