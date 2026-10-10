"""Opt-in author v0.3.5 loader; reuse separate samplers/effects without legacy migration."""
import os
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from ..freevideo_exp.runtime import canonical
from . import runtime
from .nodes import MODEL
from .profiles import AUDIO_RUNTIME, LABELS, binding


class MiniMaxH3FreeVideoAudioClockLoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FreeVideo · 新参考音频t=1加载（v0.3.5）",
            category="T8/MiniMax H3/FreeVideo/Audio v3", is_experimental=True,
            description="独立配置与新音频表；主DiT复用。接原四档分离采样／HIGH3／外置EAV、Relay。旧loader及8+2、4+4、四档默认不变，不混用旧LOW。",
            inputs=[io.String.Input("runtime_config", default="",
                tooltip="专用freevideo-runtime-audio-v3.json；先显式准备对应t=1表，不在加载时下载")],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, runtime_config=""):
        path = (runtime_config.strip() or os.environ.get("T8_FREEVIDEO_AUDIO_RUNTIME_CONFIG")
                or str(Path(folder_paths.get_user_directory()) / "default/T8/freevideo-runtime-audio-v3.json"))
        model = runtime.load_model(path, schema=AUDIO_RUNTIME)
        config = runtime.config_for(model)
        return io.NodeOutput(model, canonical(dict(producer=binding(config["schema"]),
            profiles=list(LABELS), model_revision=config["model_revision"],
            old_workflows_migrated=False, validation="metadata_only_full_SHA_at_execution")))


NODES = [MiniMaxH3FreeVideoAudioClockLoaderEXPT8]
