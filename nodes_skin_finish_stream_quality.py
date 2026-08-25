from __future__ import annotations

from comfy_api.latest import io, ui

from .skin_finish import PRESET_CONFIG
from .skin_finish_stream_quality import stream_skin_finish_quality_video


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishQualityVideoStreamT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishQualityVideoStreamT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Quality Stream / 语义低内存肤质收尾 "
                "(Advanced EXP)"
            ),
            description=(
                "Two-pass file VIDEO route that never materializes a full IMAGE batch. Pass 1 "
                "keeps only pinned YuNet face metadata; pass 2 runs pinned CPU ParseNet, Skin "
                "Finish, Frequency Split, Texture Guard and a cross-chunk Safety Audit on bounded "
                "chunks, then incrementally encodes H.264 and packet-copies verified source audio. "
                "Accepted runs preflight a reviewed 2048 MiB available-system-RAM floor when host "
                "measurement is supported; insufficient RAM returns source and writes no file."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Video.Input("source_video"),
                io.Combo.Input(
                    "preset",
                    options=[
                        name
                        for name in PRESET_CONFIG
                        if name in {"subtle", "oil_control"}
                    ],
                    default="subtle",
                ),
                io.Float.Input("amount", default=0.30, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "texture_keep", default=0.95, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "shine_control", default=0.25, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "detection_threshold",
                    default=0.45,
                    min=0.10,
                    max=0.95,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_face_height_px",
                    default=32.0,
                    min=8.0,
                    max=512.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_detail",
                    default=0.010,
                    min=0.001,
                    max=0.100,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "bbox_ema_alpha",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "scene_cut_threshold",
                    default=0.28,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input("maximum_faces", default=4, min=1, max=12, advanced=True),
                io.Float.Input(
                    "crop_expansion",
                    default=1.45,
                    min=1.0,
                    max=3.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_class_probability",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "feature_protection_px", default=4, min=0, max=64, advanced=True
                ),
                io.Int.Input(
                    "mask_feather_px",
                    default=0,
                    min=0,
                    max=8,
                    advanced=True,
                    tooltip=(
                        "Zero is the conservative semantic default; extra feathering may "
                        "approach eyes or lips and requires review."
                    ),
                ),
                io.Int.Input(
                    "proxy_long_side",
                    default=640,
                    min=128,
                    max=1280,
                    step=32,
                    advanced=True,
                ),
                io.Float.Input(
                    "low_frequency_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "source_detail_gain",
                    default=1.0,
                    min=0.0,
                    max=1.25,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "separation_radius_percent",
                    default=1.0,
                    min=0.10,
                    max=5.0,
                    step=0.10,
                    advanced=True,
                ),
                io.Int.Input(
                    "maximum_radius_px", default=32, min=1, max=128, advanced=True
                ),
                io.Float.Input(
                    "shadow_protection",
                    default=0.10,
                    min=0.0,
                    max=0.90,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "highlight_protection",
                    default=0.94,
                    min=0.10,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "minimum_texture_ratio",
                    default=0.78,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_temporal_effect_jump",
                    default=0.04,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    advanced=True,
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=2,
                    min=1,
                    max=8,
                    advanced=True,
                    tooltip="Two frames is the conservative CPU/RAM default.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="MiniMaxH3/SkinFinish/quality_stream",
                ),
                io.Float.Input("crf", default=18.0, min=0.0, max=51.0, step=1.0),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "False returns the exact source VIDEO without analysis or writes. "
                        "Enable only to render a candidate for human review; the node never "
                        "accepts aesthetic quality automatically."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        video, path, report, saved = stream_skin_finish_quality_video(**kwargs)
        if saved is None:
            return io.NodeOutput(video, path, report)
        filename, subfolder = saved
        preview = ui.PreviewVideo(
            [ui.SavedResult(filename, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(video, path, report, ui=preview)


SKIN_FINISH_QUALITY_STREAM_NODE_CLASSES = [
    MiniMaxH3SkinFinishQualityVideoStreamT8Advanced,
]
