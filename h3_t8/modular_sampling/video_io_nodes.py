"""Additive opt-in serial video I/O, separate from every sampling stage."""
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io, ui

from . import video_io

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


class MiniMaxH3VideoComponentsSerialEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Video · Serial Components (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Native file decoding with explicit per-stream thread_count=1. Original crop, "
                        "trim, rotation, float RGB, audio and bit depth retained; no global codec patch.",
            inputs=[io.Video.Input("video")],
            outputs=[io.Image.Output("images"), io.Audio.Output("audio"), io.Float.Output("fps"),
                     io.Combo.Output("bit_depth"), io.Combo.Output("color_space"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, video):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        components, report = video_io.components_serial(video, interrupt)
        report["data_identity"] = video_io.component_identity(components)
        interrupt()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate),
                             video.get_bit_depth(), video.get_color_space(), json.dumps(report))


class MiniMaxH3SaveVideoSerialEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Save Video · Serial H.264 (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicit single-thread H.264/AAC export preserving selected 8/10-bit depth, "
                        "color space, fps and original audio. Separate unique output directory; never "
                        "overwrites old media. No sampling or quality certification.",
            inputs=[io.Video.Input("video"), io.String.Input("filename_prefix", default="MiniMaxH3/serial_video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[io.Video.Output("video"), io.String.Output("video_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, video, filename_prefix):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        from comfy.cli_args import args
        metadata = None
        if not args.disable_metadata:
            metadata = dict(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        root = Path(folder_paths.get_output_directory())
        path, report = video_io.save_serial(video, root, filename_prefix, metadata, interrupt)
        locator = ui.SavedResult(path.name, path.parent.relative_to(root.resolve()).as_posix(), io.FolderType.output)
        return io.NodeOutput(video, str(path), json.dumps(report), ui=ui.PreviewVideo([locator]))


class MiniMaxH3SaveVideoIsolatedEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Save Video · Isolated H.264 (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Windows Job-owned FFmpeg export from original frame data; 8/10-bit and original audio. "
                        "Explicit disk/deadline guards. Strict complete decode before exclusive publication. "
                        "No change to old SaveVideo or sampling; no quality certification.",
            inputs=[io.Video.Input("video"), io.String.Input("filename_prefix", default="MiniMaxH3/isolated_video"),
                    io.Int.Input("timeout_seconds", default=600, min=1, max=3600),
                    io.Float.Input("max_staging_gib", default=16., min=.01, max=1024.),
                    io.Float.Input("min_free_disk_gib", default=2., min=.01, max=1024.)],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[io.Video.Output("video"), io.String.Output("video_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, video, filename_prefix, timeout_seconds, max_staging_gib, min_free_disk_gib):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        from comfy.cli_args import args
        from .video_ffmpeg import save_isolated
        metadata = None
        if not args.disable_metadata:
            metadata = dict(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        root = Path(folder_paths.get_output_directory())
        path, report = save_isolated(video, root, filename_prefix, metadata, interrupt,
            timeout=timeout_seconds, max_staging_gib=max_staging_gib, min_free_disk_gib=min_free_disk_gib)
        locator = ui.SavedResult(path.name, path.parent.relative_to(root.resolve()).as_posix(), io.FolderType.output)
        return io.NodeOutput(video, str(path), json.dumps(report), ui=ui.PreviewVideo([locator]))


NODES = [MiniMaxH3VideoComponentsSerialEXPT8, MiniMaxH3SaveVideoSerialEXPT8, MiniMaxH3SaveVideoIsolatedEXPT8]
