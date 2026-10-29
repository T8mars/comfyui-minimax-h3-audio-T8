"""Source-bound authored MASK loader; old registration prefix stays unchanged."""

import json
from pathlib import Path

from comfy_api.latest import io

from .director_mask_author import load_mask_sequence
from .director_project import contained


class MiniMaxH3AuthoredMaskSequenceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        root = Path(folder_paths.get_input_directory())
        directory = root / "t8_masks"
        choices = (
            sorted(p.relative_to(root).as_posix() for p in directory.glob("*.zip"))
            if directory.is_dir()
            else []
        )
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 · 已绘制逐帧MASK / Authored Mask (EXP/T8)",
            category="T8/MiniMax H3/Repair/Experimental",
            is_experimental=True,
            description="在导演台整片串播中绘制并导出MASK后刷新列表。连接完整原片，按源RGB24校验逐帧MASK；缩放裁时请让画面与MASK共用Source Conform。未画帧为空，不自动跟踪，不改声音。",
            inputs=[
                io.Image.Input("source_frames"),
                io.Combo.Input("mask_archive", options=choices or ["none"]),
            ],
            outputs=[io.Mask.Output("mask"), io.String.Output("frame_map_json")],
        )

    @classmethod
    def execute(cls, source_frames, mask_archive):
        import folder_paths

        root = Path(folder_paths.get_input_directory())
        if (
            not isinstance(mask_archive, str)
            or not mask_archive.startswith("t8_masks/")
            or not mask_archive.endswith(".zip")
        ):
            raise ValueError("请选择导演台导出的逐帧MASK包")
        path = contained(root, mask_archive)
        mask, manifest = load_mask_sequence(path, source_frames)
        return io.NodeOutput(
            mask, json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        )


NODES = [MiniMaxH3AuthoredMaskSequenceEXPT8]
