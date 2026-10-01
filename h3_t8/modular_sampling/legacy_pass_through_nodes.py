"""New serial-delivery nodes without modifying the frozen legacy node modules."""
from __future__ import annotations

import json

from comfy_api.latest import io

from ..native_latent_checkpoint_advanced import (
    fingerprint_native_h3_checkpoint_file,
    save_native_h3_av_checkpoint,
)
from ..nodes_motion_recovery_advanced import MiniMaxH3MotionOverloadAnalyzeT8Advanced
from ..nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
    native_h3_checkpoint_storage_root,
)


class MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8(
        MiniMaxH3MotionOverloadAnalyzeT8Advanced):
    """Keep the existing motion analysis downstream without a second sink."""

    _OUTPUT_NODE = None

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8"
        schema.display_name = (
            "MiniMax H3 Motion Overload Analyze Pass-through / "
            "动作过载串行分析 (EXP/T8)"
        )
        schema.description = (
            "Runs the same source-bound motion analysis as the original node, "
            "but only as a downstream dependency. Use for single-output graph "
            "delivery when a non-idempotent checkpoint writer is upstream."
        )
        schema.is_output_node = False
        return schema


class MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8(
        MiniMaxH3NativeLatentCheckpointSaveT8Advanced):
    """Write once on a single delivery path instead of becoming another sink."""

    _OUTPUT_NODE = None
    _NOT_IDEMPOTENT = None

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8"
        schema.display_name = (
            "MiniMax H3 Native AV Checkpoint Pass-through Save / "
            "原生音画检查点串行保存 (EXP/T8)"
        )
        schema.description = (
            "Non-output checkpoint writer for a single downstream delivery path. "
            "It preserves the exact AV latent while writing the same verified native "
            "checkpoint format. Repeated evaluation within one Core execution "
            "reuses the first file only after the exact AV manifest matches; "
            "confirm_save remains false by default."
        )
        schema.is_output_node = False
        schema.not_idempotent = True
        schema.hidden = [io.Hidden.execution_list, io.Hidden.unique_id]
        return schema

    @classmethod
    def execute(cls, **kwargs):
        execution_list = getattr(cls.hidden, "execution_list", None) if cls.hidden else None
        node_id = getattr(cls.hidden, "unique_id", None) if cls.hidden else None
        if not kwargs.get("confirm_save") or execution_list is None or node_id is None:
            return super().execute(**kwargs)

        receipts = getattr(execution_list, "_t8_native_pass_through_saves", None)
        if receipts is None:
            receipts = {}
            execution_list._t8_native_pass_through_saves = receipts
        key = str(node_id)
        options = (kwargs.get("filename_prefix", "h3_native_latent"),
                   kwargs.get("checkpoint_id", "timeline_checkpoint"),
                   kwargs.get("verify_after_write", True),
                   kwargs.get("hash_chunk_megabytes", 8))
        prior = receipts.get(key)
        if prior is not None:
            saved_options, saved = prior
            if options != saved_options:
                raise ValueError("Native checkpoint Save settings changed within one Core execution")
            probe = save_native_h3_av_checkpoint(
                storage_root=native_h3_checkpoint_storage_root(),
                **{**kwargs, "confirm_save": False},
            )
            if probe[4] != saved[4]:
                raise ValueError("Re-evaluated first-pass AV differs from its saved checkpoint")
            actual_file_sha = fingerprint_native_h3_checkpoint_file(
                native_h3_checkpoint_storage_root(), saved[2]
            )
            if actual_file_sha != saved[3]:
                raise ValueError("Saved first-pass checkpoint changed during Core execution")
            report = json.loads(saved[5])
            report["reused_within_execution"] = True
            report["files_written"] = False
            return io.NodeOutput(kwargs["av_latent"], *saved[1:5],
                                 json.dumps(report, ensure_ascii=False, sort_keys=True))

        result = save_native_h3_av_checkpoint(
            storage_root=native_h3_checkpoint_storage_root(), **kwargs
        )
        receipts[key] = (options, result)
        return io.NodeOutput(*result)
