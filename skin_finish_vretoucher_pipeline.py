from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .skin_finish_vretoucher_adapter import (
    build_vretoucher_context_plan,
    compose_vretoucher_current_frame,
    extract_vretoucher_context,
)
from .skin_finish_vretoucher_runtime import (
    VRetoucherRuntimeSession,
)


VRETOUCHER_PIPELINE_SCHEMA = "h3_t8_skin_finish_vretoucher_single_window/v1"


class VRetouchPipelineUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _hashed_report(value: dict[str, Any]) -> dict[str, Any]:
    output = dict(value)
    canonical = json.dumps(
        output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    output["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return output


class VRetoucherWindowProcessor:
    """Unregistered current-frame pipeline owning one runtime session."""

    def __init__(self, session: VRetoucherRuntimeSession):
        if not isinstance(session, VRetoucherRuntimeSession):
            raise TypeError("VRetoucherWindowProcessor requires a runtime session")
        self._session = session

    @property
    def closed(self) -> bool:
        return self._session.closed

    @property
    def close_report(self) -> dict[str, Any] | None:
        return self._session.close_report

    def process(
        self,
        frames: torch.Tensor,
        *,
        current_frame: int,
        shot_start: int,
        shot_end: int,
        track_key: str,
        frame_track_keys: list[str | None],
        face_boxes: list[list[float] | None],
        semantic_skin_mask: torch.Tensor,
        person_mask: torch.Tensor | None = None,
        context_factor: float = 1.45,
        amount: float = 1.0,
        feather_px: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.closed:
            raise VRetouchPipelineUnavailable(
                "ABSTAIN_VRETOUCHER_PIPELINE_CLOSED",
                "the VRetouchEr window processor is already closed",
            )
        plan = build_vretoucher_context_plan(
            frames,
            current_frame=int(current_frame),
            shot_start=int(shot_start),
            shot_end=int(shot_end),
            track_key=str(track_key),
            frame_track_keys=frame_track_keys,
            face_boxes=face_boxes,
            context_factor=float(context_factor),
        )
        current_record = plan["context_records"][-1]
        if int(current_record["source_frame_index"]) != int(current_frame):
            raise VRetouchPipelineUnavailable(
                "ABSTAIN_CURRENT_FRAME_RECORD_MISMATCH",
                "the newest causal record does not point to current_frame",
            )
        context = extract_vretoucher_context(frames, plan)
        proposal, inference_report = self._session.run(context)
        output, effective_mask, compose_json = compose_vretoucher_current_frame(
            frames[int(current_frame)],
            proposal,
            current_record,
            semantic_skin_mask,
            person_mask=person_mask,
            amount=float(amount),
            feather_px=int(feather_px),
        )
        compose_report = json.loads(compose_json)
        report = _hashed_report(
            {
                "schema": VRETOUCHER_PIPELINE_SCHEMA,
                "status": "CANDIDATE_REQUIRES_IDENTITY_AND_HUMAN_REVIEW",
                "plan_sha256": plan["sha256"],
                "context_indices": plan["context_indices"],
                "current_frame": int(current_frame),
                "current_frame_only": True,
                "source_batch_mutated": False,
                "checkpoint_loaded": bool(
                    self._session.load_report.get("checkpoint_loaded", False)
                ),
                "model_inference_executed": True,
                "inference": inference_report,
                "compose": compose_report,
                "semantic_skin_only": True,
                "person_track_intersection": person_mask is not None,
                "automatic_accept": False,
                "candidate_selected": False,
                "quality_validated": False,
                "audio_touched": False,
            }
        )
        return output, effective_mask, json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def close(self) -> dict[str, Any]:
        return self._session.close()

    def __enter__(self) -> VRetoucherWindowProcessor:
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self._session.__exit__(exc_type, exc_value, traceback)
