"""Immutable stage description, not a MODEL serializer or completion receipt.

Each adapter owns its numerical state/handoff. A descriptor explicitly names the
expected representation; it never converts x0, x_sigma, or another recipe's state.
No descriptor alone authorizes cache reuse or claims that sampling ran.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

SCHEMA = "t8.modular-sampling.stage-context.v1"


def clear_native_stage_descriptors(model):
    """Only inert descriptors owned by these adapters, on a new MODEL clone.

    Effect owners, V2/PDD/VDN runtime state and user attachments are untouched.
    A new stage must not retain an older descriptor for another sampling object.
    """
    for key in ("t8_modular_native_dual_v1", "t8_modular_manual_pass_v1",
                "t8_modular_rf_stage_v1", "t8_modular_native_explicit_v1", "t8_modular_pdd_stage_v1",
                "t8_modular_vdn_stage_v1"):
        model.remove_attachments(key)


@dataclass(frozen=True)
class StageContext:
    recipe: str
    stage: str
    profile: str
    start: int
    end: int
    trajectory_sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    input_semantics: str
    output_semantics: str
    denoised_semantics: str

    def __post_init__(self):
        for name in ("recipe", "stage", "profile", "input_semantics", "output_semantics", "denoised_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid stage {name}")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("Stage indices must be integers")
        noop = (self.start == self.end == 0 and self.trajectory_sigmas == ()
                and self.input_semantics == self.output_semantics == self.denoised_semantics == "identity_noop")
        if not isinstance(self.trajectory_sigmas, tuple) or not (noop or 0 <= self.start < self.end < len(self.trajectory_sigmas)):
            raise ValueError("Stage interval is outside its immutable trajectory")
        if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in self.trajectory_sigmas):
            raise ValueError("Invalid sigma trajectory")
        if any(a < b for a, b in zip(self.trajectory_sigmas, self.trajectory_sigmas[1:])):
            raise ValueError("A stage trajectory must be non-increasing; restarts use distinct trajectories")
        for value in (self.video_shift, self.audio_shift):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid AV clock shift")
        for shape in (self.video_shape, self.audio_shape):
            if not isinstance(shape, tuple) or not shape or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError("Stage shape must be an immutable positive integer tuple")

    @property
    def steps(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        # JSON-only data: no MODEL, callable, tensors, or executable payload.
        return json.loads(json.dumps({"schema": SCHEMA, **asdict(self)}, allow_nan=False))

    @classmethod
    def from_dict(cls, value: dict) -> StageContext:
        if value.get("schema") != SCHEMA:
            raise ValueError("Unknown stage context schema")
        fields = dict(value)
        fields.pop("schema")
        for name in ("trajectory_sigmas", "video_shape", "audio_shape"):
            fields[name] = tuple(fields[name])
        return cls(**fields)

    @property
    def descriptor_sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()
