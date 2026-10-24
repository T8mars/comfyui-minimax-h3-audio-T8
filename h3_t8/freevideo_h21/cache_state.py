"""Owned-worker fix for FP32 EasyCache history aliasing; upstream stays read-only."""
from __future__ import annotations

from contextlib import contextmanager

HISTORY_POLICY = "owned_FP32_history_snapshot_v1"


@contextmanager
def snapshot_history(module, *, enabled=True):
    """Keep sampled history, not views into the next in-place scheduler update.

    The pinned reference sampler updates FP32 generated rows in place. On
    FP32, strided_tensor.float() is still a view, so upstream prev_input can
    silently become the current input. Snapshot only its small strided history;
    keep the author's thresholds, windows, rates, residuals and schedules.

    Used solely inside an owned Engine child (or an explicit CPU probe). Never
    imported/installed into the host's legacy freevideo_engine. Even exceptions
    restore the exact class that was present on entry. Existing cache objects
    and other cache kinds are not replaced or reconfigured.
    """
    if not enabled:
        yield
        return
    original = module.StepCache

    class SnapshotStepCache(original):
        def record_forward(self, *args, **kwargs):
            result = super().record_forward(*args, **kwargs)
            if self.kind == "easycache-av":
                self.prev_input = tuple(value.detach().clone() for value in self.prev_input)
                self.prev_out = tuple(value.detach().clone() for value in self.prev_out)
            return result

        def stats(self):
            result = super().stats()
            if self.kind == "easycache-av":
                result["history_policy"] = HISTORY_POLICY
            return result

    module.StepCache = SnapshotStepCache
    try:
        yield
    finally:
        module.StepCache = original
