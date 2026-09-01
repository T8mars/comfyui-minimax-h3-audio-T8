"""Block-sparse attention for MiniMax-H3, as used by the SLA turbo LoRA."""

from __future__ import annotations

from .patch import patch_h3_sla

__all__ = ["patch_h3_sla"]
