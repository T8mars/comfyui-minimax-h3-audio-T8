"""Native value hashes plus explicit live-only ownership for foreign metadata."""
from __future__ import annotations

import uuid

from .modular_sampling.results import _input_identity
from .patch_stack_policy import UnverifiedModelStack


class LiveMetadataIdentity:
    """Retain unknown values; do not call, repr, serialize or certify their state."""

    def __init__(self):
        self._objects = []

    @property
    def portable(self):
        return not self._objects

    def describe(self, value):
        if isinstance(value, dict) and all(type(key) is str for key in value):
            return {key: self.describe(item) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return {"type": type(value).__name__, "items": [self.describe(item) for item in value]}
        try:
            return _input_identity(value)
        except UnverifiedModelStack:
            # Only compare live object identity. Never invent a portable digest
            # from its address, repr, class name or an unaudited callback.
            for existing, token in self._objects:
                if existing is value:
                    return {"opaque_live_token": token, "internal_state_certified": False}
            token = uuid.uuid4().hex
            self._objects.append((value, token))
            return {"opaque_live_token": token, "internal_state_certified": False}
