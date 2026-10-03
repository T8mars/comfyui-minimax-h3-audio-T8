"""Opt-in alignment guard around the installed Core, never a copied algorithm.

Imported only by the explicit Union2 apply path on a supporting Core.
"""

from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch


class AlignedUnion2Patch(MiniMaxH3FunControlPatch):
    def __init__(self, *args, expected_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_shape = tuple(expected_shape)
        self.derived_content = None

    def prepare_control_latent(self, target_shape):
        if tuple(target_shape) != self.expected_shape:
            raise ValueError(
                "Union2 actual sampler video latent differs from the declared canvas/frame grid; "
                "align source/control/MASK to this MODEL stage explicitly, no silent resize/pad"
            )
        if self.vae.spacial_compression_encode() != 16:
            raise ValueError(
                "Union2 requires a native H3 16x encoding VAE; do not use a decode-only 2x VAE"
            )
        if self.control_latent is not None:
            self.verify_derived()
        result = super().prepare_control_latent(target_shape)
        from .long_video_dual_identity import content_identity

        self.derived_content = content_identity(self.control_latent)
        return result

    def _fit_frames(self, frames, frame_count, width, height):
        if tuple(frames.shape) != (frame_count, 3, height, width):
            raise ValueError(
                "Union2 source/control RGB no longer aligns to the declared stage, no resize/pad"
            )
        return super()._fit_frames(frames, frame_count, width, height)

    def verify_derived(self):
        from .long_video_dual_identity import content_identity

        if self.control_latent is None:
            if (
                self.derived_content is not None
                or self.control_latent_shape is not None
            ):
                raise ValueError("Union2 derived cache receipt is orphaned")
        elif tuple(
            self.control_latent_shape
        ) != self.expected_shape or self.derived_content != content_identity(
            self.control_latent
        ):
            raise ValueError("Union2 derived control latent cache changed")

    def cleanup(self):
        super().cleanup()
        self.derived_content = None
