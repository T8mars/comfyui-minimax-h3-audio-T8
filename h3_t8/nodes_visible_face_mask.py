"""Two append-only post-composite operations; no sampler or detector."""
from comfy_api.latest import io
from . import visible_face_mask as visibility

CATEGORY = 'T8/MiniMax H3/Face Refine Experimental'


class MiniMaxH3VisibleFaceMaskBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Face · Bind Visible MASK (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='Bind full original RGB and full-coordinate user visible-face MASK. White permits '
                'existing changes, black preserves exact source. One-frame broadcast is explicit. No '
                'automatic segmentation/occlusion proof, sampler, resizing, crop or time conversion.',
            inputs=[io.Image.Input('source_images'), io.Mask.Input('visible_mask'),
                    io.Float.Input('fps', default=24., min=24., max=24.),
                    io.Int.Input('source_start_frame', default=0, min=0, max=5_000_000),
                    io.Boolean.Input('broadcast_single_mask', default=False)],
            outputs=[io.Custom(visibility.TYPE_NAME).Output('visible_face_mask'),
                     io.String.Output('report_json')])

    @classmethod
    def execute(cls, source_images, visible_mask, **request):
        binding = visibility.capture(source_images, visible_mask, **request)
        return io.NodeOutput(binding, binding.contract_json)

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        # Verify actual whole tensors/implementation on each explicit execution.
        return float('nan')


class MiniMaxH3VisibleFaceCompositeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name='H3 Face · Visible MASK Composite (T8 EXP)',
            category=CATEGORY, is_experimental=True,
            description='AFTER old Stitch/Gate: candidate already contains original blend alpha. Connect '
                'its changed_mask (or exact final applied support), NOT fallback mask or raw crops. '
                'Only limits existing changes, never applies original alpha twice. Full geometry must '
                'match; audio passes the same object. Changes invalidate composite/delivery only; '
                'completed sampling stages and manual approvals remain separate.',
            inputs=[io.Custom(visibility.TYPE_NAME).Input('visible_face_mask'),
                    io.Image.Input('candidate_images'), io.Mask.Input('changed_alpha', optional=True),
                    io.Audio.Input('audio', optional=True),
                    io.Custom('H3_T8_MULTIFACE_COMPOSITE').Input('multiface_composite', optional=True)],
            outputs=[io.Image.Output('images'), io.Audio.Output('audio'),
                     io.Mask.Output('effective_mask'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, visible_face_mask, candidate_images, changed_alpha=None, audio=None,
                multiface_composite=None):
        if (changed_alpha is None) == (multiface_composite is None):
            raise ValueError('Connect exactly one: original changed_alpha OR final Multi-Face composite')
        if multiface_composite is not None:
            changed_alpha = visibility.multiface_support(visible_face_mask, candidate_images, multiface_composite)
        return io.NodeOutput(*visibility.composite(visible_face_mask, candidate_images, changed_alpha, audio))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


NODES = [MiniMaxH3VisibleFaceMaskBindEXPT8, MiniMaxH3VisibleFaceCompositeEXPT8]
