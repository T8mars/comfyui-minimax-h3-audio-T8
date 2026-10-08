"""Additive Source AV lineage ports; all legacy Source AV schemas remain intact."""
from comfy_api.latest import io
import uuid

from . import source_conform as runtime
from .source_av import SHORT_AUDIO_POLICIES, SHORT_VIDEO_POLICIES

CLOCK = io.Custom("T8_H3_SOURCE_CLOCK")
FRAME_MAP = io.Custom("T8_H3_SOURCE_FRAME_MAP")
LINEAGE = io.Custom("T8_H3_CONTROL_LINEAGE")
CATEGORY = "T8/MiniMax H3/Source AV/Experimental"


def schema(cls, title, inputs, outputs, description):
    return io.Schema(node_id=cls.__name__, display_name="H3 Source · " + title + " (EXP/T8)",
        category=CATEGORY, is_experimental=True, inputs=inputs, outputs=outputs, description=description)


class MiniMaxH3SourceClockEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Decode + Clock / 同次解码时钟", [io.Video.Input("video")],
            [io.Image.Output("frames"), io.Audio.Output("audio"), io.Float.Output("source_fps"),
             CLOCK.Output("source_clock"), io.String.Output("report_json")],
            "Observe PTS at the actual native Core decode. Core owns crop/trim/rotation/RGB/audio. "
            "Strict CFR observation is bound to the emitted RGB/PCM; opaque providers keep their "
            "decoder and are unverified. Full frames are resident: not streaming or memory-safe.")

    @classmethod
    def execute(cls, video):
        from comfy.model_management import throw_exception_if_processing_interrupted
        components, clock, report = runtime.decode_clock(video, throw_exception_if_processing_interrupted)
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate), clock,
                             runtime.canonical(report))


class MiniMaxH3SourceConformEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Conform Once / 一次同源准备", [
            io.Image.Input("frames"), io.Float.Input("source_fps", default=24., min=.01, max=240.),
            io.Int.Input("width", default=448, min=32, max=3840, step=32),
            io.Int.Input("height", default=256, min=32, max=2160, step=32),
            io.Int.Input("length", default=124, min=5, max=4096, step=17),
            io.Float.Input("start_seconds", default=0., min=0., max=86400., step=.001),
            io.Combo.Input("short_video_policy", options=list(SHORT_VIDEO_POLICIES), default="strict"),
            io.Combo.Input("short_audio_policy", options=list(SHORT_AUDIO_POLICIES), default="pad_silence"),
            io.Audio.Input("source_audio", optional=True), CLOCK.Input("source_clock", optional=True)],
            [io.Image.Output("frames"), io.Audio.Output("audio"), io.Int.Output("frame_count"),
             io.Float.Output("duration_seconds"), FRAME_MAP.Output("frame_map"), io.String.Output("report_json")],
            "One legacy Source AV preparation: exact nearest frame indices, Lanczos resize, 17n+5 and "
            "32k stereo rules retained. Complete immutable map binds actual RGB/PCM. No clock input "
            "means declared fps, not verified CFR. Missing source audio is reported generated silence.")

    @classmethod
    def execute(cls, frames, source_fps, width, height, length, start_seconds=0.,
                short_video_policy="strict", short_audio_policy="pad_silence", source_audio=None, source_clock=None):
        *values, report = runtime.conform(frames, source_fps, width, height, length, start_seconds,
                                         short_video_policy, short_audio_policy, source_audio, source_clock)
        return io.NodeOutput(*values, runtime.canonical(report))


class MiniMaxH3SourceControlDeriveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Derive Control / 控制来源绑定", [io.Image.Input("source_frames"),
            io.Combo.Input("operation", options=["identity_rgb", "grayscale", "manual_mask"], default="grayscale"),
            io.Mask.Input("manual_mask", optional=True)],
            [io.Image.Output("control_image"), io.Mask.Output("control_mask"), LINEAGE.Output("lineage"),
             io.String.Output("report_json")],
            "Owned RGB channel-mean preprocessing or explicit per-frame manual MASK annotation. "
            "Connect full decoded source BEFORE Conform; no implicit mask broadcast. Annotation "
            "identity is not proof of correct object/occlusion. Unknown external preprocessors may "
            "connect directly to Map Control and run with unverified origin.")

    @classmethod
    def execute(cls, source_frames, operation="grayscale", manual_mask=None):
        result, lineage = runtime.derive_control(source_frames, operation, manual_mask)
        report = dict(source_sha=lineage.source_sha, output_sha=lineage.output_sha,
                      operation=lineage.operation, annotation_semantics_verified=False)
        return io.NodeOutput(result if operation != "manual_mask" else None,
                             result if operation == "manual_mask" else None, lineage, runtime.canonical(report))


class MiniMaxH3SourceControlMapEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Map Control / 共用帧映射", [FRAME_MAP.Input("frame_map"),
            io.Combo.Input("kind", options=["image", "mask"], default="image"),
            io.Int.Input("width", default=0, min=0, max=3840, step=32,
                         tooltip="0 uses the Conform canvas. LOW/HIGH share indices, not geometry."),
            io.Int.Input("height", default=0, min=0, max=2160, step=32),
            io.Image.Input("control_image", optional=True), io.Mask.Input("control_mask", optional=True),
            LINEAGE.Input("lineage", optional=True)],
            [io.Image.Output("control_image"), io.Mask.Output("control_mask"), io.String.Output("report_json")],
            "Reuse exact source indices without re-timing. RGB Lanczos and MASK nearest-exact use "
            "the chosen stage geometry. Equal shape/JSON alone never certifies origin. Reject stale "
            "own map/lineage; opaque external controls remain usable with unverified-origin report.")

    @classmethod
    def execute(cls, frame_map, kind="image", width=0, height=0, control_image=None, control_mask=None, lineage=None):
        control = control_image if kind == "image" else control_mask
        result, report = runtime.map_control(frame_map, control, kind, width, height, lineage)
        return io.NodeOutput(result if kind == "image" else None, result if kind == "mask" else None,
                             runtime.canonical(report))


NODES = [MiniMaxH3SourceClockEXPT8, MiniMaxH3SourceConformEXPT8,
         MiniMaxH3SourceControlDeriveEXPT8, MiniMaxH3SourceControlMapEXPT8]

# Revalidate owned content on explicit executions; do not certify an old receipt
# just because a filename/custom port was returned from Core's output cache.
for _node in NODES:
    # NaN invalidates cache but contaminates Core prompt metadata, which our
    # strict serial writer correctly refuses. A fresh JSON-safe identity has
    # the same revalidation semantics without weakening metadata validation.
    _node.fingerprint_inputs = classmethod(lambda cls, **inputs: uuid.uuid4().hex)
