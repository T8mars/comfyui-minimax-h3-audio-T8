"""Composable append-only reference assets; old nodes/sampling remain unchanged."""
from pathlib import Path
import re

import folder_paths
from comfy_api.latest import io

from .conditioning import build_conditioning
from .core import sorted_autogrow_values
from .nodes_semantic_bridge import BridgeIO
from .nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
from .prompt_relay_advanced import build_prompt_relay_conditioning
from .reference_package import (ReferencePackage, canonical, installed_package_names, load_package,
                                resolve_installed_package, save_package)
from .reference_runtime import capture_set, create_package


PACKAGE = io.Custom("T8_H3_REFERENCE_PACKAGE")
REFSET = io.Custom("T8_H3_REFERENCE_SET")
RECIPE = io.Custom("T8_TEMPORAL_NATIVE_TEXT_RECIPE")
CATEGORY = "T8/MiniMax H3/References Experimental"


def schema(cls, title, inputs, outputs, description, *, output=False):
    return io.Schema(node_id=cls.__name__, display_name="H3 Reference · "+title+" (EXP/T8)",
        category=CATEGORY, is_experimental=True, is_output_node=output,
        inputs=inputs, outputs=outputs, description=description)


class MiniMaxH3ReferenceCreateEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Create / 原生参考包", [
            io.String.Input("role_id", default="A", tooltip="Explicit stable role ID; ASCII letters/numbers/_/-, up to48."),
            io.Combo.Input("kind", options=["image", "video", "audio"], default="image"),
            io.Int.Input("width", default=256, min=32, max=2048, step=32),
            io.Int.Input("height", default=256, min=32, max=2048, step=32),
            io.Int.Input("frame_limit", default=124, min=5, max=360, step=17,
                         tooltip="Video: explicit maximum; align down to17n+5. Report actual trimming. Image ignores this."),
            io.Image.Input("frames", optional=True), io.Audio.Input("audio", optional=True),
            io.Vae.Input("video_vae", optional=True), io.Vae.Input("audio_vae", optional=True)],
            [PACKAGE.Output("reference_package"), io.String.Output("report_json")],
            "Actual native VAE encode outputs plus full RGB grounding, separate optional voice anchor. "
            "No training/LoRA/denoiser scaling or identity lock. Portable source/producer bound before+after. "
            "Unknown encoders can still use old ordinary refs, not counterfeit persistent assets.")

    @classmethod
    def execute(cls, role_id, kind="image", width=256, height=256, frame_limit=124,
                frames=None, audio=None, video_vae=None, audio_vae=None):
        package, report = create_package(role_id, kind=kind, width=width, height=height,
            frame_limit=frame_limit, frames=frames, audio=audio, video_vae=video_vae, audio_vae=audio_vae)
        return io.NodeOutput(package, canonical(report))


class MiniMaxH3ReferenceSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Save / 新文件保存", [PACKAGE.Input("reference_package"),
            io.String.Input("filename", default="role_A.safetensors"),
            io.Boolean.Input("confirm_save", default=False)],
            [PACKAGE.Output("reference_package"), io.String.Output("filename"),
             io.String.Output("file_sha256"), io.String.Output("report_json")],
            "Explicit new file in models/refmods. Never overwrite user packages or automatically accept. "
            "Unconfirmed: no file/directory write, package passes through.", output=True)

    @classmethod
    def execute(cls, reference_package, filename="role_A.safetensors", confirm_save=False):
        if type(reference_package) is not ReferencePackage:
            raise ValueError("Use an actual T8 reference package")
        reference_package.verify()
        if confirm_save is not True:
            report = {"status": "not_saved", "confirmed": False, "automatic_accept": False}
            return io.NodeOutput(reference_package, "", "", canonical(report), ui={"text": [canonical(report)]})
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\.safetensors", filename):
            raise ValueError("Use a short ASCII .safetensors filename, no directory or generated truncation")
        root = Path(folder_paths.models_dir).absolute() / "refmods"
        report = save_package(reference_package, root / filename, confirmed=True)
        return io.NodeOutput(reference_package, filename, report["sha256"], canonical(report),
                             ui={"text": [canonical(report)]})


class MiniMaxH3ReferenceLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load / 目录读取", [
            io.Combo.Input("filename", options=installed_package_names()),
            io.String.Input("expected_sha256", default="", advanced=True)],
            [PACKAGE.Output("reference_package"), io.String.Output("report_json")],
            "Exact installed T8 package from configured models/refmods directories. "
            "No arbitrary path, inferred producer or unqualified community-format import.")

    @classmethod
    def execute(cls, filename, expected_sha256=""):
        package = load_package(resolve_installed_package(filename), expected_sha256=expected_sha256 or None)
        return io.NodeOutput(package, canonical({"filename": filename, "file_sha256": package.file_sha256,
            "manifest_sha256": package.verify()["sha256"], "automatic_accept": False}))

    @classmethod
    def fingerprint_inputs(cls, filename, **_inputs):
        from .reference_package import file_sha
        return file_sha(resolve_installed_package(filename))


class MiniMaxH3ReferenceRouteEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Route / 画内画外角色", [
            io.Autogrow.Input("packages", template=io.Autogrow.TemplatePrefix(
                input=PACKAGE.Input("package"), prefix="package_", min=1, max=15)),
            io.String.Input("roles_json", multiline=True,
                default='[{"role_id":"A","visual":true,"voice":true}]')],
            [REFSET.Output("reference_set"), io.String.Output("mapping_json")],
            "Declare EVERY role including absent ones in explicit order. visual=false removes BOTH "
            "VAE visual latent and Qwen picture, voice=true retains offscreen voice anchor. "
            "No global strength, stage schedule or implicit duplicate-role merge.")

    @classmethod
    def execute(cls, packages, roles_json):
        refs, report = capture_set(sorted_autogrow_values(packages), roles_json)
        return io.NodeOutput(refs, canonical(report))


class MiniMaxH3ReferenceConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Apply / 全新原生条件", [
            REFSET.Input("reference_set"), io.Clip.Input("clip"),
            io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
            io.String.Input("prompt", multiline=True, dynamic_prompts=True),
            io.Int.Input("width", default=512, min=32, max=16384, step=32),
            io.Int.Input("height", default=288, min=32, max=16384, step=32),
            io.Int.Input("length", default=124, min=5, max=None, step=17),
            io.Combo.Input("task_type", options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"], default="auto"),
            io.Combo.Input("audio_mode", options=["native", "reference_only", "lock_source", "remix_source"], default="native"),
            io.Float.Input("audio_denoise_strength", default=0.35, min=0., max=1., step=0.01, advanced=True),
            io.Boolean.Input("add_source_as_reference", default=True),
            io.Int.Input("prompt_primary_audio_ordinal", default=0, min=0, max=9, advanced=True),
            io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
            io.Audio.Input("drive_audio", optional=True), io.Audio.Input("final_audio", optional=True),
            io.Image.Input("first_frame", optional=True), io.Image.Input("last_frame", optional=True),
            BridgeIO.Input("semantic_bridge", optional=True)],
            [io.Conditioning.Output("positive"), io.Latent.Output("av_latent"),
             io.Audio.Output("mux_audio"), io.String.Output("conditioned_prompt"),
             io.String.Output("media_map_json"), io.String.Output("report"), RECIPE.Output("native_text_recipe")],
            "Encode Qwen ONCE with only routed media; reuse stored reference VAE latents and recheck actual "
            "producer. Not an edit to old embedding. Standard positive/AV_LATENT supports ordinary and "
            "separate LOW/HIGH; Bridge once, EAV/Relay remain external. Voice reference is NOT drive/final_audio.")

    @classmethod
    def execute(cls, reference_set, clip, video_vae, audio_vae, prompt, width, height, length,
                task_type="auto", audio_mode="native", audio_denoise_strength=0.35,
                add_source_as_reference=True, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
                drive_audio=None, final_audio=None, first_frame=None, last_frame=None, semantic_bridge=None):
        result = build_conditioning(clip, video_vae, audio_vae, prompt, width, height, length,
            task_type, audio_mode, audio_denoise_strength, add_source_as_reference,
            prompt_primary_audio_ordinal, strict_prompt_tags, drive_audio=drive_audio,
            final_audio=final_audio, first_frame=first_frame, last_frame=last_frame,
            semantic_bridge=semantic_bridge, prepared_reference_set=reference_set, return_text_recipe=True)
        return io.NodeOutput(*result[:6], result[-1]["text_recipe"])


class MiniMaxH3ReferenceRelayConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Each call creates a new ordinary schema. Do not modify its class,
        # published sockets/defaults, or permit ambiguous raw/package mixing.
        result = MiniMaxH3PromptRelayConditioningT8Advanced.define_schema()
        result.node_id = cls.__name__
        result.display_name = "H3 Reference · External Prompt Relay / 外置分段条件 (EXP/T8)"
        result.category = CATEGORY
        result.description = ("Route -> explicit external Relay Plan -> fresh Qwen once, with the actual "
            "stored reference VAE latents. Paired MODEL/CONDITIONING/LATENT outputs for ordinary "
            "or independent LOW/HIGH; EAV stays downstream. Not an embedding edit, sampler, "
            "voice clone or drive/final-audio replacement. Raw refs are not mixed here.")
        replacements = {
            "width": io.Int.Input("width", default=512, min=32, max=16384, step=32),
            "height": io.Int.Input("height", default=288, min=32, max=16384, step=32),
            "task_type": io.Combo.Input("task_type", options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"], default="auto"),
            "prompt_primary_audio_ordinal": io.Int.Input("prompt_primary_audio_ordinal", default=0, min=0, max=9, advanced=True),
        }
        raw = {"ref_images", "ref_videos", "ref_video_audios", "ref_audios"}
        result.inputs = [REFSET.Input("reference_set"), *[
            replacements.get(item.id, item) for item in result.inputs if item.id not in raw]]
        return result

    @classmethod
    def execute(cls, reference_set, **kwargs):
        if any(kwargs.get(name) for name in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios")):
            raise ValueError("Use the ordered reference set, not mixed raw reference inputs")
        return io.NodeOutput(*build_prompt_relay_conditioning(
            **kwargs, prepared_reference_set=reference_set))


NODES = [MiniMaxH3ReferenceCreateEXPT8, MiniMaxH3ReferenceSaveEXPT8, MiniMaxH3ReferenceLoadEXPT8,
         MiniMaxH3ReferenceRouteEXPT8, MiniMaxH3ReferenceConditioningEXPT8,
         MiniMaxH3ReferenceRelayConditioningEXPT8]
