"""Two separate append-only AV import entries, keeping the image entry intact."""
from comfy_api.latest import io

from .external_reference_av_import import AUDIO_PROFILES, VIDEO_PROFILES, import_audio, import_video
from .nodes_reference_package import PACKAGE, schema
from .reference_package import canonical, file_sha, installed_package_names, resolve_installed_package


def _inputs(profile_name, profiles):
    return [io.Combo.Input("filename", options=installed_package_names()),
            io.String.Input("expected_sha256", default="", tooltip="Required SHA256 of the entire external file."),
            io.Int.Input("member_ordinal", default=1, min=1, max=256),
            io.String.Input("role_id", default="A"),
            io.Combo.Input(profile_name, options=list(profiles), default="native_exact"),
            io.Boolean.Input("confirm_import", default=False,
                             tooltip="Confirm source use and explicit import. Does not certify legal rights or AV synchronization.")]


class MiniMaxH3ExternalReferenceVideoImportEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Import encoded video / 外部编码视频导入", [
            *_inputs("precision_profile", VIDEO_PROFILES), io.Image.Input("source_frames"),
            io.Vae.Input("video_vae"), io.Float.Input("source_fps", default=24.0, min=1.0, max=120.0)],
            [PACKAGE.Output("reference_package"), io.String.Output("report_json")],
            "Explicit video Encode reference import. Supply actual prepared 24fps RGB at exactly the encoded "
            "geometry and native 17n+5 frame grid. Fresh connected VAE must match selected bytes; FP16 is explicit. "
            "No resize/crop/trim, audio-track import, inferred source PTS, foreign config execution or LoRA attach. "
            "Save/Route/Apply remain separate; old image and native loaders stay unchanged.")

    @classmethod
    def execute(cls, filename, expected_sha256, source_frames, video_vae, member_ordinal=1,
                role_id="A", precision_profile="native_exact", source_fps=24.0, confirm_import=False):
        package, report = import_video(resolve_installed_package(filename), source_frames, video_vae,
            expected_sha256=expected_sha256, member_ordinal=member_ordinal, role_id=role_id,
            precision_profile=precision_profile, source_fps=source_fps, confirm_import=confirm_import)
        return io.NodeOutput(package, canonical(report))

    @classmethod
    def fingerprint_inputs(cls, filename, **_inputs):
        return file_sha(resolve_installed_package(filename))


class MiniMaxH3ExternalReferenceAudioImportEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Import encoded audio / 外部编码音频导入", [
            *_inputs("encoding_profile", AUDIO_PROFILES), io.Audio.Input("source_audio"), io.Vae.Input("audio_vae")],
            [PACKAGE.Output("reference_package"), io.String.Output("report_json")],
            "Explicit audio Encode voice reference import. Supply actual prepared float32 stereo 32000Hz PCM, "
            "at most 30s. native_exact encodes once; author_10s_native_exact encodes separate 10s native chunks "
            "without image-style FP16 casting. Selected latent must match byte-for-byte. No resampling, mono "
            "duplication, trimming, inferred synchronization, foreign config execution or LoRA attach. "
            "Audio is a reference, not a replacement final soundtrack; existing Save/Route/Apply stay external.")

    @classmethod
    def execute(cls, filename, expected_sha256, source_audio, audio_vae, member_ordinal=1,
                role_id="A", encoding_profile="native_exact", confirm_import=False):
        package, report = import_audio(resolve_installed_package(filename), source_audio, audio_vae,
            expected_sha256=expected_sha256, member_ordinal=member_ordinal, role_id=role_id,
            encoding_profile=encoding_profile, confirm_import=confirm_import)
        return io.NodeOutput(package, canonical(report))

    @classmethod
    def fingerprint_inputs(cls, filename, **_inputs):
        return file_sha(resolve_installed_package(filename))


NODES = [MiniMaxH3ExternalReferenceVideoImportEXPT8, MiniMaxH3ExternalReferenceAudioImportEXPT8]
