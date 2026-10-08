"""Separate append-only community image importer; legacy reference loaders stay native."""
from comfy_api.latest import io

from .external_reference_image_import import PROFILES, import_image
from .nodes_reference_package import PACKAGE, schema
from .reference_package import canonical, file_sha, installed_package_names, resolve_installed_package


class MiniMaxH3ExternalReferenceImageImportEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Import encoded image / 外部编码图像导入", [
            io.Combo.Input("filename", options=installed_package_names()),
            io.String.Input("expected_sha256", default="", tooltip="Required SHA256 of the whole external file."),
            io.Int.Input("member_ordinal", default=1, min=1, max=256,
                         tooltip="One-based member ordinal; only image Encode members are qualified."),
            io.String.Input("role_id", default="A"),
            io.Combo.Input("precision_profile", options=list(PROFILES), default="native_exact",
                           tooltip="Author Encode casts to FP16. Explicit FP16 matching is not native-FP32 bit parity."),
            io.Boolean.Input("confirm_import", default=False,
                             tooltip="Confirm you may use this reference and its actual source image. Not a legal certificate."),
            io.Image.Input("source_image"), io.Vae.Input("video_vae")],
            [PACKAGE.Output("reference_package"), io.String.Output("report_json")],
            "Explicit v4/v5 or RefLoRA-reference-half image import. Actual original RGB at exactly the "
            "encoded geometry is re-encoded by the connected native VAE and compared byte-for-byte. "
            "No guessed normalization, resize, pooled/trained import, foreign config/path execution or LoRA attach. "
            "Only selected latent is loaded; input untouched. Existing Save is create-only; Route/Apply/EAV/Relay stay external.")

    @classmethod
    def execute(cls, filename, expected_sha256, source_image, video_vae, member_ordinal=1,
                role_id="A", precision_profile="native_exact", confirm_import=False):
        package, report = import_image(resolve_installed_package(filename), source_image, video_vae,
            expected_sha256=expected_sha256, member_ordinal=member_ordinal, role_id=role_id,
            precision_profile=precision_profile, confirm_import=confirm_import)
        return io.NodeOutput(package, canonical(report))

    @classmethod
    def fingerprint_inputs(cls, filename, **_inputs):
        return file_sha(resolve_installed_package(filename))


NODES = [MiniMaxH3ExternalReferenceImageImportEXPT8]
