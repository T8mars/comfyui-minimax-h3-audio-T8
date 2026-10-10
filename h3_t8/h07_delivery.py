"""Minimal byte-bound take sidecars. Public export is an allowlist, never raw metadata."""
import hashlib
from pathlib import Path
import re

from .h07_reports import document, identifier


def file_record(path, expected, extensions):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Supply the actual source SHA256")
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in extensions:
        raise ValueError("Explicit existing media/workflow file required, not a weight or directory")
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected.lower():
        raise ValueError("Delivery source SHA changed; do not bind another take")
    return dict(path=str(source), sha256=actual, bytes=source.stat().st_size)


def delivery_documents(video_path, video_sha256, workflow_path, workflow_sha256,
                       project_id, shot_id, take_id, recipe_id, audio_sources_json="[]", *,
                       completed_quality_stage=None, completed_modular_stage=None):
    ids = {key: identifier(value) for key, value in dict(project_id=project_id, shot_id=shot_id,
                                                       take_id=take_id, recipe_id=recipe_id).items()}
    video = file_record(video_path, video_sha256, {".mp4"})
    workflow = file_record(workflow_path, workflow_sha256, {".json"})
    sources, roles = [], set()
    for row in document(audio_sources_json, list):
        if type(row) is not dict or row.get("origin") not in ("original", "native_generated", "postprocessed"):
            raise ValueError("Audio origin is original/native_generated/postprocessed, not an implicit replacement")
        role = identifier(row.get("role_id"))
        if role in roles:
            raise ValueError("Duplicate audio role ID")
        roles.add(role)
        sources.append(dict(role_id=role, origin=row["origin"],
            file=file_record(row.get("path"), row.get("sha256"), {".mp4", ".wav", ".flac", ".mp3", ".m4a", ".ogg"})))
    if completed_quality_stage is not None and completed_modular_stage is not None:
        raise ValueError("Select one actual completed producer, not two contradictory take identities")
    producer = dict(status="unknown", actual_NFE="unknown")
    if completed_quality_stage is not None:
        from .freevideo_quality.runtime import validate_stage
        stage = completed_quality_stage
        receipt = validate_stage(stage)
        producer = dict(status="validated_completed_quality_stage_not_visual_voice_approval",
            schema=receipt["schema"], source_revision=receipt["freevideo_revision"],
            receipt_sha256=stage.receipt_sha256, request_sha256=receipt["request_sha256"],
            role=receipt["role"], actual_NFE=receipt["completed_nfe"])
    elif completed_modular_stage is not None:
        from .modular_sampling.results import StageResult
        if not isinstance(completed_modular_stage, StageResult):
            raise ValueError("Expected real typed StageResult, not an asserted JSON receipt")
        receipt = completed_modular_stage.verify()
        context = receipt["request"]["stage_context"]
        producer = dict(status="validated_completed_modular_bytes_not_generic_reuse_approval",
            schema=receipt["schema"], recipe=context["recipe"], role=context["stage"],
            receipt_sha256=receipt["receipt_sha256"], request_sha256=receipt["request_sha256"],
            actual_NFE=receipt["execution"]["denoiser_evaluations"],
            verified_recipe_completion=receipt["verified_recipe_completion"])
    common = dict(schema="t8.h07.take-delivery.v1", **ids, producer=producer,
        quality="not_automatically_accepted", binding_scope="file_bytes_not_proof_of_audio_identity_or_decode",
        external_upload_started=False, cache_reuse_authorized=False,
        audio_role_scope="explicit_declarations_not_automatic_voice_assignment")
    private = dict(common, visibility="private_local", video=video, workflow=workflow, audio_sources=sources)
    # No raw receipt/workflow JSON, prompts, filenames, usernames, paths,
    # credential-like arbitrary fields, UUID of the GPU, or cache directories.
    public = dict(common, visibility="shareable_allowlisted_copy_review_before_sharing",
        video={key: video[key] for key in ("sha256", "bytes")},
        workflow={key: workflow[key] for key in ("sha256", "bytes")},
        audio_sources=[dict(role_id=row["role_id"], origin=row["origin"],
            file={key: row["file"][key] for key in ("sha256", "bytes")}) for row in sources])
    return private, public
