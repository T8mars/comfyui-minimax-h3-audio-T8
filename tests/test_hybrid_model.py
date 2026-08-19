from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from safetensors.torch import save_file
import torch

from h3_audio_t8_pkg import hybrid_model as hybrid


def _tiny_curve_pair() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5], [2.0, 1.5]],
        dtype=torch.float32,
    )
    matrix = torch.eye(2, dtype=torch.float32)
    offset = torch.tensor([0.5, -0.25], dtype=torch.float32)
    return base, base @ matrix + offset


def _write_tiny_pair(root: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    base_curve, overlay_curve = _tiny_curve_pair()
    common = {
        "blocks.0.adaln_proj.linear.weight": torch.arange(12, dtype=torch.float16).reshape(6, 2) / 8,
        "blocks.0.adaln_proj.linear.bias": torch.arange(6, dtype=torch.float16) / 4,
    }
    base_path = root / "base.safetensors"
    overlay_path = root / "overlay.safetensors"
    save_file({"adaln_t_table": base_curve, **common}, str(base_path))
    overlay_common = {key: value + 0.125 for key, value in common.items()}
    save_file({"adaln_t_table": overlay_curve, **overlay_common}, str(overlay_path))
    return base_path, overlay_path, base_curve, overlay_curve


def _build_tiny_artifact(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    base_path, overlay_path, base_curve, overlay_curve = _write_tiny_pair(tmp_path)
    base_sha = hybrid.sha256_file(base_path, use_cache=False)
    overlay_sha = hybrid.sha256_file(overlay_path, use_cache=False)
    base_curve_sha = hybrid._tensor_sha256(base_curve)
    overlay_curve_sha = hybrid._tensor_sha256(overlay_curve)
    monkeypatch.setattr(
        hybrid,
        "PROFILE_SPECS",
        {"tiny_video_exp": {"blocks": (0, 0), "modalities": ("video",)}},
    )
    monkeypatch.setattr(hybrid, "CURVE_SHAPE", (4, 2))
    monkeypatch.setattr(hybrid, "MODALITY_ROWS", 2)
    monkeypatch.setattr(hybrid, "MODALITY_INDEX", {"video": 0, "text": 1, "audio": 2})
    monkeypatch.setattr(hybrid, "KNOWN_QUALITY_BASE_SHA256", base_sha)
    monkeypatch.setattr(hybrid, "KNOWN_REFERENCE_OVERLAY_SHA256", overlay_sha)
    monkeypatch.setattr(hybrid, "KNOWN_QUALITY_CURVE_SHA256", base_curve_sha)
    monkeypatch.setattr(hybrid, "KNOWN_REFERENCE_CURVE_SHA256", overlay_curve_sha)
    plan = {
        "schema": hybrid.PLAN_SCHEMA,
        "algorithm": hybrid.ALGORITHM,
        "compatible": True,
        "verification": "full_sha256",
        "recipe": hybrid.recipe_spec("tiny_video_exp"),
        "source": {
            "base_path": str(base_path),
            "overlay_path": str(overlay_path),
            "base_file_name": base_path.name,
            "overlay_file_name": overlay_path.name,
            "base_sha256": base_sha,
            "overlay_sha256": overlay_sha,
            "base_curve_sha256": base_curve_sha,
            "overlay_curve_sha256": overlay_curve_sha,
            "header_signature_sha256": "tiny-test-contract",
        },
        "errors": [],
        "warnings": [],
    }
    artifact = hybrid.build_hybrid_artifact(plan, tmp_path / "artifacts")
    return artifact, plan


class _FakePatcher:
    def __init__(self, *, patches=None):
        self.patches = copy.deepcopy(patches or {})
        self.attachments = {}
        self.received = None

    def clone(self):
        result = _FakePatcher(patches=self.patches)
        result.attachments = copy.deepcopy(self.attachments)
        return result

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.received = (patches, strength_patch, strength_model)
        return list(patches)

    def set_attachments(self, key, value):
        self.attachments[key] = value


def test_profile_payload_sizes_are_exact_and_neutral():
    assert hybrid.selected_slice_bytes("blocks_25_49_video_exp") == 14_515_200
    assert hybrid.selected_slice_bytes("blocks_25_49_video_audio_exp") == 29_030_400
    assert hybrid.selected_slice_bytes("blocks_0_49_all_modalities_exp") == 87_091_200
    assert all(name.endswith("_exp") for name in hybrid.PROFILE_SPECS)


@pytest.mark.parametrize(
    ("refs", "expected_profile", "has_visual", "has_audio"),
    [
        ([{"kind": "image"}], "blocks_25_49_video_exp", True, False),
        ([{"kind": "video"}], "blocks_25_49_video_exp", True, False),
        ([{"kind": "audio"}], "blocks_25_49_audio_exp", False, True),
        ([{"kind": "video_audio"}], "blocks_25_49_video_audio_exp", True, True),
        (
            [{"kind": "image"}, {"kind": "audio"}],
            "blocks_25_49_video_audio_exp",
            True,
            True,
        ),
    ],
)
def test_conditioning_reference_audit_selects_only_relevant_modalities(
    refs, expected_profile, has_visual, has_audio
):
    positive = [[torch.zeros(1), {"minimax_refs": refs}]]
    audit = hybrid.audit_conditioning_references(positive)

    assert audit["resolved_profile"] == expected_profile
    assert audit["has_visual_references"] is has_visual
    assert audit["has_audio_references"] is has_audio
    assert audit["quality_recommendation"] is False


def test_conditioning_reference_audit_ignores_hybrid_keyframe_sentinels():
    positive = [[
        torch.zeros(1),
        {
            "minimax_refs": [
                {"kind": "t8_keyframe_latent"},
                {"kind": "audio"},
            ],
            "minimax_keyframes": [{"resolved_frame_index": 0}],
        },
    ]]
    audit = hybrid.audit_conditioning_references(positive)

    assert audit["reference_count"] == 1
    assert audit["reference_kinds"] == ["audio"]
    assert audit["keyframe_count"] == 1
    assert audit["resolved_profile"] == "blocks_25_49_audio_exp"


def test_conditioning_reference_audit_fails_closed_for_unknown_kind():
    positive = [[torch.zeros(1), {"minimax_refs": [{"kind": "future_kind"}]}]]
    audit = hybrid.audit_conditioning_references(positive)

    assert audit["resolved_profile"] is None
    assert audit["unknown_reference_kinds"] == ["future_kind"]


def test_auto_profile_requires_real_extra_reference_rows(tmp_path, monkeypatch):
    base_path, overlay_path, _base_curve, _overlay_curve = _write_tiny_pair(tmp_path)
    monkeypatch.setattr(hybrid, "_validate_pruned_curve_header", lambda *_args: [])
    monkeypatch.setattr(hybrid, "_checkpoint_role", lambda *_args: "unknown")

    no_refs = [[torch.zeros(1), {"minimax_keyframes": [{"resolved_frame_index": 0}]}]]
    plan = hybrid.inspect_checkpoint_pair(
        base_path,
        overlay_path,
        hybrid.AUTO_PROFILE,
        "header_only_exp",
        no_refs,
    )

    assert plan["compatible"] is False
    assert plan["requested_profile"] == hybrid.AUTO_PROFILE
    assert any("requires extra image/video/audio references" in value for value in plan["errors"])


def test_auto_profile_resolves_audio_without_claiming_quality(tmp_path, monkeypatch):
    base_path, overlay_path, _base_curve, _overlay_curve = _write_tiny_pair(tmp_path)
    monkeypatch.setattr(hybrid, "_validate_pruned_curve_header", lambda *_args: [])
    monkeypatch.setattr(hybrid, "_checkpoint_role", lambda *_args: "unknown")
    positive = [[torch.zeros(1), {"minimax_refs": [{"kind": "audio"}]}]]

    plan = hybrid.inspect_checkpoint_pair(
        base_path,
        overlay_path,
        hybrid.AUTO_PROFILE,
        "header_only_exp",
        positive,
    )

    assert plan["requested_profile"] == hybrid.AUTO_PROFILE
    assert plan["recipe"]["profile"] == "blocks_25_49_audio_exp"
    assert plan["reference_audit"]["quality_recommendation"] is False
    assert any("minimizes patched modality rows" in value for value in plan["warnings"])


def test_curve_affine_rebase_preserves_overlay_function_after_fp16_roundtrip():
    base, overlay = _tiny_curve_pair()
    rebase = hybrid.fit_curve_rebase(base, overlay)
    weight = torch.tensor(
        [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8]], dtype=torch.float16
    )
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float16)
    rebased_weight, rebased_bias, metrics = hybrid.rebase_adaln_slice(
        weight,
        bias,
        rebase,
        0,
        3,
    )
    expected = overlay.to(torch.float64) @ weight.to(torch.float64).T + bias.to(torch.float64)
    actual = (
        base.to(torch.float64) @ rebased_weight.to(torch.float64).T
        + rebased_bias.to(torch.float64)
    )
    assert rebase.rank == 3
    assert rebase.table_relative_error < 1.0e-6
    assert torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected) < 1.0e-3
    assert metrics["effective_function_relative_rms"] < 1.0e-3


def test_header_only_pair_inspection_is_diagnostic_and_never_build_authorizing(tmp_path):
    base_path, overlay_path, _base_curve, _overlay_curve = _write_tiny_pair(tmp_path)
    plan = hybrid.inspect_checkpoint_pair(
        base_path,
        overlay_path,
        "blocks_25_49_video_audio_exp",
        "header_only_exp",
    )
    assert plan["compatible"] is False
    assert any("diagnostic only" in error for error in plan["errors"])
    with pytest.raises(ValueError, match="not compatible"):
        hybrid.build_hybrid_artifact(plan, tmp_path / "artifacts")


def test_artifact_builder_is_content_addressed_atomic_and_reusable(tmp_path, monkeypatch):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    validated = hybrid.validate_artifact_descriptor(artifact)
    assert Path(artifact["path"]).is_file()
    assert Path(artifact["sidecar_path"]).is_file()
    assert artifact["cache_hit"] is False
    assert validated["manifest"]["storage"] == "fp16_target_slices_with_offset_set"
    assert validated["manifest"]["payload_bytes"] == 12
    assert [operation["operation"] for operation in validated["manifest"]["operations"]] == [
        "set",
        "set",
    ]
    reused = hybrid.build_hybrid_artifact(plan, tmp_path / "artifacts")
    assert reused["cache_hit"] is True
    assert reused["artifact_sha256"] == artifact["artifact_sha256"]
    assert not list((tmp_path / "artifacts").glob("*.tmp-*"))
    assert not list((tmp_path / "artifacts").glob("*.lock"))


def test_artifact_validation_rejects_tampered_operation_manifest(tmp_path, monkeypatch):
    artifact, _plan = _build_tiny_artifact(tmp_path, monkeypatch)
    sidecar_path = Path(artifact["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest"]["operations"][0]["offset"] = [0, 1, 2]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical recipe"):
        hybrid.validate_artifact_descriptor(artifact)


def test_artifact_builder_refuses_to_overwrite_an_orphan(tmp_path, monkeypatch):
    base_path, overlay_path, base_curve, overlay_curve = _write_tiny_pair(tmp_path)
    base_sha = hybrid.sha256_file(base_path, use_cache=False)
    overlay_sha = hybrid.sha256_file(overlay_path, use_cache=False)
    monkeypatch.setattr(
        hybrid,
        "PROFILE_SPECS",
        {"tiny_video_exp": {"blocks": (0, 0), "modalities": ("video",)}},
    )
    monkeypatch.setattr(hybrid, "CURVE_SHAPE", (4, 2))
    monkeypatch.setattr(hybrid, "MODALITY_ROWS", 2)
    monkeypatch.setattr(hybrid, "MODALITY_INDEX", {"video": 0, "text": 1, "audio": 2})
    monkeypatch.setattr(hybrid, "KNOWN_QUALITY_BASE_SHA256", base_sha)
    monkeypatch.setattr(hybrid, "KNOWN_REFERENCE_OVERLAY_SHA256", overlay_sha)
    monkeypatch.setattr(hybrid, "KNOWN_QUALITY_CURVE_SHA256", hybrid._tensor_sha256(base_curve))
    monkeypatch.setattr(
        hybrid,
        "KNOWN_REFERENCE_CURVE_SHA256",
        hybrid._tensor_sha256(overlay_curve),
    )
    plan = {
        "schema": hybrid.PLAN_SCHEMA,
        "algorithm": hybrid.ALGORITHM,
        "compatible": True,
        "verification": "full_sha256",
        "recipe": hybrid.recipe_spec("tiny_video_exp"),
        "source": {
            "base_path": str(base_path),
            "overlay_path": str(overlay_path),
            "base_file_name": base_path.name,
            "overlay_file_name": overlay_path.name,
            "base_sha256": base_sha,
            "overlay_sha256": overlay_sha,
            "base_curve_sha256": hybrid._tensor_sha256(base_curve),
            "overlay_curve_sha256": hybrid._tensor_sha256(overlay_curve),
            "header_signature_sha256": "tiny-test-contract",
        },
    }
    output_root = tmp_path / "artifacts"
    output_root.mkdir()
    orphan = hybrid.artifact_path_for_plan(plan, output_root)
    orphan.write_bytes(b"do-not-overwrite")
    with pytest.raises(ValueError, match="incomplete hybrid artifact"):
        hybrid.build_hybrid_artifact(plan, output_root)
    assert orphan.read_bytes() == b"do-not-overwrite"


def test_artifact_maintenance_inspection_is_side_effect_free(tmp_path, monkeypatch):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    artifact_path = Path(artifact["path"])
    sidecar_path = Path(artifact["sidecar_path"])
    before = (
        artifact_path.read_bytes(),
        sidecar_path.read_bytes(),
        artifact_path.stat().st_mtime_ns,
        sidecar_path.stat().st_mtime_ns,
    )

    report = hybrid.maintain_hybrid_artifact(
        plan,
        tmp_path / "artifacts",
        "inspect_only",
        False,
        0,
        60.0,
    )

    assert report["mutation_performed"] is False
    assert report["validation"]["state"] == "valid_active_pair"
    assert report["transaction"] is None
    assert before == (
        artifact_path.read_bytes(),
        sidecar_path.read_bytes(),
        artifact_path.stat().st_mtime_ns,
        sidecar_path.stat().st_mtime_ns,
    )
    assert not (tmp_path / "artifacts" / "_recycle").exists()
    assert not (tmp_path / "artifacts" / "_maintenance_transactions").exists()


def test_artifact_maintenance_requires_confirmation_and_positive_epoch(tmp_path, monkeypatch):
    _artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="confirm_action=true"):
        hybrid.maintain_hybrid_artifact(
            plan, tmp_path / "artifacts", "quarantine_artifact_exp", False, 1
        )
    with pytest.raises(ValueError, match="operation_epoch > 0"):
        hybrid.maintain_hybrid_artifact(
            plan, tmp_path / "artifacts", "quarantine_artifact_exp", True, 0
        )


def test_artifact_maintenance_quarantine_replay_and_restore_are_transactional(
    tmp_path, monkeypatch
):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    artifact_path = Path(artifact["path"])
    sidecar_path = Path(artifact["sidecar_path"])

    quarantined = hybrid.maintain_hybrid_artifact(
        plan, tmp_path / "artifacts", "quarantine_artifact_exp", True, 11
    )
    assert quarantined["mutation_performed"] is True
    assert quarantined["transaction"]["phase"] == "quarantined"
    assert not artifact_path.exists()
    assert not sidecar_path.exists()

    replay = hybrid.maintain_hybrid_artifact(
        plan, tmp_path / "artifacts", "quarantine_artifact_exp", True, 11
    )
    assert replay["mutation_performed"] is False
    assert replay["transaction"]["phase"] == "quarantined"

    restored = hybrid.maintain_hybrid_artifact(
        plan, tmp_path / "artifacts", "restore_quarantined_exp", True, 11
    )
    assert restored["mutation_performed"] is True
    assert restored["transaction"]["phase"] == "restored"
    assert artifact_path.is_file()
    assert sidecar_path.is_file()
    hybrid.validate_artifact_descriptor(artifact)

    restore_replay = hybrid.maintain_hybrid_artifact(
        plan, tmp_path / "artifacts", "restore_quarantined_exp", True, 11
    )
    assert restore_replay["mutation_performed"] is False


def test_artifact_maintenance_recovers_after_process_like_interruption(
    tmp_path, monkeypatch
):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    artifact_path = Path(artifact["path"])
    sidecar_path = Path(artifact["sidecar_path"])
    real_replace = hybrid.os.replace

    def interrupt_before_sidecar_move(source, destination):
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if source_path == sidecar_path.resolve() and "_recycle" in destination_path.parts:
            raise SystemExit("simulated hard interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(hybrid.os, "replace", interrupt_before_sidecar_move)
    with pytest.raises(SystemExit, match="simulated hard interruption"):
        hybrid.maintain_hybrid_artifact(
            plan, tmp_path / "artifacts", "quarantine_artifact_exp", True, 12
        )
    assert not artifact_path.exists()
    assert sidecar_path.is_file()

    monkeypatch.setattr(hybrid.os, "replace", real_replace)
    recovered = hybrid.maintain_hybrid_artifact(
        plan, tmp_path / "artifacts", "recover_interrupted_exp", True, 12
    )
    assert recovered["mutation_performed"] is True
    assert recovered["transaction"]["phase"] == "recovered_active"
    assert artifact_path.is_file()
    assert sidecar_path.is_file()
    hybrid.validate_artifact_descriptor(artifact)


def test_artifact_maintenance_recovers_after_actual_worker_kill(
    tmp_path, monkeypatch
):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    artifact_path = Path(artifact["path"])
    sidecar_path = Path(artifact["sidecar_path"])
    output_root = tmp_path / "artifacts"
    plan_path = tmp_path / "maintenance-plan.json"
    contract_path = tmp_path / "maintenance-contract.json"
    ready_path = tmp_path / "maintenance-worker.ready"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    contract_path.write_text(
        json.dumps(
            {
                "profile_specs": hybrid.PROFILE_SPECS,
                "curve_shape": hybrid.CURVE_SHAPE,
                "modality_rows": hybrid.MODALITY_ROWS,
                "modality_index": hybrid.MODALITY_INDEX,
                "base_sha256": plan["source"]["base_sha256"],
                "overlay_sha256": plan["source"]["overlay_sha256"],
                "base_curve_sha256": plan["source"]["base_curve_sha256"],
                "overlay_curve_sha256": plan["source"]["overlay_curve_sha256"],
            }
        ),
        encoding="utf-8",
    )
    worker = Path(__file__).with_name("multiprocess_hybrid_maintenance_worker.py")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [
            sys.executable,
            str(worker),
            "--plan",
            str(plan_path),
            "--contract",
            str(contract_path),
            "--output-root",
            str(output_root),
            "--ready",
            str(ready_path),
            "--epoch",
            "15",
            "--hold-seconds",
            "60",
        ],
        cwd=str(Path(__file__).resolve().parents[3]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    try:
        deadline = time.monotonic() + 30.0
        while not ready_path.is_file() and process.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if not ready_path.is_file():
            stdout, stderr = process.communicate(timeout=5.0)
            pytest.fail(
                "hybrid maintenance worker did not reach the kill point: "
                f"stdout={stdout!r}, stderr={stderr!r}"
            )
        process.kill()
        process.wait(timeout=10.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10.0)

    assert not artifact_path.exists()
    assert sidecar_path.is_file()
    context = hybrid._maintenance_plan_context(plan, output_root, 15)
    assert context["maintenance_lock_path"].is_file()
    old = time.time() - 7200.0
    os.utime(context["maintenance_lock_path"], (old, old))
    recovered = hybrid.maintain_hybrid_artifact(
        plan, output_root, "recover_interrupted_exp", True, 15, 1.0
    )
    assert recovered["transaction"]["phase"] == "recovered_active"
    assert recovered["archived_stale_maintenance_lock"] is not None
    assert artifact_path.is_file()
    assert sidecar_path.is_file()
    hybrid.validate_artifact_descriptor(artifact)


def test_artifact_maintenance_rejects_tampered_journal_contract(
    tmp_path, monkeypatch
):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    root = tmp_path / "artifacts"
    hybrid.maintain_hybrid_artifact(
        plan, root, "quarantine_artifact_exp", True, 14
    )
    context = hybrid._maintenance_plan_context(plan, root, 14)
    journal_path = context["transaction_path"]
    original = json.loads(journal_path.read_text(encoding="utf-8"))

    invalid_count = copy.deepcopy(original)
    invalid_count["moved_count"] = 0
    journal_path.write_text(json.dumps(invalid_count), encoding="utf-8")
    with pytest.raises(ValueError, match="phase/count mismatch"):
        hybrid.maintain_hybrid_artifact(
            plan, root, "restore_quarantined_exp", True, 14
        )

    incomplete_pair = copy.deepcopy(original)
    incomplete_pair["items"] = incomplete_pair["items"][:1]
    incomplete_pair["moved_count"] = 1
    journal_path.write_text(json.dumps(incomplete_pair), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain exactly"):
        hybrid.maintain_hybrid_artifact(
            plan, root, "restore_quarantined_exp", True, 14
        )

    journal_path.unlink()
    journal_path.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink file"):
        hybrid.maintain_hybrid_artifact(
            plan, root, "restore_quarantined_exp", True, 14
        )
    journal_path.rmdir()

    journal_path.write_text(json.dumps(original), encoding="utf-8")
    restored = hybrid.maintain_hybrid_artifact(
        plan, root, "restore_quarantined_exp", True, 14
    )
    assert restored["transaction"]["phase"] == "restored"
    hybrid.validate_artifact_descriptor(artifact)


def test_artifact_maintenance_rejects_internal_directory_symlink(
    tmp_path, monkeypatch
):
    _artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    root = tmp_path / "artifacts"
    external = tmp_path / "external-transactions"
    external.mkdir()
    transaction_root = root / "_maintenance_transactions"
    try:
        transaction_root.symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="may not be a symbolic link"):
        hybrid.maintain_hybrid_artifact(
            plan, root, "inspect_only", False, 0
        )


def test_stale_build_residue_is_quarantined_without_deletion_and_can_rebuild(
    tmp_path, monkeypatch
):
    artifact, plan = _build_tiny_artifact(tmp_path, monkeypatch)
    artifact_path = Path(artifact["path"])
    sidecar_path = Path(artifact["sidecar_path"])
    sidecar_path.unlink()
    lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
    lock_path.write_text(
        json.dumps({"pid": 999999999, "created_unix": time.time() - 7200}),
        encoding="utf-8",
    )
    old = time.time() - 7200
    os.utime(artifact_path, (old, old))
    os.utime(lock_path, (old, old))

    report = hybrid.maintain_hybrid_artifact(
        plan,
        tmp_path / "artifacts",
        "quarantine_stale_build_residue_exp",
        True,
        13,
        1.0,
    )
    assert report["mutation_performed"] is True
    assert report["transaction"]["item_count"] == 2
    assert not artifact_path.exists()
    assert not lock_path.exists()
    recycled = list((tmp_path / "artifacts" / "_recycle").rglob("*"))
    assert any(path.name == artifact_path.name for path in recycled)
    assert any(path.name == lock_path.name for path in recycled)

    rebuilt = hybrid.build_hybrid_artifact(plan, tmp_path / "artifacts")
    assert rebuilt["cache_hit"] is False
    assert artifact_path.is_file()
    assert sidecar_path.is_file()


def test_artifact_applies_offset_set_to_clone_and_attaches_provenance(tmp_path, monkeypatch):
    artifact, _plan = _build_tiny_artifact(tmp_path, monkeypatch)
    original = _FakePatcher()
    patched, attachment = hybrid.apply_artifact_to_model(original, artifact)
    assert patched is not original
    assert original.received is None
    patches, strength_patch, strength_model = patched.received
    assert strength_patch == strength_model == 1.0
    assert len(patches) == 2
    assert all(value[0] == "set" for value in patches.values())
    assert all(isinstance(key, tuple) and key[1] == (0, 0, 2) for key in patches)
    assert attachment["operation_count"] == 2
    assert patched.attachments[hybrid.ATTACHMENT_KEY] == attachment


def test_artifact_rejects_existing_patch_on_same_adaln_tensor(tmp_path, monkeypatch):
    artifact, _plan = _build_tiny_artifact(tmp_path, monkeypatch)
    model_key = artifact["manifest"]["operations"][0]["model_key"]
    original = _FakePatcher(patches={model_key: [(1.0, (torch.ones(1),), 1.0, None, None)]})
    with pytest.raises(ValueError, match="existing whole-tensor patch"):
        hybrid.apply_artifact_to_model(original, artifact)


def test_base_only_loader_is_a_stock_loader_control(tmp_path, monkeypatch):
    import comfy.sd

    base_path = tmp_path / "base.safetensors"
    base_path.write_bytes(b"stock-loader-control")
    sentinel = object()
    calls = []

    def fake_load(path, model_options):
        calls.append((path, model_options))
        return sentinel

    monkeypatch.setattr(comfy.sd, "load_diffusion_model", fake_load)
    model, report = hybrid.load_hybrid_model(base_path, "base_only", "default")
    assert model is sentinel
    assert calls == [(str(base_path.resolve()), {})]
    assert report["artifact_applied"] is False


def test_file_stat_fingerprint_changes_after_file_replacement(tmp_path):
    path = tmp_path / "tracked.bin"
    path.write_bytes(b"a")
    before = hybrid.file_stat_fingerprint([path])
    path.write_bytes(b"replacement")
    after = hybrid.file_stat_fingerprint([path])
    assert before != after


def test_hybrid_api_example_is_isolated_and_routes_the_patched_model():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "hybrid_model_advanced_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["class_type"]: (node_id, node) for node_id, node in workflow.items()}
    assert "UNETLoader" not in by_type
    assert "LoraLoaderBypassModelOnly" not in by_type
    inspector_id, inspector = by_type["MiniMaxH3HybridPairInspectorT8Advanced"]
    builder_id, builder = by_type["MiniMaxH3HybridArtifactBuilderT8Advanced"]
    loader_id, loader = by_type["MiniMaxH3HybridModelLoaderT8Advanced"]
    conditioning_id, conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    sampler_id, sampler = by_type["MiniMaxH3DualClockSamplerT8"]
    _guider_id, guider = by_type["BasicGuider"]
    _advanced_id, advanced = by_type["SamplerCustomAdvanced"]
    assert inspector["inputs"] == {
        "quality_base": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "reference_overlay": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "profile": "blocks_25_49_video_audio_exp",
        "verification": "full_sha256",
    }
    assert builder["inputs"]["hybrid_plan"] == [inspector_id, 0]
    assert loader["inputs"]["hybrid_artifact"] == [builder_id, 0]
    assert loader["inputs"]["mode"] == "apply_artifact_exp"
    assert sampler["inputs"]["model"] == [loader_id, 0]
    assert guider["inputs"]["model"] == [sampler_id, 0]
    assert guider["inputs"]["conditioning"] == [conditioning_id, 0]
    assert advanced["inputs"]["latent_image"] == [conditioning_id, 1]
    assert sampler["inputs"]["steps"] == 20
    assert conditioning["inputs"]["task_type"] == "Ref2VA"


def test_hybrid_frontend_workflow_is_importable_and_link_consistent():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "09-hybrid-model"
        / "2026-08-09_H3_Hybrid_Model_Advanced_Stock20_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    types = [node["type"] for node in workflow["nodes"]]
    assert types[:3] == [
        "MiniMaxH3HybridPairInspectorT8Advanced",
        "MiniMaxH3HybridArtifactBuilderT8Advanced",
        "MiniMaxH3HybridModelLoaderT8Advanced",
    ]
    assert "UNETLoader" not in types
    assert not any("LoraLoader" in node_type for node_type in types)
    links = {link[0]: link for link in workflow["links"]}
    for node in workflow["nodes"]:
        for input_value in node.get("inputs", []):
            link_id = input_value.get("link")
            if link_id is not None:
                assert link_id in links
                assert links[link_id][3] == node["id"]
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert link_id in links
                assert links[link_id][1] == node["id"]


def test_hybrid_audio_reference_api_uses_conditioning_aware_minimal_profile():
    path = (
        Path(__file__).resolve().parents[1]
        / "tests" / "fixtures" / "api" / "hybrid_model_audio_reference_api.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["class_type"]: (node_id, node) for node_id, node in workflow.items()}
    inspector_id, inspector = by_type["MiniMaxH3HybridPairInspectorT8Advanced"]
    _builder_id, builder = by_type["MiniMaxH3HybridArtifactBuilderT8Advanced"]
    _loader_id, loader = by_type["MiniMaxH3HybridModelLoaderT8Advanced"]
    conditioning_id, conditioning = by_type["MiniMaxH3AudioConditioningT8"]

    assert inspector["inputs"]["profile"] == hybrid.AUTO_PROFILE
    assert inspector["inputs"]["positive"] == [conditioning_id, 0]
    assert builder["inputs"]["hybrid_plan"] == [inspector_id, 0]
    assert loader["inputs"]["mode"] == "apply_artifact_exp"
    assert conditioning["inputs"]["ref_audios.ref_audio_0"] == [
        by_type["LoadAudio"][0],
        0,
    ]
    assert not any(
        name.startswith(("ref_images.", "ref_videos.", "ref_video_audios."))
        for name in conditioning["inputs"]
    )

    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "workflows"
            / "09-hybrid-model"
            / "2026-08-09_H3_Hybrid_Model_Audio_Reference_Stock20_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in contract["nodes"]}
    links = {link[0]: link for link in contract["links"]}
    assert contract["last_node_id"] == max(nodes)
    assert contract["last_link_id"] == max(links)
    assert "LoadAudio" in {node["type"] for node in nodes.values()}
    for node in nodes.values():
        for input_value in node.get("inputs", []):
            link_id = input_value.get("link")
            if link_id is not None:
                assert link_id in links
                assert links[link_id][3] == node["id"]


def test_hybrid_mixed_reference_examples_are_auto_routed_and_importable():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (root / "tests" / "fixtures" / "api" / "hybrid_model_mixed_reference_api.json").read_text(
            encoding="utf-8"
        )
    )
    by_type = {node["class_type"]: (node_id, node) for node_id, node in workflow.items()}
    conditioning_id, conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    inspector = by_type["MiniMaxH3HybridPairInspectorT8Advanced"][1]
    assert inspector["inputs"]["profile"] == hybrid.AUTO_PROFILE
    assert inspector["inputs"]["positive"] == [conditioning_id, 0]
    assert "ref_images.ref_image_0" in conditioning["inputs"]
    assert "ref_audios.ref_audio_0" in conditioning["inputs"]

    frontend = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "09-hybrid-model"
            / "2026-08-09_H3_Hybrid_Model_Mixed_Reference_Stock20_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in frontend["nodes"]}
    links = {link[0]: link for link in frontend["links"]}
    assert frontend["last_node_id"] == max(nodes)
    assert frontend["last_link_id"] == max(links)
    assert {"LoadImage", "LoadAudio"}.issubset(
        {node["type"] for node in nodes.values()}
    )


def test_hybrid_artifact_maintenance_examples_are_safe_and_link_consistent():
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (root / "tests" / "fixtures" / "api" / "hybrid_artifact_maintenance_api.json").read_text(
            encoding="utf-8"
        )
    )
    inspector_id = next(
        node_id
        for node_id, node in api.items()
        if node["class_type"] == "MiniMaxH3HybridPairInspectorT8Advanced"
    )
    maintenance = next(
        node
        for node in api.values()
        if node["class_type"] == "MiniMaxH3HybridArtifactMaintenanceT8Advanced"
    )
    assert maintenance["inputs"] == {
        "hybrid_plan": [inspector_id, 0],
        "action": "inspect_only",
        "confirm_action": False,
        "operation_epoch": 0,
        "stale_after_minutes": 60.0,
    }

    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "09-hybrid-model"
            / "2026-08-12_H3_Hybrid_Artifact_Maintenance_Advanced.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    assert nodes[2]["type"] == "MiniMaxH3HybridArtifactMaintenanceT8Advanced"
    assert nodes[2]["widgets_values"] == ["inspect_only", False, 0, 60.0]
    assert not any(
        value in {"quarantine_artifact_exp", "restore_quarantined_exp"}
        for node in nodes.values()
        for value in node.get("widgets_values", [])
        if isinstance(value, str)
    )
    for node in nodes.values():
        for input_value in node.get("inputs", []):
            link_id = input_value.get("link")
            if link_id is not None:
                assert link_id in links
                assert links[link_id][3] == node["id"]
