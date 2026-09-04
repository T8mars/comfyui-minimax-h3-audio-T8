from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from h3_audio_t8_pkg import dlss_nr_advanced as dlss
from h3_audio_t8_pkg import nodes_dlss_nr_advanced as nodes_dlss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_built_in_allowlist_contains_only_audited_full_archives():
    assert set(dlss.APPROVED_RELEASES) == {"1.2", "1.3"}
    assert dlss.APPROVED_RELEASES["1.2"]["archives"] == {
        "video2dlssnr_release.zip": {
            "bytes": 247409690,
            "sha256": "df9de64b92b62ad381e7b83a995c6b84d5bbb6bf4db59e7d46cb75c6e3f5feb3",
        },
        "video2dlssnr-comfyui.zip": {
            "bytes": 247400780,
            "sha256": "26f30045a89bb9f957411303d8f4beb1a5aecf572b189455f7a94b60cdd20ce5",
        },
    }
    assert dlss.APPROVED_RELEASES["1.3"]["archives"] == {
        "video2dlssnr_release.zip": {
            "bytes": 247420277,
            "sha256": "1cab80a30927421c7fb36f42eb5f11d50fef2f2dbc8c1236424a0ccab2eff0bd",
        }
    }
    for release in dlss.APPROVED_RELEASES.values():
        assert all("light" not in name for name in release["archives"])


def test_v12_example_manifest_matches_the_official_full_archive_layout():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "runtime-manifests"
        / "dlss-nr-v1.2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["release_archive"] == {
        "path": "video2dlssnr_release.zip",
        "bytes": 247409690,
        "sha256": "df9de64b92b62ad381e7b83a995c6b84d5bbb6bf4db59e7d46cb75c6e3f5feb3",
    }
    assert {
        logical: spec["archive_path"] for logical, spec in manifest["files"].items()
    } == {
        "executable": "video2dlssnr/out/video2dlssnr.exe",
        "dlss_sr": "video2dlssnr/out/nvngx_dlss.dll",
        "dlss_nr": "video2dlssnr/out/nvngx_dlssnr.dll",
        "forwarder": "video2dlssnr/out/nvngx.dll_dlssnr.dll",
    }


def test_v13_example_manifest_matches_the_official_full_archive_layout():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "runtime-manifests"
        / "dlss-nr-v1.3.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_archive"] == {
        "path": "video2dlssnr_release.zip",
        "bytes": 247420277,
        "sha256": "1cab80a30927421c7fb36f42eb5f11d50fef2f2dbc8c1236424a0ccab2eff0bd",
    }
    assert {
        logical: spec["archive_path"] for logical, spec in manifest["files"].items()
    } == {
        "executable": "out/video2dlssnr.exe",
        "dlss_sr": "out/nvngx_dlss.dll",
        "dlss_nr": "out/nvngx_dlssnr.dll",
        "forwarder": "out/nvngx.dll_dlssnr.dll",
    }


def _fake_runtime(
    tmp_path: Path,
    monkeypatch,
    *,
    runtime_version: str = "1.2",
    archive_backslashes: bool = False,
) -> tuple[Path, dict]:
    root = tmp_path / "models" / "DLSS-NR" / runtime_version
    root.mkdir(parents=True)
    payloads = {
        "bin/video2dlssnr.exe": f"fake-executable-v{runtime_version}".encode(),
        "bin/nvngx_dlss.dll": f"fake-dlss-sr-v{runtime_version}".encode(),
        "bin/nvngx_dlssnr.dll": f"fake-dlss-nr-v{runtime_version}".encode(),
        "bin/nvngx.dll_dlssnr.dll": f"fake-forwarder-v{runtime_version}".encode(),
    }
    archive = root / "video2dlssnr_release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for member, payload in payloads.items():
            archive_member = (
                member.replace("/", "\\") if archive_backslashes else member
            )
            bundle.writestr(archive_member, payload)
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    tag_commit = (
        dlss.V13_RELEASE_TAG_COMMIT
        if runtime_version == "1.3"
        else dlss.V12_RELEASE_TAG_COMMIT
    )
    source_commit = (
        dlss.V13_NATIVE_SOURCE_AUDIT_COMMIT
        if runtime_version == "1.3"
        else dlss.V12_NATIVE_SOURCE_AUDIT_COMMIT
    )
    approved = {
        "runtime_version": runtime_version,
        "release_tag": f"v{runtime_version}",
        "release_tag_commit": tag_commit,
        "source_audit_commit": source_commit,
        "archives": {
            archive.name: {
                "bytes": archive.stat().st_size,
                "sha256": _sha256(archive),
            }
        },
    }
    monkeypatch.setattr(dlss, "APPROVED_RELEASES", {runtime_version: approved})
    manifest = {
        "schema": dlss.RUNTIME_MANIFEST_SCHEMA,
        "runtime_version": runtime_version,
        "release_tag": f"v{runtime_version}",
        "release_tag_commit": tag_commit,
        "source_audit_commit": source_commit,
        "release_archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
        },
        "files": {
            logical: {
                "path": member,
                "archive_path": member,
            }
            for logical, member in {
                "executable": "bin/video2dlssnr.exe",
                "dlss_sr": "bin/nvngx_dlss.dll",
                "dlss_nr": "bin/nvngx_dlssnr.dll",
                "forwarder": "bin/nvngx.dll_dlssnr.dll",
            }.items()
        },
    }
    (root / dlss.RUNTIME_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root, manifest


def _host(*, driver: str = "616.56", free_mib: float = 8192.0) -> dict:
    return {
        "platform": "Windows",
        "windows_release": "11",
        "nvidia_smi": {
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 4060 Ti",
                    "driver_version": driver,
                    "memory_total_mib": 16380.0,
                    "memory_free_mib": free_mib,
                    "pci_bus_id": "00000000:01:00.0",
                    "uuid": "GPU-test",
                }
            ],
        },
        "torch_cuda": {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 4060 Ti",
                    "memory_total_mib": 16380.0,
                    "pci_bus_id": "00000000:01:00.0",
                }
            ],
        },
    }


def test_relative_runtime_paths_reject_escape_absolute_and_symlink(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "real.bin").write_bytes(b"x")
    (root / "link.bin").symlink_to(root / "real.bin")

    assert dlss.resolve_runtime_member(root, "real.bin") == root / "real.bin"
    for unsafe in ("../escape.bin", "/absolute.bin", "C:/escape.bin", "a\\b.bin", ""):
        with pytest.raises(ValueError, match="relative POSIX"):
            dlss.resolve_runtime_member(root, unsafe)
    with pytest.raises(ValueError, match="symbolic link"):
        dlss.resolve_runtime_member(root, "link.bin")


def test_runtime_discovery_lists_only_allowlisted_version_directories(tmp_path):
    (tmp_path / "DLSS-NR" / "1.2").mkdir(parents=True)
    (tmp_path / "DLSS-NR" / "1.3").mkdir()
    assert dlss.available_runtime_versions(tmp_path) == ["1.2", "1.3"]


def test_static_runtime_audit_requires_explicit_license_acceptance(
    tmp_path, monkeypatch
):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=False,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
    )
    assert ready is False
    assert report["status"] == "BLOCKED"
    assert any("license" in item.lower() for item in report["errors"])
    assert report["mutations"] == []
    assert report["downloads"] == []


def test_static_runtime_audit_verifies_archive_and_installed_file_identity(
    tmp_path, monkeypatch
):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
    )
    assert ready is False
    assert report["status"] == "STATIC_PASS_REAL_PROBE_REQUIRED"
    assert report["static_validation"]["passed"] is True
    assert set(report["runtime_files"]) == set(dlss.REQUIRED_RUNTIME_FILES)
    assert all(item["matches_archive"] for item in report["runtime_files"].values())

    (root / "bin" / "nvngx_dlss.dll").write_bytes(b"tampered")
    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
    )
    assert ready is False
    assert report["status"] == "BLOCKED"
    assert any("does not match" in item for item in report["errors"])


def test_runtime_audit_requires_all_loaded_files_in_one_directory(
    tmp_path, monkeypatch
):
    root, manifest = _fake_runtime(tmp_path, monkeypatch)
    relocated = root / "other" / "nvngx.dll_dlssnr.dll"
    relocated.parent.mkdir()
    (root / "bin" / "nvngx.dll_dlssnr.dll").replace(relocated)
    manifest["files"]["forwarder"]["path"] = "other/nvngx.dll_dlssnr.dll"
    (root / dlss.RUNTIME_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
    )
    assert ready is False
    assert report["status"] == "BLOCKED"
    assert any("one directory" in item for item in report["errors"])


def test_v13_windows_zip_member_separators_are_canonicalized(tmp_path, monkeypatch):
    root, _manifest = _fake_runtime(
        tmp_path,
        monkeypatch,
        runtime_version="1.3",
        archive_backslashes=True,
    )
    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.3",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
    )
    assert ready is False
    assert report["status"] == "STATIC_PASS_REAL_PROBE_REQUIRED"
    assert report["static_validation"]["passed"] is True
    assert all(item["matches_archive"] for item in report["runtime_files"].values())


@pytest.mark.parametrize(
    ("host", "message"),
    [
        (_host(driver="591.74"), "616.56"),
        (_host(free_mib=511.9), "512"),
        ({**_host(), "platform": "Linux"}, "Windows"),
    ],
)
def test_runtime_audit_fails_closed_on_host_gates(tmp_path, monkeypatch, host, message):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=host,
    )
    assert ready is False
    assert report["status"] == "BLOCKED"
    assert any(message in item for item in report["errors"])


def test_feature_probe_uses_argument_array_and_requires_matching_gpu(
    tmp_path, monkeypatch
):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    calls = []

    def probe_runner(command, *, timeout_seconds):
        calls.append((command, timeout_seconds))
        return {
            "returncode": 0,
            "stdout": "  gpu:    NVIDIA GeForce RTX 4060 Ti (16109 MB)\n"
            "Neural Rendering RAN via route F (forwarder).\n",
            "stderr": "",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
        probe_runner=probe_runner,
    )
    assert ready is True
    assert report["status"] == "READY"
    command, timeout = calls[0]
    assert isinstance(command, list)
    assert command[0] == str((root / "bin" / "video2dlssnr.exe").resolve())
    assert command[1:] == [
        "--probe-nr",
        "--dll-dir",
        str((root / "bin").resolve()),
        "--adapter",
        "0",
        "--nr-in",
        "64x64",
        "--nr-out",
        "64x64",
        "--nr-preset",
        "0",
    ]
    assert timeout == dlss.PROBE_TIMEOUT_SECONDS
    assert report["probe_memory_tolerance_mib"] == 512.0

    def wrong_gpu(command, *, timeout_seconds):
        del command, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "gpu: NVIDIA RTX 4090 (24564 MB)\n"
            "Neural Rendering RAN via route F (forwarder).\n",
            "stderr": "",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
        probe_runner=wrong_gpu,
    )
    assert ready is False
    assert report["status"] == "BLOCKED"
    assert any("adapter" in item.lower() for item in report["errors"])

    def same_name_wrong_capacity(command, *, timeout_seconds):
        del command, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "gpu: NVIDIA GeForce RTX 4060 Ti (8188 MB)\n"
            "Neural Rendering RAN via route F (forwarder).\n",
            "stderr": "",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
        probe_runner=same_name_wrong_capacity,
    )
    assert ready is False
    assert any("adapter" in item.lower() for item in report["errors"])


def test_feature_probe_requires_forwarder_route_f_marker(tmp_path, monkeypatch):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)

    def route_a_only(command, *, timeout_seconds):
        del command, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "gpu: NVIDIA GeForce RTX 4060 Ti (16109 MB)\n",
            "stderr": "Neural Rendering RAN via route A.\n",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
        probe_runner=route_a_only,
    )
    assert ready is False
    assert report["feature_probe"]["route_f_success_marker"] is False
    assert any("route F" in item for item in report["errors"])


def test_feature_probe_rejects_ambiguous_same_model_same_memory_gpus(
    tmp_path, monkeypatch
):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    host = _host()
    second = dict(host["nvidia_smi"]["gpus"][0])
    second.update(
        {
            "index": 1,
            "pci_bus_id": "00000000:02:00.0",
            "uuid": "GPU-test-duplicate",
        }
    )
    host["nvidia_smi"]["gpus"].append(second)

    def successful_probe(command, *, timeout_seconds):
        del command, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "gpu: NVIDIA GeForce RTX 4060 Ti (16109 MB)\n"
            "Neural Rendering RAN via route F (forwarder).\n",
            "stderr": "",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=host,
        probe_runner=successful_probe,
    )
    assert ready is False
    assert report["feature_probe"]["dxgi_identity_candidate_count"] == 2
    assert report["feature_probe"]["dxgi_identity_unique"] is False
    assert any("uniquely" in item for item in report["errors"])


def test_execution_revalidation_rejects_new_same_signature_gpu(tmp_path, monkeypatch):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)

    def successful_probe(command, *, timeout_seconds):
        del command, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "gpu: NVIDIA GeForce RTX 4060 Ti (16109 MB)\n"
            "Neural Rendering RAN via route F (forwarder).\n",
            "stderr": "",
            "timed_out": False,
        }

    ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=_host(),
        probe_runner=successful_probe,
    )
    assert ready is True
    handle = dlss.runtime_handle_from_report(report)
    assert handle is not None

    changed_host = _host()
    duplicate = dict(changed_host["nvidia_smi"]["gpus"][0])
    duplicate.update(
        {"index": 1, "pci_bus_id": "00000000:02:00.0", "uuid": "GPU-duplicate"}
    )
    changed_host["nvidia_smi"]["gpus"].append(duplicate)
    with pytest.raises(RuntimeError, match="revalidated uniquely"):
        dlss.revalidate_runtime_handle(handle, host_info=changed_host)


def test_cuda_to_nvidia_mapping_uses_pci_not_assumed_dxgi_order(tmp_path, monkeypatch):
    root, _manifest = _fake_runtime(tmp_path, monkeypatch)
    host = _host()
    host["nvidia_smi"]["gpus"] = [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "driver_version": "616.56",
            "memory_total_mib": 24564.0,
            "memory_free_mib": 20000.0,
            "pci_bus_id": "00000000:01:00.0",
            "uuid": "GPU-4090",
        },
        {
            "index": 1,
            "name": "NVIDIA GeForce RTX 4060 Ti",
            "driver_version": "616.56",
            "memory_total_mib": 16380.0,
            "memory_free_mib": 8000.0,
            "pci_bus_id": "00000000:02:00.0",
            "uuid": "GPU-4060",
        },
    ]
    host["torch_cuda"]["devices"][0]["pci_bus_id"] = "0000:02:00.0"
    _ready, report = dlss.audit_dlss_nr_runtime(
        root,
        "1.2",
        accept_external_runtime_license=True,
        probe_mode="static_only",
        dxgi_adapter_index=0,
        cuda_device_index=0,
        host_info=host,
    )
    assert report["static_validation"]["passed"] is True
    assert report["device_mapping"]["nvidia"]["index"] == 1


def test_runtime_audit_node_is_first_append_only_dlss_node():
    classes = nodes_dlss.DLSS_NR_ADVANCED_NODE_CLASSES
    schemas = [node.define_schema() for node in classes]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3DLSSNRRuntimeAuditT8Advanced",
        "MiniMaxH3DLSSNRImageSuperResolutionT8Advanced",
        "MiniMaxH3DLSSNRVideoFramesT8Advanced",
        "MiniMaxH3DLSSNRVideoFileT8Advanced",
    ]
    schema = schemas[0]
    assert schema.is_experimental is False
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["runtime_version"].default == "1.3"
    assert inputs["accept_external_runtime_license"].default is False
    assert inputs["probe_mode"].default == "static_only"
    for schema in schemas:
        assert schema.is_experimental is False
        assert schema.category == nodes_dlss.CATEGORY
    image_inputs = {item.id: item for item in schemas[1].inputs}
    frame_inputs = {item.id: item for item in schemas[2].inputs}
    file_inputs = {item.id: item for item in schemas[3].inputs}
    assert image_inputs["mode"].default == "sr_nr"
    assert image_inputs["scale"].default == "2.0"
    assert image_inputs["quality_profile"].default == "standard"
    assert image_inputs["sr_preset"].default == "default"
    assert image_inputs["nr_intensity"].default == 1.5
    assert image_inputs["nr_detail"].default == 1.0
    assert image_inputs["nr_detail"].max == 1.0
    assert image_inputs["nr_color"].default == 1.0
    assert image_inputs["nr_ui_correction"].default is False
    assert frame_inputs["motion_engine"].default == "auto"
    assert file_inputs["crf"].default == 18.0


def _runtime_handle(root: Path, runtime_version: str = "1.3") -> dict:
    return {
        "schema": dlss.RUNTIME_HANDLE_SCHEMA,
        "runtime_root": str(root.resolve()),
        "runtime_version": runtime_version,
        "dxgi_adapter_index": 0,
        "cuda_device_index": 0,
        "runtime_files": {
            "executable": {"path": "bin/video2dlssnr.exe"},
            "dlss_sr": {"path": "bin/nvngx_dlss.dll"},
            "dlss_nr": {"path": "bin/nvngx_dlssnr.dll"},
            "forwarder": {"path": "bin/nvngx.dll_dlssnr.dll"},
        },
        "audit_fingerprint": "test",
    }


def test_dimensions_and_mode_contract_reject_fallback_paths():
    assert dlss.target_dimensions(5, 7, 1.5) == (8, 11)
    assert dlss.target_dimensions(960, 544, 2.0) == (1920, 1088)
    with pytest.raises(ValueError, match="supported values"):
        dlss.target_dimensions(960, 544, 4.0)
    with pytest.raises(ValueError, match="NR only"):
        dlss.validate_processing_contract("nr_only", 2.0)
    with pytest.raises(ValueError, match="upscale"):
        dlss.validate_processing_contract("sr_only", 1.0)


def test_reference_profiles_are_exact_and_custom_is_strict():
    standard = dlss.resolve_quality_profile("standard")
    assert standard == {
        "nr_preset": 0,
        "nr_style": 0,
        "nr_intensity": 1.5,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": -1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 1.0,
        "nr_color": 1.0,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    }
    custom = dict(standard)
    custom["nr_intensity"] = 1.75
    assert dlss.resolve_quality_profile("custom", custom)["nr_intensity"] == 1.75
    with pytest.raises(ValueError, match="exactly"):
        dlss.resolve_quality_profile("custom", {"nr_intensity": 1.0})
    with pytest.raises(ValueError, match="nr_color"):
        invalid = dict(standard)
        invalid["nr_color"] = 1.1
        dlss.resolve_quality_profile("custom", invalid)
    with pytest.raises(ValueError, match="nr_detail"):
        invalid = dict(standard)
        invalid["nr_detail"] = 1.01
        dlss.resolve_quality_profile("custom", invalid)

    requested, effective = dlss.effective_quality_profile("sr_only", "standard")
    assert requested["nr_detail"] == 1.0
    assert effective["nr_detail"] == 0.0
    assert requested is not effective


def test_v12_scaled_output_is_rejected_instead_of_using_wrong_sr_mode(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="wrong DLSS SR quality mode"):
        dlss.build_raw_video_command(
            _runtime_handle(root, "1.2"),
            width=64,
            height=48,
            frame_count=1,
            mode="sr_nr",
            scale=2.0,
            motion_engine="auto",
        )


def test_bhwc_sdr_and_fps_inputs_fail_closed_before_process_start(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    for invalid in (
        torch.zeros((3, 4, 3)),
        torch.full((1, 3, 4, 3), float("nan")),
        torch.full((1, 3, 4, 3), 1.01),
    ):
        with pytest.raises(ValueError, match=r"BHWC|NaN|0\.\.1"):
            dlss.process_video_frame_batch(
                runtime,
                invalid,
                fps=24.0,
                mode="sr_nr",
                scale=2.0,
                process_factory=lambda *_args, **_kwargs: pytest.fail(
                    "process must not start"
                ),
            )
    with pytest.raises(ValueError, match="fps"):
        dlss.process_video_frame_batch(
            runtime,
            torch.zeros((1, 3, 4, 3)),
            fps=float("inf"),
            mode="sr_nr",
            scale=2.0,
            process_factory=lambda *_args, **_kwargs: pytest.fail(
                "process must not start"
            ),
        )


def test_common_arguments_are_literal_and_mode_specific(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    command = dlss.build_raw_video_command(
        runtime,
        width=64,
        height=48,
        frame_count=3,
        mode="sr_nr",
        scale=1.5,
        motion_engine="auto",
    )
    assert isinstance(command, list)
    assert command[:5] == [
        str((root / "bin" / "video2dlssnr.exe").resolve()),
        "--nr-video",
        "--dll-dir",
        str((root / "bin").resolve()),
        "--adapter",
    ]
    assert command[command.index("--nr-in") + 1] == "64x48"
    assert command[command.index("--nr-scale") + 1] == "1.5"
    assert "--frames" not in command
    assert command[command.index("--nr-sr-preset") + 1] == "default"
    assert command[command.index("--nr-style") + 1] == "0"
    assert command[command.index("--nr-intensity") + 1] == "1.5"
    assert command[command.index("--nr-detail") + 1] == "1"
    assert command[command.index("--nr-color") + 1] == "1"
    assert command[command.index("--nr-ui-correction") + 1] == "0"
    assert "--nr-hdr" not in command
    assert "--nr-motion-vis" not in command


def test_image_batch_preserves_order_and_resizes_extra_channels(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    images = torch.zeros((2, 3, 5, 5), dtype=torch.float32)
    images[0, ..., 0] = 1.0
    images[1, ..., 1] = 1.0
    images[0, ..., 3] = torch.linspace(0.0, 1.0, 15).reshape(3, 5)
    images[1, ..., 3] = 0.25
    images[0, ..., 4] = 0.75
    images[1, ..., 4] = 0.50
    calls = []

    def runner(command, *, cwd, timeout_seconds):
        from PIL import Image

        calls.append((command, cwd, timeout_seconds))
        source = Path(command[command.index("--in") + 1])
        output_dir = Path(command[command.index("--out") + 1])
        scale = float(command[command.index("--nr-scale") + 1])
        with Image.open(source) as opened:
            size = dlss.target_dimensions(opened.width, opened.height, scale)
            candidate = opened.resize(size, resample=Image.Resampling.NEAREST)
            candidate.save(output_dir / f"{source.name}_nr.png")
        return {"returncode": 0, "stdout": "ok", "stderr": "", "timed_out": False}

    candidate, source, report = dlss.process_image_batch(
        runtime,
        images,
        mode="sr_nr",
        scale=2.0,
        runner=runner,
    )
    assert source is images
    assert candidate.shape == (2, 6, 10, 5)
    assert candidate[0, ..., 0].mean() > 0.99
    assert candidate[1, ..., 1].mean() > 0.99
    assert torch.equal(
        candidate[..., 3:], dlss.resize_extra_channels(images[..., 3:], 6, 10)
    )
    assert report["frame_order_exact"] is True
    assert report["rgb_bridge"] == "rounded_uint8_rgba_png"
    assert len(calls) == 2


def test_sr_only_command_and_report_use_effective_zero_detail(tmp_path):
    from PIL import Image

    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)

    def runner(command, *, cwd, timeout_seconds):
        del cwd, timeout_seconds
        assert command[command.index("--nr-detail") + 1] == "0"
        source = Path(command[command.index("--in") + 1])
        output_dir = Path(command[command.index("--out") + 1])
        with Image.open(source) as opened:
            opened.resize((8, 8)).save(output_dir / f"{source.name}_nr.png")
        return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}

    _candidate, _source, report = dlss.process_image_batch(
        runtime,
        torch.zeros((1, 4, 4, 3)),
        mode="sr_only",
        scale=2.0,
        runner=runner,
    )
    assert report["requested_nr_parameters"]["nr_detail"] == 1.0
    assert report["nr_parameters"]["nr_detail"] == 0.0
    assert report["sr_only_execution"] == {
        "nr_evaluation_required_by_upstream": True,
        "output_composite_nr_detail": 0.0,
    }


def test_image_rejects_upstream_bilinear_fallback_even_on_zero_exit(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)

    def runner(command, *, cwd, timeout_seconds):
        del command, cwd, timeout_seconds
        return {
            "returncode": 0,
            "stdout": "",
            "stderr": "DLSS pass refused; finishing the rest bilinear",
            "timed_out": False,
        }

    with pytest.raises(RuntimeError, match="bilinear fallback"):
        dlss.process_image_batch(
            runtime,
            torch.zeros((1, 4, 6, 3)),
            mode="sr_nr",
            scale=2.0,
            runner=runner,
        )


def _write_raw_passthrough_helper(path: Path) -> None:
    path.write_text(
        """
import os
import sys

input_bytes = int(sys.argv[1])
output_bytes = int(sys.argv[2])
frame_count = int(sys.argv[3])
partial = int(sys.argv[4])
backend = sys.argv[5] if len(sys.argv) > 5 else "nvof"
if backend == "nvof":
    sys.stderr.write("optical flow: NVIDIA NVOFA (hardware)\\n")
else:
    sys.stderr.write("optical flow: Lucas-Kanade (GPU compute, NVOFA unavailable)\\n")
sys.stderr.flush()
for index in range(frame_count):
    data = bytearray()
    while len(data) < input_bytes:
        chunk = os.read(0, input_bytes - len(data))
        if not chunk:
            sys.exit(7)
        data.extend(chunk)
    output = (data * ((output_bytes + input_bytes - 1) // input_bytes))[:output_bytes]
    if partial and index == frame_count - 1:
        os.write(1, output[: output_bytes // 2])
        sys.exit(9)
    os.write(1, output)
sys.exit(0)
""".strip(),
        encoding="utf-8",
    )


def test_video_frames_use_one_process_keep_order_fps_and_audio_identity(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    frames = torch.rand((3, 4, 6, 4), generator=torch.Generator().manual_seed(260903))
    helper = tmp_path / "raw_helper.py"
    _write_raw_passthrough_helper(helper)
    processes = []

    def factory(_command, **kwargs):
        process = subprocess.Popen(
            [sys.executable, str(helper), str(4 * 6 * 4), str(12 * 18 * 4), "3", "0"],
            **kwargs,
        )
        processes.append(process)
        return process

    audio = {"waveform": torch.ones(1, 1, 10), "sample_rate": 32000}
    candidate, source, returned_audio, report = dlss.process_video_frame_batch(
        runtime,
        frames,
        fps=24.0,
        audio=audio,
        mode="sr_nr",
        scale=3.0,
        process_factory=factory,
    )
    assert source is frames
    assert returned_audio is audio
    assert candidate.shape == (3, 12, 18, 4)
    assert report["input_frame_count"] == report["output_frame_count"] == 3
    assert report["fps"] == 24.0
    assert report["single_persistent_process"] is True
    assert report["motion_engine_requested"] == "auto"
    assert report["motion_engine_resolved"] == "nvof"
    assert len(processes) == 1
    assert processes[0].poll() == 0
    expected_alpha = dlss.resize_extra_channels(frames[..., 3:], 12, 18)
    assert torch.equal(candidate[..., 3:], expected_alpha)


def test_video_frames_partial_pipe_fails_and_reaps_process(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    frames = torch.zeros((2, 4, 6, 3), dtype=torch.float32)
    helper = tmp_path / "raw_helper.py"
    _write_raw_passthrough_helper(helper)
    processes = []

    def factory(_command, **kwargs):
        process = subprocess.Popen(
            [sys.executable, str(helper), str(4 * 6 * 4), str(8 * 12 * 4), "2", "1"],
            **kwargs,
        )
        processes.append(process)
        return process

    with pytest.raises(RuntimeError, match="incomplete|exit code"):
        dlss.process_video_frame_batch(
            runtime,
            frames,
            fps=24.0,
            mode="sr_nr",
            scale=2.0,
            process_factory=factory,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_explicit_nvof_request_rejects_silent_lk_fallback(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    frames = torch.zeros((2, 4, 6, 3), dtype=torch.float32)
    helper = tmp_path / "raw_helper.py"
    _write_raw_passthrough_helper(helper)
    processes = []

    def factory(_command, **kwargs):
        process = subprocess.Popen(
            [
                sys.executable,
                str(helper),
                str(4 * 6 * 4),
                str(8 * 12 * 4),
                "2",
                "0",
                "lk",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    with pytest.raises(RuntimeError, match="actually used 'lk'"):
        dlss.process_video_frame_batch(
            runtime,
            frames,
            fps=24.0,
            mode="sr_nr",
            scale=2.0,
            motion_engine="nvof",
            process_factory=factory,
        )
    assert len(processes) == 1 and processes[0].poll() is not None


def test_default_image_runner_cancellation_reaps_process(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []

    def factory(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(dlss.subprocess, "Popen", factory)
    checks = 0

    def interrupted():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("test image cancellation")

    with pytest.raises(RuntimeError, match="test image cancellation"):
        dlss._default_media_runner(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path),
            timeout_seconds=60.0,
            interrupt_check=interrupted,
        )
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_video_frames_cancellation_reaps_process(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    frames = torch.zeros((8, 4, 6, 3), dtype=torch.float32)
    helper = tmp_path / "raw_helper.py"
    _write_raw_passthrough_helper(helper)
    processes = []

    def factory(_command, **kwargs):
        process = subprocess.Popen(
            [sys.executable, str(helper), str(4 * 6 * 4), str(8 * 12 * 4), "8", "0"],
            **kwargs,
        )
        processes.append(process)
        return process

    def interrupted():
        raise RuntimeError("test cancellation")

    with pytest.raises(RuntimeError, match="test cancellation"):
        dlss.process_video_frame_batch(
            runtime,
            frames,
            fps=24.0,
            mode="sr_nr",
            scale=2.0,
            process_factory=factory,
            interrupt_check=interrupted,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_cfr_validation_rejects_timestamp_jitter():
    assert (
        dlss.validate_cfr_pts(
            [0, 512, 1024], dlss.Fraction(1, 12288), dlss.Fraction(24, 1)
        )["cfr"]
        is True
    )
    with pytest.raises(ValueError, match="VFR"):
        dlss.validate_cfr_pts(
            [0, 512, 1025], dlss.Fraction(1, 12288), dlss.Fraction(24, 1)
        )


def _write_source_video(path: Path) -> None:
    import av
    import numpy as np

    with av.open(str(path), mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
        video.codec_context.max_b_frames = 0
        video.codec_context.thread_count = 1
        video.codec_context.color_primaries = 1
        video.codec_context.color_trc = 1
        video.codec_context.colorspace = 1
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "mono"
        for index in range(4):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[..., index % 3] = 40 + index * 40
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        phase = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False, dtype=np.float32)
        samples = (0.1 * np.sin(phase))[None, :]
        for offset in range(0, samples.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(
                samples[:, offset : offset + 1024], format="fltp", layout="mono"
            )
            frame.sample_rate = 32000
            frame.pts = offset
            frame.time_base = dlss.Fraction(1, 32000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def test_audio_identity_includes_packet_and_decoded_frame_timestamps(tmp_path):
    source_path = tmp_path / "source.mp4"
    _write_source_video(source_path)
    packets = dlss._audio_packet_digests(source_path)
    pcm = dlss._audio_pcm_digests(source_path)
    assert packets[0]["packet_timeline_sha256"]
    assert packets[0]["first_packet_pts_seconds"] is not None
    assert pcm[0]["decoded_timeline_sha256"]
    assert pcm[0]["first_frame_pts_seconds"] is not None

    changed_packets = [dict(item) for item in packets]
    changed_packets[0]["packet_timeline_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="packet timestamps"):
        dlss._validate_audio_identity(packets, changed_packets, pcm, pcm)

    changed_pcm = [dict(item) for item in pcm]
    changed_pcm[0]["decoded_timeline_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="decoded audio timestamps"):
        dlss._validate_audio_identity(packets, packets, pcm, changed_pcm)


class _FileVideo:
    def __init__(self, path: Path):
        self.path = path

    def get_stream_source(self):
        return str(self.path)

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_frame_count(self):
        return 4

    def get_dimensions(self):
        return 64, 48

    def get_bit_depth(self):
        return 8

    def get_frame_rate(self):
        return 24.0


def test_video_file_streams_and_packet_copies_audio_atomically(tmp_path):
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    for name in dlss.REQUIRED_RUNTIME_FILES.values():
        (root / "bin" / name).write_bytes(b"x")
    runtime = _runtime_handle(root)
    source_path = tmp_path / "source.mp4"
    _write_source_video(source_path)
    source_video = _FileVideo(source_path)
    helper = tmp_path / "raw_helper.py"
    _write_raw_passthrough_helper(helper)
    processes = []

    def factory(_command, **kwargs):
        process = subprocess.Popen(
            [
                sys.executable,
                str(helper),
                str(64 * 48 * 4),
                str(128 * 96 * 4),
                "4",
                "0",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    output = tmp_path / "published.mp4"
    published, returned_source, report = dlss.process_video_file(
        runtime,
        source_video,
        output_path=output,
        mode="sr_nr",
        scale=2.0,
        process_factory=factory,
    )
    assert published == output.resolve()
    assert returned_source is source_video
    assert output.is_file()
    assert report["video"]["input_frame_count"] == 4
    assert report["video"]["output_frame_count"] == 4
    assert report["video"]["full_image_batch_materialized"] is False
    assert report["audio"]["packet_payload_exact"] is True
    assert report["audio"]["packet_timeline_exact"] is True
    assert report["audio"]["decoded_pcm_exact"] is True
    assert report["audio"]["decoded_timeline_exact"] is True
    assert report["motion_engine_resolved"] == "nvof"
    assert report["atomic_publish"] is True
    assert len(processes) == 1 and processes[0].poll() == 0
    assert not list(tmp_path.glob("*.partial-*.mp4"))
    assert not list(tmp_path.glob("*.video-only-*.mp4"))
