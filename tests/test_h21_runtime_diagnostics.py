"""Read-only live ABI observations are not model/device execution gates."""
import json
import types

import pytest
import torch

from h3_audio_t8_pkg.h3_runtime_diagnostics import collect_h3_runtime_contracts


def test_actual_loaded_core_abis_and_precision_paths_are_separate():
    import comfy.ldm.minimax.model
    import comfy.ldm.minimax.vae
    import comfy.model_prefetch
    before = torch.cuda.is_initialized()
    facts = collect_h3_runtime_contracts()
    assert facts["callable_abis"]["final_layer"]["present"]
    assert "sample_sigmas" in facts["callable_abis"]["final_layer"]["signature"]
    assert facts["callable_abis"]["prefetch_cleanup"]["accepts_two_positional"] is True
    assert callable(comfy.model_prefetch.cleanup_prefetched_modules)
    assert facts["precision"]["dit_modulation_scaling"]["turing_numerical_qualification"] == "not_measured"
    assert facts["precision"]["vae_fast_accumulation"]["actual_branch_executed"] == "not_measured"
    assert facts["execution_counts"]["status"] == "not_measured_by_snapshot"
    assert torch.cuda.is_initialized() is before is False
    assert not any(facts["changes"].values())
    assert json.loads(json.dumps(facts))["schema"].endswith(".v1")


@pytest.mark.parametrize("count", [1, 2])
def test_old_or_new_cleanup_abi_observed_without_executing_body(count):
    def one(modules):
        raise AssertionError("Read-only audit executed cleanup")
    def two(model, modules):
        raise AssertionError("Read-only audit executed cleanup")
    function = one if count == 1 else two
    facts = collect_h3_runtime_contracts(modules={
        "comfy.model_prefetch": types.SimpleNamespace(cleanup_prefetched_modules=function)})
    cleanup = facts["callable_abis"]["prefetch_cleanup"]
    assert cleanup["accepts_one_positional"] is (count == 1)
    assert cleanup["accepts_two_positional"] is (count == 2)
    assert not cleanup["execution_verified"]


def test_unhashable_unknown_owner_and_missing_backends_remain_observations():
    class Owner:
        __hash__ = None
        def __call__(self, *args, **kwargs):
            raise AssertionError("Unknown callable was executed")
    owner = Owner()
    facts = collect_h3_runtime_contracts(modules={
        "comfy.model_prefetch": types.SimpleNamespace(cleanup_prefetched_modules=owner)})
    assert facts["callable_abis"]["prefetch_cleanup"]["present"]
    assert facts["callable_abis"]["prefetch_cleanup"]["accepts_two_positional"]
    assert not facts["callable_abis"]["kitchen_fp16_conv3d"]["present"]
    assert facts["precision"]["fp16_accumulation_opt_in"] is None
    assert facts["changes"]["blocks_sampling"] is False


def test_flags_are_observed_not_enabled_and_fake_kernel_is_never_called():
    calls = []
    def kernel(*args):
        calls.append(args)
        raise AssertionError("No kernel execution allowed")
    matmul = types.SimpleNamespace(allow_fp16_accumulation=True)
    modules = {
        "torch": types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=matmul))),
        "comfy.quant_ops": types.SimpleNamespace(ck=types.SimpleNamespace(fp16_conv3d=kernel)),
    }
    facts = collect_h3_runtime_contracts(modules=modules)
    assert facts["precision"]["fp16_accumulation_opt_in"] is True
    assert facts["precision"]["vae_fast_accumulation"]["kernel_symbol_present"] is True
    assert facts["precision"]["vae_fast_accumulation"]["actual_branch_executed"] == "not_measured"
    assert calls == [] and matmul.allow_fp16_accumulation is True


def test_unreadable_or_changed_disk_does_not_claim_memory_identity(tmp_path):
    source = tmp_path / "core.py"
    source.write_text("# source v1\n", encoding="utf8")
    modules = {"comfy.ops": types.SimpleNamespace(__file__=str(source))}
    before = collect_h3_runtime_contracts(modules=modules)["loaded_modules"]["comfy.ops"]
    source.write_text("# source v2\n", encoding="utf8")
    after = collect_h3_runtime_contracts(modules=modules)["loaded_modules"]["comfy.ops"]
    assert before["disk_sha256"] != after["disk_sha256"]
    assert after["memory_matches_disk"] == "not_proven_by_file_hash"
    source.unlink()
    missing = collect_h3_runtime_contracts(modules=modules)["loaded_modules"]["comfy.ops"]
    assert missing["disk_sha256"] is None and missing["loaded"]


def test_missing_distribution_metadata_never_imports_or_installs_backend(monkeypatch):
    import importlib.metadata
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", missing)
    facts = collect_h3_runtime_contracts(modules={})
    assert all(row["version"] is None for row in facts["installed_distributions"].values())
    assert not any(row["loaded"] for row in facts["loaded_modules"].values())
    assert not facts["changes"]["blocks_sampling"]


@pytest.mark.parametrize("module,expected", [
    (None, None), (types.SimpleNamespace(args=types.SimpleNamespace()), None),
    (types.SimpleNamespace(args=types.SimpleNamespace(fast=set())), []),
])
def test_unknown_startup_flags_are_not_reported_as_explicitly_disabled(module, expected):
    facts = collect_h3_runtime_contracts(modules={"comfy.cli_args": module})
    assert facts["precision"]["startup_fast_features"] == expected


def test_uninspectable_user_signature_is_reported_unknown_without_invocation():
    class Owner:
        @property
        def __signature__(self):
            raise RuntimeError("Opaque user signature cannot be inspected")
        def __call__(self, *args):
            raise AssertionError("Must not execute an unknown kernel or cleanup owner")
    facts = collect_h3_runtime_contracts(modules={
        "comfy.model_prefetch": types.SimpleNamespace(cleanup_prefetched_modules=Owner())})
    cleanup = facts["callable_abis"]["prefetch_cleanup"]
    assert cleanup["present"] and cleanup["signature"] is None
    assert cleanup["signature_error"] == "RuntimeError"
    assert cleanup["accepts_one_positional"] is None
    assert not facts["changes"]["blocks_sampling"]


def test_existing_environment_snapshot_includes_abi_facts_without_changing_model():
    from h3_audio_t8_pkg.environment_audit import collect_environment_snapshot
    from tests.test_fast_h3_v2_core_sampler import model as tiny_model
    model = tiny_model()
    options, patches = model.model_options, model.patches
    snapshot = collect_environment_snapshot(model=model)
    assert snapshot["runtime_contracts"]["schema"].endswith(".v1")
    assert model.model_options is options and model.patches is patches
    assert not torch.cuda.is_initialized()
