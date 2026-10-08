"""Actual source-backed identities; admission fixtures are NOT CUDA evidence."""
from copy import deepcopy

import torch

from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from h3_audio_t8_pkg.res_recovery_identity import scoped_history_identity, history_recovery_allowed
from test_res_sol_normalization import selected
from test_res_original_sol_identity import snapshot, assert_untouched
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- source/dispatch fixture


def identity(module, tmp_path):
    branch = selected(module, tmp_path)
    before = snapshot(branch)
    observed = {}
    result = loaded_model_identity(branch, observed_artifacts=observed)
    assert_untouched(branch, before)
    return branch, result, observed


def test_real_source_bound_MODEL_qualifies_only_scoped_history_not_generic_CUDA_or_Stage(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    _, content, observed = identity(module, tmp_path)
    scoped = content["scoped_RES_history_content"]
    assert scoped["content_reconstruction_verified"]
    assert not content["portable_cache_reuse"]
    assert not scoped["CUDA_numerical_qualified"] and not scoped["generic_Stage_reuse_qualified"]
    assert set(scoped["required_program_producer_sha256"]) == {entry["producer_sha256"] for entry in observed.values()}
    assert loaded_model_identity(selected(module, tmp_path)) == content
    assert not torch.cuda.is_initialized()


def test_actual_foreign_hook_or_weight_owner_never_receives_scoped_admission(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    branch, content, _ = identity(module, tmp_path)
    assert content["scoped_RES_history_content"]["content_reconstruction_verified"]
    calls = []
    handle = branch.model.diffusion_model.blocks[0].register_forward_pre_hook(lambda *_: calls.append(True))
    before = snapshot(branch)
    assert not loaded_model_identity(branch)["scoped_RES_history_content"]["content_reconstruction_verified"]
    assert_untouched(branch, before)
    handle.remove()
    branch.set_attachments("unknown_recovery_owner", object())
    assert not loaded_model_identity(branch)["scoped_RES_history_content"]["content_reconstruction_verified"]
    assert not calls and not torch.cuda.is_initialized()


def test_incomplete_or_corrupt_calculation_declarations_cannot_replace_actual_inspection(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    _, content, _ = identity(module, tmp_path)
    for name in ("coordinates", "latent_format", "actual_class_forward"):
        changed = deepcopy(content)
        changed[name] = {"portable_cache_reuse": False}
        assert not scoped_history_identity(changed)["content_reconstruction_verified"]
    changed = deepcopy(content)
    changed["original_sol_structure"]["kitchen_execution_chain"]["dispatch_content_verified"] = False
    assert not scoped_history_identity(changed)["content_reconstruction_verified"]
    assert not torch.cuda.is_initialized()


def test_admission_requires_actual_positions_both_warm_producers_and_prior_file_validation(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    _, content, observed = identity(module, tmp_path)
    # Literal gate fixture only; it is not a saved GPU program or qualification.
    program = dict(compiler_key_independently_derived=True, native_files_and_memory_content_consistent=True,
        actual_launcher=dict(native_launcher_extension_content_bound=True, loaded_GPU_handle_types_observed=True))
    snapshot_data = {"kernels": {name: {"producer_sha256": entry["producer_sha256"], "programs": [deepcopy(program)]}
                                 for name, entry in observed.items()}}
    position, checked = {"portable_condition_layout": True}, {"file_contents_verified": True}
    assert history_recovery_allowed(content, position, snapshot_data, checked)
    assert not history_recovery_allowed(content, {}, snapshot_data, checked)
    assert not history_recovery_allowed(content, position, snapshot_data, None)
    assert not history_recovery_allowed(content, position, None, checked)
    changed = deepcopy(snapshot_data)
    changed["kernels"].pop(next(iter(changed["kernels"])))
    assert not history_recovery_allowed(content, position, changed, checked)
    for field in ("compiler_key_independently_derived", "native_files_and_memory_content_consistent"):
        changed = deepcopy(snapshot_data)
        next(iter(changed["kernels"].values()))["programs"][0][field] = False
        assert not history_recovery_allowed(content, position, changed, checked)
    changed = deepcopy(snapshot_data)
    next(iter(changed["kernels"].values()))["programs"][0]["actual_launcher"]["loaded_GPU_handle_types_observed"] = False
    assert not history_recovery_allowed(content, position, changed, checked)
    assert not torch.cuda.is_initialized()
