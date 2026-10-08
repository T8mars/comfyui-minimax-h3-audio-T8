"""Scoped RES history reconstruction, separate from universal CUDA approval.

Only identities produced by the actual MODEL inspectors enter this adapter.
It does not whitelist filenames, remove patches, run kernels or grant generic
Stage reuse. Resume still binds every run-contract field and literal tensor.
"""
import hashlib
from pathlib import Path

from .res_kitchen_identity import _digest


def _digest_matches(value):
    return (type(value) is dict and value.get("sha256") ==
            _digest({key: item for key, item in value.items() if key != "sha256"}))


def _portable(value):
    if type(value) is dict:
        return value.get("portable_cache_reuse") is not False and all(_portable(item) for item in value.values())
    if type(value) in (list, tuple):
        return all(_portable(item) for item in value)
    return True


def scoped_history_identity(identity):
    """Qualify recognized calculation content, not GPU numerics or aesthetics."""
    result = {"schema": "t8.minimax_h3.RES_scoped_history_content.v1",
              "content_reconstruction_verified": False,
              "required_program_producer_sha256": [],
              "generic_Stage_reuse_qualified": False, "CUDA_numerical_qualified": False,
              "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not identity.get("automatic_loaded_weights_verified"):
        return result
    if "RES_external_effects" in identity and not _portable(identity["RES_external_effects"]):
        return result
    for name in ("weights", "coordinates", "actual_class_forward", "latent_format", "placement"):
        if name not in identity or not _portable(identity[name]):
            return result
    sol = identity.get("original_sol_structure")
    projection = identity.get("original_sol_normalization")
    if (type(sol) is not dict or not _digest_matches(sol)
            or sol.get("schema") != "t8.minimax_h3.RES_original_Sol_structure.v1"
            or sol.get("python_composition_verified") is not True
            or sol.get("unwrapped_attention_delegates_authenticated") is not True
            or type(projection) is not dict
            or projection.get("readonly_normalization_verified") is not True
            or projection.get("executing_MODEL_mutated") is not False
            or projection.get("original_Sol_content_sha256") != sol["sha256"]):
        return result
    caches = sol.get("actual_cache_integrity", {})
    if (not all(caches.get(name) is True for name in ("lookup_metadata_verified",
            "derived_permutation_content_verified", "selected_kernel_parameter_cache_verified"))
            or caches.get("actual_global_caches_changed") is not False):
        return result
    kitchen = sol.get("kitchen_execution_chain", {})
    if (not _digest_matches(kitchen) or kitchen.get("dispatch_content_verified") is not True):
        return result
    raw = sol.get("raw_KJ_calculation_functions", {})
    paths = sol.get("raw_KJ_calculation_paths", {})
    if (not raw or not paths or set(paths) != set(sol.get("composed_attention_paths", ()))
            or set(paths.values()) != set(raw)):
        return result
    producers = set()
    for key, calculation in raw.items():
        if (key != _digest(calculation) or not _digest_matches(calculation)
                or calculation.get("schema") != "t8.minimax_h3.RES_raw_KJ_Sage_content.v1"
                or calculation.get("calculation_content_verified") is not True
                or calculation.get("selected_branch") != "sm89"
                or len(calculation.get("native_binaries", {})) != 2
                or len(calculation.get("native_operations", {})) != 3):
            return result
        rope = calculation.get("RoPE_execution_chain", {})
        if not _digest_matches(rope) or rope.get("dispatch_content_verified") is not True:
            return result
        quantizers = calculation.get("quantizers", {})
        if set(quantizers) != {"_quant_query_per_thread_int8_i64_kernel", "_quant_key_per_thread_int8_i64_kernel"}:
            return result
        for value in quantizers.values():
            if (value.get("kind") != "source_backed_native_Triton_JIT"
                    or value.get("native_parameter_cache_owners", {}).get("native_parameter_and_default_factory_verified") is not True
                    or value.get("derived_cache_inspector", {}).get("inspection_writes_cache_or_initializes_driver") is not False):
                return result
            producers.add(_digest(value))
    result.update(content_reconstruction_verified=True,
                  required_program_producer_sha256=sorted(producers),
                  original_Sol_content_sha256=sol["sha256"])
    return result


def history_recovery_allowed(identity, position, snapshot, asset_check):
    """Called after bound-file validation, before loading/using solver history."""
    if identity.get("portable_cache_reuse") is True:
        return True
    scoped = identity.get("scoped_RES_history_content", {})
    if (not scoped.get("content_reconstruction_verified")
            or position.get("portable_condition_layout") is not True
            or not asset_check or asset_check.get("file_contents_verified") is not True
            or type(snapshot) is not dict):
        return False
    required = set(scoped["required_program_producer_sha256"])
    observed = set()
    for entry in snapshot.get("kernels", {}).values():
        if not entry.get("programs"):
            continue
        if all(program.get("compiler_key_independently_derived") is True
               and program.get("native_files_and_memory_content_consistent") is True
               and program.get("actual_launcher", {}).get("native_launcher_extension_content_bound") is True
               and program.get("actual_launcher", {}).get("loaded_GPU_handle_types_observed") is True
               for program in entry["programs"]):
            observed.add(entry["producer_sha256"])
    return bool(required) and required <= observed
