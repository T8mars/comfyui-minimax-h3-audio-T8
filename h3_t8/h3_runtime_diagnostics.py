"""Read already-loaded ABI/precision facts; never select or execute a backend."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import sys


SCHEMA = "t8.h3.runtime_contract_observation.v1"


def _resolve(modules, module, path):
    value = modules.get(module)
    for name in path.split("."):
        value = getattr(value, name, None)
        if value is None:
            break
    return value


def _callable_fact(value):
    fact = {"present": callable(value), "owner": None, "signature": None}
    if not callable(value):
        return fact
    target = getattr(value, "__func__", value)
    owner = target if inspect.isroutine(target) or inspect.isclass(target) else type(target)
    fact["owner"] = f"{owner.__module__}.{getattr(owner, '__qualname__', type(owner).__qualname__)}"
    try:
        fact["signature"] = str(inspect.signature(value))
    except Exception as error:
        # This is introspection, not execution. A user-defined __signature__ may
        # be opaque; a diagnostic must not turn that into a MODEL execution ban.
        fact["signature_error"] = type(error).__name__
    # A callable owner/name is not an authenticated executable identity.
    fact["execution_verified"] = False
    return fact


def _module_fact(module):
    fact = {"loaded": module is not None, "path": None, "disk_sha256": None,
            "memory_matches_disk": "not_proven_by_file_hash"}
    path = getattr(module, "__file__", None)
    if path is None:
        return fact
    try:
        path = Path(path).resolve()
        fact["path"] = str(path)
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            fact["disk_sha256"] = digest.hexdigest()
    except (OSError, TypeError, ValueError) as error:
        fact["inspection_error"] = type(error).__name__
    return fact


def _binds(value, count):
    if not callable(value):
        return None
    try:
        signature = inspect.signature(value)
    except Exception:
        return None
    try:
        signature.bind(*[object() for _ in range(count)])
    except TypeError:
        return False
    return True


def collect_h3_runtime_contracts(*, modules=None):
    """Missing modules remain unknown; do not import a kernel just to inspect it.

    ``modules`` is injectable for old/new ABI counterexamples. No inspected
    callable, CUDA device query, model loader or dtype conversion is invoked.
    """
    modules = sys.modules if modules is None else modules
    names = ("comfy.ldm.minimax.model", "comfy.ldm.minimax.vae",
             "comfy.ops", "comfy.model_prefetch", "comfy.quant_ops")
    targets = {
        "dit_block": (names[0], "DiTBlock.forward"),
        "dit_modulation": (names[0], "_mod_scale_shift"),
        "final_layer": (names[0], "FinalLayer.forward"),
        "video_vae_decode": (names[1], "MiniMaxH3VideoVAE.decode"),
        "video_vae_decode_tiled": (names[1], "MiniMaxH3VideoVAE.decode_tiled"),
        "video_vae_fast_accumulation_gate": (names[1], "_fp16_accum_conv"),
        "linear_fast_accumulation_gate": (names[2], "_fp16_linear_wanted"),
        "linear_fused_input_activation": (names[2], "linear_input_act"),
        "prefetch_cleanup": (names[3], "cleanup_prefetched_modules"),
        "prefetch_pop": (names[3], "prefetch_queue_pop"),
        "kitchen_fp16_linear": (names[4], "ck.fp16_linear"),
        "kitchen_fp16_conv3d": (names[4], "ck.fp16_conv3d"),
    }
    callables = {key: _callable_fact(_resolve(modules, *path)) for key, path in targets.items()}
    cleanup = _resolve(modules, names[3], "cleanup_prefetched_modules")
    callables["prefetch_cleanup"].update(
        accepts_one_positional=_binds(cleanup, 1), accepts_two_positional=_binds(cleanup, 2))

    distributions = {}
    for distribution in ("torch", "comfy-kitchen", "sageattention", "triton"):
        try:
            distributions[distribution] = {"version": importlib.metadata.version(distribution)}
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = {"version": None, "status": "no_distribution_metadata"}
        except (OSError, ValueError) as error:
            distributions[distribution] = {"version": None, "inspection_error": type(error).__name__}

    torch = modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    initialized = getattr(cuda, "is_initialized", None)
    initialized = bool(initialized()) if callable(initialized) else None
    backends = getattr(torch, "backends", None)
    matmul = getattr(getattr(backends, "cuda", None), "matmul", None)
    opt_in = getattr(matmul, "allow_fp16_accumulation", None)
    opt_in = opt_in if type(opt_in) is bool else None
    cli_args = getattr(modules.get("comfy.cli_args"), "args", None)
    fast = getattr(cli_args, "fast", None)
    fast = sorted(str(getattr(value, "value", value)) for value in fast) if isinstance(fast, (tuple, list, set, frozenset)) else None
    return {
        "schema": SCHEMA,
        "loaded_modules": {name: _module_fact(modules.get(name)) for name in names},
        "callable_abis": callables,
        "installed_distributions": distributions,
        "precision": {
            "fp16_accumulation_opt_in": opt_in,
            "startup_fast_features": fast,
            "cuda_already_initialized": initialized,
            "dit_modulation_scaling": {"observation": callables["dit_modulation"],
                                       "turing_numerical_qualification": "not_measured"},
            "vae_fast_accumulation": {
                "gate_present": callables["video_vae_fast_accumulation_gate"]["present"],
                "kernel_symbol_present": callables["kitchen_fp16_conv3d"]["present"],
                "actual_branch_executed": "not_measured"},
        },
        "execution_counts": {"status": "not_measured_by_snapshot",
                             "source": "Use the sampling-dependent Spark/Veda/cache audit, not installed symbols."},
        "changes": {"models": False, "dtype": False, "backends": False,
                    "environment": False, "kernel_executed": False, "blocks_sampling": False},
        "boundaries": [
            "Loaded source, callable ABI and distribution versions are observations, not kernel compatibility or quality proof.",
            "DiT modulation/scaling and VAE fp16 fast accumulation are different numerical paths.",
            "An fp16 opt-in or callable symbol does not prove that a CUDA/fp16 branch was consumed.",
            "Missing/unknown devices or providers do not restrict the user's selected MODEL, LoRAs or callables.",
        ],
    }
