"""RES-only identity of Relay's unchanged original-Sol selected delegate.

This recognizes the concrete source-bound delegate; it does not change the
Relay/Sol execution policy, invoke its attention, or certify arbitrary user
delegates. Unknown adapters remain runnable and nonportable.
"""
from collections import Counter
import hashlib
from pathlib import Path

from . import relay_sol_backend as backends
from .patch_stack_policy import UnverifiedModelStack
from .relay_kj_backend import _codes, _live_code_matches
from .res_sol_identity import inspect_original_sol


def project_original_sol_delegate(model, backend):
    from .res_memory_effects import inspect_owner, RESMemoryDelegate
    from .relay_kj_memory import HeadGroupedBackend
    memory = None
    if type(backend) is HeadGroupedBackend and type(backend.delegate) is RESMemoryDelegate:
        _, memory = inspect_owner(model, backend)
        backend = backend.delegate.original
    if type(backend) is not backends.UserSelectedBackend or set(vars(backend)) != {"override", "counters"}:
        raise UnverifiedModelStack("RES Relay backend lacks its source-bound original Sol adapter")
    if type(backend.counters) is not Counter:
        raise UnverifiedModelStack("RES Relay backend counters have another execution owner")
    source = Path(backends.__file__).read_bytes()
    compiled = tuple(_codes(compile(source, backends.__file__, "exec", dont_inherit=True)))
    for name in ("__init__", "attention", "report"):
        method = vars(backends.UserSelectedBackend).get(name)
        if not _live_code_matches(method, compiled, backends):
            raise UnverifiedModelStack("RES Relay selected delegate executable changed")
    view = model.clone()
    view.model_options["transformer_options"]["optimized_attention_override"] = backend.override
    inspected = inspect_original_sol(view)
    if inspected is None or not inspected[0]["python_composition_verified"]:
        raise UnverifiedModelStack("RES Relay selected delegate is not the authenticated original Sol")
    # Full raw computation/weights and compiled files are separately inspected
    # by loaded_model_identity. Here only factor the authenticated Relay layer.
    contract = dict(kind="source_bound_original_Sol_Relay_delegate",
        original_Sol_content_sha256=inspected[0]["sha256"],
        delegate_source_sha256=hashlib.sha256(source).hexdigest(),
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        attention_executed_by_inspection=False, executing_MODEL_mutated=False,
        CUDA_numerical_qualified=False)
    if memory is not None:
        contract["RES_memory_effects"] = memory
    return view, contract
