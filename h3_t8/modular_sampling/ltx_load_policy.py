"""Explicit native LTX load profile, without Core or sampling-math patches.

Legacy default is an identity operation. The opt-in profile delegates Core's
force_offload preparation, keeping all existing wrappers and selected LoRAs.
Its observations are not sampler completion, quality or portable-cache proof.
"""
import json

import comfy.patcher_extension as extension

KEY = "t8_ltx_native_consistent_streaming_v1"
MODES = ("legacy_default", "consistent_streaming_exp")


class LTXLoadPolicyRuntime:
    def __init__(self, mode):
        self.mode = mode
        self.prepare_calls = 0

    def __call__(self, executor, model, noise_shape, conds, model_options=None,
                 force_full_load=False, force_offload=False):
        if force_full_load:
            raise ValueError("Consistent streaming cannot also request native full-load")
        self.prepare_calls += 1
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=False, force_offload=True)

    def report(self, model):
        wrappers = ([] if self.mode == MODES[0] else model.get_wrappers(
                    extension.WrappersMP.PREPARE_SAMPLING, KEY))
        installed = self in wrappers
        return json.dumps({"schema": "t8.ltx.native-load-policy.observations.v1", "mode": self.mode,
            "status": "legacy_profile_unchanged" if self.mode == MODES[0] else
                      "native_preparation_delegated_not_certified" if installed and self.prepare_calls else
                      "installed_not_executed" if installed else "unverified_wrapper_not_present",
            "native_preparation_delegations": self.prepare_calls, "wrapper_present": installed,
            "Core_force_offload_requested": self.mode == MODES[1],
            "sampler_completion_verified": False, "candidate_provenance_verified": False,
            "portable_cache_reuse_authorized": False, "quality_accepted": False,
            "warning": "Explicit streaming changes Core residency and can change LoRA rounding versus mixed placement. "
                       "Not a legacy bit-equivalence or universal reproducibility/speed guarantee. "
                       "Foreign wrappers remain delegated and may override the request."}, sort_keys=True)


def apply_ltx_load_policy(model, mode="legacy_default"):
    if mode not in MODES:
        raise ValueError("Unknown native LTX load policy")
    runtime = LTXLoadPolicyRuntime(mode)
    if mode == MODES[0]:
        return model, runtime, runtime.report(model)
    if not callable(getattr(model, "clone", None)) or not callable(getattr(model, "add_wrapper_with_key", None)):
        raise TypeError("Selected native load policy requires the real ModelPatcher wrapper API")
    cloned = model.clone()
    cloned.add_wrapper_with_key(extension.WrappersMP.PREPARE_SAMPLING, KEY, runtime)
    return cloned, runtime, runtime.report(cloned)


def audit_ltx_load_policy(model, candidate_latent, runtime):
    if not isinstance(runtime, LTXLoadPolicyRuntime):
        raise ValueError("Native load policy audit needs its actual runtime")
    return candidate_latent, runtime.report(model)
