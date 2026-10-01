"""Read-only identities and visible-forward clocks for original Bias/STG.

Never removes effects on the executing MODEL. Unknown user callbacks remain
executable and unverified; only exact owned closures can be projected away.
"""
import torch

from .. import detail_sampling_advanced as legacy
from ..vdn_attention_compat import _factory_closure


def describe(model):
    wrapper = model.model_options.get("model_function_wrapper")
    biases, stgs, known = [], [], True
    seen = set()
    while wrapper is not None:
        if id(wrapper) in seen or len(seen) > 32:
            known = False
            break
        seen.add(id(wrapper))
        state = _factory_closure(wrapper, legacy.setup_model_time_bias_sampling, "model_function_wrapper")
        fields = {"bias", "start_progress", "end_progress", "shift_video", "bias_domain", "previous_wrapper"}
        if state is None or set(state) != fields or vars(wrapper):
            known = False
            break
        biases.append({key: state[key] for key in sorted(fields - {"previous_wrapper"})})
        wrapper = state["previous_wrapper"]
    functions = model.model_options.get("sampler_post_cfg_function", [])
    if type(functions) is not list:
        known = False
        functions = []
    for function in functions:
        state = _factory_closure(function, legacy.apply_h3_spatiotemporal_guidance, "post_cfg_function")
        fields = {"blocks", "end_progress", "marker_key", "marker_value", "scale", "shift_video", "skip_block", "start_progress"}
        if (state is None or set(state) != fields or vars(function)
                or state["marker_key"] is not None or state["marker_value"] is not None
                or _factory_closure(state["skip_block"], legacy.apply_h3_spatiotemporal_guidance, "skip_block") != {}
                or vars(state["skip_block"])):
            known = False
            continue
        stgs.append({key: state[key] for key in sorted(fields - {"marker_key", "marker_value", "skip_block"})})
    return {"known": known, "biases": biases, "stg": stgs}


def project(model):
    contract = describe(model)
    view = model.clone()
    if contract["known"]:
        view.model_options.pop("model_function_wrapper", None)
        view.model_options.pop("sampler_post_cfg_function", None)
    return view, contract


def forward_plan(contract, sigmas, block_count, *, model_time_dtype=torch.float32):
    records = []
    for sigma in sigmas[:-1]:
        # Native Euler multiplies each scalar by s_in of the initial state dtype.
        raw = torch.tensor([float(sigma)], dtype=model_time_dtype)
        visible = raw
        for bias in contract["biases"]:
            visible = legacy.model_time_bias_sigma(visible, bias=bias["bias"],
                start_progress=bias["start_progress"], end_progress=bias["end_progress"],
                shift_video=bias["shift_video"], domain=bias["bias_domain"])
        records.append({"sigma": float(visible[0]), "attention_blocks": block_count, "weak": False})
        for stg in contract["stg"]:
            progress = float(legacy._flow_progress(raw, stg["shift_video"])[0])
            if stg["start_progress"] <= progress <= stg["end_progress"]:
                skipped = sum(0 <= index < block_count for index in set(stg["blocks"]))
                records.append({"sigma": float(visible[0]), "attention_blocks": block_count - skipped, "weak": True})
    return {"known": contract["known"], "forwards": records,
            "boundary": "Native CFG=1 main/owned STG weak plan; other guider or user forwards remain unverified."}
