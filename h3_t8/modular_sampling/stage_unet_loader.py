"""Load a separate stage model only after a verified preceding stage completed."""

import json

from .results import StageResult


def _load_unet(unet_name, weight_dtype):
    from nodes import UNETLoader

    return UNETLoader().load_unet(unet_name, weight_dtype)[0]


def load_unet_after_stage(completed_stage, unet_name, weight_dtype):
    if type(completed_stage) is not StageResult:
        raise ValueError("Stage model load requires a sampler-produced StageResult")
    receipt = completed_stage.verify()
    if receipt["verified_recipe_completion"] is not True:
        raise ValueError("Stage model load requires a completed preceding stage")
    # Deliberately call Core's loader after the dependency, not MODEL.clone().
    # Core may cache two identical UNETLoader nodes as the same patcher/network;
    # cloning that patcher cannot isolate two live model_sampling owners.
    model = _load_unet(unet_name, weight_dtype)
    report = {"schema": "t8.modular-sampling.stage-unet-load.v1",
              "status": "independent_model_loaded_after_completed_stage",
              "preceding_stage": receipt["request"]["stage_context"]["stage"],
              "preceding_receipt_sha256": receipt["receipt_sha256"],
              "unet_name": unet_name, "weight_dtype": weight_dtype,
              "boundary": "A new Core model load, not a shared patcher clone or hidden sampler. "
                          "It may require another CPU/RAM/VRAM model allocation until the earlier stage is released."}
    return model, json.dumps(report, ensure_ascii=False, sort_keys=True)
