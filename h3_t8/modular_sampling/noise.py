"""Explicit FreeNoise on a NOISE stream, never an invisible second sampling pass."""
from dataclasses import dataclass
import json

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, Noise_EmptyNoise

from .. import freenoise_advanced as legacy
from ..patch_stack_policy import _execution_selection

MODES = ("disabled", "paper_permutation", "variance_preserving_blend", "from_model_plan")


class _PlanCarrier:
    # Legacy builder owns validation/config serialization. This is inert data,
    # not a loaded MODEL or a fake execution owner.
    model_options = {}

    def clone(self):
        return _PlanCarrier()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class StageNoise:
    source: object
    config_json: str
    segment_index: int

    @property
    def seed(self):
        return self.source.seed

    def generate_noise(self, input_latent):
        original = self.source.generate_noise(input_latent)
        return legacy.reschedule_h3_noise(original, config=json.loads(self.config_json),
                                          segment_index=self.segment_index)[0]


def build_noise(noise, mode="disabled", base_seed=123456789, reuse_ratio=.65, segment_index=0, model=None):
    if mode not in MODES:
        raise ValueError("Unknown stage noise mode")
    if type(segment_index) is not int or segment_index < 0:
        raise ValueError("Stage noise segment_index must be a nonnegative integer")
    if not callable(getattr(noise, "generate_noise", None)) or not hasattr(noise, "seed"):
        raise ValueError("Connect a Core-compatible NOISE provider")
    config = None
    if mode == "from_model_plan":
        if model is None:
            raise ValueError("from_model_plan requires the MODEL carrying the old FreeNoise plan")
        config = legacy.free_noise_config(model)
        if config is not None:
            # Validate executable fields through the original public builder.
            legacy.build_free_noise_model(_PlanCarrier(), mode=config["mode"],
                base_seed=config["base_seed"], reuse_ratio=config["reuse_ratio"])
    elif mode != "disabled":
        carrier, _ = legacy.build_free_noise_model(_PlanCarrier(), mode=mode,
                                                  base_seed=base_seed, reuse_ratio=reuse_ratio)
        config = legacy.free_noise_config(carrier)
    output = noise if config is None else StageNoise(noise, _canonical(config), segment_index)
    return output, json.dumps({"schema": "t8.modular-sampling.stage-noise.v1",
        "status": "bypass_identity" if config is None else "configured_not_yet_generated",
        "config": config, "segment_index": segment_index, "noise_generated": False,
        "audio": "delegate audio noise remains the same object; only video is rescheduled",
        "boundary": "Input NOISE owns seed/batch_index. This wraps one generate_noise call, not sampling; "
                    "disabled/absent legacy plan returns the exact input provider."}, ensure_ascii=False, indent=2)


def operator_identity(noise):
    """Portable only for exact known provider state; opaque sources still run."""
    from .core_sampler_identity import native_random_noise_class
    native_random = native_random_noise_class()
    if type(noise) in (Noise_RandomNoise, Noise_EmptyNoise, native_random) and set(vars(noise)) == {"seed"}:
        return {"portable": True, "provider": type(noise).__name__, "seed": int(noise.seed)}
    if type(noise) is StageNoise and set(vars(noise)) == {"source", "config_json", "segment_index"}:
        source = operator_identity(noise.source)
        config = json.loads(noise.config_json)
        # config_json and source are rechecked after sampling as well.
        return {"portable": source["portable"], "provider": "t8_stage_freenoise_v1", "source": source,
                "config": config, "segment_index": noise.segment_index}
    return {"portable": False, "provider": "unverified_user_noise", "selection": _execution_selection(noise)}
