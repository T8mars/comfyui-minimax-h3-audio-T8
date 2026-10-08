"""Pre-sampling RES descriptor for existing external Stage EAV and Relay.

No sampler is executed or replaced. Only this inert owner is removed from an
inspection clone. The complete original trajectory remains the clock during
remaining-step recovery; observed coverage still uses the actual suffix.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .long_video_dual_identity import _v2_sampling_identity
from .modular_sampling.contracts import StageContext
from .modular_sampling.results import _input_identity, canonical
from .patch_stack_policy import UnverifiedModelStack
from .sampling import model_uses_raw_audio_velocity, nested_av_parts

KEY = "t8_RES_external_effect_context_v1"


@dataclass(frozen=True)
class RESEffectOwner:
    context: StageContext
    sampling_object: object
    sampling_json: str
    sampler: object
    source: dict
    selection_json: str
    audio_velocity_is_raw: bool

    @property
    def runtime(self):
        return None

    @property
    def profile(self):
        return self.context.profile


def capture_owner(model):
    from .res_stage_exp import _context, _selection
    owner = model.get_attachment(KEY)
    if type(owner) is not RESEffectOwner or set(vars(owner)) != set(RESEffectOwner.__dataclass_fields__):
        raise ValueError("RES external effects require their actual pre-sampling binding")
    if model.object_patches.get("model_sampling") is not owner.sampling_object:
        raise ValueError("RES effect MODEL sampling object was replaced")
    if canonical(_v2_sampling_identity(owner.sampling_object)) != owner.sampling_json:
        raise ValueError("RES effect AV coordinates changed")
    sigmas = torch.tensor(owner.context.trajectory_sigmas[owner.context.start:], dtype=torch.float32)
    state, selected = _selection(owner.sampler, sigmas, owner.source)
    if canonical(selected) != owner.selection_json or _context(state, model, owner.source) != owner.context:
        raise ValueError("RES effect source, sampler or complete trajectory changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"), options.get("minimax_h3_sigma_shift_audio")) != (
            owner.context.video_shift, owner.context.audio_shift):
        raise ValueError("RES effect MODEL AV shifts differ from its context")
    return owner


def bind_effects(model, sampler, sigmas, av_latent):
    from .res_stage_exp import _context, _selection
    if model.get_attachment(KEY) is not None or model.get_attachment("t8_modular_stage_eav_v1") is not None:
        raise ValueError("Bind the RES setup before applying its external Stage EAV")
    from .res_memory_effects import prepare
    model = prepare(model)
    state, selected = _selection(sampler, sigmas, av_latent)
    context = _context(state, model, av_latent)
    branch = model.clone()
    sampling = model.get_model_object("model_sampling")
    branch.set_attachments(KEY, RESEffectOwner(context, sampling, canonical(_v2_sampling_identity(sampling)),
        sampler, dict(av_latent), canonical(selected), model_uses_raw_audio_velocity(model)))
    capture_owner(branch)
    report = dict(schema="t8.minimax_h3.RES_external_effect_context.v1", stage_context=context.to_dict(),
        sampled=False, additional_NFE=0, sampler_and_sigmas_unchanged=True,
        cache_reuse_authorized=False, human_accepted=False,
        wiring="Paired external Relay MODEL -> RES History -> RES Effect Bind -> external Stage EAV -> BasicGuider. "
               "Use the paired Relay CONDITIONING, original NOISE and original AV input.")
    return branch, sampler, sigmas, context, json.dumps(report, ensure_ascii=False, indent=2)


def validate_stage(model, sigmas, av_latent, context):
    owner = capture_owner(model)
    if type(context) is not StageContext or context != owner.context:
        raise ValueError("RES external EAV context differs from its actual MODEL")
    expected = torch.tensor(context.trajectory_sigmas[context.start:context.end + 1]).to(sigmas)
    if sigmas.shape != expected.shape or not torch.equal(sigmas, expected):
        raise ValueError("RES external EAV sigmas differ from the actual remaining trajectory")
    if _input_identity(av_latent) != _input_identity(owner.source):
        raise ValueError("RES external EAV source differs from its original AV input")
    video, audio = nested_av_parts(av_latent)
    return video, audio, owner


def project_owner(model):
    """Authenticate the inert owner; never delete an execution effect."""
    if model.get_attachment(KEY) is None:
        return model, None
    try:
        owner = capture_owner(model)
    except ValueError as error:
        raise UnverifiedModelStack(str(error)) from error
    branch = model.clone()
    branch.remove_attachments(KEY)
    context = owner.context.to_dict()
    # Resume has a different observed suffix, not a new numerical effect or
    # input. Bind full AV clock/config/source so the original checkpoint can be
    # resumed. validate_stage still checks the exact actual suffix at execution.
    context.update(start=0, stage="res_complete")
    return branch, dict(stage_context=context, source=_input_identity(owner.source),
        audio_velocity_is_raw=owner.audio_velocity_is_raw,
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def project_identity(model):
    """Reuse authenticated existing effect projection, scoped only to RES."""
    from .modular_sampling.effect_identity import project_stage_effects
    branch, effects = project_stage_effects(model)
    from .res_eav_memory import project
    branch, standalone = project(branch)
    if standalone is not None:
        effects = {**(effects or {}), "standalone_EAV_memory": standalone}
    branch, descriptor = project_owner(branch)
    if descriptor is None:
        raise UnverifiedModelStack("RES effects lost their actual pre-sampling owner")
    if effects is not None:
        effects = dict(effects)
        if "stage_context" in effects:
            if effects["stage_context"] != capture_owner(model).context.to_dict():
                raise UnverifiedModelStack("RES EAV and MODEL descriptors differ")
            effects["stage_context"] = descriptor["stage_context"]
    return branch, dict(binding=descriptor, effects=effects)
