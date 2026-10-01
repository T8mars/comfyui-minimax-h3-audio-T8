"""Live-only native sampler proof for the separate learned H3→LTX refiner.

The older source-bound Bind/Audit contract remains unchanged. This opt-in path
observes one actual Core sampler return; it does not authorize persisted reuse.
"""

from dataclasses import dataclass
import hashlib
import json

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .ltx_latent_stage import SCHEMA, _tensor, audit_ltx_latent_stage
from .results import _input_identity, canonical


@dataclass(frozen=True)
class LTXLearnedSampleProof:
    boundary_sha256: str
    source_identity: dict
    candidate_identity: dict
    candidate_object_id: int
    callbacks: tuple[int, ...]
    guider_sample_calls: int


def _verified_boundary(boundary):
    if not isinstance(boundary, dict) or boundary.get("schema") != SCHEMA:
        raise ValueError("Learned LTX candidate lacks a source-bound stage")
    unsigned = dict(boundary)
    digest = unsigned.pop("boundary_sha256", None)
    if digest != hashlib.sha256(canonical(unsigned).encode()).hexdigest():
        raise ValueError("Learned LTX boundary SHA256 differs")
    return unsigned, digest


def sample_ltx_latent_stage(noise, guider, sampler, sigmas, latent_image, stage_boundary):
    """Delegate once to Core SamplerCustomAdvanced and record three callbacks."""
    boundary, digest = _verified_boundary(stage_boundary)
    if (id(noise) != boundary["noise_object_id"]
            or id(guider) != boundary["guider_object_id"]
            or id(sampler) != boundary["sampler_object_id"]
            or id(guider.model_patcher) != boundary["model_object_id"]
            or _input_identity(latent_image) != boundary["ltx_video"]
            or [float(value) for value in sigmas.detach().cpu().tolist()] != boundary["sigmas"]):
        raise ValueError("Learned LTX sampler inputs differ from bound external controls")
    source_identity = _input_identity(latent_image)
    sigma_identity = _input_identity(sigmas)
    observed = {"callbacks": [], "guider_sample_calls": 0}

    class ObservedGuider:
        model_patcher = guider.model_patcher

        def sample(self, actual_noise, *args, callback=None, **kwargs):
            observed["guider_sample_calls"] += 1

            def record(*items, **options):
                step = items[0] if items else options.get("step")
                observed["callbacks"].append(int(step))
                if callback is not None:
                    return callback(*items, **options)

            return guider.sample(actual_noise, *args, callback=record, **kwargs)

    output, denoised = SamplerCustomAdvanced.execute(
        noise, ObservedGuider(), sampler, sigmas, latent_image).result
    if (_input_identity(latent_image) != source_identity
            or _input_identity(sigmas) != sigma_identity
            or observed["guider_sample_calls"] != 1
            or observed["callbacks"] != [0, 1, 2]):
        raise ValueError("Learned LTX three-step sampler completion was not observed")
    if (not isinstance(output, dict) or "samples" not in output
            or not isinstance(denoised, dict) or "samples" not in denoised):
        raise ValueError("Learned LTX Core sampler returned no LATENT pair")
    for label, value in (("output", output), ("denoised", denoised)):
        sample = _tensor(value["samples"], f"Learned LTX {label}", 5)
        if list(sample.shape) != boundary["video_shape"]:
            raise ValueError(f"Learned LTX {label} shape differs from bound source")
    proof = LTXLearnedSampleProof(digest, source_identity, _input_identity(output),
                                  id(output), tuple(observed["callbacks"]),
                                  observed["guider_sample_calls"])
    report = {"schema": SCHEMA, "status": "native_ltx_sampler_returned",
              "observed_step_callbacks": list(proof.callbacks),
              "sampler_execution_proven": True, "portable_cache_reuse_authorized": False,
              "boundary": "One Core sampler return and three step callbacks; not a proof of "
                          "internal network forward counts, quality or cross-process cache safety."}
    return output, denoised, proof, canonical(report)


def audit_completed_ltx_latent_stage(boundary, h3_latent, original_h3_av,
                                     ltx_video_latent, adapter_report_json, model,
                                     noise, guider, sampler, sigmas, setup_report_json,
                                     candidate_latent, sample_proof):
    """Keep the old source audit, then require the matching live sampler proof."""
    audited = audit_ltx_latent_stage(boundary, h3_latent, original_h3_av,
                                     ltx_video_latent, adapter_report_json, model,
                                     noise, guider, sampler, sigmas, setup_report_json,
                                     candidate_latent)
    _unsigned, digest = _verified_boundary(boundary)
    if (type(sample_proof) is not LTXLearnedSampleProof
            or sample_proof.boundary_sha256 != digest
            or sample_proof.source_identity != boundary["ltx_video"]
            or sample_proof.candidate_object_id != id(candidate_latent)
            or sample_proof.candidate_identity != _input_identity(candidate_latent)
            or sample_proof.callbacks != (0, 1, 2)
            or sample_proof.guider_sample_calls != 1):
        raise ValueError("Learned LTX candidate lacks matching actual sampler proof")
    report = json.loads(audited[3])
    report.update(status="source_bound_sampler_completed", sampler_execution_proven=True,
                  portable_cache_reuse_authorized=False,
                  boundary="Actual Core sampler return and three step callbacks are live-only; "
                           "no cross-process cache, model quality or A/V acceptance is inferred.")
    return audited[0], audited[1], audited[2], canonical(report)
