"""Per-MODEL residual gate: retain the original control tower and all owners."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib

import torch

from .control_regions import canonical
from .patch_stack_policy import warn_patch_stack


REGION_ATTACHMENT = "t8_h3_control_region_runtime_v1"


def target_span(args):
    options = args.get("transformer_options") or {}
    layout = args.get("layout")
    if layout is None:
        layout = options.get("minimax_h3_layout")
    if layout is None or not hasattr(layout, "signature") or not hasattr(layout, "segments"):
        return None
    signature = tuple(layout.signature)
    if len(signature) < 4:
        return None
    spans = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "video"]
    if len(spans) != 1:
        raise ValueError("Control region needs one actual target-video span, not text/reference rows")
    start, stop = spans[0]
    latent_t, latent_h, latent_w = map(int, signature[1:4])
    if not (0 <= start < stop) or stop - start != latent_t * (latent_h // 2) * (latent_w // 2):
        raise ValueError("Control region target span disagrees with actual PackedLayout geometry")
    positions, update = getattr(layout, "img_pos", None), getattr(layout, "img_update", None)
    if positions is not None and update is not None:
        if (not torch.is_tensor(positions) or not torch.is_tensor(update)
                or positions.ndim != 1 or update.ndim != 1 or update.dtype != torch.bool
                or positions.shape != update.shape):
            raise ValueError("Control region visual-row update mask has an invalid layout")
        target = positions.detach().cpu()[update.detach().cpu()]
        if not torch.equal(target, torch.arange(start, stop, dtype=target.dtype)):
            raise ValueError("Control region target rows are not the published raster video span")
    return start, stop, latent_t, latent_h, latent_w


@dataclass(eq=False)
class ControlRegionRuntime:
    stage: str = "SINGLE"
    gate: object = None
    geometry: dict | None = None
    backend: str | None = None
    run: int = 0
    calls: int = 0
    gated_calls: int = 0
    target_row_evaluations: int = 0
    reasons: Counter = field(default_factory=Counter)
    errors: list = field(default_factory=list)
    cleanup_callbacks: list = field(default_factory=list, repr=False)
    bypass: str | None = None
    owner: object = field(default=None, repr=False)
    _integrity: dict | None = field(default=None, repr=False)

    def pre_run(self, *_args, **_kwargs):
        self.run += 1
        self.calls = self.gated_calls = self.target_row_evaluations = 0
        self.reasons.clear()
        self.errors.clear()
        self.cleanup()
        if self.owner is not None:
            from .modular_sampling.control_owner_identity import integrity_snapshot
            self._integrity = integrity_snapshot(self.owner)

    def cleanup(self, *_args, **_kwargs):
        try:
            if self._integrity is not None:
                from .modular_sampling.control_owner_identity import integrity_snapshot
                if self._integrity != integrity_snapshot(self.owner):
                    raise ValueError("Control owner numerical contents changed during sampling")
        finally:
            self._integrity = None
            for callback in self.cleanup_callbacks:
                callback()
            if self.gate is not None:
                self.gate._cache.clear()

    def retain(self, reason):
        first = reason not in self.reasons
        self.reasons[reason] += 1
        if first:
            warn_patch_stack("H3 regional control keeps the original control unchanged: " + reason)

    def consume(self, skip, args, *, span=None, full_packed=False):
        self.calls += 1
        if self.gate is None:
            return skip
        options = args.get("transformer_options") or {}
        if options.get("sol_morton"):
            self.retain("non_raster_Morton_region_not_applied")
            return skip
        actual = target_span(args)
        if actual is None:
            self.retain("target_layout_unavailable_region_not_applied")
            return skip
        start, stop, latent_t, latent_h, latent_w = actual
        if span is not None and tuple(span) != (start, stop):
            raise ValueError("Control region injection span is not the actual target-video span")
        if skip.ndim != 2 or (full_packed and skip.shape[0] < stop) or (not full_packed and skip.shape[0] != stop - start):
            raise ValueError("Control residual rows disagree with actual target layout")
        weights = self.gate.rows(latent_t, latent_h, latent_w).to(device=skip.device, dtype=skip.dtype)
        if full_packed:
            # Do not clear old text/reference contributions: only target video
            # is gated. Core's existing audio-zero rule remains its own policy.
            skip[start:stop].mul_(weights)
        else:
            skip = skip * weights
        self.gated_calls += 1
        self.target_row_evaluations += stop - start
        return skip

    def report(self):
        requested = self.gate is not None
        return {"schema": "t8.h3.control_region_audit.v1", "stage": self.stage, "backend": self.backend,
                "status": "failed" if self.errors else "bypass" if self.bypass else
                          "observed_region_gated" if self.gated_calls else
                          "observed_control_region_not_applied" if self.calls else "not_executed",
                "bypass_reason": self.bypass, "run": self.run,
                "region_requested": requested, "region_applied": self.gated_calls > 0,
                "residual_calls": self.calls, "gated_calls": self.gated_calls,
                "target_row_evaluations": self.target_row_evaluations,
                "preserved_reasons": dict(self.reasons), "errors": list(self.errors),
                "geometry": self.geometry,
                "geometry_sha256": None if self.geometry is None else self.geometry["geometry_sha256"],
                "crop_prepared": bool(self.geometry and self.geometry["crop_requested"]),
                "crop_consumed_by_observed_control": bool(self.calls and self.geometry and self.geometry["crop_requested"]),
                "portable_cache_identity": False, "quality_accepted": False,
                "boundary": "Direct target-video control residual only; joint attention may change other pixels or generated audio. No sampler, sigma, audio routing or output RGB compositing change."}


def register_runtime(model, runtime):
    from comfy.patcher_extension import CallbacksMP
    owner = "t8_control_region_" + str(id(runtime))
    if callable(getattr(model, "add_callback_with_key", None)):
        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, owner, runtime.pre_run)
        model.add_callback_with_key(CallbacksMP.ON_CLEANUP, owner, runtime.cleanup)
    else:
        runtime.retain("request_callbacks_unavailable_execution_local_only")
    if callable(getattr(model, "set_attachments", None)):
        model.set_attachments(REGION_ATTACHMENT, runtime)


def official_region_patch(base_class, runtime):
    """Use the current Core lifecycle, intercepting only its existing skip add.

    Core remains unmodified. Its normal wrapper, VAE preparation, previous block
    owner, prefetch pause scopes, to/models/cleanup paths stay inherited.
    """
    class RegionalPatch(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.derived_content = None

        def verify_derived(self):
            from .control_regions import tensor_sha
            if self.control_latent is None:
                if self.derived_content is not None or self.control_latent_shape is not None:
                    raise ValueError("Control owner has an orphan derived latent cache")
            elif (self.derived_content != tensor_sha(self.control_latent)
                  or tuple(self.control_latent.shape) != self.control_latent_shape):
                raise ValueError("Control owner derived latent cache changed")

        def prepare_control_latent(self, target_shape):
            self.verify_derived()
            result = super().prepare_control_latent(target_shape)
            from .control_regions import tensor_sha
            self.derived_content = tensor_sha(self.control_latent)
            return result

        def cleanup(self):
            super().cleanup()
            self.derived_content = None

        def diffusion_model_wrapper(self, *args, **kwargs):
            try:
                return super().diffusion_model_wrapper(*args, **kwargs)
            except Exception as error:
                runtime.errors.append(f"{type(error).__name__}: {error}")
                raise

        def after_block(self, block_index, args, out):
            if not self.active:
                return out
            try:
                index = self.model_patch.model.injection_layers.index(block_index)
                if index == 0:
                    self.control_latent = self.control_latent.to(out["img"].device)
                    self.control_stream = self.model_patch.model.init_stream(
                        self.pristine_stream, self.control_latent, args["layout"], args["t_emb"])
                    self.pristine_stream = None
                self.control_stream, skip = self.model_patch.model.step(
                    index, self.control_stream, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                    transformer_options=args["transformer_options"])
                skip[args["layout"].audio_pos.to(skip.device)] = 0
                skip = runtime.consume(skip, args, full_packed=True)
                out["img"].add_(skip, alpha=self.strength)
                return out
            except Exception as error:
                runtime.errors.append(f"{type(error).__name__}: {error}")
                raise

    return RegionalPatch


def apply_h3_fun_control_region(*args, region_mask=None, crop_plan_json="", stage="SINGLE", **kwargs):
    from .h3_fun_control_advanced import apply_h3_fun_control
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("Regional control stage must be an explicit nonempty label")
    state = ControlRegionRuntime(stage=stage)
    model, positive, report_json = apply_h3_fun_control(
        *args, **kwargs, region_mask=region_mask, crop_plan_json=crop_plan_json, region_runtime=state)
    # This is configuration, not evidence that any sampler called the effect.
    import json
    report = json.loads(report_json)
    report["regional_control"] = state.report()
    report["configuration_sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return model, positive, state, canonical(report)
