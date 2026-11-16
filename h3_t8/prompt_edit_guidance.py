"""Explicit original/edited prompt guidance on Core's conditioning consumers."""
from __future__ import annotations

import json
import math

import torch
import comfy.samplers
import comfy.model_patcher


def _same_media(a, b):
    if a is b:
        return True
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_media(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_media(x, y) for x, y in zip(a, b))
    if isinstance(a, (str, int, float, bool, type(None))) and type(a) is type(b):
        return a == b
    return False


def validate_text_only_edit(original, edited):
    """Compare actual metadata contents, not reference counts or tensor shapes."""
    if not isinstance(original, (list, tuple)) or not isinstance(edited, (list, tuple)):
        raise ValueError("Both inputs must be CONDITIONING sequences")
    if len(original) != len(edited) or not original:
        raise ValueError("Original and edited conditioning must have matching entries")
    lengths = []
    for index, (a, b) in enumerate(zip(original, edited)):
        if len(a) != 2 or len(b) != 2 or not isinstance(a[1], dict) or not isinstance(b[1], dict):
            raise ValueError("Expected [embedding, metadata] conditioning entries")
        ta, tb = a[0], b[0]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor) or ta.ndim != 3 or tb.ndim != 3:
            raise ValueError("Text embeddings must be [batch, tokens, features] tensors")
        if ta.shape[0] != tb.shape[0] or ta.shape[2] != tb.shape[2] or ta.dtype != tb.dtype:
            raise ValueError("Text embedding batch, feature width and dtype must match")
        # These two fields describe text, not input media. Everything else must
        # retain actual values (including clocks, refs, guide tensors and masks).
        skip = {"minimax_token_tags", "pooled_output"}
        ma = {k: v for k, v in a[1].items() if k not in skip}
        mb = {k: v for k, v in b[1].items() if k not in skip}
        if not _same_media(ma, mb):
            raise ValueError(f"Conditioning entry {index} changed media, clock or non-text metadata")
        lengths.append([int(ta.shape[1]), int(tb.shape[1])])
    return lengths


def sampling_step(timestep, options):
    schedule = options.get("transformer_options", {}).get("sample_sigmas")
    if schedule is None:
        raise ValueError("first_step needs the sampler's actual sample_sigmas schedule")
    grid = torch.as_tensor(schedule).detach().flatten()
    current = torch.as_tensor(timestep, device=grid.device).detach().flatten()
    if grid.numel() < 2 or not torch.isfinite(grid).all() or (grid[1:] > grid[:-1]).any():
        raise ValueError("sample_sigmas must be a finite nonincreasing schedule")
    if not current.numel() or not torch.isfinite(current).all() or not torch.all(current == current[0]):
        raise ValueError("One guidance call requires a common finite sigma")
    return max(0, min(grid.numel() - 2,
                      int(torch.searchsorted(-grid[:-1].contiguous(), -current[0], right=True)) - 1))


class PromptEditGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, original, edited, edit_strength=1.1, first_step=0):
        if not math.isfinite(edit_strength):
            raise ValueError("edit_strength must be finite")
        if not isinstance(first_step, int) or isinstance(first_step, bool) or first_step < 0:
            raise ValueError("first_step must be a nonnegative integer")
        self.text_lengths = validate_text_only_edit(original, edited)
        super().__init__(model)
        self.inner_set_conds({"negative": original, "positive": edited})
        self.edit_strength = float(edit_strength)
        self.first_step = first_step
        self.calls = []
        self.physical_model_forwards = 0
        self.model_options = comfy.model_patcher.create_model_options_clone(model.model_options)
        prior = self.model_options.get("model_function_wrapper")

        def count_forward(apply, options):
            def counted(*args, **kwargs):
                self.physical_model_forwards += 1
                return apply(*args, **kwargs)
            if prior is not None:
                return prior(counted, options)
            return counted(options["input"], options["timestep"], **options["c"])

        if prior is not None and not callable(prior):
            raise TypeError("Existing model_function_wrapper must be callable")
        self.model_options["model_function_wrapper"] = count_forward

    def sample(self, *args, **kwargs):
        # Comfy may reuse a cached GUIDER object for a later queue submission.
        # Reports describe this invocation, not a sum of all previous renders.
        self.calls = []
        self.physical_model_forwards = 0
        return super().sample(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        options = dict(model_options or self.model_options)
        # Retain the caller's schedule and all delegates. Isolate two model
        # evaluations so native H3 payload/layout objects cannot be co-batched.
        step = sampling_step(timestep, options) if self.first_step else 0
        strength = 0.0 if step < self.first_step else self.edit_strength
        original, edited = self.conds.get("negative"), self.conds.get("positive")
        branches = []

        def evaluate(conds, label):
            branch_options = comfy.model_patcher.create_model_options_clone(options)
            branch_options.setdefault("transformer_options", {})["t8_prompt_edit_branch"] = label
            branches.append(label)
            return comfy.samplers.calc_cond_batch(self.inner_model, [conds], x, timestep, branch_options)[0]

        if strength == 0.0:
            prediction = evaluate(original, "original")
        elif strength == 1.0:
            prediction = evaluate(edited, "edited")
        else:
            a = evaluate(original, "original")
            b = evaluate(edited, "edited")
            prediction = a + strength * (b - a)
        self.calls.append({"step": step, "sigma": float(torch.as_tensor(timestep).flatten()[0]),
                           "strength": strength, "branches": branches})
        return prediction

    def report(self):
        return json.dumps({"schema": "t8.h3.prompt_edit.v1", "text_token_lengths": self.text_lengths,
                           "first_step": self.first_step, "edit_strength": self.edit_strength,
                           "actual_predict_calls": self.calls,
                           "physical_model_forwards": self.physical_model_forwards,
                           "media_position_policy": "native_independent_text_lengths",
                           "quality_validated": False})
