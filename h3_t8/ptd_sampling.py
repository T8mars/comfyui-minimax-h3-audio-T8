"""PTD fixed-knot mean-velocity recurrence; no ordinary Euler/PDD substitution."""

from __future__ import annotations
import math
import torch
from .ptd_checkpoint import recipe_grid

T_MAX = 0.999


def _time(time):
    value = float(time)
    if not math.isfinite(value) or not 0 <= value <= T_MAX:
        raise ValueError("PTD shared time must be finite and within [0,.999]")
    return value


def audio_sigma(time):
    t = _time(time)
    return T_MAX if t == T_MAX else 0.25 * t * T_MAX / (T_MAX - 0.75 * t)


def audio_sigma_prime(time):
    t = _time(time)
    return 0.25 * T_MAX**2 / (T_MAX - 0.75 * t) ** 2


def native_audio_shift(time):
    t = _time(time)
    if t == 0:
        return 0.25
    if t == T_MAX:
        return 1.0
    a = audio_sigma(t)
    return a * (1 - t) / (t * (1 - a))


def ptd_sigmas(recipe, device=None):
    return torch.tensor(recipe_grid(recipe), dtype=torch.float64, device=device)


def streams(value):
    if getattr(value, "is_nested", False):
        result = tuple(value.unbind())
        if len(result) != 2:
            raise ValueError("PTD requires exactly video and audio streams")
        return result
    if not isinstance(value, torch.Tensor):
        raise TypeError("PTD state must be a Tensor or actual AV NestedTensor")
    return (value,)


def map_streams(value, function):
    result = [function(part) for part in streams(value)]
    return type(value)(result) if getattr(value, "is_nested", False) else result[0]


def pair_streams(left, right, function):
    a, b = streams(left), streams(right)
    if (
        len(a) != len(b)
        or bool(getattr(left, "is_nested", False))
        != bool(getattr(right, "is_nested", False))
        or any(x.shape != y.shape or x.device != y.device for x, y in zip(a, b))
    ):
        raise ValueError(
            "PTD prediction must preserve every state stream shape and device"
        )
    result = [function(x, y) for x, y in zip(a, b)]
    return type(left)(result) if getattr(left, "is_nested", False) else result[0]


def initial_state(noise):
    return map_streams(noise, lambda value: (value.double() * T_MAX).float())


def validate_grid(sigmas, recipe):
    if (
        not isinstance(sigmas, torch.Tensor)
        or sigmas.ndim != 1
        or sigmas.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(
            "PTD requires its complete one-dimensional float32/float64 grid"
        )
    expected = ptd_sigmas(recipe).to(sigmas.dtype)
    if sigmas.shape != expected.shape or not torch.equal(
        sigmas.detach().cpu(), expected
    ):
        raise ValueError(
            f"PTD requires the complete unmodified {recipe} grid; no split, partial or 4+4 reinterpretation"
        )


@torch.no_grad()
def sample_ptd(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    controller,
    recipe=None,
):
    chosen = controller.recipe if recipe is None else recipe
    if controller.recipe != chosen:
        raise ValueError("PTD sampler recipe differs from the loaded experts")
    validate_grid(sigmas, chosen)
    extra = {} if extra_args is None else dict(extra_args)
    if extra.get("denoise_mask") is not None:
        raise ValueError(
            "PTD fixed pure-noise recurrence has no partial denoise-mask initial-state contract"
        )
    original = getattr(model, "latent_image", None)
    if original is not None and any(
        bool(torch.count_nonzero(part)) for part in streams(original)
    ):
        raise ValueError(
            "PTD requires empty generated latent; reference/keyframe images belong in CONDITIONING"
        )
    raw = getattr(model, "noise", None)
    state = (
        initial_state(raw)
        if raw is not None
        else map_streams(x, lambda value: value.float())
    )
    pair_streams(state, x, lambda a, b: a)
    parts = streams(state)
    first = parts[0]
    if (
        first.ndim < 1
        or first.shape[0] < 1
        or any(
            part.shape[0] != first.shape[0] or part.device != first.device
            for part in parts
        )
    ):
        raise ValueError("PTD state streams must share a nonempty batch and device")
    grid = ptd_sigmas(chosen, first.device)
    controller.begin_request()
    try:
        for hop, time in enumerate(recipe_grid(chosen)[:-1]):
            with controller.activate(hop, time):
                predicted = model(state, grid[hop].expand(first.shape[0]), **extra)
            velocity = pair_streams(
                state, predicted, lambda a, b: (a.float() - b.float()) / time
            )
            if callback is not None:
                callback(
                    {
                        "x": state,
                        "i": hop,
                        "sigma": grid[hop],
                        "sigma_hat": grid[hop],
                        "denoised": predicted,
                    }
                )
            dt = (grid[hop + 1] - grid[hop]).float()
            state = pair_streams(
                state, velocity, lambda a, b: a.float() + dt * b.float()
            )
            if any(not bool(torch.isfinite(part).all()) for part in streams(state)):
                raise ValueError(f"Non-finite PTD state after hop {hop}")
            controller.completed_hop(hop)
    except BaseException:
        controller.end_request(False)
        raise
    controller.end_request(True)
    return state


def make_sampler(controller):
    from comfy.samplers import KSAMPLER

    return KSAMPLER(
        sample_ptd,
        extra_options={"controller": controller, "recipe": controller.recipe},
    )
