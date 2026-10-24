"""Actual selected embedding width at owned-worker validation boundaries.

Original source files, caller tensors, sampler and model remain unchanged.
The owned child temporarily swaps four validation references and restores
them; parent/user interpreters are untouched. Default5120 keeps originals.
"""
from contextlib import contextmanager


@contextmanager
def selected_width(text_dim):
    if type(text_dim) is not int or text_dim < 1:
        raise ValueError("Actual selected H3 text_dim must be a positive integer")
    if text_dim == 5120:
        yield
        return
    import torch
    from freevideo_engine import conditioning, keyframes, media_conditioning
    from freevideo_engine.media_request import TASKS

    def to_cache(value, prompt=None, task="t2va"):
        if task not in TASKS or len(value) != 1:
            raise ValueError("Expected one native H3 conditioning sequence with an actual task")
        embeds, extra = value[0]
        tags = extra.get("minimax_token_tags")
        if (not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1
                or embeds.shape[-1] != text_dim or not embeds.shape[1]):
            raise ValueError("Actual selected H3 prompt width requires [1,L," + str(text_dim) + "]")
        if not embeds.is_floating_point() or not bool(torch.isfinite(embeds).all()):
            raise ValueError("H3 conditioning must contain finite floating point embeddings")
        if (not isinstance(tags, torch.Tensor) or tags.ndim != 1 or len(tags) != embeds.shape[1]
                or not bool(((tags == 0) | (tags == 1)).all()) or (task == "t2va" and not bool((tags == 1).all()))):
            raise ValueError("Invalid native H3 modality token tags for " + task)
        if task in ("i2va", "l2va", "fl2va") and not bool((tags == 0).any()):
            raise ValueError("Keyframe tasks require jointly encoded vision tokens")
        result = dict(prompt_embeds=embeds[0].detach().to("cpu", torch.bfloat16),
                      text_token_tags=tags.detach().to("cpu", torch.long))
        if prompt is not None:
            result["prompt"] = prompt
        return result

    def validate_conditioning(prompt, tags, conditions, task, width, height):
        to_cache([[prompt.unsqueeze(0), dict(minimax_token_tags=tags)]], task=task)
        if task == "t2va":
            if conditions:
                raise ValueError("Keyframe input requires its actual native task")
            return
        anchors_by_task = {"i2va": ("first",), "l2va": ("last",), "fl2va": ("first", "last")}
        if task not in anchors_by_task or not conditions:
            raise ValueError("Keyframe task requires its encoded anchors")
        anchors, latents = conditions
        if tuple(anchors) != anchors_by_task[task] or len(latents) != len(anchors):
            raise ValueError("Native H3 keyframe anchor order/count differs from actual task")
        expected = (1, 24, 1, height // 16, width // 16)
        for value in latents:
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Native keyframe latent must be finite float32 at the actual canvas shape")

    previous = (conditioning.to_cache, keyframes.validate_conditioning,
                media_conditioning.to_cache, media_conditioning.validate_conditioning)
    conditioning.to_cache = media_conditioning.to_cache = to_cache
    keyframes.validate_conditioning = media_conditioning.validate_conditioning = validate_conditioning
    try:
        yield
    finally:
        (conditioning.to_cache, keyframes.validate_conditioning,
         media_conditioning.to_cache, media_conditioning.validate_conditioning) = previous
