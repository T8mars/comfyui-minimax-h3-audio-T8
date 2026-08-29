from __future__ import annotations

# Qwen-only timed-reference transport adapted for this GPL project from the
# public ComfyUI-MiniMaxH3-Timed-References design by Ethanfel (GPL-3.0-only).
# Native H3 reference construction remains delegated to ComfyUI core.

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import comfy.utils
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer, VISION_END, VISION_START


REFERENCE_SIZES = (
    "64",
    "96",
    "128",
    "192",
    "256",
    "384",
    "512",
    "768",
    "1024",
    "1280",
    "source",
)
DEFAULT_ANALYSIS_FPS = 2.0
MAX_REFERENCE_SECONDS = 15.0
TIMESTAMP_DECIMALS = 6


def normalize_prompt_tag(value: str) -> str:
    tag = str(value).strip().removeprefix("#")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", tag):
        raise ValueError(
            "prompt_tag must start with a letter and contain only letters, numbers, _ or -"
        )
    return tag


def _fraction(value, field: str) -> Fraction:
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field} must be finite")
        result = Fraction(value)
    else:
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"{field} must be finite") from error
        if not parsed.is_finite():
            raise ValueError(f"{field} must be finite")
        result = Fraction(parsed)
    return result


def format_timestamp(value) -> str:
    result = _fraction(value, "timestamp")
    if result < 0:
        raise ValueError("reference timestamps must be non-negative")
    text = f"{float(result):.{TIMESTAMP_DECIMALS}f}".rstrip("0").rstrip(".")
    return text or "0"


def _resize_frames(frames, size: str):
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError("timed references require IMAGE frames [T,H,W,C>=3]")
    frames = frames[..., :3]
    if size == "source":
        return frames
    edge = int(size)
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    scale = math.sqrt((edge * edge) / max(1, source_w * source_h))
    target_w = max(32, round(source_w * scale / 32) * 32)
    target_h = max(32, round(source_h * scale / 32) * 32)
    samples = frames.movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, target_w, target_h, "lanczos", "disabled"
    )
    return samples.movedim(1, -1)


def _nearest_indices(timestamps, analysis_fps: float) -> list[int]:
    values = [_fraction(value, "video timestamp") for value in timestamps]
    if len(values) < 2:
        raise ValueError("timed video references require at least two frames")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError("video timestamps must be strictly increasing")
    rate = _fraction(analysis_fps, "analysis_fps")
    if rate <= 0:
        raise ValueError("analysis_fps must be greater than zero")
    relative = [value - values[0] for value in values]
    selected: list[int] = []
    target = Fraction(0)
    right = 0
    step = Fraction(1) / rate
    while target <= relative[-1]:
        while right < len(relative) and relative[right] < target:
            right += 1
        if right == 0:
            index = 0
        elif right >= len(relative):
            index = len(relative) - 1
        else:
            left = right - 1
            index = left if target - relative[left] <= relative[right] - target else right
        if not selected or selected[-1] != index:
            selected.append(index)
        target += step
    return selected


def prepare_timed_video_frames(
    frames,
    source_fps: float,
    target_start_seconds: float,
    size: str,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
):
    if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[-1] < 3:
        raise ValueError("timed video references require at least two IMAGE frames")
    source_rate = _fraction(source_fps, "source_fps")
    analysis_rate = _fraction(analysis_fps, "analysis_fps")
    start = _fraction(target_start_seconds, "target_start_seconds")
    if source_rate <= 0 or analysis_rate <= 0:
        raise ValueError("source_fps and analysis_fps must be greater than zero")
    if analysis_rate > source_rate:
        raise ValueError("analysis_fps cannot exceed source_fps")
    if start < 0:
        raise ValueError("target_start_seconds must be non-negative")
    duration = Fraction(int(frames.shape[0]), 1) / source_rate
    if duration > MAX_REFERENCE_SECONDS:
        raise ValueError("timed video references are limited to 15 seconds")
    source_times = [Fraction(index, 1) / source_rate for index in range(frames.shape[0])]
    indices = _nearest_indices(source_times, analysis_rate)
    return _resize_frames(frames[indices], size), [start + source_times[i] for i in indices]


def _replace_tag(prompt: str, tag: str, label: str) -> str:
    pattern = rf"(?<![A-Za-z0-9_-])#{re.escape(tag)}(?![A-Za-z0-9_-])"
    return re.sub(pattern, lambda _match: label, prompt)


def _validated_times(values, count: int) -> list[Fraction]:
    if values is None or len(values) != count:
        raise ValueError("timed-reference timestamp count must match its frame count")
    result = [_fraction(value, "timestamp") for value in values]
    if any(value < 0 for value in result):
        raise ValueError("reference timestamps must be non-negative")
    if any(current < previous for previous, current in zip(result, result[1:])):
        raise ValueError("reference timestamps must be ordered")
    return result


class MiniMaxH3TimedReferenceTokenizerT8:
    def __init__(self, tokenizer, references):
        self.tokenizer = tokenizer
        self.references = tuple(references)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        if "minimax_ref_items" not in kwargs:
            return self.tokenizer.tokenize_with_weights(
                text, return_word_ids, **kwargs
            )

        native_items = list(kwargs.get("minimax_ref_items") or [])
        first_index = sum(item.get("type") == "video" for item in native_items)
        prompt = str(text)
        for offset, reference in enumerate(self.references, start=1):
            prompt = _replace_tag(
                prompt,
                reference["prompt_tag"],
                f"<Video {first_index + offset}>",
            )

        # Older/newer tokenizer variants can still accept exact timestamp items.
        # Use that public route when internal Qwen helpers are unavailable.
        if not hasattr(self.tokenizer, "qwen3vl_32b") or not hasattr(
            self.tokenizer, "_vision_entry"
        ):
            for reference in self.references:
                native_items.append(
                    {
                        "type": "video",
                        "data": reference["frames"],
                        "timestamps": reference["timestamps"],
                    }
                )
            kwargs["minimax_ref_items"] = native_items
            return self.tokenizer.tokenize_with_weights(
                prompt, return_word_ids, **kwargs
            )

        entries = []
        if native_items:
            native_kwargs = dict(kwargs)
            native_kwargs["minimax_ref_items"] = native_items
            native = self.tokenizer.tokenize_with_weights("", False, **native_kwargs)
            entries.extend(native["qwen3vl_32b"][0])

        def add_text(value: str) -> None:
            if not value:
                return
            batches = self.tokenizer.qwen3vl_32b.tokenize_with_weights(
                value, return_word_ids=False, disable_weights=True
            )
            if len(batches) != 1:
                raise ValueError("MiniMax H3 text segment exceeds the prompt length")
            entries.extend(batches[0])

        for offset, reference in enumerate(self.references, start=1):
            frames = reference["frames"]
            times = _validated_times(reference["timestamps"], int(frames.shape[0]))
            if frames.shape[0] % 2:
                import torch

                frames = torch.cat([frames, frames[-1:]], dim=0)
                times.append(times[-1])
            add_text(f"<Video {first_index + offset}>: ")
            for index in range(0, frames.shape[0], 2):
                block_time = (times[index] + times[index + 1]) / 2
                add_text(f"<{format_timestamp(block_time)} seconds>")
                entries.append((VISION_START, 1.0))
                entries.append(
                    (self.tokenizer._vision_entry(frames[index : index + 2], True), 1.0)
                )
                entries.append((VISION_END, 1.0))
        add_text(prompt)
        if not entries:
            entries.append((151643, 1.0))
        if return_word_ids:
            entries = [entry + (0,) for entry in entries]
        return {"qwen3vl_32b": [entries]}


def append_timed_reference(clip, reference):
    tokenizer = clip.tokenizer
    references = []
    if isinstance(tokenizer, MiniMaxH3TimedReferenceTokenizerT8):
        references = list(tokenizer.references)
        tokenizer = tokenizer.tokenizer
    if not isinstance(tokenizer, MiniMaxH3Tokenizer):
        raise ValueError("timed references require a MiniMax H3 CLIP model")
    if any(item["prompt_tag"] == reference["prompt_tag"] for item in references):
        raise ValueError("each chained timed reference requires a unique prompt_tag")
    references.append(reference)
    output = clip.clone()
    output.tokenizer = MiniMaxH3TimedReferenceTokenizerT8(tokenizer, references)
    return output
