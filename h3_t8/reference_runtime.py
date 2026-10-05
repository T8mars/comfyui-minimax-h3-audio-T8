"""Native reference encoding and fresh-conditioning inputs, never old embedding edits.

Portable asset creation is a separate opt-in from ordinary user refs. Unknown
encoder owners remain usable on the old path, but are not certified by a label.
"""
from dataclasses import dataclass
import json

import torch
import torchaudio

from .core import (FPS, align_frame_count_down, encode_audio_once,
                   ensure_h3_audio_vae_non_aligned_crop_compat, resize_image, validate_audio)
from .progressive_producers import native_producer_identity
from .reference_package import (IDENTIFIER, MAX_BYTES, MAX_MEMBERS, NORMALIZATION, ReferencePackage,
                                canonical, capture_package, route_packages, tensor_record)


PRODUCER_SCOPE = "actual_native_weights_tokenizer_configuration_and_implementation"


def portable_producer(component, role):
    identity = native_producer_identity(component, role)
    if identity.get("portable_cache_reuse") is False or identity.get("scope") != PRODUCER_SCOPE:
        raise ValueError("This encoder has no portable reference-asset identity; use ordinary native refs instead")
    return {"role": role, "sha256": identity["sha256"], "scope": PRODUCER_SCOPE}


def create_package(role_id, *, kind="image", frames=None, audio=None,
                   video_vae=None, audio_vae=None, width=256, height=256, frame_limit=124):
    """Capture the actual native VAE output; report all explicit preprocessing.

    No denoiser scaling, training, pooling, audio replacement, Qwen inference or
    DiT sampling. A visual role can optionally carry one separate voice anchor.
    """
    if kind not in ("image", "video", "audio"):
        raise ValueError("Choose an explicit image, video or audio asset")
    if not isinstance(role_id, str) or not IDENTIFIER.fullmatch(role_id):
        raise ValueError("Use an explicit short ASCII role ID before encoding")
    if (type(width) is not int or type(height) is not int or width < 32 or height < 32
            or width % 32 or height % 32 or type(frame_limit) is not int or not 5 <= frame_limit <= 360):
        raise ValueError("Reference dimensions use positive 32-grid; explicit video frame limit is 5..360")
    if kind == "audio" and (frames is not None or audio is None):
        raise ValueError("Audio asset needs only an explicit AUDIO source")
    tensors, members, before, report = {}, [], {}, {"role_id": role_id, "kind": kind,
        "normalization": NORMALIZATION, "sampling_executed": False, "automatic_accept": False}
    if kind != "audio":
        if (type(frames) is not torch.Tensor or frames.ndim != 4 or frames.shape[-1] != 3
                or frames.dtype != torch.float32 or not bool(torch.isfinite(frames).all())
                or not bool(((frames >= 0) & (frames <= 1)).all())):
            raise ValueError("Visual source needs finite RGB IMAGE [N,H,W,3] in 0..1")
        count = int(frames.shape[0])
        if (kind == "image" and count != 1) or (kind == "video" and count < 5):
            raise ValueError("Image asset uses exactly one frame; video uses at least five")
        used = 1 if kind == "image" else align_frame_count_down(min(count, frame_limit))
        if used * width * height * 3 * 4 > MAX_BYTES:
            raise ValueError("Selected reference RGB exceeds the bounded asset budget; reduce size/frames explicitly")
        before["video_vae"] = portable_producer(video_vae, "video_vae")
        rgb = resize_image(frames[:used], width, height).detach().cpu().contiguous().clone()
        latent = video_vae.encode(rgb).detach().cpu().contiguous()
        tensors.update(visual_latent=latent, visual_rgb=rgb)
        members.append({"id": "visual", "role_id": role_id, "kind": kind,
            "latent": "visual_latent", "grounding": "visual_rgb",
            "source": {"sha256": tensor_record(rgb)["sha256"], "scope": "actual_decoded_rgb_tensor"},
            "producer": before["video_vae"], "normalization": NORMALIZATION})
        report.update(input_frames=count, used_frames=used, aligned_grid="17n+5" if kind == "video" else "single",
                      reference_dimensions=[width, height], trimmed_frames=count-used)
    if audio is not None:
        waveform, rate = validate_audio(audio)
        if (not torch.isfinite(waveform).all() or waveform.numel() * waveform.element_size() > MAX_BYTES
                or waveform.shape[1] not in (1, 2)):
            raise ValueError("Voice source must be bounded finite mono/stereo AUDIO")
        # The same existing compatibility boundary is established before the
        # producer snapshot; it must not look like a mid-encode state change.
        ensure_h3_audio_vae_non_aligned_crop_compat(audio_vae)
        before["audio_vae"] = portable_producer(audio_vae, "audio_vae")
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if rate <= 0 or vae_rate <= 0 or waveform.shape[1] * round(waveform.shape[-1] * vae_rate / rate) * 4 > MAX_BYTES:
            raise ValueError("Resampled voice exceeds the bounded asset budget")
        pcm = waveform.detach().cpu().float().contiguous().clone()
        if rate != vae_rate:
            pcm = torchaudio.functional.resample(pcm, rate, vae_rate)
        encoded = encode_audio_once(audio_vae, {"waveform": pcm, "sample_rate": vae_rate})
        tensors["voice_latent"] = encoded.detach().cpu().contiguous()
        members.append({"id": "voice", "role_id": role_id, "kind": "audio",
            "latent": "voice_latent", "grounding": None,
            "source": {"sha256": tensor_record(pcm)["sha256"], "scope": "actual_decoded_pcm_tensor"},
            "producer": before["audio_vae"], "normalization": NORMALIZATION})
        report.update(voice_input_sample_rate=rate, encoded_sample_rate=vae_rate,
                      voice_samples=int(pcm.shape[-1]), voice_channels=int(pcm.shape[1]))
    for role, original in before.items():
        component = video_vae if role == "video_vae" else audio_vae
        if portable_producer(component, role) != original:
            raise ValueError("Actual reference encoder changed during encoding")
    package = capture_package(members, tensors)
    report.update(manifest_sha256=package.verify()["sha256"], producer=before,
                  encoded_reference_members=len(members))
    return package, report


@dataclass(frozen=True)
class ReferenceSet:
    packages: tuple
    roles_json: str
    allow_missing_grounding: bool = False

    def routed(self):
        return route_packages(list(self.packages), json.loads(self.roles_json),
                              allow_missing_grounding=self.allow_missing_grounding)

    def conditioning_inputs(self, video_vae, audio_vae):
        routed = self.routed()
        if routed["latent_only_exp"]:
            raise ValueError("Latent-only visual Apply requires its separately qualified token/ordinal adapter; complete RGB packages work here")
        actual = {}
        selected = {(row["role_id"], row["member_id"]) for row in routed["mapping"]}
        for package in self.packages:
            for member in package.verify(allow_missing_grounding=self.allow_missing_grounding)["members"]:
                if (member["role_id"], member["id"]) not in selected:
                    continue
                role = member["producer"]["role"]
                if role not in actual:
                    actual[role] = portable_producer(video_vae if role == "video_vae" else audio_vae, role)
                if member["producer"] != actual[role]:
                    raise ValueError("Reference package encoder differs from the actual selected VAE; rebuild this asset explicitly")
        items = []
        for item in routed["qwen_ref_items"]:
            if item["type"] != "video":
                items.append(dict(item))
            else:
                indices = list(range(0, item["data"].shape[0], FPS // 2))
                items.append({"type": "video", "data": item["data"][indices],
                              "timestamps": [index / FPS for index in indices]})
        return {**routed, "qwen_ref_items": items, "actual_producers": actual}


def capture_set(packages, roles_json, *, allow_missing_grounding=False):
    if not isinstance(roles_json, str) or len(roles_json.encode()) > 16384:
        raise ValueError("Use a bounded explicit role presence JSON list")
    if type(packages) is not list or not 0 < len(packages) <= MAX_MEMBERS:
        raise ValueError("Use a bounded ordered reference package list")
    if any(type(package) is not ReferencePackage for package in packages):
        raise ValueError("Use actual T8 reference packages, not a provider label")
    result = ReferenceSet(tuple(packages), canonical(json.loads(roles_json)), allow_missing_grounding)
    routed = result.routed()
    return result, {key: value for key, value in routed.items() if key not in ("refs", "qwen_ref_items")}
