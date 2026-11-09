"""Optional native denoise-mask conditions at the actual delivered endpoints.

No pixel paste or post-sampling latent bridge. A latent token covers a span of
decoded frames, so conditioning it is not a promise of RGB endpoint equality.
"""
import json
import math

import torch
from comfy.nested_tensor import NestedTensor

from .sampling import nested_av_parts
from .progressive_masking import normalize_av_masks
from .source_av import frame_count_from_video_latent_t


def token_frame_spans(latent_t):
    expected = frame_count_from_video_latent_t(latent_t)
    cursor = 0
    spans = []
    for index in range(latent_t):
        count = (1, 4, 4, 4, 4)[index % 5]
        spans.append((cursor, cursor + count))
        cursor += count
    if cursor != expected:
        raise ValueError("H3 latent token spans do not match the physical frame count")
    return spans


def condition_refine_endpoints(av_latent, enabled=False, delivery_start_frame=0, delivery_frame_count=0,
                               first_frames=1, last_frames=1, endpoint_strength=0.):
    if not enabled:
        return av_latent, json.dumps(dict(status="identity_bypass", source_modified=False))
    parameters = (delivery_start_frame, delivery_frame_count, first_frames, last_frames)
    if any(type(value) is not int or value < 0 for value in parameters):
        raise ValueError("Endpoint frame coordinates must be nonnegative integers")
    if (isinstance(endpoint_strength,bool) or not math.isfinite(endpoint_strength)
            or not 0 <= endpoint_strength <= 1):
        raise ValueError("Endpoint strength must be finite in [0,1]: 0 conditions source, 1 generates")
    video, audio = nested_av_parts(av_latent)
    if not all(bool(torch.isfinite(p).all()) for p in (video,audio)):
        raise ValueError("Endpoint conditioning source contains non-finite latent values")
    spans = token_frame_spans(video.shape[2])
    total = spans[-1][1]
    count = delivery_frame_count or total - delivery_start_frame
    end = delivery_start_frame + count
    if count <= 0 or end > total or first_frames > count or last_frames > count:
        raise ValueError("Delivered endpoint window must fit inside the source frames")
    windows = [(delivery_start_frame,delivery_start_frame+first_frames), (end-last_frames,end)]
    tokens = [index for index,(start,stop) in enumerate(spans)
              if any(a < b and max(start,a) < min(stop,b) for a,b in windows)]
    masks = normalize_av_masks(av_latent.get("noise_mask"),video,audio)
    if masks is None:
        vm,am = torch.ones_like(video,dtype=torch.float32),torch.ones_like(audio,dtype=torch.float32)
    else:
        vm,am = masks.unbind()
        vm=vm.clone()
    if tokens:
        vm[:,:,tokens] = torch.minimum(vm[:,:,tokens],vm.new_tensor(endpoint_strength))
    result=dict(av_latent)
    result["noise_mask"]=NestedTensor((vm,am))
    report=dict(schema="t8.refine.endpoint_conditions.v1",status="prepared_not_sampled",
                generated_frames=total,delivery_start_frame=delivery_start_frame,delivery_frame_count=count,
                delivered_last_frame=end-1,requested_windows=windows,conditioned_latent_indices=tokens,
                conditioned_frame_spans=[list(spans[i]) for i in tokens],endpoint_strength=endpoint_strength,
                video_source_modified=False,audio_mask_modified=False,post_sampling_bridge=False,
                semantics="Native sampling condition mask. Existing stronger protection is preserved. "
                          "Temporal VAE context can affect adjacent RGB; exact pixel endpoints and motion quality require review.")
    return result,json.dumps(report,ensure_ascii=False)
