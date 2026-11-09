# Adapted from BenderBlender/ComfyUI-NAG-MiniMaxH3 c63a046, MIT.
# Original NAG formula by Dar-Yen Chen; see THIRD_PARTY_NOTICES/NAG-H3-LICENSE.txt.
import torch


def nag_(z_positive, z_negative, scale, tau, alpha):
    """Normalized Attention Guidance, same math as ComfyUI-NAG's utils.nag (MIT, Dar-Yen Chen).

    Computed in place over z_negative, which is returned holding the guidance: H3 attention
    outputs span ~40k tokens, so the extra temporaries of the out-of-place version add up.
    """
    eps = 1e-6
    z_guidance = z_negative.mul_(1.0 - scale).add_(z_positive, alpha=scale)

    norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True).clamp_min(eps)
    norm_guidance = torch.norm(z_guidance, p=1, dim=-1, keepdim=True).clamp_min(eps)

    s = norm_guidance / norm_positive
    z_guidance.mul_(s.clamp(max=tau) / s * alpha).add_(z_positive, alpha=1.0 - alpha)
    return z_guidance
