# FlashVSR attribution and modifications

The optional `MiniMaxH3FlashVSR*` nodes include a modified copy of the public
FlashVSR inference core.

- Official project: https://github.com/OpenImagingLab/FlashVSR
- Official revision reviewed: `0ef95713cd336ee89a921fc2fbc2933aaf9e5f4a`
- Public TE-Speed core reviewed: https://github.com/tl2012tl/TE-Speed-FlashVSR
- TE-Speed revision reviewed: `912f1ad5829ab7c9d442ac4ffb912ea1fe3c7f42`
- License: Apache License 2.0

The included source is under `flashvsr_vendor/`, together with its Apache-2.0
license. Local modifications replace the unpublished TE binary attention bridge
with an auditable `spas_sage_attn` adapter, add an explicit per-chunk budget
controller, expose staged memory handling, and integrate ComfyUI loading and
reporting. The unpublished TE `.pyd` modules are not bundled, reverse engineered,
or represented as reproduced.

The separately installed `spas_sage_attn` wheel provides the CUDA block-sparse
Sage2 kernel. It is not redistributed here and must match the user's Torch, CUDA,
Python, and GPU architecture. FlashVSR-v1.1 model weights are also not bundled.
