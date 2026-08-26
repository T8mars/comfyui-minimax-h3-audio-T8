# VRetouchEr pinned inference source

This directory contains the minimal Python inference source copied from the official
[`Davidcoach/VRetouchEr_CVPR_2024`](https://github.com/Davidcoach/VRetouchEr_CVPR_2024)
repository at revision `ae25b5475680ed01958c017b32b669b4e46d7f9b`.

- Upstream license: MIT; see `LICENSE` in this directory.
- Copyright: Wen Xue, 2025.
- Line endings are normalized from CRLF to LF. Runtime verification hashes normalized text so both
  the pinned checkout and this bundled copy produce the same logical-source identity.
- No checkpoint, training data, sample media, compiled CUDA/C++ operator or separate SPyNet weight
  is included.
- `op/fused_act.py` and `op/upfirdn2d.py` are retained only as pinned formula provenance. The T8
  runtime supplies audited pure-PyTorch implementations and never compiles or imports their native
  extensions.
- The bundled files must not be imported directly. The unregistered T8 research bridge temporarily
  isolates the upstream generic `model` package name, removes the unused `turtle/tkinter` import,
  disables the absent separate SPyNet preload, verifies the complete source set and restores any
  pre-existing module tree.

The official `gen_best.pth` is not redistributed. Until a user-supplied official checkpoint passes
the locked size, trusted SHA-256, 411-entry state structure, numerical, identity, temporal, memory
and human-quality gates, this source remains an unregistered research dependency rather than a
usable ComfyUI node.
