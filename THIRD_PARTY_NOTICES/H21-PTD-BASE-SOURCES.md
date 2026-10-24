# H21 plain MiniMax-H3 base conversion

The layout mapping is adapted from EPFL VITA Lab's PTD-inference, fixed commit
`43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85`, under Apache-2.0. The independent T8
streaming implementation does not call its full-state loader or remote model code.

- [Official mapping](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/ptd_comfy/base.py):
  verbatim test reference `tests/fixtures/h21_ptd_base_reference.py`, SHA256
  `87502acb79bc95665f9bc211a253f226acb64b01712882cd2911dbfa3e0b596f`.
- [Official CLI](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/scripts/convert_comfy_base.py)
  and [integration contract](https://github.com/vita-epfl/PTD-inference/blob/43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85/docs/COMFYUI.md)
  were read as text, not installed or run on trained weights.
- Full [Apache-2.0 license](H21-PTD-BASE-APACHE2.txt) retained. Upstream bytes SHA256
  `3733abdbdcf808bdd34e30446340e9993780f7c3f9990a03e22ddfc7ab3e7d66`;
  this text copy adds one final newline, SHA256
  `0df6ff49f4b1719d07d164a71e9fa31800736dc883e39751a804e94ba5bd5275`.
- Full [upstream NOTICE](H21-PTD-BASE-NOTICE.txt) retained byte-for-byte, SHA256
  `1582b4cc1b5ac2e5b0987340aa614eccbf61d5a434e1f0a6959c8b89f96f9258`.

T8 changes2026-10-21: metadata-only preparation, explicit source manifests,
streamed raw tensor ranges, source-block content proofs, finite chunk validation,
create-only resumable parts, atomic publication and failure recovery. Existing
Meridian mapping/IO primitives are reused, but its DMD/ConvRot conversion job and
model metadata are not used. Explicit Core configuration preserves actual eps
and constructor dimensions rather than assuming alternate sources share defaults.

MiniMax-H3 model weights, derivatives and outputs remain subject to the separate
MiniMax H3 Community License. The code license does not relicense the base model,
PTD experts or outputs. No model is distributed by this source integration.
