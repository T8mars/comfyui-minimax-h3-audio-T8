# Third-party notices

## Alibaba PAI MiniMax-H3 PDD acceleration adapters

The optional `MiniMax H3 PDD 8-Step Setup (T8 Advanced EXP)` node implements a clean-room ComfyUI
integration of the Parallel Decoding Distillation schedule and dynamic output-head behavior published
with [`alibaba-pai/MiniMax-H3-Acc-LoRAs`](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
The implementation was reviewed against upstream revision
`78db175437ee05df7ec492ee366f01b68b8d20e6`, whose reference code is distributed under the
Apache License 2.0. No PDD model weight is committed or redistributed by this repository. Users must
install the matching FL2VA or Ref2VA adapter and base model themselves. The node is optional and does
not alter stable nodes or legacy workflows when it is not connected.

## LightX2V MiniMax H3 Turbo-SLA and SpargeAttn kernel

The optional MiniMax H3 LightX2V SLA Advanced node implements a clean-room ComfyUI adapter for the
dynamic sparse-attention routing math reviewed at fixed LightX2V revision
[`lightx2v/LightX2V@f8aee98b`](https://github.com/ModelTC/LightX2V/tree/f8aee98b5462cca8d7288888146ebd95592bf266).
It authenticates a user-installed LoRA from fixed Hugging Face model revision
[`lightx2v/Minimax-h3-Turbo-SLA@10ade67c`](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/tree/10ade67cd15ff7a135fa35c2a0673ea96c839247).
Neither model weights nor LightX2V source files are redistributed by this repository.

Sparse execution optionally calls the separately installed
[`thu-ml/SpargeAttn`](https://github.com/thu-ml/SpargeAttn) `spas-sage-attn` package, which is
distributed under the BSD 3-Clause License. The package is loaded only by the explicit sparse SLA
route; stable nodes and the SLA disabled path do not depend on it. Users remain responsible for
installing a wheel compatible with their ComfyUI Torch, CUDA and GPU architecture.

The optional SLA + KJ Sage Composer interoperates with a separately installed
[`kijai/ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes) MiniMax H3 memory-efficient
SageAttention patch, distributed under GPL-3.0. The composer authenticates and conditionally
delegates that installed bound forward at runtime; no KJNodes source file is copied or redistributed
by this repository. The ordinary SLA node remains independent of KJNodes.

## ComfyUI-ClipProj interoperability

The optional ClipProj audit and bridge workflows interoperate with a separately installed
[`nicolab28/ComfyUI-ClipProj`](https://github.com/nicolab28/ComfyUI-ClipProj) tree. Local validation
used version 0.1.13 at fixed revision `c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c`, whose source is
distributed under the MIT License. This repository does not copy or redistribute its source or any
Qwen/ClipProj model weights. Stable MiniMax H3 workflows do not import it; the external
`ClipProjApply` node and a user-supplied dimension-matched projection matrix are required only by
explicitly selected ClipProj workflows.

## ComfyUI-sol-attn interoperability

The optional Sol-Attn audit and conservative workflow interoperate with a separately installed
[`Saganaki22/ComfyUI-sol-attn`](https://github.com/Saganaki22/ComfyUI-sol-attn) tree. Local validation
used version 0.6.2 at fixed revision `930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`, whose source is
distributed under the Apache License 2.0. No upstream source or compiled kernel is redistributed by
this repository. Stable, dense and disabled routes do not import Sol-Attn; users must install a
Torch/CUDA/GPU-compatible build and explicitly select the Sol workflow.

## MiniMax H3 RAVEN Streaming interoperability

The optional guarded RAVEN workflow interoperates with a separately installed
[`YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming`](https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming)
version 0.1.0 at fixed revision `bcfa38138ddf1a5041af9880760815874138d4e1`, which is
distributed under the MIT License. This repository does not copy its causal DiT, KV-cache,
streaming VAE/preview implementation or any model weights. The external `RAVENStreamingSampler`
remains the execution node; the T8 integration only delegates loading after a preflight and audits
the request through the external runtime's own contracts.

The research repository
[`mvp-ai-lab/RAVEN`](https://github.com/mvp-ai-lab/RAVEN) was reviewed at revision
`5a71a3cb0588ce2a9696ac23af6c78ac3f9929f3` and is licensed CC BY-NC 4.0. No source from that
repository is included or adapted into this GPL package. Users must separately review the licenses
and terms of the full MiniMax H3 base model and mandatory RAVEN Streaming LoRA.

## Enhance-A-Video / FETA research

The isolated MiniMax H3 Enhance-A-Video Advanced implementation is a clean-room H3 adaptation of
the equations described in [`Enhance-A-Video` (arXiv:2502.07508v3)](https://arxiv.org/abs/2502.07508v3).
The reference implementation was reviewed at fixed revision
[`NUS-HPC-AI-Lab/Enhance-A-Video@16a7899e`](https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video/tree/16a7899e6f55f85ea19f1d3a415c6dc0c4096176),
which is licensed under Apache-2.0. That repository is not a runtime dependency, no source file is
copied, and the H3 joint audio-video adapter is explicitly experimental because the paper did not
evaluate MiniMax H3 or joint audio-video generation.

## Prompt Relay research and community implementations

The MiniMax H3 Prompt Relay Advanced implementation derives the temporal-penalty equation from
[`Prompt Relay` (arXiv:2604.10030v1)](https://arxiv.org/abs/2604.10030v1) and was designed after
reviewing the public interaction and implementation approaches in the following fixed revisions:

- [`GordonChen19/Prompt-Relay@0ad2b227`](https://github.com/GordonChen19/Prompt-Relay/tree/0ad2b22741ab09e89e7981aba8980ced29a707b9)
- [`kijai/ComfyUI-PromptRelay@ca5d4e3e`](https://github.com/kijai/ComfyUI-PromptRelay/tree/ca5d4e3edb6abd9c2a4c68a3a6798eec1980f450)
- [`yichengup/ComfyUI-YCNodes-MiniMax-H3@ca9447bd`](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/tree/ca9447bd21048f37539f9f15250831fcfdf481c7)

The runtime code in this repository is H3-specific: it binds the exact native Qwen token stream,
targets MiniMax H3 packed joint audio-video self-attention, streams target-video query chunks, and
does not copy a whole Wan/LTX/H3 Transformer forward or create a dense sequence-squared mask. The
three community repositories are not runtime dependencies and no model weights are redistributed.

## SPEED: Spectral Progressive Diffusion

The isolated MiniMax H3 SPEED Advanced implementation is a clean-room adaptation of the public
equations and reference algorithms in
[`howardhx/speed@ca7801c9`](https://github.com/howardhx/speed/tree/ca7801c9bdffe681742e9592345bcf4885959be5).
No source code is copied from `StanLukuvka/ComfyUI-MiniMax-H3-SPEED`; its WIP implementation and
defaults are not runtime dependencies.

Copyright (c) 2026 Howard Xiao

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ComfyUI-H3-FaceRefine

The isolated Face Refine Parity Advanced implementation is a clean-room adaptation of the public
node contracts and algorithms in
[`Carasibana/ComfyUI-H3-FaceRefine@79a97ce5`](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/tree/79a97ce5ee4b393ce26313bd1280b706fe8b4f2c).

Copyright (c) 2026 Carasibana

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ComfyUI_CineStyle interaction reference and Skin Finish provenance

The T8 Skin Finish node interaction and product boundary were designed after reviewing
[`chflame163/ComfyUI_CineStyle@e7d5fac`](https://github.com/chflame163/ComfyUI_CineStyle/tree/e7d5facafd95c97190fcf54171960f25c21b3043),
whose top-level repository is licensed under MIT. The T8 implementation does not import that
repository, copy or vendor its Beauty implementation, download its models, or depend on its Python
requirements.

In particular, CineStyle's `py/vfx_beauty.py` describes itself as a Torch port of the
`crok_beauty` Matchbox shader and points through a public shader lineage that includes
`Ls_Dollface`. The per-item provenance and license of that shader lineage were not sufficiently
clear for direct reuse. T8 therefore uses an independently written, generic masked low-frequency
tone/highlight correction with explicit source fallback, rather than the referenced 19-pass shader
code or constants. No CineStyle Beauty, BiSeNet, ParseNet or Matchbox source/weights are redistributed.

The optional `face_refine_plan` route only derives a conservative inner-face proxy mask from this
repository's existing plan geometry. It is explicitly not a semantic skin parser. Users who supply
external masks or separately installed model outputs remain responsible for the licenses and usage
terms of those inputs.

## VRetouchEr CVPR 2024 research bridge

The unregistered Skin Finish research bridge includes the minimal Python inference source from
[`Davidcoach/VRetouchEr_CVPR_2024`](https://github.com/Davidcoach/VRetouchEr_CVPR_2024) at fixed
revision `ae25b5475680ed01958c017b32b669b4e46d7f9b`. The upstream repository and bundled source are
licensed under the MIT License; the complete upstream notice is retained at
`vendor/vretoucher_upstream/LICENSE`.

Copyright (c) 2025 Wen Xue

The source is bundled with CRLF line endings normalized to LF and otherwise kept as the pinned
minimal inference files. T8 does not include the official `gen_best.pth`, training data, sample
media, compiled native operators or a separate SPyNet checkpoint. The two upstream operator Python
files are retained as formula provenance, while the T8 bridge uses audited pure-PyTorch equivalents
and does not compile their CUDA/C++ extensions. The bridge remains unregistered until a separately
installed official checkpoint passes identity, structure, numerical, perceptual, temporal and
memory validation.

## Optional FaceXLib ParseNet semantic-mask runtime

The opt-in `MiniMaxH3SkinFinishSemanticMaskT8Advanced` and
`MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced` nodes import the user's separately installed
[FaceXLib](https://github.com/xinntao/facexlib) ParseNet implementation. FaceXLib source code is
published under the MIT License. T8 does not copy or vendor that source and does not make FaceXLib a
mandatory dependency.

The node accepts only the separately installed FaceXLib v0.2.2 release checkpoint
`parsing_parsenet.pth`, placed at `ComfyUI/models/facedetection/parsing_parsenet.pth`. It requires an
exact size of 85,331,193 bytes and SHA-256
`3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2`, performs no runtime download,
and requires PyTorch `weights_only=True` loading. The checkpoint release page does not explicitly
state a checkpoint-specific license, and training-data terms may impose additional restrictions.
Users must review those terms before installation or use. T8 does not redistribute the checkpoint
or claim that the FaceXLib source-code license automatically licenses the checkpoint or its training
data.

The 19-class indices are interpreted using the ParseNet/CelebAMask-HQ mapping documented by the
[CelebAMask-HQ face parsing project](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/README.md).
This differs from the BiSeNet label ordering shown in some FaceXLib examples; the two lists must not
be interchanged.

The multi-person node includes the standard five FFHQ alignment coordinates published in FaceXLib's
MIT-licensed [`face_restoration_helper.py` at v0.2.2](https://github.com/xinntao/facexlib/blob/v0.2.2/facexlib/utils/face_restoration_helper.py).
Those five numeric points are used only as the 512x512 alignment target for OpenCV's independently
called similarity transform. T8 does not vendor FaceXLib's restoration helper, detector, warping or
restoration implementation. YuNet eye and mouth pairs are normalized by image x-coordinate before
alignment, and missing or unstable landmarks fail closed rather than being propagated.

## OpenCV Zoo YuNet and SFace models

The optional local multi-person identity-suggestion route uses OpenCV Zoo's YuNet face detector and SFace
face-recognition model through the user's OpenCV installation. SFace is distributed by OpenCV Zoo under the
Apache License 2.0; YuNet's model directory carries its own MIT notice. Model weights are installed under the
user's ComfyUI model directories and are not distributed in this repository.

OpenCV Zoo: https://github.com/opencv/opencv_zoo

Copyright 2020-2026 OpenCV

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0 . Unless
required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the License.

## ComfyUI native SAM3.1

The multi-person tracker calls the SAM3.1 implementation supplied by the installed ComfyUI version. This
repository does not copy that implementation or distribute `sam3.1_multiplex_fp16.safetensors`. Users remain
responsible for the upstream checkpoint license and applicable use restrictions.

## MiniMax H3 Fun ControlNet research and compatibility references

The append-only H3 Fun ControlNet Advanced nodes were implemented against the public native-core
design in [`Comfy-Org/ComfyUI#15860`](https://github.com/Comfy-Org/ComfyUI/pull/15860) and the
documented interaction and packed-video-row findings in
[`wyzborrero/ComfyUI-H3-FunControl`](https://github.com/wyzborrero/ComfyUI-H3-FunControl), which is
published under the Apache License 2.0. The T8 compatibility implementation is independently
integrated with this repository's current ComfyUI extension API; the community package is not a
runtime dependency.

ControlNet weights are not distributed in this repository. Users may separately obtain Kijai's
MiniMax H3 experimental ControlNet checkpoint under the terms stated on its Hugging Face model
page. Filename, byte size and fingerprint are diagnostic only: the installed ComfyUI framework
loader and the checkpoint tensor structure are authoritative. The fallback keeps the five-block
control tower dense, injects only into target-video rows and rejects Sol-Attn Morton token
reordering because it changes the row order that the ControlNet was trained against.

## RAFT and RealBasicVSR optional runtimes

The motion-audit nodes call torchvision's RAFT Small/Large implementation and load user-installed
weights from `ComfyUI/models/optical_flow`. Torchvision and its model weights remain subject to
their upstream terms; neither is copied or downloaded at runtime by this repository.

The optional RealBasicVSR node contains a dependency-light inference adaptation of OpenMMLab
MMagic's RealBasicVSR, BasicVSR and SPyNet architectures under Apache License 2.0. The detailed
notice is in [`THIRD_PARTY_NOTICES/RealBasicVSR.md`](THIRD_PARTY_NOTICES/RealBasicVSR.md). Model
weights are not bundled or downloaded at runtime, and filename, size or hash is never an execution
allowlist.

## NVIDIA Sol-Engine H3 Super Acceleration reference

The append-only H3 Super Stage-2 nodes implement the public pipeline contract documented by
[NVIDIA Sol-Engine H3 Super Acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)
and audited against the `sol-engine` branch of [`NVlabs/Sana`](https://github.com/NVlabs/Sana) at
commit `d0c0a4685ab5dc2336d18b7213d85f13def92418`. No NVIDIA model weights, LTX model weights,
Sol-Engine runtime, or CUDA kernel source is copied into this repository.

`sol_engine_taehv.py` is a reduced adaptation of [`madebyollin/taehv`](https://github.com/madebyollin/taehv)
commit `32ac0146b11007cda5a57b60a3b35653361fb8a4`, used by NVIDIA's published Stage-2 path.
It is distributed under the MIT License:

> Copyright (c) 2025 Ollin Boer Bohan
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without restriction,
> including without limitation the rights to use, copy, modify, merge, publish, distribute,
> sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: The above copyright notice and this
> permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The optional sparse-attention route interoperates at runtime with a separately installed
[`kijai/ComfyUI-SolAttn_triton`](https://github.com/kijai/ComfyUI-SolAttn_triton). T8 discovers only
an already-loaded module and does not install, download, reload, or redistribute it. When it is not
available, the LTX refiner stays dense and the node reports that fallback. Users are responsible for
the licenses and terms of all separately installed code and model assets.
