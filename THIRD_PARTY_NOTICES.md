# Third-party notices

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
