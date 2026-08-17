# Third-party notices

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
