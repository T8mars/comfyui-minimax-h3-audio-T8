# Motion Context experimental operator attribution

Copyright (C) 2026 NikoDemon80.

The optional T8 MC seam operators are adaptations of `refine_blend.py`,
`join_match.py`, and `sigma_tools.py` from
[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context),
fixed commit `518c083d11d80fa5920222ad89889924ee4401c9`.
They are not represented as a clean-room implementation.

The original full GPL version 3 license and copyright notice are preserved
verbatim in [MC_MOTION_CONTEXT_GPL3.txt](MC_MOTION_CONTEXT_GPL3.txt).
The preserved file has SHA256
`ca7e33c5bd9b43b42e075f33271452aefd3d61da005738fac21127a55f878d90`.

T8 modifications dated 2026-10-21: independent opt-in execution and reports;
no implicit refine sampling; native AV metadata/container preservation;
explicit original whole-clip versus T8 keep-pin Join Match variants;
actual shape/finite-input checks; exact native cycle arithmetic without
the source's arbitrary 4096-step search limit. Existing T8 workflows,
audio, trained sigma tables, and long-video defaults are not replaced.
The source author's quality observations are not T8 validation results.

Original source SHA256 (UTF-8 raw text from the fixed commit):

- `refine_blend.py`: `6d11748b9e29219cec9cd145e2b516a8d868bf07016f835554e6ccc69fb190f1`
- `join_match.py`: `3ad13d22bbf6f700488c979f13f2604648e818f0f417f49e769dbe6737151eab`
- `sigma_tools.py`: `7900243938bfb7d521eb8138555e1fcc31ca600acadc1005a96ec7d2a0ab034e`

These adaptations are distributed under GNU GPL version 3. They come with
ABSOLUTELY NO WARRANTY, including no warranty of merchantability or fitness
for a particular purpose. See the full license for redistribution terms.
