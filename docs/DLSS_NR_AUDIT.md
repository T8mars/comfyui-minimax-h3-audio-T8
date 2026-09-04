# DLSS-NR external runtime boundary

This feature is an optional Windows/RTX Advanced post-processing route. It is not a MiniMax H3
generator and it does not claim to improve lip sync. T8 talks to an external program through its
public command-line protocol. No `video2dlssnr` source, executable, forwarder, NVIDIA DLL, model, or
runtime archive is copied into or distributed with this project.

## Pinned audit

- Native reference: `DaniilSokolyuk/video2dlssnr` v1.3 commit
  `55a4ceb588a419b9b56497aa0b563d0c9e2b6c77`. The older v1.2 source audit remains pinned at
  `e1946117699c4e6dcd531f5e042401d04268320e` and its release tag at
  `1f49c3429e4a2f4dd62e28f09f5f21decb7bb38f`.
- ComfyUI wrapper reference: `piscesbody/ComfyUI-DLSS-NR` commit
  `4d329a864e99267734fffa2ee4b7ddeafc005c4a`.
- The wrapper contains an MIT license. The native reference has no root `LICENSE` at the pinned
  commit and relies on proprietary NVIDIA components. Its code and binaries are therefore not
  vendored, ported, compiled, or redistributed by T8.
- The full official `video2dlssnr` v1.2 and v1.3 release archives are allowlisted by exact size and
  SHA-256. The incomplete light archives remain rejected. v1.3 is the default/recommended runtime
  because it fixes the v1.2 2x/3x SR-mode selection and exposes explicit SR presets.

## User-supplied layout

After obtaining the runtime yourself and accepting every applicable upstream and NVIDIA term, use:

```text
ComfyUI/models/DLSS-NR/1.3/
  t8-runtime-manifest.json
  video2dlssnr_release.zip
  bin/
    video2dlssnr.exe
    nvngx_dlss.dll
    nvngx_dlssnr.dll
    nvngx.dll_dlssnr.dll
```

Nested paths may differ if the manifest names them exactly. Every manifest path must be a relative
POSIX path inside the version folder. Absolute paths, `..`, backslashes in the manifest, symbolic
links, duplicate paths, unknown fields, missing files, and unknown versions fail closed. The
official v1.3 ZIP stores member names with Windows separators; those verified archive names are
canonicalized to POSIX only for exact member lookup after traversal/collision validation.

The executable, DLSS SR DLL, DLSS NR DLL, and forwarder must all resolve to one directory. This
binds the byte-verified files to the exact directory searched by the audited executable instead of
allowing a verified DLL in one directory and a different loadable DLL beside the executable.

Keep the original full official archive in the version folder. T8 first checks its built-in release
size/SHA-256 allowlist, opens it read-only without extraction, and then requires each installed
EXE/DLL to be byte-identical to the corresponding member of that verified archive. This prevents a
renamed or substituted DLL from passing without embedding proprietary inner files in this project.

Use [`examples/runtime-manifests/dlss-nr-v1.3.json`](../examples/runtime-manifests/dlss-nr-v1.3.json)
for v1.3 or [`examples/runtime-manifests/dlss-nr-v1.2.json`](../examples/runtime-manifests/dlss-nr-v1.2.json)
for the older release. Change only the relative installed/member paths if a separately allowlisted
official archive uses another layout.

## Runtime Audit behavior

`static_only` performs no media work. It checks Windows, driver 616.56+, at least 512 MiB currently
free VRAM, the selected NVIDIA/Torch CUDA device mapping, manifest, archive, and runtime files. A
successful static audit deliberately returns `STATIC_PASS_REAL_PROBE_REQUIRED`, not ready.

`feature_probe_1_frame` is explicit and launches the verified executable once with a 64x64
feature-creation/evaluation probe. It uses an argument array, never a shell command. The probe must
exit successfully and report the same adapter selected by the host checks before the result can be
`READY`. Route F must be reported explicitly because the media paths use the forwarder even when
other probe routes happen to work. Exact normalized adapter names must match. A bounded 512 MiB total-memory tolerance accounts
for the normal difference between `nvidia-smi` physical VRAM and DXGI's WDDM-reserved figure while
still rejecting the 8 GB and 16 GB variants of an otherwise identically named GPU. If more than one
installed GPU shares the probed name and memory signature, the audit and execution revalidation
fail closed because the upstream CLI does not expose a PCI/LUID identity for the DXGI adapter.

The node never downloads a runtime, accepts a license for the user, installs a package, changes a
driver, unloads every ComfyUI model, processes source media during a static audit, or silently falls
back to bilinear scaling.

## P1 execution contracts

The three media nodes passed the fixed-material P2-P4 gates and are released as Advanced nodes. `standard` now
reproduces the reference ComfyUI wrapper's effective default parameters: style 0, intensity 1.5,
detail/color/local structure/local tone 1.0, skin/global tone -1.0 and UI correction off. Named
`max_detail`, `portrait`, `night`, and `light` profiles plus a strictly validated `custom` profile
are available. Named profiles ignore manual widgets so a saved profile cannot silently drift.

- Image accepts finite BHWC SDR values in 0..1. Every batch item is a separate still run, so no
  temporal state crosses images. RGB is rounded to an 8-bit RGBA PNG bridge. Source alpha and any
  further channels are bilinear-resized and reattached instead of being silently dropped. Image
  helpers are polled for ComfyUI cancellation and are terminated/reaped immediately on interruption.
- Video Frames uses one persistent helper for the entire ordered batch. Raw RGBA input/output is
  bounded by four-frame queues. It keeps frame count, order and fps metadata, performs no frame
  interpolation, and returns the exact same optional `AUDIO` object. Reports contain both the
  requested and the resolved `nvof`/`lk` backend; an explicit `nvof` request may not silently fall
  back to LK.
- Video File accepts one untrimmed, file-backed, square-pixel SDR/8-bit/Rec.709-compatible CFR video
  stream. FFprobe and decoded PTS must agree on CFR. Rotation, crop metadata, HDR/wide gamut,
  high-bit-depth input, changing dimensions and odd target sizes are rejected. It streams frames to
  one helper, encodes a video-only temporary, copies the original compressed audio packets, checks
  packet payload, normalized packet PTS/DTS timeline, decoded PCM, and decoded-frame timeline
  identity, jointly decodes the result, and only then publishes with
  an atomic replace. It never creates a full `IMAGE` batch.
- Supported routes are exactly 1x NR-only and 1.5x/2x/3x SR-only or SR+NR. SR-only uses the
  upstream <=3x DLSS path with NR composition forced to zero. The upstream still evaluates NR in
  that route; reports therefore distinguish requested parameters from the effective detail=0
  composite instead of describing it as a compute-independent SR path. Custom detail is limited to
  the reference wrapper's 0..1 range. v1.3 also exposes `default` and
  `E/F/J/K/L/M` SR presets. The audited v1.2 source incorrectly forced MaxQuality for every scaled
  ratio, so v1.2 is now fail-closed for all scaled SR/SR+NR work and remains usable only for 1x
  NR-only. 4x, chained passes, HDR, VFR and any RGB bilinear fallback are rejected.
- Before a media helper starts, ComfyUI may free memory only for the selected CUDA device and the
  runtime/device/archive identity is checked again. The integration never calls
  `unload_all_models()`.

Timeout, interruption, broken pipe, nonzero exit, short/long raw output, wrong dimensions, wrong
frame count, audio drift, or strict-decode failure terminates/reaps the helper, removes temporary
files and returns no candidate.

## P2-P4 validation tools

The real gates use developer-only preparation, execution and review tools. They are validation
aids, not end-user workflows, and their existence does not mark a real gate complete.

`tools/prepare_dlss_nr_validation_inputs.py` prepares the fixed P2/P3 inputs without loading a GPU
model or starting DLSS. It requires an already reviewed 124-frame, near-0.5 MP speech source whose
audible line is confirmed as “你在哪里”. It copies that source byte-for-byte, extracts frame 61 as
the P2 image, joins 62 frames from each of two separately generated H3 clips as an intentional hard
cut, and overlays fixed small text, one-pixel lines and a two-pixel checkerboard on the speech clip.
The output is staged and strict-video-validated before one atomic directory publish; an existing
target is never overwritten.

The current hash-bound input bundle is
`artifacts/dlss-nr-validation-inputs-20260903/validation_inputs.json` (SHA-256
`9BA1B6E9907EFE69869D5D51B83395DA3A18F50DF013CDA3F88ED0301D935206`). Its speech source is
960x544, 124 frames at 24 fps with one decodable AAC stream and remains byte-identical to the
reviewed H3 source. The prepared hard cut is at frame 62 and is the strongest transition at the
96x54 screening scale (mean absolute delta 0.15044758). The fine-detail clip is also
960x544x124 and has a deterministic text/line/checker overlay. This closes only input preparation;
the manifest explicitly leaves P2, P3, P4 and automatic promotion false.

To reproduce it in a new evidence directory:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' tools\prepare_dlss_nr_validation_inputs.py `
  --speech-video artifacts\native-masked-context-new-ema-b-classical-single-utterance-0p5mp-real-ab-20260903\20260903-011628\segment0\segment0.mp4 `
  --hard-cut-second-video artifacts\native-masked-context-color-match-v2-final-0p5mp-real-ab-20260903\20260903-122721\segment0\segment0.mp4 `
  --output-dir artifacts\dlss-nr-validation-inputs-NEW `
  --ffmpeg F:\AI-T8-video-onekey\ffmpeg\bin\ffmpeg.exe `
  --font-file C:\Windows\Fonts\consola.ttf `
  --confirm-speech-phrase '你在哪里'
```

`tools/run_dlss_nr_validation.py` first performs the selected allowlisted runtime's one-frame feature audit. If
that audit is not `READY`, it writes only `runtime_audit.json` and starts no media process. Once the
host is ready, this command runs every GPU media job strictly one after another:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' tools\run_dlss_nr_validation.py `
  --output-dir artifacts\dlss-nr-real-validation-NEW `
  --runtime-version 1.3 `
  --stage all `
  --accept-external-runtime-license `
  --input-manifest artifacts\dlss-nr-validation-inputs-20260903\validation_inputs.json `
  --input-manifest-sha256 9ba1b6e9907efe69869d5d51b83395da3a18f50df013cda3f88ed0301d935206 `
  --image artifacts\dlss-nr-validation-inputs-20260903\p2_representative_frame_061.png `
  --speech-video artifacts\dlss-nr-validation-inputs-20260903\p3_speech_960x544_124f.mp4 `
  --hard-cut-video artifacts\dlss-nr-validation-inputs-20260903\p3_hard_cut_960x544_124f.mp4 `
  --fine-texture-video artifacts\dlss-nr-validation-inputs-20260903\p3_fine_texture_960x544_124f.mp4 `
  --confirm-speech-phrase '你在哪里' `
  --confirm-hard-cut-source `
  --confirm-fine-texture-source
```

After the feature audit is READY and before any P2/P3 media run, the runner verifies the supplied
input-manifest SHA-256, its non-promotion status and operator confirmations, and every requested
input path, byte size and SHA-256. This prevents a prepared source from being silently swapped
between input review and real execution.

P2 produces the same-source RGB8 comparison set: 1x NR-only, 2x Lanczos, 2x SR-only and 2x SR+NR.
Each DLSS run receives a fresh runtime/archive/device revalidation and a 100 ms `nvidia-smi` VRAM
trace. The report records elapsed time, minimum free/peak used VRAM, hashes, exact dimensions,
8-bit PNG output and RGB quantization.

P3 requires a near-0.5 MP, exactly 124-frame speech clip whose audible line has been checked as
“你在哪里”, plus a confirmed real hard cut and a confirmed subtitle/fine-texture clip. It checks
strict CFR decode, exact frame count/fps/duration, source audio packet and decoded-PCM identity,
black/frozen regressions and a thumbnail-domain hard-cut history screen. It then creates
`p3-video/p3_review.html`. That page must still be watched at normal speed for mouth/lip sync,
face/identity/skin, text/fine texture, color, temporal behavior, cut history and audio. Speech
content and visual quality are never inferred from the mechanical report.

After exporting `dlss_nr_p3_human_review.json`, bind it to the real validation evidence and enforce
the per-clip required fields with:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' tools\analyze_dlss_nr_p3_review.py `
  --review X:\review-results\dlss_nr_p3_human_review.json `
  --validation-report artifacts\dlss-nr-real-validation-NEW\validation_report.json `
  --output artifacts\dlss-nr-real-validation-NEW\p3_review_analysis.json
```

The P3 analyzer requires all eight review fields to be resolved for every clip. Speech must pass
overall, mouth/lip sync, face/identity/skin, color, temporal stability and audio; the hard-cut clip
must pass overall, color, temporal stability and cut history; the subtitle/fine-texture clip must
pass overall, text/fine texture, color and temporal stability. Any explicit failure, pending or
unsure required value, or failed mechanical run keeps P3 at `NOT_MET`. A fixed-material P3 pass only
allows preparation of P4; it is not a general quality claim.

For P4, first render the same hash-bound source with 2x Lanczos, RealBasicVSR `conservative`,
FlashVSR `quality_locked`, and DLSS-NR `standard`. Copy
[`examples/validation-manifests/dlss-nr-p4.template.json`](../examples/validation-manifests/dlss-nr-p4.template.json),
fill every path and SHA-256, then run:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' tools\build_dlss_nr_blind_review.py `
  --manifest X:\review-inputs\dlss-nr-p4.json `
  --output-dir artifacts\dlss-nr-p4-blind-NEW
```

The builder refuses a missing speech, hard-cut or fine-texture group; a source/candidate hash or
profile mismatch; non-2x geometry; or unequal frame count/fps/duration. It ignores candidate audio,
decodes and encodes all four candidates serially with the same one-thread `libx264`, yuv420p,
Rec.709-limited metadata and fixed CBR contract, then packet-copies and packet/PCM-verifies the
authoritative source audio.
A seeded cyclic Latin schedule assigns anonymous A-D filenames. Method/profile/path mappings and
mechanical screens remain in `blind_key.json` and `mechanical_screening.json`; neither is exposed in
the HTML. The review page requires complete viewing and records best-arm judgements plus per-arm
regressions for face/identity/skin, mouth/lip sync, text/fine texture, color, temporal stability and
hard failures. Mechanical screens never choose a winner.

After exporting the review JSON, reveal and enforce the fixed-material gate with:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
& 'F:\AI-T8-video-onekey\python\python.exe' tools\analyze_dlss_nr_blind_review.py `
  --review X:\review-results\dlss_nr_p4_blind_review.json `
  --blind-key artifacts\dlss-nr-p4-blind-NEW\blind_key.json `
  --mechanical-screening artifacts\dlss-nr-p4-blind-NEW\mechanical_screening.json `
  --output artifacts\dlss-nr-p4-blind-NEW\review_analysis.json
```

The analyzer hash-binds all three inputs and verifies every private A-D mapping, profile and
normalized-media hash. P4 can pass only when all three required clip types are assessable, all four
arms were marked completely watched, all six judgements were completed, all four mechanical
screens are clean, and DLSS-NR has no checked human regression. A pass says only
`eligible_for_p5_release_decision`; it never promotes a node automatically. Any face, identity,
skin, mouth/lip-sync, text, color, temporal or blocking regression would have kept the feature experimental.

Every output directory is immutable by convention: the tools refuse a non-empty target. A report
ending in `HUMAN_REVIEW_REQUIRED` is not a P2-P4 quality PASS and must not be used to promote the
nodes or create P5 workflows.

## Current machine

The RTX 4060 Ti 16 GB machine now has Microsoft WHQL-signed driver 616.56; `nvidia-smi`, Windows
PnP, and a ComfyUI PyTorch/CUDA smoke test pass. After explicit user acceptance of the upstream
research/educational-use restriction, the full v1.2 and v1.3 release assets were size/SHA-verified
and installed in separate version folders. v1.2 was not overwritten. Static identity validation and
the real v1.3 route-F feature probe pass.

The allowlisted release bytes are the trust boundary. The v1.2 audit found that the upstream-built
executable and forwarder were unsigned, `nvngx_dlss.dll` had a valid NVIDIA Authenticode signature,
and that release's exact `nvngx_dlssnr.dll` reported `HashMismatch` to Windows Authenticode despite
matching the GitHub asset digest and verified archive member byte-for-byte. This is not NVIDIA
endorsement; retain the user acceptance and exact-archive requirements for every version.

The first real run exposed three mock-gap fixes: official archive members include the
`video2dlssnr/out/` prefix, image outputs append `_nr.png` to the complete input filename, and DXGI
reports 16109 MiB while `nvidia-smi` reports 16380 MiB for this same WDDM adapter. Exact names plus a
512 MiB memory tolerance now pass that reservation difference while rejecting the 8 GB variant.

`artifacts/dlss-nr-v13-standard-real-validation-20260904` records the corrected serial v1.3
P2/P3 mechanical pass with the `standard` profile and default SR preset. The still matrix produced
the expected 1x and 2x 8-bit outputs. All three 124-frame videos retained exact 24 fps and audio
packet/PCM identity, showed no black/freeze regression, and preserved the hard cut. The three video
runs took 16.80/17.00/17.12 seconds with 3505/3327/3686 MiB peak total GPU use reported by
`nvidia-smi` and at least 12424 MiB free.

After the final upstream-parity audit, the additional safety and observability fixes were rerun
serially as `artifacts/dlss-nr-v13-audit-fixes-real-validation-20260904` (report SHA-256
`CD817AC29BEC6DF128F35C57CA6F2646E8F0EEFC512BBF886F851A8C8E75CE78`). The three P2 images and
three P3 MP4 candidates are byte-identical to the previous review set. The new report proves the
required route-F marker, unique GPU signature, actual NVOFA backend for all three videos, and exact
audio packet payload, packet timeline, decoded PCM, and decoded-frame timeline. After the P4
candidate and encoding additions, the focused DLSS suite is 76 tests. The last complete-project run
before those isolated P4-tool additions was 2123 tests with six existing dependency warnings; the
next full run is part of P5 after P4 human acceptance.
These results close the implementation audit findings. The exported P2 and P3 human records were
subsequently hash-bound and passed; the P3 analyzer returned `p3_fixed_material_gate=PASS`.

The earlier weak visual result was not a missing model call. It came from a T8-only profile that
combined Natural style, 0.5 intensity/detail and 0.25 final composition, plus a comparison that did
not match the wrapper's Standard preset. A separate game-scene/character rerun now shows
99.81%-99.96% of pixels changed relative to Lanczos and logs successful DLSS upscale plus NR
evaluation. The stronger outputs also have lower full-reference PSNR/SSIM on these synthetic
downsample tests, so visible extra detail must not be described as more faithful beyond the fixed
reviewed material.

`artifacts/dlss-nr-v13-final-review-20260904/review.html` is the current human gate page. It shows
the formal P2 1x NR, 2x SR-only and 2x SR+NR comparisons as fixed half splits, adds the game
scene/character calibration, and renders the three P3 Lanczos/DLSS videos as synchronized half
frames. It can export a P2 note file and an analyzer-compatible
`dlss_nr_p3_human_review.json`; neither export may be fabricated or inferred from metrics. The
accepted files have SHA-256 `FEA7496149EE22059673B0ADD90D977B8E149CB80ACA8C0745CFAADC9E4BB0C8`
and `26979CAAC2B59398F090D8042349879CE0B12697211BE8DBA401E10033D40E5F` respectively.

The real P4 package is at
`artifacts/dlss-nr-p4-blind-review-v13-final-20260904`. Its candidate manifest SHA-256 is
`939640724916200B9BDCB72A48A7FE0C807145A97E936EAA7691B308933D6C452`. All twelve normalized
1920x1088x124@24 fps arms strictly decode, retain exact source audio packet payload/timeline and
decoded PCM/timeline, add no screened black/frozen frames, and preserve the intentional hard cut.
The user completed all three A-D panels. The raw export is preserved byte-for-byte with SHA-256
`03B678911CE063DF3A70DEAD020C2D371251600735FEA44B5B247B73256D00EB`. Its `unsure` overall and
skin selections were resolved only in a separate adjudicated copy after the user explicitly said
all four were acceptable, differed mainly in skin/texture character, and had no definitive winner.
That copy has SHA-256 `D437C936769EBBED69DF3B93256008648C96C48476E1462A6ECCFA74AD285D70`;
the analysis has SHA-256 `C1B0C336F062837637FBD237AB8D75AC56F6F4A722B7B6530213D04C0F879D5C`
and returns `p4_fixed_material_gate=PASS`. The user's anonymous B/D preference is recorded but not
translated into a method preference because the A-D mapping rotates by clip.

P5 provides four independent workflows under `examples/workflows/25-dlss-nr/`: runtime audit,
image, video frames and file-backed video. Directory 24 was already assigned to MV LipSync. The
nodes are released as Advanced after the fixed-material P2-P4 gates, while all platform, license,
runtime-integrity and non-generalization boundaries remain mandatory.

The final v1.70.0 release gate passed 81 DLSS-specific tests, 130 focused tests and 2,133 full-suite
tests. Ruff, 652 delivery Python compilation checks, 274 delivery JSON parses, 203 source/user
workflow mirrors, version synchronization, diff checking and `comfy node validate` also passed.
The Registry package contains 530 files and is 2,311,484 bytes with SHA-256
`7BC370FFBE58EC65B7B365CA1F448D3A035D6EF67AA1103E8472714DAA146CF3`. All 530 package files
match the Git index byte-for-byte; an isolated extraction imports 288 unique nodes with the four
DLSS-NR nodes at the registration tail. The archive contains the four workflows and no runtime
binary, model, media, artifact, test, tool, developer document, or GitHub workflow.
