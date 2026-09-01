# LoRA and sampler verification report

This report records the historical LoRA, stable sampler, and multi-rate sampler
verification checkpoint. For the current plugin version, node inventory, and
Ref2VA still-image status, also read the project-root `README.md` and
`features.json`.

## 2026-09-01 — Fully local MV / lip-scene route

Three append-only Advanced EXP nodes now perform local CPU song-boundary analysis, deterministic
six-section Ref2VA prompt compilation, and strictly serial H3 scene rendering through the connected
ComfyUI `MODEL`. The implementation does not import a network client, submit `/prompt`, or call a
remote H3, LLM, TTS, music, or video service. Each scene uses `<Picture 1>` and `<Audio 1>`; generated
segment audio is discarded from delivery and the complete original song is muxed exactly once after
accepted video assembly. The accepted-manifest and contract hashes support resume without changing
the model, media, prompt, geometry, sampler, or seed contract.

One bounded real run used the local Ref2VA INT8 model plus the local Turbo4 EMA LoRA,
`dual_clock_euler/native_flow`, shifts `12/3`, 736×416, 124 frames at 24fps, and one scene. The local
ComfyUI prompt `a4a9aa2d-1092-4a2e-820f-1e255c34ee09` completed in about 94 seconds without concurrent
generation. The accepted state is `complete`, contains one accepted scene, records
`external_api_used=false`, and records `source_audio_policy=full_original_song_muxed_once`. The final
H.264/AAC file passes FFmpeg `-xerror` strict decode and has SHA-256
`5d9b7f6b7884bef1bff582b20080382c76b0286acc238fa4f1cbbcad48453fe4`.

After updating append-only registration expectations, the focused affected scope passed 230 tests.
A final resume audit then bound accepted scenes to the state contract, prompt, model ID, seed and
geometry; missing contract state and changed accepted fields now fail closed. Three new CPU tests cover
that boundary and non-string exact-lyrics input. The corrected five-file scope passed 101 tests and the
full main-worktree suite passed 1,931 tests with five existing Triton/PyTorch warnings. Ruff,
596 publishable Python syntax checks, 251 publishable JSON parses, `git diff --check`, Comfy node
configuration/security validation, and the CPU whitelist import gate passed. All 184 frontend
workflow JSON files are byte-identical in the project source and user workflow mirror. No additional
H3 generation or stress test was run for this closeout.

The five-timepoint contact sheet is mechanically normal. In full-speed user review, the user reported
no obvious visual problem. Lip sync is recorded separately as `ABSTAIN_NO_ASSESSABLE_VOCAL`: this
validation source contained music without spoken or clearly assessable vocal content, so it cannot
support an exact-lip-sync statement. The route remains Advanced EXP and is not a phoneme-level lip model.

## 2026-08-30 — Mask-preserving low-Sigma two-pass v4

The old low-Sigma route could start pass one with a valid nested H3 video `noise_mask`, then lose that
mask after learned spatial upscale and reopen the protected background during pass two. The new v4 plan
is append-only: v1, v2 and v3 IDs, schemas, widgets, defaults and workflows remain unchanged. Its effective
video mask is the inherited mask multiplied by spatial and temporal ownership. A one-frame static mask may
expand through latent time; mismatched dynamic masks are rejected rather than temporally interpolated.

One serial real run used 576×320 pass one, 1152×640 pass two, 124 frames, eight first-pass NFE and three
Euler updates at denoise 0.30. Strict H.264 video, AAC audio and combined decode passed. The final returned
audio tensor and decoded PCM matched the draft first pass. Against the same masked-first-pass v3 route,
mean adjacent protected-background MAD fell from 0.00169030 to 0.00080846 (52.2%); mean drift from frame
zero fell from 0.02095937 to 0.00620405 (70.4%). Minimum free GPU memory was only about 201MiB, below the
512MiB gate. These metrics support the mask-loss diagnosis but do not establish visual acceptance. Unknown
floating objects, mask edges, subject motion and generated subtitles remain full-video human-review gates.

## 2026-08-30 — Chunked two-pass global noise v2

An opt-in plan creates one full target-resolution H3 video-noise tensor. The released v1 plan and all
old workflows remain unchanged. Per-piece audio noise is zero and the executor returns the original
audio tensor. The v2 default is now explicitly `full_frame_safe + full_clip_safe`: one spatial canvas
and one complete H3 Transformer trajectory, with no final-latent spatial or temporal stitching.

The final single isolated FL2VA PDD8 run rendered 416×256 to 832×512 for 124 frames (5.1667 seconds).
It used three overlapping temporal segments and two 512×512 spatial tiles per segment. Runtime evidence
recorded one `[1,24,37,32,52]` global-noise generation, exact coordinate slicing on every piece, zero
per-piece audio noise and exact audio tensor passthrough. The 124-frame H.264 and finite 32kHz stereo AAC
passed strict video, audio and combined decode in 184.860 seconds. Peak GPU use was 15,541MiB with 569MiB
minimum free. Static samples showed no black frame or hard seam, but full-motion human review failed: the
person and background both exhibit visible low-amplitude non-rigid warping. The test face also crosses the
384–512-pixel overlap between two independently denoised spatial canvases; shared initial noise does not
make their Transformer predictions or temporal trajectories identical. This route therefore passes only
mechanical/media gates and is not quality-accepted. Ancestral/SDE internal step noise remains outside this
feature; Euler is the documented validation sampler.

The first same-seed 416×256→832×512×124 full-canvas/full-timeline run completed in 170.703 seconds,
generated the global `[1,24,37,32,52]` noise once, preserved the exact audio tensor and passed strict
H.264/AAC decode. Human review nevertheless rejected it because the person and scene widened over time.
The 736×416 source (aspect 1.76923) had been connected directly to a 416×256 canvas (aspect 1.625).
Native H3 preprocesses the first endpoint with direct resize and the last with center crop, so the graph
encoded two different geometries from one image. Their ratio predicts 8.8757% horizontal expansion;
background affine measurement found 8.719% in LOW-4 and 8.861% in the final. This proved that the
deformation already existed in pass 1 and was not caused by the learned upscaler, pass-2 tiling or MP4.

The corrected append-only workflow first center-crops the source once to 832×512, then reuses that exact
IMAGE for both endpoints of both Conditioning nodes. One serial rerun completed in 169.735 seconds and
again produced 124-frame H.264 plus finite 32kHz stereo AAC with strict combined decode and exact audio
passthrough. Tail background expansion fell to 0.0206% in LOW-4 and 0.0027% in the final; vertical change
stayed within 0.05%. Peak GPU use was 15,811MiB with only 299MiB minimum free, so the 512MiB headroom
gate and any general 16GB claim fail. The widening defect is mechanically removed; full-motion person
quality remains pending explicit human review.

A repository-wide endpoint audit found the same resize-policy risk in the dated PDD FL2VA learned-latent
4+4 workflow: its bundled 3027×1531 images were connected directly to 512×288 LOW and 1024×576 HIGH
Conditioning. First and last inputs now pass through separate center-crop `ImageScale` nodes at 1024×576
before reuse by both passes, preserving independent user-replaceable endpoints while giving both native
resize policies one shared aspect. The PDD workflow builder was updated to reproduce the committed graph
exactly, and a regression test compares the builder output with the dated frontend workflow.

Two temporal-only candidates were then run serially with a full spatial canvas and the same seed. A
fully locked overlap produced hard publication jumps about 2.8× and 4.4× its own frame-difference P95.
A guarded smoothstep takeover reduced the first hard seam but the short final segment still reached
2.84× P95. Both media files decoded cleanly, but both fail the temporal quality gate. Current ComfyUI H3
does not expose global time coordinates or a per-sigma consensus interface, so merging independent final
latent trajectories is not presented as a solved long-video method. `guarded_overlap_exp` remains an
explicit diagnostic option only; the dated workflow uses `full_clip_safe`.

Final prealignment regression: 1,890 full-project tests pass. Changed-scope Ruff, compileall and JSON
parsing pass, and the dated source/user workflow copies are byte-identical.

## 2026-08-30 — H3 compatibility batch and real FastH3 VSA route

Four compatibility-first changes were added without reordering prior node IDs or changing old workflow
defaults. H3 Audio VAE encoding disables the legacy aligned-length tail crop only for the H3 audio VAE;
Fun Control resolves the new official `MODEL_PATCH` contract and the old ControlNet contract structurally;
and an append-only long-video Sampling Plan supplies disabled, tail-subdivision and independent manual
second-pass modes while preserving Prompt Relay, EAV ownership and the bounded preview cache. Focused
source, schema and frontend regression checks passed before the real render.

FastH3 VSA now consumes the official VSA/Data-Free adapter's 340 applicable patches and all 50 learned
compression gates. A first short real attempt exposed that Comfy Kitchen returns `(B,T,H,D)` and H3's
output projection needs `(T,H*D)`; that run stopped at the first forward and produced no candidate. After
adding the explicit fusion and a regression test, exactly one serial isolated render was run at
832×480×124 (399,360 pixels, 5.1667 seconds), four Euler NFE and shifts 12/3. The runtime receipt reported
`comfy_kitchen_vsa_h3_90pct_tile64`, 50/50 gates, 90% sparsity and no filename/hash/byte-size gate.

The final file contains exactly 124 H.264 frames at 24fps and finite 32kHz stereo AAC. Strict video-only,
audio-only and combined FFmpeg decode all passed. End-to-end execution was 111.641 seconds; observed peak
GPU use was 15,494MiB with 616MiB minimum free on the local RTX 4060 Ti 16GiB. Sampled frames showed no
black screen, corruption or tail collapse. This is one mechanical/media pass, not a universal speed,
quality or 16GiB safety claim; complete motion and listening review remain human gates.

Evidence is retained locally under
`artifacts/community-update-real-validation-20260829/20260830-170521-fast_h3_vsa` and is excluded from Git.
The generated MP4 SHA-256 is
`CB143B873D0B085F301657326D635BE1D6A4A327E2C0E5718F3DBCFE69E042C8`.

## 2026-08-30 — LTX-2.5 low-Sigma identity-preserve Stage 2

An independent append-only setup node and frontend workflow now expose the exact schedule
`0.5 -> 0.412 -> 0.350 -> 0`: four Sigma points and exactly three Euler updates. The
official `0.909375 -> 0.725 -> 0.421875 -> 0` node and workflow are unchanged. The new
route defaults to Dense Attention so Sigma is the only changed algorithmic variable; its
optional Sol-Attn mode is explicitly EXP and holds tau at 1.0 instead of incorrectly mapping
the custom knots to NVIDIA's per-step tau sequence. H3 audio remains outside Stage 2.

One serial low-load run reused the corrected full-LTX-VAE 320x192x22 fixture and produced
640x384x17 H.264 plus 32 kHz stereo AAC in 35.96 seconds. All three model updates completed,
strict FFmpeg decode passed, and decoded audio PCM exactly matched the official-schedule
control. In the sampled first/middle/final frames, the low-Sigma candidate remained visibly
closer to the H3 source face than the official full-denoise control. This is a one-short-clip
smoke observation, not a universal identity, high-resolution, multi-person or memory-safety
claim; the optional Sol-Attn mode was not part of this run.

## 2026-08-29 — NVIDIA H3 Super Acceleration Stage-2 validation

Correction on 2026-08-30: the original workflow incorrectly used TAEHV Encode as the
LTX Refiner input. NVIDIA keeps the full LTX-2.5 Video VAE encoder because the Refiner
was trained on its latent distribution; TAEHV belongs only at the final decode. The
2026-08-29 small mechanical run is therefore invalid as a Stage-2 quality/parity proof.
The corrected workflow uses `VAEEncode` with `ltx-2.5-video-vae-conv-bf16.safetensors`
before `LTXVLatentUpsampler` and retains TAEHV only after the Refiner.

The corrected route was then executed in an isolated ComfyUI process on 2026-08-30.
The low-load smoke used 17 frames at 640x384 and 24 fps. Runtime logs confirmed that
ComfyUI loaded `VideoVAE` before `LatentUpsampler`, followed by the exact three Euler
updates and final TAEHV decode. The resulting H264/AAC file passed strict video and audio
decode, contained all 17 frames, and retained 32 kHz stereo source audio. Contact-frame
inspection found a coherent subject across the clip with no first-frame-only collapse.
This is a corrected-chain mechanical validation, not a claim of NVIDIA's 4xGB200 speedup
or a full-resolution perceptual benchmark.

The original five append-only H3 Super nodes were audited against `NVlabs/Sana` sol-engine commit
`d0c0a4685ab5dc2336d18b7213d85f13def92418`. The corrected Stage-2 route uses the full
LTX-2.5 Video VAE encoder, the official x2 LTX latent upscaler, LTX-2.5 Dev with distilled LoRA strength 0.8, CFG 1, and
the three Euler updates `0.909375 -> 0.725 -> 0.421875 -> 0`. H3 audio bypasses LTX and is
trimmed only to the kept video duration.

One serial low-VRAM real run used a 320x192, 22-frame H3 draft and the official Lightricks
Comfy INT8 ConvRot Dev Transformer/text encoder. The 8n+1 policy kept 17 frames; Stage 2
completed all three updates in a 41.29-second prompt and emitted 640x384 H.264 plus the
original 32kHz stereo AAC. Strict decoding passed for both streams, and the first, middle,
and final frames were visually coherent. This is a small mechanical validation only, not a
1080p benchmark or a claim that INT8 reproduces NVIDIA's fixed 4xGB200 BF16 result.

## 2026-08-29 — Official ComfyUI native PDD FinalLayer compatibility

ComfyUI PR #15908 was validated against official core commit
`e7051b03758a1247e3adb84a5b784ffacb9a23bd`. The merged contract differs from
the PR's early `set_weight/set_bias` draft: it uses a schedule-aware native
FinalLayer, shape-changing padded `diff` patches, and ModelPatcher handling for
resized weight and bias tensors in both normal and low-VRAM load/unload paths.

`MiniMaxH3PDD8StepSetupT8Advanced` now probes those runtime semantics. When
present, all 258 backbone adapters use ComfyUI's ordinary LoRA mapping while
the four converted-file absolute head banks are transformed in memory to the
native row-0-full-head plus row-1-to-31-offset layout. Four padded patches with
`strength_model=0` replace the base one-head tensors through ModelPatcher. An
older core continues to use the already validated dynamic-bypass and T8
final-head fallback. No ComfyUI version, model SHA-256, file size, or filename
is an execution allowlist.

Focused tests prove the native and fallback head mathematics agree for all
eight blocks at video shift 12 and audio shift 3. Both installed FL2VA and
Ref2VA files pass serial meta assembly with 258 backbone targets, four native
head targets, zero fallback hooks, nine sigma values and exact block indices
0 through 7. The complete repository passes 1,813 tests.

Two guarded real runs were then executed serially, never concurrently, at
736x416x22, Euler/simple, eight NFE, shifts 12/3 and CFG 1. FL2VA and Ref2VA
both selected the official native path and emitted exactly 22 finite H.264
frames with finite 32kHz stereo AAC; strict video, audio and combined decoding
passed. FL2VA peaked at 15,628MiB with 482MiB minimum free and therefore missed
the project's 512MiB comfort margin. Ref2VA peaked at 15,477MiB with 633MiB
minimum free and passed that margin. These are short compatibility renders,
not stress, quality-superiority, repeated-use, or universal 16GiB evidence.

## 2026-08-27 — Alibaba PAI PDD 8-step integration checkpoint

Two additional importable frontend workflows now compose PDD with the learned 3D latent upscaler for
FL2VA and Ref2VA. They do not run PDD 8+8. One official nine-value sigma trajectory is split at index
4: LOW consumes blocks 0 through 3, pass-1 `denoised_output` is learned-upscaled, HIGH rebuilds the
geometry-specific dual-clock sampler, and PASS 2 consumes blocks 4 through 7. Total joint AV Transformer
work remains eight model forwards. HIGH Conditioning is rebuilt at the upscaler's actual dimensions and
the native `legacy_policy` audio continuation remains active.

One requested low-load Ref2VA smoke ran serially under `--lowvram`, with no repetition or stress test:
256x256x22 LOW to 512x512x22 HIGH. The server log contains two completed four-step sampler phases and the
published combined output strictly decodes as exactly 22 H.264 frames at 24fps plus finite 32kHz stereo
AAC. Audio peak is `0.830137` with zero decoded samples at or above `0.999`. The six-frame contact sheet is
visually coherent. The first validator revision failed only while reading a changed report key after the
video had already been saved; the completed media was recovered and checked without rerunning inference.
This proves the mechanical Ref2VA 4+4 handoff at the small smoke contract only, not FL2VA quality,
full-duration stability, production-resolution VRAM safety or superiority over single-pass PDD.

A second user-requested one-shot Ref2VA check used 864x480x22 LOW (414,720 pixels, about 0.4MP)
with `scale_by=1.5`. The learned upscaler preserved the source aspect as closely as its 32-pixel
alignment permits and therefore produced 1312x736x22 HIGH (965,632 pixels), with effective scale
`1.525908` and anisotropy `1.009756`. The complete 4+4 run finished in 70.922 seconds. Strict decode
passes for exactly 22 H.264 frames and finite 32kHz stereo AAC; audio peak is `0.590970` with zero
samples at or above `0.999`. Peak GPU use was 15,476MiB and minimum free VRAM was 634MiB, passing the
512MiB project gate by only 122MiB. The contact sheet is visually coherent. This was one serial run,
not a repeat or pressure test, so it establishes only this exact short contract on this machine.
The user reviewed this result and reported no issue. The matching 864x480x22, `scale_by=1.5`
Ref2VA frontend workflow is therefore published as the formal Stable PDD two-pass preset. FL2VA
two-pass remains Advanced EXP because it has not received an equivalent real render and human review.

The converted FL2VA and Ref2VA adapters were re-hashed after installation into `models/loras`:

- FL2VA: `95b79e73dbad645f4f4ccd7fb8c5d864e7b978022a4c372f8cfaba82d3ff40bf`, 1,658,719,696 bytes;
- Ref2VA: `f4522e368ad7da1af19a283a728fbeb1f2b18866569ef9169b73786c3d69e4d2`, 1,658,719,704 bytes.

Each file contains exactly 778 tensors: 258 complete LoRA A/B/alpha triples and four 32-interval
video/audio head-bank tensors. Current-Comfy CPU/meta integration maps all 258 adapters, creates 258
dynamic bypass hooks, verifies 206 rank-64 plus 52 rank-192 fused-QKV adapters with alpha/rank 1,
patches one PDD final layer, and installs one diffusion wrapper. The native 8-step schedule matches
the official PDD boundaries with maximum error `2.50966925e-08`, selects blocks 0 through 7 exactly,
and is bit-identical to this project's native-flow sigma helper.

Eleven focused PDD tests cover plan normalization, schedule refusal, interval fusion, dtype normalization,
variant-strict header validation, head selection, the PDD-only file picker, normal/failed-injection offload,
current-Comfy per-token modulation rows, node schema and both frontend workflows; the separate append-only registry test also passes. The local
lifecycle wrapper restores all hooks and moves bypass-only adapter tensors to the MODEL offload device after
normal ejection or a partially failed injection, because current ComfyUI's generic bypass ejection only
restores module forwards. An isolated CPU ComfyUI launch exposes the
expected five inputs and four outputs through `/object_info`. Both real adapter files independently pass the
serial meta integration tool.

The subsequent isolated real-render gate used the matching full non-pruned INT8 ConvRot base, strength 1,
Euler/simple, 8 NFE, 12/3 shifts, CFG 1, 736x416x124 and one process at a time. FL2VA completed in
147.156 seconds with 15,663MiB peak used and 447MiB minimum free VRAM; Ref2VA completed in 139.718
seconds with 15,600MiB peak used and 510MiB minimum free. Both setup reports contain 258 mapped
adapters/hooks and exact block indices 0..7. Both outputs strictly decode as 124 H.264 frames at 24fps
plus finite 32kHz stereo AAC. FL2VA has zero decoded samples at or above 0.999 absolute; Ref2VA has
0.01395% at or above that diagnostic threshold and requires listening rather than an automatic quality claim.
Both runs therefore pass setup/media mechanics but fail the fixed 512MiB residual-VRAM gate. The later
visual review was accepted; listening remains independent, and no equal-contract speed control or repeated-use
test was run.

A requested higher-resolution Ref2VA confirmation then ran once at 1152x640x124 (0.737MP), with all
other PDD sampling terms unchanged. It completed in 414.156 seconds, peaked at 15,610MiB used and left
500MiB minimum free, again failing the fixed 512MiB residual gate. Strict video/audio/combined decode,
exact 124-frame geometry and finite 32kHz stereo AAC all pass; decoded audio peak is 0.749909 with zero
samples at or above the 0.999 clipping diagnostic. The contact sheet visibly contains generated Chinese
dialogue subtitles even though the prompt requested no subtitles. Increasing resolution therefore does
not resolve that adherence issue; it is not a low-resolution display artefact. The user accepted the 0.7MP
visual result and did not treat the subtitle behavior as a visual hard failure; full listening remains an
independent human-review gate.

## 2026-08-25 — Skin Finish learned-proposal route stopped at the single-frame gate

The post-reveal reviewer correction for the dichromatic candidate was recorded as “basically the
same”, without rewriting the original blind form. A materially different local-model audit then
found that the installed GFPGAN v1.4 checkpoint loads through ComfyUI's bundled Spandrel, while the
installed CodeFormer checkpoint does not. GFPGAN is a generative face-restoration prior, so direct
restored-RGB or whole-face paste was excluded: that would import generated colour, facial detail and
identity risk rather than establish a skin-only finish.

An unregistered clean-room prototype used only zero-mean bounded low-frequency luminance, bounded
low-frequency chromaticity and local detail energy from the learned proposal. It retained the
source's high-frequency phase, excluded parsed facial features, stayed inside the semantic skin
mask and applied a direction-preserving 0.10 RGB delta cap both before and after affine return to the
source frame. Nine deterministic tests pass. The return-path cap was necessary: an earlier probe
showed that inverse-warped luma scale could raise an aligned 0.10 limit to 0.1324 in the full frame.

The final fixed frame-66 v6 probe reports masked mean RGB change 0.00580815, peak 0.10000002,
texture ratio 1.01476622, effective full-frame area 0.23996247 and maximum applied chroma-component
shift 0.008. One GFPGAN inference took 0.428602 seconds; GFPGAN and ParseNet were released. The
report is
`artifacts/skin-finish-learned-surface-probe-20260825-v6/calibration_report.json`, SHA-256
`2B0332A951A19C37848ACC7A9458A3BA4575AD1A6F8971D4F334C0A626E525D4`.

The guarded candidate was still visually close to source. The visibly different raw GFPGAN output
depended on the generated RGB/colour/face reconstruction that the safe route intentionally refused
to paste. The predeclared visibility gate therefore stopped the experiment before six frames, full
video, stress/repeat work, node registration or workflow integration. The 211-node registry and all
existing workflows remain unchanged; this is negative research evidence, not a released feature.

A second, materially different single-frame route then retained the proposal's non-centred broad
low-frequency RGB skin-surface field instead of tuning the first decomposition. Source RGB remained
the pixel carrier; proposal geometry, facial features and generated high-frequency detail were not
pasted. ParseNet limited the field to semantic skin, a direction-preserving 0.10 cap applied after
full-frame return, and a pinned local SFace model provided a separate same-frame safety gate. Nine
focused prototype/calibration tests pass.

On the same fixed frame 66, full-frame masked mean RGB change reached 0.01792077, above the
predeclared 0.012 visibility floor. Peak change was 0.05065823, aligned texture ratio 0.99734342,
new clipping zero and mask exterior bit-exact. The result nevertheless remained only subtly
different on direct inspection, and SFace source/candidate cosine was 0.77901375 against the
predeclared 0.90 gate. The route therefore records `ABSTAIN` without lowering the threshold or
running six frames. Evidence is
`artifacts/skin-finish-learned-rgb-surface-probe-20260825-v1/calibration_report.json`, SHA-256
`7F26BC07C3306CC4BED20609B8E6617834E97F307B30FA9216ED3D2FD4EF590A`.

A third representation split the proposal into a 12-pixel broad surface, a 2-to-12-pixel middle
band and excluded generated fine detail. It used the official OpenCV SFace cosine threshold 0.363
plus a fixed 0.20 project safety margin, and additionally required the guarded candidate to improve
on the raw GFPGAN proposal by at least 0.02. The fixed frame passed those gates: full-frame masked
mean RGB change 0.02354103, peak 0.10000002, source-fine cosine 0.98669648, texture ratio
1.00310755 and full-frame SFace cosine 0.93614626 versus raw-proposal 0.84714699. The report is
`artifacts/skin-finish-learned-mid-surface-probe-20260825-v1/calibration_report.json`, SHA-256
`737D2AA8DD6B6003AF49D4B2A560EA8755225F6AC343FED298CE6EB7F3E3FC9C`.

Mechanical passage did not establish usefulness: direct inspection of the full-resolution source
and candidate still found the surface change effectively indistinguishable. The route therefore
stopped before six frames rather than promoting a metric-only pass.

A fourth, explicitly higher-risk experiment allowed generated RGB and texture only on ParseNet
semantic skin while excluding eyebrows, eyes, nose, mouth, ears and hair. It required source and
proposal edge agreement, expanded any mismatched-edge risk before blending, capped RGB change,
and retained the same exact mask-exterior contract. Deterministic tests first exposed and then
closed a bug where source-flat pixels could admit new proposal edges without protecting their
neighbourhood.

On the fixed frame, the tentative semantic-skin fusion had masked mean RGB change 0.02308811,
peak 0.09696823, texture ratio 0.98752564, structural-gradient cosine 0.99513686, zero clipping and
exact exterior. It nevertheless failed the predeclared 0.025 visibility floor, returned the exact
source and did not continue. Consequently the saved candidate/source PNGs intentionally share a
SHA-256; they are proof of fail-closed fallback, not evidence of an accepted reconstruction. The
report is
`artifacts/skin-finish-learned-skin-reconstruction-probe-20260825-v1/calibration_report.json`,
SHA-256 `AA99162C49B5F0EE8B2BBDBACACD7EC49922B80E10544544DD0588C050B0CD87`.

Across all four representations, the only strongly visible image was the raw whole-face GFPGAN
proposal, which also changed identity-relevant eyes, iris appearance, facial detail and geometry.
That is face restoration, not a scientifically established skin-only finish. No threshold was
lowered, no parameter sweep or video run was used to force a pass, and the 211-node registry plus
all existing workflows remain unchanged.

The final low-load scope passed 79 learned-route calibration, registration and frontend-workflow
tests. Ruff, targeted `py_compile`, `features.json` parsing and `git diff --check` also passed. Four
existing Triton deprecation warnings were unchanged. No H3, SAM, CUDA, GFPGAN inference or video
generation was run during this closing regression.

## 2026-08-26 — MiniMax H3 Audio Refine bounded quality pair

Four append-only experimental nodes now cover preflight Audit, deterministic partial-tail Plan,
dual-clock Setup and a source-first Quality Gate. The old runtime node prefix at positions 0 through
210 is retained. The initial route is deliberately narrow: the connected model only, CFG 1,
`native_flow`, `dual_clock_euler`, video/audio shifts 12/3, deterministic noise, video mask zero and
audio mask one. Protected final/locked/remix audio and unknown transformer patch stacks fail closed.

One bounded quality pair used 1056x608, 124 frames at 24fps, a clear Chinese dialogue prompt, four
Turbo first-pass steps and four refine steps at `audio_denoise=0.50`. It completed in 414.14 seconds;
whole-device telemetry observed a 14,468MiB GPU peak and 1,642MiB minimum free VRAM. Original and
raw-refine H.264/AAC files both strictly decoded at 32kHz stereo. The raw candidate's decoded video
hash differed from the source, proving that a zero video mask alone is not an exact preservation
contract in the current ComfyUI path. The Quality Gate therefore defaults to the original result;
its fallback decoded video and audio hashes were exact matches to the original. After explicit
acceptance it reconstructs the output from the exact original video latent and candidate audio
latent, subject to finite/shape/rate/channel/duration checks.

Evidence is kept locally under
`artifacts/audio-refine-quality-pair-20260826/20260826-041947-c544a81e`. A separate randomized A/B
page contained no reveal mapping during listening. One reviewer initially reported that the two
sides were approximately the same while the right side was slightly quieter, then explicitly
clarified that the left side was better. Reveal maps A/left to Audio Refine and B/right to the
original, so this fixed case records a slight subjective preference for the refined arm and a lower
perceived level in the original arm. The adjudication is
`LIMITED_HUMAN_PREFERENCE_AUDIO_REFINE_KEEP_MANUAL_GATE`; the Quality Gate remains false by default
because one prompt, seed and reviewer do not establish general improvement or non-inferiority. The
ignored private adjudication record SHA-256 is
`C6EC049B71A56B2D166F0C40CE1C563DDA23D225213F8952EF843630AB071AB0`.
This single prompt/seed/reviewer does not establish equivalence, non-inferiority, transcript/voice
preservation, lip-sync preservation, general 16GB safety or superiority over a conventional
eight-step baseline. Frozen Cache remains deferred.

## 2026-08-24 — Creator and external-bridge human-review adjudication

Three current `final` reviewer exports were checked against their exact private review IDs and
SHA-bound keys, then analyzed with `tools/analyze_external_bridge_blind_review.py`. The Creator
package returned one tie; reveal maps A to separate AV decode/media composition and B to native
latent concat/one AV decode. The reviewer clarified that the first attached review package was too
unclear to determine whether there was a problem, so this is
`ABSTAIN_SOURCE_MATERIAL_INSUFFICIENT`, not equivalence or Creator acceptance.

The ClipProj 8B/Sol package returned four ties. Reveal maps the three multimodal candidate arms to
ClipProj Qwen3-VL 8B and their controls to native Qwen3-VL 32B. The review notes report broken faces
on both sides for I2VA/FL2VA/Ref2VA, with broken pseudo-subtitles and/or overall picture quality in
I2VA/FL2VA. These shared defects do not establish 8B noninferiority. The fourth pair maps A to
Scheduled Sol-Attn at 5,139 tokens and B to native dense attention; it tied without a failure note,
which is insufficient for quality or performance promotion.

The same-input ClipProj 4B package returned two ties. Reveal maps 4B against 8B and 4B against native
32B, respectively. The reviewer clarification identifies the third attached review package as too
unclear for reliable judgement, so both are treated as source-material-insufficient abstentions,
not 4B/8B/32B equivalence. The immutable source/analysis hashes and this interpretation boundary are
recorded in ignored local evidence at
`artifacts/blind-review-human-adjudication-20260824/report.json`. Native 32B remains the default;
Creator auto-accept, ClipProj default replacement and Sol promotion remain denied.

The review builder and analyzer now support an explicit per-pair assessability field with
`assessable`, `source_material_insufficient`, `playback_problem` and `unsure`. Only assessable pairs
contribute preference or blocking-failure counts. The other states produce explicit abstention
decisions and cannot satisfy the promotion panel gate. Legacy v1 exports remain count/decision
compatible by assuming assessable while marking that assumption separately; rebuilding an existing
keyed package no longer overwrites its review page.

Replacement evidence now uses a bright, frame-filling red mechanical metronome on white instead of
the dark rain material. A Creator probe samples one 256x256x22 latent at eight NFE, reuses that exact
latent twice, and compares native 22+22->39 AV-latent concat followed by one VAE decode against one
source VAE decode followed by exact 5-video-frame/7200-audio-sample media composition. The isolated
run completed in 35.125 seconds, peaked at 15,114 MiB with 996 MiB minimum sampled free VRAM, and
returned below its baseline. Both review arms strictly decode as 39-frame H.264 plus 32 kHz stereo
AAC; the lossless sources contain exactly 52,000 samples. An initial report failure was only the
validator reusing the source probe's 22-frame constant; the corrected 39-frame contract and four
focused tests pass without rerunning the model. Human seam/audio preference remains pending at
`artifacts/creator-clear-av-runtime-v1/20260824-003322-885d6026/blind/blind_review.html`.

The same clear prompt, seed `2608241001`, 256x256x22 geometry, eight NFE and 12/3 shifts also produced
strictly decoded 4B and 8B ClipProj arms. Their single observed times were 55.563/44.219 seconds,
peaks 14,953/14,991 MiB and minimum sampled free VRAM 1,157/1,119 MiB. An explicitly incomplete 4B
versus 8B review page exists under `artifacts/clipproj-clear-triplet-runtime-v1/`; native 32B is not
in that package. Its conservative preflight has twice observed only about 13.25-13.4 GiB free against
a 14.5 GiB gate, so no forced 32B run or three-way quality claim was made. That gate is derived from
the prior fixed native run: 15,127.725 MiB peak minus 1,156.5 MiB baseline equals 13,971.225 MiB
incremental use; adding 512 MiB required headroom gives 14,483.225 MiB, rounded up to 14,500 MiB.
The probe records this derivation and refuses CLI attempts to lower an arm below its reviewed floor.
The latest check had only 12,997 MiB free, so a run would not have had evidence-backed capacity.

The reviewer then submitted the replacement Creator and 4B-versus-8B exports with
`assessability=unsure` and the same explicit reason: one-second material without a clearly visible
human face is not sufficient. These exports are recorded as `ABSTAIN_REVIEWER_UNSURE`; their default
tie fields are not counted as preferences. The fixed replacement contract is now one SHA-locked
front-facing portrait (`10A.jpg`), I2VA, 512x256, 124 frames (5.167 seconds), eight NFE, shifts 12/3,
seed `2608245001` and a Mandarin utterance. The 4B and 8B arms each strictly decoded 124 H.264 frames
plus 32 kHz stereo AAC in 78.797/74.219 seconds. Their sampled peaks were 15,341/15,388 MiB with
769/722 MiB minimum free. The keyed 5.167-second 4B-versus-8B page is at
`artifacts/human-face-5s-clipproj-runtime-v1/blind-4b-vs-8b/review/blind_review.html`.

The same 124-frame 4B latent was reused twice for the corrected Creator comparison. Native AV latent
concat removes five repeated video frames and nine 40 Hz audio-latent steps, producing exactly 243
frames and 405 audio-latent steps (324,000 samples, 10.125 seconds). Both the one-decode candidate and
the separate-decode/media-compose control strictly decode as 512x256 H.264 plus 32 kHz stereo AAC.
The isolated run completed in 94.375 seconds, peaked at 15,534 MiB, left 576 MiB minimum sampled free
and returned below its baseline. Early, join and late frame inspection shows a clear chest-up human
face rather than black, stretched or placeholder material. This is mechanical/assessability evidence;
human seam, identity, lip, spoken-text and audio preference remain pending at
`artifacts/human-face-5s-creator-av-runtime-v1/20260824-010638-22a8e5e5/blind/blind_review.html`.
The latest native-32B preflight observed 13,809 MiB free against the unchanged evidence-derived
14,500 MiB floor, so no forced 32B run or three-way claim was made.

A no-model objective pass then strictly decoded all four replacement arms and found zero near-black,
near-white or frozen-transition frames. Creator candidate/control full-video SSIM is about 0.98785
and zero-lag audio cosine about 0.99998. At the 124-frame join, video absolute-difference means are
2.793 for one-decode and 6.656 for separate-decode composition; both audio arms move from dialogue
tail to near-silence across the join (about -61.8/-62.3 dB over the adjacent 250 ms windows), so the
5.17-second point still requires human listening. ClipProj 4B/8B full-video SSIM is about 0.92467,
while zero-lag audio cosine is only about 0.11389 and 8B RMS is about 60.2% of 4B. These metrics prove
route difference and reject gross playback material failure; they do not rank perceptual quality,
spoken-text accuracy or audio noninferiority. The report is at
`artifacts/human-face-replacement-objective-analysis-20260824/objective_analysis.json`.

For reviewer convenience, the two keyed long-material pairs are also combined into one immutable
page with independent per-pair A/B randomization and assessability fields. The builder revalidates
all three PASS reports, the common generation contract, reference-image SHA and media contracts;
its private key contains two pairs and has SHA-256
`9B8C80C2F3200DA99D58438A3F7FED3724B48A5E0C766BC75DBB226C91DAC845`. The public page does not expose
the A/B method mapping and exports one `human_face_replacement_final_blind_review.json` at
`artifacts/human-face-replacement-final-review-20260824/review/blind_review.html`.

## 2026-08-23 — Real H3 NFE boundary interruption and fresh-process resume

The earlier real probe exposed one workflow error before sampling: the plain-text H3 Conditioning
`report` output had been connected to `run_contract_json`, whose contract is a non-empty JSON
object. The sampler schema was not changed. A new final append-only
`MiniMaxH3NFERunContractT8Advanced` node now compiles the exact conditioned prompt, canonical media
map, Conditioning report and supported positive-conditioning tensor contents into deterministic
JSON. Tensor hashing is chunked; chunk size changes transfer memory and speed but not the digest.
The first 190 node IDs remain in their original positions and the compiler is node 191.

Focused compiler, NFE, real-probe, registration and workflow tests pass 31/31; changed-scope Ruff
also passes. The corrected real probe then fully SHA-256 hashed all six declared weight files and,
after rechecking 13,835 MiB free VRAM and isolated port 8197, executed one fixed
256x256x22 T2VA, four-NFE, 12/3-shift contract. The uninterrupted control completed in 18.875 s.
The second graph was interrupted exactly after progress 2/4 and its checkpoint recorded two
completed plus two remaining steps. A distinct ComfyUI process resumed the remaining two steps in
12.937 s.

The resumed and control native `samples_video` and `samples_audio` tensors are exact. Both MP4 files
pass strict video-only, audio-only and combined decode, and decoded RGB plus PCM SHA-256 values are
exact. Whole-device usage was 2,275 MiB before the isolated processes and 2,278 MiB after cleanup.
The report is local evidence at
`artifacts/nfe-resume-real-runtime-v1/20260823-220903-7aecbd23/validation_report.json`.

The refreshed local Registry candidate is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-nfe-real-resume-final42.zip`: 301 entries,
1,512,609 bytes, SHA-256
`8D8399D0ACF472D786DB036D9987B360532CAC4FCB42E9027B9E416A5D30F82F`. Official validate/pack,
zero duplicate/unsafe/excluded entries, 301/301 worktree byte parity, 133 workflows, seven Quick
Start subgraphs and isolated 191/191 unique-node import all pass; the first 190 IDs remain in place.
The real Git index remains empty and no root `node.zip` remains.

This proves only the declared model/LoRA/Conditioning/seed/sampler/kernel contract. It does not prove
mid-forward recovery, DPM++ or SDE/ancestral history, arbitrary wrappers/modalities/resolutions,
cross-GPU determinism, repeated-run memory stability, lower VRAM or universal 16 GiB safety.

The retained 1.40.0 checkpoint appends EAV + BlockCache, EAV + STG, and EAV + Long Video as nodes
146-148 while preserving the first 145 node IDs and stable sampling. It retains the isolated EAV +
Prompt Relay composer and importable Stock20 T2VA workflow from 1.39.2, the complete learned-latent
two-pass workflow set with the already-validated 4+4 schedule, and
the 1.39.0 isolated EAV + Strict Sage composer and its real 0.7MP T2VA no-fallback Sage
call/runtime/media probe, the 1.38.2 real Ref2VA and
task-Hybrid EAV pairs, the deterministic strict diagnostic decoder, and the
1.37.1 learned two-pass eight-call update and the 1.36.2 partial-start audio-clock repair. It is
validated against ComfyUI source tree `187eda8ef5e588c6a5765cad53e482765edae052`; its real
learned-latent generation used the one-key runtime entry point
`0f1fa67ad8a68b62c65ebc97a7bf485df2459c3a`. The preceding Qwen-cache/H3 compatibility probe ran
on 2026-08-14 against ComfyUI `v0.32.0-16@ddbaa8752874c275290d054ee4fddd6e004f5fdf`, while the wider
historical generation matrix below remains anchored to
`0.31.0@cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`. Historical LoRA conversion evidence was originally
recorded on 2026-08-06 against source commit `563b98eefbe643a4cd510ee7f0b43e79880d5a3f`.

## Long Video AAC process-isolation checkpoint (2026-08-23)

A full-suite rerun terminated the Windows Python process with native floating-point exception
`0xc0000093` at `long_video_delivery.py` while PyAV encoded AAC. The exact five-frame,
128x128, 32kHz-stereo input succeeded when run alone. This is evidence of a repeated in-process
native encoder-state boundary, not evidence that the candidate geometry or audio tensor was invalid;
the exception cannot be caught by Python once the host process is terminated.

Both Long Video candidate writing and final accepted composition now keep only PyAV/libx264 video
encoding in the host. Normalized planar stereo is written as interleaved little-endian float32 to a
temporary file, and a no-shell, cancellable FFmpeg child performs AAC encoding and stream-copy mux.
Child failure, Python error and Comfy interruption terminate the child and clean video, audio, mux
and log temporaries; only a successful, fsynced MP4 is atomically placed. One bounded 1024-sample
AAC pad prevents codec frame quantization from making decoded audio shorter than the absolute
manifest boundary. Composition continues to trim every accepted segment to its exact logical sample
range, so the pad is codec-only and remains below one AAC frame.

The complete Long Video Delivery module now passes 28 tests, including the formerly crashing legacy
manifest migration, fractional five-frame sample boundaries, two-segment accepted composition,
decoded final audio length, simulated child failure cleanup and raw stereo interleave. A second scope
of the real media case plus all registration tests passes 35/35. Ruff and `py_compile` pass. The full
repository suite was deliberately not repeated under the user's active low-load requirement; this
checkpoint therefore closes the observed host-process AAC crash path, not every native codec,
filesystem, FFmpeg-build, power-loss or whole-suite stability claim. FFmpeg on `PATH` is now an
explicit requirement for Long Video candidate/final delivery.

The resulting local Registry candidate is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-aac-isolation-final37.zip`: 299 entries,
1,504,589 bytes, SHA-256
`3879785D870CD3E41440B3BF01BC1C8FB01A14EB284705E4A8BCEFEA6369326A`. Registry validation,
zero duplicate/unsafe/excluded entries, exact source-byte parity and isolated 190/190-node import
all pass. Relative to `final36`, no entry was added or removed; only `long_video_delivery.py`, the
root/category README files, `features.json` and `meta.json` changed. All 133 workflow JSON files and
all seven Quick Start subgraphs are byte-identical. The real Git staged set remains empty and no root
`node.zip` exists. `final36` is retained as the historical pre-isolation package.

## ClipProj 4B asset and workflow gate (2026-08-23)

The local files named `qwen_3_4b.safetensors` were not accepted merely because their hidden width
is 2560. Their safetensors headers contain no visual, vision, merger or DeepStack tensors, proving
that they are text-only Qwen3 checkpoints and therefore unsafe for this bridge. The reviewed
ComfyUI single-file Qwen3-VL encoder was downloaded from Comfy-Org/Krea-2 revision
`e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96`; its 5,242,467,968 bytes hash to
`54BD5144DF0BBC25DD6CCADFCB826B521445A1B06AE5A42570BDD2974CA87094`, and its header includes the
visual tower plus a 2560-dimensional merger output. The v3.1 projection was downloaded from
NicoLab28 revision `2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808`; its 26,256,128 bytes hash to
`0184E5C8D666A131962506D21949C2D8A8C6F33445B7B5E347E9A7E0A5BAA819`, with W shaped
`[2560,5120]`.

A dated frontend workflow is deterministically derived from the reviewed 8B T2VA workflow. The
source 8B file remains byte-identical at SHA-256
`F3245007CAAE3868B1811A541FAC9EFD392AE2EC56D4503B84A4499619E2E862`; the 4B derivative switches
only the encoder family/file, projection and audit declaration, and uses a bounded 256x256x22 smoke
geometry. It originally recorded `ASSET_CONTRACT_PASS_RUNTIME_PENDING`; after the guarded run below,
the generator now records `ASSET_AND_SINGLE_T2VA_RUNTIME_PASS`. Twenty focused external-bridge and
frontend-workflow tests passed before the runtime-state update. The earlier insufficient-headroom
abstention remains evidence that the gate refused to compete with the user-owned service, not the
current runtime conclusion.

The follow-up adds `tools/run_clipproj_4b_real_probe.py` without changing a node, workflow or stable
sampler. Its default invocation performs only dependency, reviewed byte-size, isolated-port and GPU
headroom checks. An explicit `--confirm-run` first hashes only the experimental 4B encoder and
projection against the fixed hashes above, then repeats the port and 12,000MiB free-VRAM gate before
starting one private-database ComfyUI process on port 8197. It runs the fixed 256x256x22, four-NFE,
12/3 dual-clock graph, strictly decodes video/audio/combined streams, requires exactly 22 decoded
256x256 RGB frames plus 32kHz stereo AAC, records whole-device peak usage and terminates only its own
process. Port 8188 is observation-only.

Six focused tool tests pass; together with the reused NFE isolation infrastructure, 14 probe tests
pass, and the external bridge plus registration scope passes 47 tests. Ruff and `py_compile` pass.
An earlier no-consent preflight correctly returned `ABSTAIN_INSUFFICIENT_FREE_VRAM` at 3,360MiB free
and started no process. On 2026-08-23, a later preflight observed localhost:8188 stopped, port 8197
free and 13,649MiB free VRAM. Full encoder/projection hashes matched, then exactly one isolated process
ran seed 123456789 at 256x256x22, four NFE and 12/3 shifts. It completed in 43.812 seconds; whole-device
peak was 15,015MiB, minimum free was 1,095MiB and final use was 14MiB above the 2,461MiB baseline.
Exact 22-frame H.264 plus 32kHz stereo AAC passed strict decode, and audio contained no NaN or Inf.
The media SHA-256 is
`839442EB88BC05C5433A579BC55C0E34803AEC45DCF7979B20EB3CA80B035E5A`. Evidence is retained in ignored
artifact `artifacts/clipproj-4b-real-runtime-v1/20260823-213048-8066329d/`. This is a single mechanical
runtime PASS, not a quality, listening, speedup, memory-saving, repeat-stability or general 16GiB claim.

The probe now accepts an explicit `--seed` while retaining `123456789` as its default. A second
guarded run used seed `2608228001`, exactly matching the existing ClipProj-8B and native-32B rain
T2VA prompt, 256x256x22 geometry, four NFE and 12/3 shifts. It completed in 55.750 seconds, peaked at
15,019MiB with 1,091MiB minimum free, and returned to the 2,199MiB baseline. The exact H.264/AAC
container passed strict video/audio decode and hashes to
`45E819736A025F23BE45854FF268375480D3CDD8A9EF7283CF4E68DCBF229F06`; evidence is in ignored artifact
`artifacts/clipproj-4b-real-runtime-v1/20260823-223125-e90c8e04/`.

Those three same-input outputs are packaged as two anonymous direct comparisons at
`artifacts/clipproj-4b-vs-8b-32b-review-v1/blind-final/blind_review.html`: 4B versus 8B, and 4B versus
native 32B. The deterministic seed places the 4B arm on opposite sides in the two pairs. All four
copied AV files match the private-key SHA and strictly decode; the public page contains no source
path or side-to-method mapping. This closes the confounded-seed mechanical gap only. Human visual
and listening export, 4B multimodal behavior, repeated runs and any general equivalence or 16GiB
claim remain open.

`tools/analyze_clipproj_same_input_three_way.py` makes the comparison reproducible without loading
any model. It extracts and exactly compares prompt, seed, geometry, task/audio mode, NFE and both
shifts from all three API graphs; strictly decodes each joint-AV media file three times; preserves
the structured runtime scope; and reports all three pairwise video/audio differences. The common
contract hashes to `F836B902500CAB16925810A3471A16A6F9008908EE1A01D6341E30415213A62B`, and
every mechanical check passes.

Observed prompt-to-terminal times for 4B/8B/native-32B were 36.172/37.797/35.125 seconds, while
whole-device peaks were 15,019.0/13,926.7/15,127.7MiB. Thus this single evidence does not show a 4B
speed advantage, and 8B—not 4B—had the lowest observed peak. Mean SSIM for 4B-vs-8B,
4B-vs-32B and 8B-vs-32B was about 0.731/0.730/0.653; zero-lag audio cosine was about
0.316/0.759/0.299. These values show route differences and cannot rank perceptual quality or prove
audio noninferiority. The report is
`artifacts/clipproj-4b-vs-8b-32b-review-v1/objective_analysis.json`, 7,515 bytes, SHA-256
`33A36467CA261B55FCEBF749DFE9FB6AE8BB7F8AF745C8A888022E5C41E029D7`.

The objective-state Registry refresh is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-clipproj-objective-final44.zip`: 301 entries,
1,512,931 bytes, SHA-256
`9924AF1703DA2BEB80D2FA2642B97E5D2C0FE770ACE20024426086E6742A787C`. Official validation and pack
pass. Its entry list exactly matches `final43`, and `features.json` is the only changed member and
matches the current worktree bytes; nodes, workflows, subgraphs and stable sampling remain
byte-identical. The real Git index remains untouched, there is no root `node.zip`, and the package
is not committed or published.

The local Registry refresh is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-clipproj-comparison-final43.zip`: 301 entries,
1,512,540 bytes, SHA-256
`CB821E7035DA9CE2402D408E2508E2587742C0FF44020B746B1D6DC3BFA41126`. Official validation and pack
pass from an isolated tracked tree. Its entry list is identical to `final42`; the only changed
archive member is `features.json`, which matches the worktree byte-for-byte. Therefore all 133
workflows, seven Quick Start subgraphs, 191 runtime modules/nodes and the stable sampler remain
byte-identical to the already imported `final42` candidate. No root `node.zip` remains, the real Git
index is untouched, and this candidate is neither committed nor published.

The corresponding local Registry candidate is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-clipproj-4b-final35.zip`: 299 entries,
1,500,923 bytes, SHA-256
`F6116C88CCE332817C79B38A86C84F2683822D70FBD376540A01B30973A7CB7D`. Registry validation passes;
the archive contains 133 frontend workflows, seven Quick Start subgraphs and 190/190 unique nodes
in the same order as source and `final34`. There are no duplicate, unsafe, excluded, missing-source
or source-byte-mismatch entries. Relative to `final34`, it only appends the 4B workflow and updates
`README.md`, the system-memory category guide and `features.json`. The real Git staged set remains
empty and no root `node.zip` exists. This packaging result does not change the 4B runtime boundary.

## Creator AV Review Quick Start checkpoint (2026-08-23)

A seventh dated ASCII ComfyUI-native Quick Start subgraph now wraps the existing synchronized
Creator audio/video review workflow. It exposes two aligned videos, their labels and seeds, the
human winner and notes, the equal-geometry guard, and the silent-comparison filename prefix. Its
outputs keep A and B audio separate, expose labelled comparison frames and the deliberately silent
comparison video, and return both the human visual-review JSON and reference-relative audio-drift
report. The winner remains `ABSTAIN` by default and no automatic quality or acceptance decision is
introduced.

The generator remains deterministic. All six previously released Quick Start JSON files retain
their exact byte-level SHA-256 values; only
`subgraphs/2026-08-23_H3_Quick_Creator_AV_Review.json` is appended. The underlying dated Creator AV
workflow is source-hash locked. Twelve focused Quick Start and Creator workflow tests pass. This
checkpoint performs no GPU generation and does not convert a mechanical drift PASS into a human
quality result.

The complete low-load release audit passes 1,153 tests with four existing Triton warnings, full
Ruff, compileall over 292 non-artifact Python files, 197 non-artifact JSON parses, 132/132
project-to-menu workflow SHA parity, unchanged stable `sampling.py`, and `git diff --check` with the
existing CRLF notice. The final local candidate is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-creator-quickstart-final34.zip`: 298 entries,
1,495,056 bytes, SHA-256
`63D91180DE28E881F367B1182B98D0FBEC7BFAF1DDCB7E30B75A5B1332834B14`. Registry validation passes;
the archive has 132 workflows, seven subgraphs, zero unsafe/duplicate/excluded/missing/mismatched
entries, and isolated import returns the same ordered 190 node IDs as source and `final32`. Two
earlier attempts were rejected before promotion: one packed only the real Git-tracked 1.44 tree,
and one normalized 100 CRLF files instead of preserving source bytes. Neither is a release
candidate. The real Git staged set remains empty and no root `node.zip` remains.

## Prompt semantic contract audit checkpoint (2026-08-23)

One independent Advanced EXP node is appended as runtime node 190 after the complete prior
189-node prefix. It does not modify Prompt Provider Router, Prompt Budget, Prompt Rewriter 8B,
stable sampling, or an existing workflow. The user supplies bounded strict JSON containing
required and forbidden phrase groups scoped to the complete candidate or one H3 rewrite field.
Latin phrases use token boundaries so `turn` does not match `return`; Chinese, Japanese and Korean
phrases use normalized substring matching. Empty anchors ABSTAIN. Invalid JSON, unknown keys,
duplicate IDs, missing required groups, present forbidden groups, exact-dialogue changes and lost
source media tags fail closed.

The `safe_prompt` output remains the exact original by default. A candidate is forwarded only when
the mechanical audit passes and `accept_candidate_after_review` is explicitly enabled. The
previously captured real CPU-only Ollama 8B result that changed `turns` into `stands still` now
returns `REJECT`, with both `required_group_missing` and `forbidden_group_present`, even when the
accept switch is enabled. This proves the declared lexical contract only; it is not a universal
semantic-equivalence or prompt-quality classifier.

Twelve focused tests and the complete 1,137-test CPU suite pass. Full Ruff, compileall over 288
Python files, 196 non-artifact JSON parses, 190-node registration, 132-workflow schema and installed
menu SHA parity, unchanged stable-sampling Git blob identity, and `git diff --check` also pass. The
new Provider-to-Audit frontend workflow uses one official multiline-string source for both routes,
contains three NOTE nodes, and defaults to local passthrough plus manual review.

A clean alternate Git index produced
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-semantic-final25.zip`: 297 entries,
1,485,318 bytes, SHA-256
`5EB9D9BBE9220ACF19AB619C26E042F30DFD9124EB4682C17DC08BCE02C3F872`. It contains 132 workflows,
six Quick Start subgraphs and all 190 runtime nodes, while excluded development, model and media
paths are absent. Registry validation, source-byte parity and isolated 190/190 unique import pass;
the first 189 node IDs exactly match `final23`, and the semantic audit is the sole new final mapping.
The real Git index remains empty and no root `node.zip` remains. After this package was built, the
external-bridge blind-review builder gained immutable-key replacement protection and
`features.json` changed accordingly. `final25` is therefore a historical validated snapshot, not a
byte-identical package of the current source; the next authorized publication must rebuild and
repeat Registry/source-parity checks.

That rebuild is now complete as the local-only `final27` candidate at
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-review-evidence-final27.zip`: 297 entries,
1,485,460 bytes, SHA-256
`D253E8FC2AEDB0DB15C553A88DD07B59CC454F4EEE278DA29F8F35314CA6FF93`. It contains 132 workflows,
six Quick Start subgraphs and all 190 runtime nodes. There are zero duplicate, unsafe, excluded,
missing-source or source-byte-mismatch entries. UTF-8 Registry validation passes; isolated import
returns 190 unique nodes in exact source order, and the first 189 IDs exactly match `final23`.
The real Git index remains empty and no root `node.zip` exists. Exact archive identity is recorded
only outside the package so its metadata cannot recursively change its own hash.

After the Prompt Budget contract was corrected to distinguish the current official 7000-character
CLI submission ceiling from the local open-weight tokenizer boundary, the complete source passed
1138 CPU tests, full Ruff, 196 non-artifact JSON parses and compileall. That source change makes
`final27` historical. A separate temporary Git index then produced the local-only `final29` package
at `artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-prompt-boundary-final29.zip`: 297 entries,
1,486,285 bytes, SHA-256
`BBC60ABAE51199FA00B5FDDF188EBA87ED16F702066A0FFEE96AF46B54580889`. It contains 132 workflows,
six Quick Start subgraphs and all 190 runtime nodes. Registry validation passes in UTF-8 mode; there
are zero duplicate, unsafe, excluded, missing-source or source-byte-mismatch entries. Isolated import
returns 190 unique nodes in exact source/final27 order with zero input/output-contract differences
when display text and tooltips are excluded. The real Git staged set remains empty and no root
`node.zip` exists. Precise evidence is stored in the excluded `final29-prompt-boundary-results.json`.

One later same-input, CPU-only `deepseek-r1:1.5b` probe exposed a thinking-only Ollama envelope and
led only to a fail-closed diagnostic improvement; no node schema or default changed. The complete
source then passed 1,140 CPU tests, full Ruff, compileall over 288 Python files, 196 non-artifact
JSON parses, 190/190 unique registration, 132/132 workflow SHA parity and the unchanged stable
sampling Git blob. The refreshed local-only `final30` package is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-provider-thinking-final30.zip`: 297 entries,
1,483,238 bytes, SHA-256
`F6130C50D8974FEB255113031ADB2DFA638F23237964E4A83AE0CC7FEB43325A`. Registry validation,
security exclusions and current-source byte parity pass. Isolated import returns the same 190 IDs
in the same order as `final29`, with zero input/output-contract differences after display text and
tooltips are excluded. Only `prompt_provider_advanced.py`, `features.json` and the Prompt Relay
directory README differ from `final29`; the real Git staged set remains empty and no root
`node.zip` exists. Exact evidence is stored in the excluded
`final30-provider-thinking-results.json`.

## Dual-clock Euler NFE checkpoint/resume checkpoint (2026-08-23)

One append-only Advanced EXP sampler setup is registered as node 188 after the three RAVEN bridge
nodes. Its default `disabled` mode performs no file I/O and preserves the stable sampler path. The
only resumable contract is this project's first-order, history-free `dual_clock_euler` integration
with the `native_flow` schedule. Opt-in checkpoint mode writes after each completed joint-AV Euler
step; each no-pickle safetensors file contains the post-step packed state, original processed noise
and latent, optional packed denoise mask, and complete sigma schedule. A non-blocking same-path OS
lock, output-root path confinement, write/read verification and atomic replacement protect the
checkpoint boundary.

Deterministic CPU tests prove that the disabled implementation is bit-exact against the stable Euler
loop, a simulated exception on call three leaves only completed step two on disk, and resuming that
state finishes bit-exact against an uninterrupted four-step control. A separate process then wrote
the two-step checkpoint and exited; a second Python process loaded it and again produced a bit-exact
final tensor. Valid safetensors payload tampering, traversal, symlinks, non-finite tensors, schema,
seed, sigma, AV-layout, shift, audio-protocol, runtime-signature and explicit model/run-contract
mismatches fail closed before continuation.

This evidence does not cover DPM++ or any multistep derivative history, ancestral/SDE RNG state,
native or third-party sampler internals, an interruption inside a Transformer forward, automatic
cryptographic hashing of every loaded model weight, or real H3 CUDA restart media equivalence. The
new complete frontend workflow remains default-disabled and explains these boundaries on canvas.

The full low-load source gate now passes 1,106 CPU tests, full Ruff, compileall, 195 non-artifact
JSON parses, 188-node registration, 131-workflow structure and installed-menu hash parity,
unchanged stable `sampling.py` Git-blob identity, and `git diff --check`. A clean alternate-index
Registry archive was written to
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-nfe-final20.zip`; it has 292 entries,
1,465,447 bytes and SHA-256
`71F1AD5E71AFD64556A5A75FA648B4543509DCA77D6F51EB6C8F94A05A2ADE64`. The archive contains
131 workflows, six Quick Start subgraphs and the complete 188-node runtime, while tests, tools,
docs, artifacts, agent metadata, `roadmap.md`, `SKILL.md`, local experiment data, models and media
are absent. Registry validation passed. Isolated extraction registered 188/188 unique nodes in
source order; all first 187 IDs exactly matched the preceding RAVEN package and the NFE setup was
the sole final mapping. No root `node.zip` remains and the real Git index was not staged.

## RAVEN external integration checkpoint (2026-08-23)

Three Advanced nodes were appended after the previous 184-node prefix: one published/manual
profile emitter, one guarded external-loader delegate and one strict request audit. The integration
does not copy or replace the separately installed MIT RAVEN causal runtime, and the external
`RAVENStreamingSampler` remains in the workflow so its streaming preview and incremental VAE path
stay authoritative. Deterministic tests cover exact published values, manual deviations, pre-load
resource blocking, quantized-base rejection, exact delegation, T2VA pass-through identity,
mandatory 266-module/rank-128/alpha-128/strength-1 adapter checks, object-patch rejection and the
explicit over-192-frame acknowledgement.

The resulting source passes 1,098 CPU tests, full Ruff, compileall, 194 non-artifact JSON parses,
187-node registration, 130-workflow structure and installed-menu hash parity, and `git diff --check`.
The pinned external 0.1.0 API plus the current Comfy feature probe pass without loading weights.

A clean alternate Git index then produced a 289-entry, 1,448,039-byte Registry archive at
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-raven-final19.zip`, SHA-256
`181DB774DAA554A00CF5012245F849A03AC6D2434E229EB4D317CFEA904CF2A2`. It contains all 130
frontend workflows, six Quick Start subgraphs, the web extension and the 187-node runtime. Tests,
tools, docs, artifacts, agent metadata, `roadmap.md`, `SKILL.md`, local experiment data, archives,
model weights and generated media are absent. `comfy node validate` passed. Isolated extraction
registered 187/187 unique nodes in exact source order, all first 184 positions matched, and the
three RAVEN nodes were last. The real Git index remained empty and no root `node.zip` remains.

The external plugin and weights are not installed in the current runtime. Static upstream review
was pinned to adapter revision `bcfa38138ddf1a5041af9880760815874138d4e1` and research revision
`5a71a3cb0588ce2a9696ac23af6c78ac3f9929f3`. The local RTX 4060 Ti 16GB and approximately 128GB
host RAM are below the reviewed envelope, so the default loader correctly refuses before loading.
This checkpoint is schema/contract/workflow evidence only, not a RAVEN generation, image-detail,
streaming-latency, physical-24GB or OOM-safety result.

## 1.45.0 local creator, audit and prompt-budget checkpoint (2026-08-22)

The existing Environment Audit now preserves the exact failure stage behind KJNodes' generic
SageAttention architecture error when `sage_attention` is explicitly selected. It checks the
package and `sageattention.core` imports, the six callable symbols consumed by the installed KJ
MiniMax H3 patch, the wheel-reported `smXX` list and its match to the active GPU. The current embedded
Python exposes all six symbols and reports `sm89` for the RTX 4060 Ti. Deterministic tests separately
cover core-import failure, missing symbols, architecture mismatch and a matching contract. The audit
does not import KJNodes, load H3 or execute an attention kernel, so a pass is not output-correctness
or speed evidence.

The same read-only audit now makes the MiniMax H3 `audio_scale` compatibility boundary explicit.
Current ComfyUI `187eda8ef5e588c6a5765cad53e482765edae052` contains FLOW_AV merge
`bdcb886a4705a03cf40f4a7226de9fc7c059fc90`; native `ModelSamplingAV` reports `4.0` for video/audio
shift `12/3`. T8 stable and multi-rate custom samplers each report the intentional neutral value
`1.0`, because those samplers already maintain separate video and audio clocks and must not apply
the native carry scale a second time. The live source-contract snapshot returns
`custom_sampling_compatible=true`; deterministic tests also cover unsupported and inconclusive
states. This is a no-model/no-CUDA compatibility diagnostic, not a new sampling formula or a
quality claim. The original GitHub issue still requires the reporter's environment to update and
retest before it can be closed.

The Audio Lock Source workflow and generated Quick Audio Drive subgraph now state the actual H3
boundary: `lock_source` plus the final `mux_audio` preserves the input soundtrack, while
`add_source_as_reference`, `<Audio N>` and an optional exact `<d>...</d>` transcript only strengthen
the generative condition. H3 is not a deterministic phoneme/lip solver; exact production lip sync
still requires a dedicated post-process. This documentation correction changes no conditioning
schema, sampler default or saved-user workflow contract.

Four append-only audit/compiler nodes preserve all 163 v1.44 IDs and stable sampling. Audio Integrity Audit is a
CPU signal report for exact samples/duration, optional frame-boundary delta, non-finite values,
opening discontinuity, persistent DC-offset boundaries using 100ms context on each side, clipping,
and tail/head similarity. Speaker
Routing Audit compiles first-use dialogue speakers to explicit `<Audio N>` ordinals and abstains on
missing/duplicate references, unstructured vocalization text, or ambiguous same-descriptor routes.
Audio Perceptual Drift Audit compares synchronized same-content candidates against an accepted
reference using 500ms/100ms windows, a 24-band gain-normalized log-power envelope, level delta,
an active-audio floor and a three-window persistence gate. Prompt Budget + Role Compiler reports Unicode/UTF-8 size, a labelled planning token estimate,
optional connected-CLIP exact counts, media count/order, and subject-to-media assignments. None of
the four modifies source audio, a dialogue plan, or silently truncates prompt text.

The prompt compiler now also preserves leading/trailing prompt whitespace, reports connected but
unassigned and source-prompt-only media ordinals, and distinguishes exact compiled-text counts from
the visual/timestamp tokens inserted later by H3 Conditioning. Exact 7000-character PASS and
7001-character ABSTAIN cases preserve the input byte-for-byte; a three-subject Picture/Video/Audio
contract and explicit shared-audio path also pass deterministically. The validated ComfyUI MiniMax
tokenizer source does not impose a 7000-character guard. The default 7000 ceiling nevertheless
matches the current official MiniMax H3 CLI submission contract. These are distinct boundaries:
the official CLI submission rule is enforced by default for compatibility, but is not presented as a local
open-weight tokenizer hard limit or architectural quality guarantee. A configured ceiling above
7000 produces an explicit official-compatibility warning and cannot raise the current official CLI limit.

The connected-tokenizer path was also executed once against the existing
`qwen3vl_8b_fp8_scaled.safetensors` using ComfyUI's Qwen3-VL 8B/Boogu loader on CPU. A mixed
Chinese/English prompt containing `<Picture 1>`, `<Video 1>`, `<Audio 1>` plus one explicit
picture/video/audio role binding compiled without truncation. The node returned `PASS`, 296 Unicode
codepoints, a 153-token planning estimate and 140 tokenizer-exact `qwen3vl_8b` tokens, with no
findings or warnings. ComfyUI reported a complete 10,097.97MB CPU load and 0.93-second execution;
whole-device VRAM did not rise. This proves the real connected-tokenizer/count parser for this
encoder, not that the official 7000-character submission rule is the open-weight tokenizer's hard
limit or that other encoders tokenize identically.

Four additional append-only Creator Workspace nodes reuse the existing Studio Timeline. Their
non-destructive overlay records prompt/seed variants, media roles, retention and hold metadata;
the workspace compiler records an explicit run window and reproducible sidecar; the selector emits
the existing prompt/length/seed contract. The CPU synchronized comparison adds A/B labels, preserves
source pixels by center-padding rather than resizing, trims only to the common frame count, and can
force `ABSTAIN` on unequal geometry. These nodes never queue a prompt, load a model, write/delete
media, replace Conditioning or alter sampling mathematics.

Deterministic synthetic tests cover exact A/V boundary accounting, injected pop/DC/clipping/tail
copy findings, unique and duplicate speaker bindings, role compilation, connected tokenizer
counting, mapping overflow, and non-truncation. These tests prove implementation contracts only.
The three user-reviewed Motion Recovery WAVs were then checked at low CPU load. The first
short-window implementation falsely labelled normal speech amplitude as a DC jump; after changing
the check to compare persistent 100ms context means, `pass1_original`, `pass2_recovered_exp` and
`blend_exp` all pass the structural audit. The user-heard middle section of
`pass2_recovered_exp` that sounds distant is a perceptual timbre/reverb change and is not detected
by the structural checks. The new reference-relative drift audit was then run against the same
three 5.152-second/32kHz stereo WAVs. `pass1_original` returned PASS with zero drift;
`pass2_recovered_exp` returned ABSTAIN over 1.4-3.6s, spectral-drift p90 0.4385, level-delta p90
7.66dB and waveform correlation 0.5006; `blend_exp` returned PASS with p90 0.1318/1.83dB and
correlation 0.9861. This matches this one human label set but is not a general detector benchmark.
Transcript scoring was also corrected so Unicode scripts are not silently discarded. CJK
ideographs, kana and hangul use character-error units; accented Latin, Cyrillic, Arabic and other
scripts use Unicode word units, while mixed CJK/Latin text uses a Unicode-alphanumeric CER
denominator. Deterministic tests cover these normalization contracts and punctuation-only expected
text now fails closed. This is metric-infrastructure evidence only: no 30-case-per-language H3
generation corpus or multilingual listening gate has been completed.
The standalone multilingual validator now adds a strict v2 design audit before ASR execution. By
default each language needs 30 unique audio hashes, ten utterances, both described and clone modes,
consistent text per utterance and three distinct seeds inside each identical utterance/mode/voice
condition. It records manifest/audio SHA-256 identity,
mean/median/p90 error and the fraction of cases below the per-case threshold. `--validate-only`
checks those facts without importing Faster Whisper. Duplicate audio or an incomplete design denies
the stable gate; it is not misreported as a speech-model quality failure.
All fourteen previously generated real-H3 speech files were then transcribed with the pinned local
multilingual Faster-Whisper small model on CPU INT8 using two threads and beam size one. Both English
described-voice cases had WER 0 and the single Chinese described-voice case had CER 1/14. The eleven
English clone cases had median WER 0 but mean WER 0.7841: six were at or below the 0.15 case threshold,
one was 0.25, and four contained severe extra or non-target speech at 1.625-2.5. The validator now
emits language-by-mode, condition and sorted outlier summaries so a favorable median cannot hide
that split. The strict design and stable multilingual gates remain false: this is an unbalanced
historical sample with only one Chinese case and no three-seed repeated cells. The fixed report is
`artifacts/speech-multilingual-v2-pilot/historical-strict-asr-report-v2-breakdowns.json`, SHA-256
`C2530B91F06CD3A371CD0FEA9E54003D13849F3DCB8EC571091AFB11F91E5902`.

### Multilingual speech formal execution-plan checkpoint (2026-08-23)

`tools/build_speech_multilingual_formal_matrix.py` now closes the reproducible execution-planning
layer without submitting work to ComfyUI. The preregistered English/Chinese specification contains
ten reviewed utterances per language, described and reference-clone modes, and three fixed seeds per
cell, producing 120 deterministic API prompt files. Clone cases rotate ten local CC-BY-4.0
LibriSpeech speakers whose source paths, durations and unique content hashes are validated before a
plan is written. Chinese clone cases are explicitly cross-lingual references; the reviewed text set
is lexically varied but not phonetically balanced. This design therefore does not satisfy the
separate ten-speaker-by-ten-utterance identity or formal ABX requirement.

The plan writer is byte-idempotent and rejects existing prompt drift. The collector rejects unsafe
prefixes, missing or ambiguous output matches, undecodable/under-two-second audio, and duplicate
audio content. It writes the strict multilingual manifest only when all 120 cases resolve uniquely;
an incomplete later collection removes any stale derived manifest. Seven focused tool tests plus
four existing Unicode/multilingual tests passed, as did Python compilation and Ruff through the
available system Ruff executable. The generated `plan.json` SHA-256 is
`119735A82B59ED5F1EDDBBD68A74B5A19B8742CBECCCCCB49F32DF6338C9CBE8`. A read-only scan of the
current ComfyUI output found zero of 120 cases, with all rows `PENDING_MISSING_OUTPUT`; no strict
manifest exists, `execution_started=false`, and `stable_multilingual_gate_pass=false`. No model,
GPU, queue or running 8188 service was touched. This checkpoint proves the experiment and collection
contracts only; generation, ASR, human transcript review, speaker identity and perceptual listening
remain pending.

`tools/run_speech_multilingual_formal_batch.py` now provides the missing bounded execution entry
without weakening that boundary. It defaults to preflight-only, accepts loopback hosts only, rejects
port 8188, requires a free private port and at least 12,000MiB free VRAM, and selects one missing case
by default with a hard per-invocation limit of six. A confirmed run starts one tool-owned ComfyUI
process on the private port with dedicated user/temp directories, an in-memory database and a matrix-
local output root. It submits cases serially, atomically records every attempt, skips already collected
cases, stops after the first failure and removes only its own process and lock. Every prompt is
revalidated against the immutable plan SHA immediately before selection; source/reference identities,
standard model/input paths, `unload_all_models`, output uniqueness and stale execution state are also
fail-closed gates.

Ten CPU tests cover safe defaults, prompt drift and mandatory unload, explicit 8188 rejection before
GPU/model work, complete preflight gates, dry-run and failed-confirm behavior, serial success plus
resume without duplicate submission, fail-fast recoverable state, active-lock rejection and ambiguous
output rejection, plus startup-failure and timeout cleanup of the owned process and lock. Ruff and Python compilation pass. The real default preflight found all reviewed
files and reference hashes, a free 8197 port, no output conflict and one correctly selected pending
case. It observed the user's 8188 service and only 4,628MiB free VRAM, below the 12,000MiB gate, so it
returned `ABSTAIN_INSUFFICIENT_FREE_VRAM` without starting 8197, loading a model or submitting a case.
The formal collection therefore remains zero of 120 and no quality gate changes.

### Voice-clone identity and ABX formal-plan checkpoint (2026-08-23)

`tools/build_voice_clone_identity_formal_matrix.py` now closes the separate fixed-speaker identity
planning gap without submitting any ComfyUI work. The immutable design contains ten licensed
LibriSpeech `test-clean` targets, ten reviewed English utterances per target and three fixed seeds,
producing 300 reference-voice API prompts. The associated 90-case ABX schedule assigns three distinct
impostors and all three known seeds to every target. LibriSpeech's corpus-provided F/M label is used
only as a coarse blocking variable: the selected source set contains six M-labelled and four
F-labelled speakers, so every target has at least three same-label alternatives and no cross-label
pitch shortcut is required. The label is not treated as inferred biological sex or personal identity.

The 90 candidates are unique: nine utterances per target cover the 3-impostor-by-3-seed grid, while
the tenth utterance remains a held-out generated cell inside the complete 300-case matrix. The plan
pre-registers at least three independent reviewers, 0.80 identity accuracy, a 0.65 Wilson 95% lower
bound, at most 0.20 abstentions and at most 0.05 invalid responses. Regardless of those future
results, `high_fidelity_clone_claim` remains `NOT_ESTABLISHED` because fixed-set identity
discrimination cannot establish naturalness, acting control, consent, safety or generalization.

The plan and all 300 prompt files are byte-idempotent. `plan.json` SHA-256 is
`3D9814B221BBDC1BE5D92766140B00AF24BFF6B5BAD88004DA80B6411C33F335`; the independent
`identity_design.json` SHA-256 is
`7AC44DCAEB30FA26A0D7E193C6682CC98002F1FD20ECC85917E65946BFF224FD`. Five identity-plan CPU tests cover
the complete grid, same-label balanced impostors, prompt/source identity, compatibility with the
bounded serial executor, byte-idempotence, drift rejection, empty collection and a complete reduced
collection.

Collection is deliberately two-stage. All 300 outputs must first be uniquely matched and decoded;
only then can the tool emit a 32kHz mono FLAC standardization-job contract with no loudness
normalization. `tools/materialize_voice_clone_abx_standardized.py` is now the only reviewed execution
path for that contract. It defaults to preflight-only; explicit confirmation processes files serially,
defaults to at most ten unique files per invocation, hard-limits the invocation to 25, atomically
promotes each 32kHz mono FLAC and records resumable state after each file. It rejects input SHA drift,
unsafe paths, symlinks, untracked/stale outputs, output hash drift and concurrent execution. It writes
`abx_manifest.json` only after every A/B/X set is contract-identical and content-distinct and every
candidate is unique. Formal manifests also request target-reference A/B positions balanced per target
and globally; legacy manifests retain their previous independent-random behavior.

The complete related scope passes 37 CPU tests plus standalone Ruff and Python compilation. The
combined regression found and fixed a latent full-collection contract defect: generated manifests did
not carry `audio_sha256/audio_contract`, so standardization-job creation would have raised only after
all outputs had been collected. A reduced complete collection now exercises collection, bounded
standardization, resume, final manifest and blind-package creation end to end. This is synthetic
mechanical evidence, not a listening result. The current real collection is zero of 300 with every row
`PENDING_MISSING_OUTPUT`. A real
preflight reused the safe executor, observed the user 8188 service, selected exactly one pending clone
case and found 4,250MiB free VRAM against the 12,000MiB gate. It returned
`ABSTAIN_INSUFFICIENT_FREE_VRAM`; private port 8197 remained closed and no generation started.

The resulting local Registry candidate is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-audio-asr-final36.zip`: 299 entries,
1,501,774 bytes, SHA-256
`1072FAF336FA2022F77E1CDA7032E51E3810D6AF65DE1A1EA348946BFAB96DAA`. Relative to `final35`,
only `features.json` and the speech-workflow category guide changed; runtime sources, workflow JSON,
all 190 node IDs and their order remain exact. Registry, exclusions and source-byte parity pass.

After the formal identity standardizer and latest 4B no-consent preflight were documented, a fresh
alternate-index build produced
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-voice-standardization-final38.zip`: 299 entries,
1,507,229 bytes, SHA-256
`D4A2ABA15A87997802678D41F176F4E6E8103EF225C91DD20E439ED3941FD858`. Official
`comfy node validate` and `comfy node pack` ran under UTF-8 without touching the real Git index.
The archive has zero duplicate, unsafe or excluded paths, exact byte parity with all 299 source
files, 133 frontend workflows, seven Quick Start subgraphs and an isolated 190/190 unique-node
import whose final ID remains `MiniMaxH3PromptSemanticContractAuditT8Advanced`. Relative to
`final37`, no path was added or removed; only `features.json` and
`examples/workflows/05-speech-dialogue/README.md` changed. The package does not include tests,
tools, docs, artifacts, model weights, generated media, `roadmap.md` or `SKILL.md`; no root
`node.zip` remains. This packages the mechanical implementation and its honest status text only;
it does not convert the zero-of-300 collection or 4B resource abstention into a runtime/quality pass.
Fifty-one changed-scope speech/audio tests, full Ruff, compileall and static JSON/workflow checks pass.
A complete pytest rerun is not reported as passing because Windows terminated Python with native
floating-point exception `0xc0000093` inside the unrelated long-video PyAV MP4 writer; it was not
repeated under the user's active workload.
Real opening-pop/tail-wrap calibration, multilingual/multi-speaker listening, wider false-positive
testing and model-level causality remain unvalidated, so `PASS` is not perceptual certification and
`ABSTAIN` is not a diagnosis.

The voice-clone evidence now has a separate CPU-only blinded A/B/X packaging and analysis contract.
The builder accepts audio-only target reference, impostor reference and generated candidate files,
requires the same codec/sample-rate/channel/container contract, binds every copied file by SHA-256,
and exposes only opaque case numbers in the browser. The private key retains speaker and target-side
mapping. The analyzer merges uniquely named reviewer exports, preserves unanswered cases, and reports
per-speaker accuracy, Wilson 95% lower bounds, abstain/invalid rates and explicit design coverage. Its
default formal gate requires at least ten targets, three distinct impostors and three known seeds per
target, plus three independent reviewers. Even a passing identity panel always leaves
`high_fidelity_clone_claim=NOT_ESTABLISHED`.

Applying that contract to the historical ten-speaker pilot found a confound before any listening:
the human references were 16kHz mono while generated X clips were 32kHz stereo. The old package is
therefore not countable. Originals remain untouched; separate 32kHz-mono FLAC copies produced a new
ten-case opaque pilot page with thirty hash-verified media files. This upgraded pilot still has only
one impostor and one utterance per target, unknown seed provenance and no human export, so its formal
decision remains `ABSTAIN` and it provides no new clone-quality claim.

Creator tests cover source-object immutability, deterministic variant seeds, run-window boundaries,
hold maps, shot selection, pixel preservation, geometry abstention, frontend NOTE content and exact
link-slot wiring. A real-data API session then compiled a three-shot 22/22/39-frame timeline over
the native-latent review material, generated three deterministic shot-B variants, recorded explicit
video/audio context roles plus a five-frame hold-first map and selected variant 2/seed 2608229136.
The synchronized inputs were both `[39,256,256,3]`; the output was `[39,288,520,3]` with source
pixels preserved exactly. All eight nodes executed in 1.688 seconds, whole-device VRAM stayed at
the 1,156.5MiB baseline and the labelled MP4 passed strict video decode three of three times. The
session deliberately retained `ABSTAIN`; human usability, audio review, real-generation cancel/cache
interaction and any authorized filesystem cleanup remain pending. A later non-destructive retention
plan now closes candidate classification only, as described below.

Two further append-only Creator Runtime nodes now record explicit execution outcomes in a
hash-bound immutable ledger and compile the next `render`, `review`, `retry` or `complete` action.
They do not inspect ComfyUI history, infer cache hits, cancel a queue or touch artifact paths. An
isolated CPU-only API graph used a 73/124-frame two-shot plan, recorded shot 0 variant 0 attempt 1 as
`completed` and then `accepted`, and selected shot 1 variant 0 attempt 1 next. Requeueing the exact
same graph reported nodes 1-5 cached; changing only the base seed reported no cached nodes. This
validates ledger transitions, deterministic resume selection and ordinary ComfyUI graph-cache
identity/invalidation. It does not validate H3-internal partial compute reuse, automatic history
binding or cancellation of a live generation; those remain explicit future integration work.

A ninth append-only Creator node now compiles artifact retention decisions from the validated
workspace and immutable run ledger. It deduplicates canonical artifact manifests, applies the
per-shot retention policy and emits separate keep and proposed-delete JSON manifests. It keeps
completed-but-unreviewed candidates, rejects multiple accepted runs for one shot, rejects any
path hint present in both lists and rejects deletion candidates that do not expose an explicit
`path` field. The review-confirmation input defaults to false. Even after confirmation, the only
positive terminal state is `READY_FOR_EXTERNAL_EXECUTOR`; `files_mutated=false`,
`files_deleted=false` and `destructive_executor_included=false` remain invariant. Targeted tests
cover awaiting acceptance, winner retention, metadata-only policy, keep/delete collisions and a
rehashed invalid ledger. No test or runtime path opens, moves or deletes candidate media.

Two more opt-in nodes now bridge Creator to the existing Long Video background runtime instead of
introducing a second queue implementation. The Start node persists only a JSON-safe, 4KiB-bounded
binding containing the Creator workspace hash and run count. The selector validates that binding,
uses the durable accepted-manifest count as the next Creator run position, and maps the current
automatic retry count to a deterministic seed variant. Cross-workspace chain reuse, impossible
progress, early manifest completion and unsupported state all fail closed. `review_only` remains
the default and blocks downstream generation. If explicitly enabled, queue deletion, targeted
running-prompt interruption, history monitoring, retries, process leases and accepted promotion
remain owned by the already-tested Long Video manager and terminal. Unit/schema/frontend tests
cover this binding and selection path. A later live H3 probe, described below, closes exact
active-sampler interruption and observed release for one low-load run only; unattended acceptance
quality remains open.

An isolated CPU-only ComfyUI service then loaded both live node schemas. The default
`review_only` graph completed successfully and created no background state directory. A second
lightweight graph explicitly enabled automatic mode but deliberately omitted the required Long
Video Auto Accept terminal. Its foreground prompt completed, but the background history monitor
correctly changed the job to `failed`, retained the exact foreground prompt ID and 64-character
workspace binding, emitted the missing-terminal diagnosis and queued no continuation. This proves
the live fail-closed history hook without loading H3; it is still not a live sampler-interrupt test.

The same isolated service then loaded a temporary CPU-only output node that waited in 50ms
increments while calling ComfyUI's normal interruption check. A bound Creator prompt genuinely
reached `runtime_location=running`. The background cancel route targeted the identical prompt ID,
returned `deleted_from_queue=false` and `interrupt_signalled=true`, and history ended with
`execution_interrupted`. The durable job stayed `cancelled` with accepted/retry counts both zero;
running and pending queues were empty. The temporary node, bytecode and test chain state were
removed after validation. This proves the real PromptServer running-prompt interruption path and
its Creator identity binding on CPU.

A separate isolated localhost:8197 run then used the real Qwen3-VL 8B ClipProj path, INT8 H3
model, Turbo4 LoRA and native dual-clock sampler at 256x256x22. The bound sampler reported progress
1/4 before the cancel route targeted the exact prompt ID. History ended with
`execution_interrupted`; accepted/retry counts stayed zero, both queues emptied, and the durable
state recorded `unload_all_models` without a release error. Coarse whole-device memory observations
were 2,999MiB before submission, 10,794MiB after interruption and 3,089MiB after the bounded
release observation, a final delta of 90MiB. The isolated service and exact test-chain state were
removed afterward. This validates one live H3 cancellation/release path, not repeated stability,
media-quality parity or a universal 16GiB tier.

The follow-up control-plane probe kept the production `124..362` render-window contract instead of
weakening it for testing. It used ComfyUI's native empty MiniMax H3 AV latent, 64x64 deterministic
frames and exact stereo silence to run two legal segments: 124 frames followed by 119 new frames
after a five-frame context prefix. The first attempt exposed a real lifecycle bug: a failed job had
requested model release but retained its process chain lease, so a new job could not bind. Failed
states now release the lease and in-memory prompt snapshot only after the terminal state is durably
written. A focused regression reattaches a fresh prompt in the same manager and preserves
`previous_job_id`.

After restarting only the isolated localhost:8197 service, a fresh job reattached to the same
Creator workspace, atomically saved and accepted both candidates, queued exactly one second prompt,
persisted the parent context and candidate identity, and composed a 243-frame/10.125-second MP4.
Both observed histories completed successfully, the queue ended empty, `unload_all_models` was
recorded without error, and ffprobe found H.264 video plus 44.1kHz stereo AAC. Whole-device memory
ended at the probe baseline. This closes terminal reattachment and complete Candidate Save/Auto
Accept/composition mechanics with lightweight media; it does not claim that a cancelled real H3
sample can resume internal diffusion state, that unattended candidates have acceptable quality, or
that real H3 resumed media has passed visual/listening review.

A separate importable synchronized AV review workflow then used the same two 39-frame/256x256
candidates through native `LoadVideo -> GetVideoComponents`. The visual branch kept the existing
center-pad-without-resize contract and saved a deliberately silent 520x288/24fps/39-frame labelled
comparison; it passed strict single-thread video decode three of three times and retained
`winner=ABSTAIN`. The audio branch emitted independent PreviewAudio assets for A and B, then compared
the aligned same-content tracks with the reference-relative drift audit. It returned PASS with
waveform correlation 0.9842, spectral-drift p90 0.0735, absolute level-delta p90 0.2308dB and no
persistent section. The candidate containers differed by 25ms of AAC duration, so this controlled
probe allowed up to 42ms while preserving the node's ordinary 21ms workflow default. Output SHA-256
was `0479AC59B8A8B5A69FD018F8A04C7E035F9B8932614473ACDD07ECADE7784D17`. This closes only the
mechanical audio-preview/report wiring: PASS is not perceptual equivalence, the comparison video
contains no audio by design, and no human winner or automatic acceptance is claimed.

The same fixed 39-frame pair now has a standalone Creator synchronized-AV blind review package at
`artifacts/creator-workspace-av-ab-v1/blind`. Both source containers and their opaque A/B copies
match the private SHA-256 mapping and strictly decode complete video plus audio. The public HTML
contains no source path, method name or side mapping. A new optional display contract binds the
Creator-specific title, instructions and safe export basename into the immutable key; an optional
analysis contract similarly binds the exact scientific generalization boundary. This closed a
real report-semantic defect found during the mechanical reveal smoke test: a Creator page must not
emit the older ClipProj/Sol-specific boundary. Existing external-bridge manifests that omit the
new fields retain their previous private-key shape and idempotent rebuild behavior. Nine focused
builder/analyzer tests, Ruff, py_compile and diff check passed. The all-tie smoke export is marked
synthetic and proves only the page-to-analyzer contract; no human vote has been counted.

Because the Creator status text changes `features.json`, `final38` is now historical. A fresh
alternate-index `final39` candidate at
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-creator-review-final39.zip` contains 299
entries, 133 frontend workflows, seven Quick Start subgraphs and 190 unique nodes. It is 1,507,243
bytes with SHA-256
`866F42C6F03702EAA7F76D8144909F87D4394E003FA065B2E8FF18B73A71C4D7`. Official validate/pack,
zero duplicate/unsafe/excluded paths, all-source byte parity, and isolated 190/190 import with exact
`features.json` node order passed. Relative to `final38`, no entry was added or removed and only
`features.json` changed. The real Git index remained at zero staged files and no root `node.zip`
remains. Review builders, tests, private keys and media are excluded from the Registry archive, so
this package records the honest status without embedding evidence or converting the synthetic tie
export into a human decision.

Two pass-through external compatibility audits are also appended. ClipProj Audit requires one
separately installed ComfyUI-ClipProj 0.1.13+ tree and checks ProjectedCLIP identity, Qwen3-VL
declaration, safetensors header input/output dimensions and load mode without loading either model.
Sol-Attn Audit requires one ComfyUI-sol-attn 0.6.2+ tree and checks CUDA/BF16 architecture, complete
outer H3 patch ownership, fallback/fused markers and unreviewed wrapper conflicts without importing
or executing its kernel. Synthetic plugin and header fixtures prove these fail-closed contracts only;
no quality, speed, memory, audio or hardware-runtime claim is made.

The current upstream trees were subsequently installed without modifying their source: ClipProj
0.1.13 at `c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c` and Sol-Attn 0.6.2 at
`930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`. A real header-only ClipProj probe identified the
existing `qwen3vl_8b_fp8_scaled.safetensors` as Qwen3-VL 8B/Boogu. The pinned v3.1 matrix from HF
revision `2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808` is 41,990,896 bytes, has SHA-256
`DF0661849D0FD51DB66B0C9AA76F2C1C3EABD81B9A4745EDD2A4617AB24C87F7`, and contains `W [4096,5120]`.
The real Sol source and RTX 4060 Ti SM89/BF16 hardware passed against a synthetic complete 50-block
owner after the parser was fixed for upstream's bold Markdown version header.

A separate low-load service then ran one fixed-seed 256x256x22 T2VA probe with four NFE, shifts
12/3 and the same H3 base/LoRA controls. The 8B ClipProj route and native 32B control both passed
strict video, audio and combined-container decoding. Cold whole-device peaks were 13,926.7 and
15,127.7MiB respectively; process-private peaks were 60.44 and 69.10GiB. The 8B route therefore
used about 1,201MiB less device peak in this one short run but took about 2.7 seconds longer. This
does not establish visual quality, speed, modality-general behavior or a memory-safe tier.

A separate ClipProj 8B I2VA probe then enabled the visual contract explicitly with
`has_reference_images=true`. One real first frame entered both the H3 video-VAE keyframe path and
the projected Qwen3-VL vision path. The fixed controls were 256x256x22, four NFE, shifts 12/3 and
seed 2608228201; the prompt included Chinese text, the proper name `Lin Lan`, `<Picture 1>` and one
very short Chinese utterance. The complete run finished in 36.125 seconds and saved exact 22-frame
H264 plus 32kHz stereo AAC. Video-only, audio-only and combined-container FFmpeg decoding each
passed three of three times. SHA-256 was
`2D89E7B8535B7D0A2AE9C4E3B97E4E799650C2F45E3169D87D5EC442618EA8B8`.
The source image resized to the 256x256 model canvas versus generated frame zero measured SSIM
0.833372 and PSNR 26.7323dB; this is recorded only as an anchor-presence signal, not a quality score.
Whole-device peak was 15,207.4MiB, leaving about 1,172.6MiB of the reported 16,380MiB device total;
process-private peak was 63.43GiB and no thermal-throttle sample was observed. A same-seed native-32B
control then completed in 33.718 seconds, peaked at 15,591.9MiB and passed strict decode 3/3; its
SHA-256 was `0E91A2D693F5F8CD1818AA765A7EC330BDAA913742B9EFAD571E2769D122C57C`.
The 8B/32B videos measured SSIM 0.9293, source-to-frame-zero SSIM 0.8334/0.8394, and PCM correlation
0.8984. Both visual outputs showed an unwanted pseudo-Chinese subtitle-like overlay. No human speech
listening or ASR acceptance was performed, so these mechanical similarities do not prove quality or
spoken-text parity.

A separate ClipProj 8B FL2VA probe then connected two distinct keyframes. The first and last images
entered the H3 Conditioning `first_frame`/`last_frame` inputs, the audit kept
`has_reference_images=true`, and the prompt bound them with `<Picture 1>` and `<Picture 2>`. Fixed
controls were 256x256x22, four NFE, shifts 12/3 and seed 2608228301. The full run finished in 23.953
seconds, produced H264 plus 32kHz stereo AAC, and passed video-only, audio-only and combined-container
decoding three of three times. SHA-256 was
`F656969E71B678B710FAD4B13385474D283ABD1F3E2509434A680AE1FE9902D2`. Whole-device peak was
15,358.0MiB with about 1,022.0MiB headroom; process-private peak was about 63.87GiB, maximum sampled
GPU temperature was 57C and no thermal-throttle sample was observed. First/last anchor SSIM was
about 0.8354/0.5095 after the documented resize/center-cover transforms. These values only show that
both visual anchors influenced the path; they are not quality scores. A same-seed native-32B control
then completed in 31.078 seconds, peaked at 15,670.6MiB and passed strict decode 3/3; its SHA-256 was
`684500D393C6EDA9CFC414C69280560AF8DA668929B3C3ABEFC96FEF794D6257`. The 8B/32B videos measured
SSIM 0.7090; first-anchor SSIM was 0.8354/0.8462 and last-anchor SSIM 0.5095/0.5302. PCM correlation
was only 0.5608 and RMS differed materially (0.1994/0.0858), so no audio non-inferiority claim is
made. Long-duration interpolation, listening, 0.7MP and general 16GiB validation remain open.

ClipProj 8B Ref2VA was then compared against the native 32B encoder with the same reference image,
prompt, Ref2VA pruned INT8 model, 256x256x22 canvas, Stock20 schedule, shifts 12/3 and seed
2608228401. No Turbo LoRA or external attention wrapper was used. Both routes produced exact
22-frame H264 plus 32kHz stereo AAC and passed video-only, audio-only and combined-container decode
three of three times. The 8B route completed in 22.907 seconds, peaked at 16,318.2MiB and left about
61.8MiB headroom; its SHA-256 was
`310AFE14E2673AAD0C2D25AAA8C831287DC32CAA619082EFC972FBA463607F6B`. The native-32B route took
27.812 seconds, peaked at 16,018.2MiB and left about 361.8MiB; its SHA-256 was
`8BF6034C8AE5AF820B128E6D6A2B976EBFC71EA9305E3F5027869EBFEF5D6F49`. YuNet+SFace similarity to
the reference at frames 0/10/21 was 0.263/0.349/0.346 for 8B and 0.320/0.326/0.261 for 32B. These
three-frame means (0.319/0.302) are matching aids, not quality scores, and neither path consistently
cleared the project's usual 0.36 suggestion line. The 8B route fails the 512MiB safety gate and used
about 300MiB more peak device memory in this sample. Human review, 0.7MP, multiple references,
repeated runs and general quality/performance claims remain open.

The first Sol run used `dense_percent=0.2`; logs showed only the dense prefix and its output SHA-256
was identical to the native 32B control. With four NFE, that percentage kept all calls dense, so a
successful node execution was not proof that Sol ran. Repeating the same short probe with
`dense_percent=0.0` and mechanically lowering `min_tokens` from the production default 4096 to 256
logged `Sol active (547 tokens)` and passed strict media decoding. The four-step frontend now keeps
`dense_percent=0.0`; the 547-token probe closes only strict SM89 kernel execution.

A subsequent single fixed-seed T2VA pair used 1152x640x22 (0.737MP), four NFE, shifts 12/3, the
native 32B encoder, the same INT8 H3 base/Turbo LoRA and the production `min_tokens=4096`. The Sol
route kept blocks 0-2 and -1 dense, exact conditioning KV, FP16/BF16 QK/PV and strict mode. Runtime
logged `Sol active (5139 tokens)` on 46 of 50 blocks with no fallback. Sol and dense produced exact
22-frame 24fps H264 plus 32kHz stereo AAC; video-only, audio-only and combined-container decoding
each passed three of three times for both outputs. Telemetry durations were 49.281/41.828 seconds
and whole-device peaks were 16,004.9/16,008.7MiB, leaving about 374.6/370.8MiB headroom. The Sol
sampler-stage peak was 15,936.7MiB versus 14,588.2MiB for dense. Sol ran first and dense second, so
timing and stage peaks are cache/order-confounded; by user policy no cold/warm repetition or stress
run was added. The observed pair therefore shows neither a speed nor a memory advantage on this
RTX 4060 Ti sample. Full-video SSIM was about 0.558, PCM zero-lag cosine about 0.719 and Sol RMS was
about 1.290 times dense. These values prove that the sparse route materially changed the output;
they do not rank visual quality, motion, rain-sound fidelity or audio non-inferiority. Human blind
review, other modalities, repeated-run behavior and a general 16GiB safety claim remain open.
Output SHA-256 values were
`9EBB855A1CC5FF9183F135ECCEFFD5C637E93FFA854E986D8123F368B2DB5B4A` (Sol) and
`31252919CB4F77BD2F9BEB1D0C30FCD04ADA644AFE372FD566784DE4BBA9320E` (dense).

The three existing ClipProj 8B versus native-32B visual-reference controls and this Sol versus dense
pair were then packaged into one deterministic four-pair anonymous review page under the ignored
`artifacts/external-bridge-blind-review-v1/blind` directory. The public page contains only generic
A/B media names, the common prompt and the relevant first/last/identity references; the private key
retains source paths and SHA-256 mappings. Browser validation found four groups, eight decodable
videos, four intact references, no method-name leak and all ratings in their default tie/none state.
Synchronized muted playback advanced both videos together, and the package builder/export contract
has dedicated tests. No human vote has been submitted, so none of these routes has crossed a
perceptual-quality or audio-non-inferiority gate.

`tools/analyze_external_bridge_blind_review.py` completes the local reveal path. It requires matching
review/key schemas, the exact `review_id` and pair set, one A/B-to-control/candidate mapping per pair,
known reference metrics and valid media SHA-256 fields. It maps omitted values to ties exactly as the
page states, keeps private source media paths out of the analysis body, records the submitted review
and key hashes, and reports ClipProj and Sol-Attn separately rather than pooling unlike treatments.
An all-tie four-pair smoke export passed against the real private key and correctly kept quality,
automatic-enable and generalization claims denied.

A later evidence-lifecycle audit found one complete user export at
`D:/Backup/Downloads/external_bridge_blind_review.json`, created at 00:22:45 local time with
`review_id=clipproj-sol-20260823-v1`. The package in the same directory had been rebuilt at 00:23:51
with `review_id=clipproj-sol-20260823-final`, so the strict analyzer correctly refused to reveal the
older votes against the newer key. The ID was not edited and those votes are not counted. The
builder now constructs the full prospective key before writing media and permits an existing output
directory only when its stored key is exactly equal; changed review IDs, blind seeds, pair/source
hashes or contracts must use a new directory, and a non-empty unkeyed directory is also rejected.
Focused tests cover exact idempotent rebuild and both replacement failures.

The current `clipproj-sol-20260823-final` package was then re-audited without running H3 or using
the GPU. Rebuilding the exact manifest with blind seed `2608230101` succeeded idempotently against
the immutable stored key. The HTML contains the `final` review ID and export schema but not the old
`v1` ID. All eight A/B videos and four reference images matched the SHA-256 values of both their
source files and the private key. FFmpeg strictly decoded every video; the first six were
256x256x22 and the Sol pair was 1152x640x22, all at 24 fps with 32 kHz stereo audio as declared by
their per-pair contracts. This proves that the current package is internally valid and does not
repair or transfer the old votes. A new human export from the `final` page is still required before
any ClipProj or Sol perceptual decision can be counted.

The tenth appended node is a mechanical native H3 AV-latent timeline concat. Batch-1 segments must
share canvas and dtype, video latent time must satisfy `5n+2`, and audio time must exactly match the
rounded 24fps-to-40Hz clock. Each later segment loses exactly two video latent steps/five frames;
audio removal is calculated from cumulative global rounding rather than a fixed 8/9-step guess.
Deterministic tests cover 22+22=39 frames and three 124-frame segments=362 frames, exact mask parity
and invalid geometry. A separate low-load real probe sampled two related-prompt, independently
seeded 256x256x22 T2VA segments at Turbo8 and then performed one decode after native-grid concat.
The combined output was exactly 39 frames. Source lossless audio had 29,600 samples/37 latent steps
each; combined audio had 52,000 samples/65 steps after removing exactly nine steps/7,200 samples
from segment B. An identical-seed repeat reproduced source A, source B and combined H264/AAC MP4s
byte-for-byte. Each source and the combined file passed video-only, audio-only and combined strict
FFmpeg decode three of three times.

Compared with decoding the two latents separately and composing A plus B-from-frame-5, the one-decode
boundary video MAD was 0.04269 versus 0.11164. The one-decode lossless audio single-sample jump was
0.00104 versus 0.02231, but its adjacent 100ms RMS still changed by 8.32dB versus 8.85dB. This is a
local smoothing signal, not evidence that independent diffusion states became continuous. The two
runs completed in 72.41/70.58 seconds and peaked at 16,038.39/16,235.32MiB, leaving only
341.11/144.18MiB whole-device headroom; both fail the 512MiB project safety gate. Continuation-
conditioned quality, human review and general 16GiB safety remain open.

On 2026-08-23 a separate guarded probe changed only that graph's text path from the native 32B
encoder to the already audited Qwen3-VL 8B plus ClipProj v3.1 path. The two prompts, seeds
`2608229101/2608229102`, 256x256 geometry, 22+22-to-39-frame contract, eight NFE per segment,
12/3 dual clock, CPU concat and one final AV decode remained fixed. Before starting its private
8197 process, the tool fully checked the 10,588,637,512-byte encoder as
`4BA424CF...F66BCD` and the 41,990,896-byte projection as `DF066184...4C87F7`, then repeated the
port and 12,000MiB free-VRAM gates. The only real run completed in 86.735 seconds. Across 288
whole-device samples at 0.25-second intervals, peak usage was 15,505MiB and minimum free memory was
605MiB, passing the fixed 512MiB sampled screen by 93MiB. After the private process stopped, usage
was below its 2,203MiB baseline. Ports 8188 and 8197 were left inactive; the existing 11434 service
was observed only and remained active.

The accepted media is exactly 39 frames of 256x256 H264 with 32kHz stereo AAC and passes strict
video, audio and combined decode; its SHA-256 is
`DE9E61E44137A4C8A5FEE88DB62A8E25B1B29B4CEC8EFABB806439001D39842D`. The first validator
incorrectly required the 52,000-sample lossless tensor/FLAC length from the AAC container and
therefore returned a mechanical false negative. Read-only reanalysis established that both the
prior 32B combined MP4 (`55BB1F7...19110`) and this 8B MP4 decode to exactly 51,200 AAC samples,
while the prior lossless FLAC remains exactly 52,000 samples. The corrected validator preserves
these as two different contracts and records that no model rerun occurred. This is one fixed local
mechanical and sampled-headroom pass, not seamless-continuation evidence, an 8B/32B quality result,
repeatability, a longer-chain result or a general 16GiB safety tier. Raw evidence is under ignored
`artifacts/native-latent-clipproj-8b-real-runtime-v1/20260823-150501-92cbf70d`.

The corresponding byte-exact local Registry refresh is
`artifacts/releases/minimax-h3-audio-t8-v1.45.0-local-native-latent-8b-headroom-final45.zip`:
301 entries, 1,513,672 bytes, SHA-256
`B2A2555789649DD7C9E7EB4956CC06A8696108053D5261E72C361A68014CB13E`. Official UTF-8
Registry validate/pack pass. All 301 archive members match current worktree bytes; there are zero
duplicate, unsafe, excluded or missing-source paths, with 133 workflows and seven Quick Start
subgraphs. Isolated import returns 191 registered/unique IDs matching `features.json`, ending with
`MiniMaxH3NFERunContractT8Advanced`. Relative to `final44`, the entry list is exact and only
`README.md` plus `features.json` changed. The real Git index stayed empty and no root `node.zip`
remains. This is a local package candidate only; it has not been committed or published.

An eighteenth append-only node now closes the content-identity half of the crash-resume contract
without changing the earlier concat schema or any of the first 180 node IDs. Native Latent Resume
Manifest hashes complete H3 video/audio latent samples, optional nested AV masks and supported
non-volatile metadata in bounded CPU chunks. The checkpoint ID, shape, dtype and mask-presence
facts are compared alongside the exact content SHA-256. An empty expected manifest creates a
baseline; a supplied manifest defaults to `error` on mismatch, while explicit `report_only` returns
`MISMATCH` with `resume_verified=false`. Deterministic tests prove identical hashes at one- and
eight-MiB chunk sizes, exact-match acceptance, and rejection of sample, mask, metadata,
checkpoint-ID, malformed-JSON and unsupported-metadata changes. The node writes no file, performs
no sampling/VAE decode and excludes only its two declared volatile T8 report keys. It therefore
does not persist a latent, resume diffusion-internal NFE state, release cached inputs or prove that
an externally saved checkpoint survives a real process crash; those remain separate gates.

One final append-only Prompt Provider Router occupies registry position 178 without changing the
preceding 178 IDs or any stable workflow. Its default `local_passthrough` path returns the source
prompt exactly and performs no network request. Opt-in routes use the pinned H3 Prompt Rewriter
three-field contract with either OpenAI-compatible chat completions (OpenAI, LM Studio or
llama.cpp) or native Ollama chat. Every request requires explicit confirmation; a non-loopback
endpoint additionally requires explicit remote permission, HTTPS and an API key read only from a
named uppercase environment variable. Loopback HTTP bypasses system proxies. Embedded URL
credentials, redirects, oversized responses and invalid envelopes fail closed.

T2VA sends no image. I2VA, L2VA and FL2VA require exactly the matching first/last IMAGE inputs and
send bounded, downscaled JPEGs; raw audio is never uploaded. Strict output validation requires all
three sections in order, required and supported Picture ordinals, unchanged source `<d>` dialogue
blocks, and retained source Video/Audio tags. Ollama defaults `keep_alive=0` so the server is asked
to unload after its response. The OpenAI-compatible protocol has no standardized model-unload
operation, and GGUF remains owned by the external LM Studio/llama.cpp/Ollama service.

Deterministic tests cover both provider envelopes, confirmation order, local/remote endpoint
policy, environment-only keys, image packing, response bounds, interruption checkpoints and
malformed-output refusal. An ephemeral real loopback HTTP server also completed the actual urllib
request/response path.

A single negative real-provider probe then started an isolated Ollama 0.32.15 listener on
`127.0.0.1:11435`. `CUDA_VISIBLE_DEVICES=-1`, `GGML_VK_VISIBLE_DEVICES=-1`,
`OLLAMA_VULKAN=false` and `OLLAMA_LLM_LIBRARY=cpu` produced a server log with `library=cpu` and
`total_vram=0 B`; the existing listeners on 8188 and 11434 were not reused or stopped. The router
sent one six-second T2VA prompt to `deepseek-r1:1.5b` with temperature zero, a 768-token bound and
`keep_alive=0`. The real request completed in 25.295 seconds. The model exhausted all 768 output
tokens, omitted `overall_soundscape` and `non_diegetic_music`, and changed the exact Chinese
`<d>` dialogue block. Non-strict collection reported both findings, and re-validating the identical
response with the default strict contract raised the intended `ValueError`. Request SHA-256 was
`6F8A04E4...D394`; response-text SHA-256 was `570DEC73...0769`.

Immediately after the response, the isolated server's `ollama ps` list was empty, which is bounded
evidence that this Ollama version honored the requested unload for this one CPU model. The listener
was then stopped; 8188 remained PID 44872 and the pre-existing 11434 service remained PID 16276.
This closes one live Ollama transport/fail-closed/unload probe, not acceptable rewrite quality,
OpenAI-compatible unload, blocking-socket cancellation latency or cross-provider compatibility.
The dated importable workflow contains the complete privacy, upload and unload NOTE guidance.

A second single-request probe used the complete local `deepseek-r1:8b` Q4_K_M blob
(`6340DC32...F2BE`) under the same isolated CPU-only server. The router received all three fields in
92.392 seconds; Ollama logged 1,161 prompt tokens and 672 generated tokens with no truncation.
However, the response removed the source Chinese `<d>` dialogue entirely. Re-validating that same
response with the default strict contract therefore raised the intended dialogue-preservation
`ValueError`. Request SHA-256 was `D920DE7B...E058`; response-text SHA-256 was
`A23FC430...C4C`. `ollama ps` was again empty after `keep_alive=0`, and the isolated listener was
stopped. The stronger model improved field completeness but still did not pass the H3 contract.

The blocking-I/O boundary was then exercised with two real `ThreadingHTTPServer` loopback sockets.
A server that accepted the POST but withheld response headers exceeded a 0.1-second test timeout
and is now normalized to an actionable `ValueError` instead of leaking a raw `TimeoutError`. A
second server sent 16 response bytes, flushed, and stalled for 0.4 seconds. Using
`HTTPResponse.read1()` rather than fill-oriented `read()` returned after that first socket read;
the third ComfyUI interruption checkpoint stopped the request before 0.3 seconds and before the
remaining body arrived. No worker thread is retained by the node. Cancellation before headers or
the first response byte still cannot preempt the operating-system socket read and may wait for the
configured timeout; the UI minimum remains one second. This closes the declared real socket
timeout and streamed-chunk cancellation behavior without claiming impossible instantaneous cancel.

The two live model failures shared one root risk: a generic provider could alter or remove an
immutable source `<d>` block before the caller had any safe opportunity to recover it. The provider
route now protects every occurrence with a deterministic token containing its ordinal and a
12-hex SHA-256 prefix. Only tokens appearing exactly once inside the parsed
`integrated_multimodal_description` are restored. Missing, edited, duplicated, unresolved or
soundscape/music-misplaced tokens add findings and fail under the default strict policy. The node
does not infer a speaker, timestamp or insertion point, and the local passthrough and pinned local
8B rewriter paths are unchanged. Unit coverage also confirms the serialized provider body contains
the token but not the source dialogue text.

One final single-request CPU-only `deepseek-r1:8b` probe exercised the current guard. The request
contained one protected token and no raw Chinese dialogue. Ollama logged 1,278 prompt tokens and
753 generated tokens without truncation; the router completed in 99.915 seconds. The model again
omitted the speech and therefore omitted the token. The router reported
`protected_dialogue_tokens=1`, `restored_dialogue_tokens=0`, both the missing-token and original
dialogue findings, and rejected the same output under strict validation. Request SHA-256 was
`3C508816...4B8D`; raw/unchanged output SHA-256 was `ADBA1ADB...43F2`. The model unloaded after
`keep_alive=0`, and the isolated listener was stopped. This proves the guard does not leak or
invent dialogue when a model ignores it; a positive provider-quality gate remains open.

The router now also exposes an append-only, default-zero `contract_repair_attempts` control. A
failed candidate can be returned to the same provider for at most two deterministic text-only
repair passes. The retry payload contains the protected source, opaque dialogue tokens, validation
findings and the invalid raw candidate, but no reference-image bytes or restored source dialogue.
Every candidate is re-parsed and revalidated from scratch; exhaustion remains a hard failure under
the default strict policy. Unit tests cover one-failure-then-success, three-response exhaustion,
request-count/hash reporting, no image re-upload and exact local dialogue restoration. This closes
the retry contract mechanically, not the still-open real-provider quality gate.

A final isolated CPU-only `deepseek-r1:8b` T2VA probe used the literal Chinese source dialogue
`你在干嘛呢，我在这里呀。`, one protected token, temperature zero and a 1,024-token cap. It returned
all three fields in 81.964 seconds, preserved the token exactly once in the integrated description,
restored the original dialogue byte-for-byte, and produced zero deterministic findings. The first
response passed, so `contract_repair_attempts_used=0`. Server logs again recorded `library=cpu` and
`total_vram=0 B`; `ollama ps` was empty after `keep_alive=0`, port 11435 was stopped, and existing
8188/11434 processes were preserved. The output nevertheless changed the requested turning action
into standing still. This closes the open positive strict-contract gate, but it is not a human
semantic-fidelity pass, cross-provider result or recommendation to make this model the default.

The same source prompt was subsequently sent once to the already-present
`deepseek-r1:1.5b` Q4_K_M model through a separate CPU-only listener on port 11435. This was a
bounded cross-model negative probe, not an attempt to replace the still-deferred LightX2V 8B run:
only 31.37 GiB of host memory and 484 MiB of VRAM were free, so loading the roughly 20 GiB BF16
base-plus-adapter stack would have violated the low-resource gate. The small model received 1,272
prompt tokens and generated the full 1,024-token allowance without server-side truncation. Its
thinking-enabled chat template exposed no non-empty standard `message.content`, so the router
raised `Provider returned an empty rewritten prompt` before parsing or human review. This is the
intended fail-closed result; private reasoning is not silently promoted to a production H3 prompt.
The response parser now distinguishes this observed thinking-only envelope and recommends either
raising `max_new_tokens` or configuring the model to emit a final answer; an actually empty Ollama
response keeps the original generic error. The change adds no node input and does not expose or
return the private reasoning text.
`keep_alive=0` again left zero loaded Ollama models, port 11435 was stopped, the existing 8188 and
11434 PIDs were preserved, and GPU allocation stayed at 15,626 MiB. The result closes one
same-input compatibility boundary but is neither a cross-provider comparison nor a semantic
quality result.

The authoritative Git history also corrected one stale planning record: Prompt Rewriter 8B,
LanPaint AV, the external BlockSwap bridge and all six Quick Start subgraphs already entered
`origin/main` through ancestor commit `620d87d`. They are not waiting inside this local checkpoint.
The unpublished v1.45 batch instead contains the later Creator, ClipProj/Sol, Audio Integrity,
native-latent timeline/checkpoint and prompt-budget additions described above.

A pre-publication Registry audit first reproduced six omitted Unicode-named Quick Start files when
packing through a Windows/GBK console. Only their package paths were renamed to dated ASCII names;
their bilingual titles, NOTE text and graph contents remain unchanged. The pre-continuation checkpoint then
added Native Latent Checkpoint Save/Load after the previous 181 IDs and rebuilt from the clean
scratch index. `comfy node pack` produced a 284-entry, 1,423,698-byte ZIP containing all six
subgraphs, 128 workflows, the web extension and required runtime modules. Tests, tools, docs,
artifacts, `roadmap.md`, `SKILL.md`, `LOCAL_EXPERIMENT_DATA.md`, local agent metadata, model
weights, generated media and archives were absent. The extracted ZIP imported through
`comfy_entrypoint` with 183 registered and 183 unique nodes; all previous 181 IDs remained in exact
order and Checkpoint Save/Load were the final two mappings. Optional dependency metadata and
third-party notices pin the separately installed ComfyUI-ClipProj 0.1.13/MIT and ComfyUI-sol-attn
0.6.2/Apache-2.0 revisions. The final local audit ZIP SHA-256 is
`B0667461A8F474996C405B3B0ABA440BDFF6339CFA61F431A92D24268EDA6702`. `comfy node validate`
passed under an explicit UTF-8 console environment. The real project Git index remained empty
throughout the scratch build.

The checkpoint Save/Load path also passed a distinct-process evidence gate. PID 25280 wrote and
verified a completed 22-frame nested H3 AV latent, exited, and PID 8884 strictly loaded it using the
independently retained manifest plus whole-file SHA. Load returned `MATCH_EXTERNAL`; video/audio
tensors, FP16/BF16 dtypes, nested masks and supported metadata matched exactly. The stored file SHA
was `C5079378B7DCA981CE361E0F8C155107BCEED219DA7BC51B24A9F0777C108DE6` and its exact-content
SHA was `865085D76D92AF20548C21E6F28511D988F9FD69CCF915279249CF7FB82D1DB9`. This proves only
completed-latent process replacement. It does not recover an interrupted diffusion iteration,
Transformer/sampler derivative state, ComfyUI queue, model residency, CUDA allocator state or
perceptual continuation.

The next append-only node occupies position 183 without changing any of those first 183 mappings.
Native Latent Continuation Concat accepts one complete accumulated AV latent, one sampled Long
Video continuation, and the directly connected Planner and Conditioning reports that produced it.
It requires matching schema, chain, segment index, render length, timeline start/end, active motion
context and the native keyframe count. It then removes the full proven 5/22/39-frame context rather
than the ordinary five-frame H3 prefix. Audio removal is derived from the cumulative 24fps-to-40Hz
phase. The safe default additionally requires `video_and_audio` context and
`timeline_audio_ref=true`; an explicit video-only policy keeps the structural trim but denies an
audio-continuity claim. CPU coverage proves 124+124 with 22-frame context becomes 226 frames,
video T=67 and audio T=377; chaining a 39-frame-context segment becomes 311 frames. Five-frame
rounding, nested masks, exact-grid and hidden-tail final segments, closed-chain rejection, stale
reports and mismatched audio policy are also covered. The workflow connects both reports directly
and explains that the final hidden tail is trimmed only after one AV decode.

A fresh `source-final17-native-latent-continuation` scratch index then produced a 285-entry,
1,431,556-byte Registry ZIP with SHA-256
`FA57D947640B7E544CA020AD33D2A16F9E437604ACD5AD94AFD30E0932882D42`. It contains 129 workflows,
all six dated Quick Start subgraphs, the web extension and the 184-node runtime. Tests, tools, docs,
artifacts, `roadmap.md`, `SKILL.md`, local experiment data, models and generated media remain absent.
`comfy node validate` passed, and an isolated extraction imported 184 registered/184 unique nodes;
its full ID order matches the source, all first 183 positions are unchanged, and the continuation
concat is last. The package includes the new workflow and no root-level accidental `node.zip` was
created.

The completed current local checkpoint has 184 registered nodes and 129 importable frontend
workflows. The full CPU suite passed 1088 tests with four pre-existing Triton deprecation warnings.
Full-repository Ruff excluding ignored artifacts, compileall over 273 Python files, 193 non-artifact
JSON parses, version consistency, unchanged stable-sampling Git blob, `git diff --check`, Registry
validation, isolated package import and 129/129 project-to-user-workflow SHA parity passed. No GPU
generation or pressure test was run for this increment. These gates prove the structural and
packaging contract, not that a third-party sampler consumed the report, human continuity,
interrupted-NFE recovery, lower VRAM or universal 16GiB safety. The batch remains uncommitted and
unpublished.
Subsequent isolated low-load
probes added one ClipProj route, one strict Sol kernel execution, one repeated native-latent
single-decode route, one Creator Workspace real-data session and one CPU-only real Qwen tokenizer
count; none is publication, pressure, perceptual-quality or general memory-safety evidence.

## 1.44.0 LightX2V SLA + KJ Sage Composer checkpoint (2026-08-22)

Three append-only nodes add a strict LightX2V Turbo-SLA route and an isolated KJ Sage composer
without changing the first 160 node IDs,
their input order, defaults, or stable sampling code. The loader authenticates the fixed 1.956GB
ComfyUI LoRA by size, SHA-256, metadata, tensor structure and exact 208-patch mapping. The original
loader remains fail-closed on every external attention/object patch. The new composer accepts only
the complete 50-block KJNodes MiniMax H3 Sage whole-forward set with matching bindings, structure
and one source fingerprint. For SLA apply calls it selects the stock H3 forward so the call reaches
the block-sparse Sage2 override; for dense control and calls outside the SLA route it delegates the
installed KJ forward. No call evaluates both kernels. Sol-Attn and all other attention owners remain
rejected. The audit token is single-use and additionally rejects any KJ bypass of the SLA apply path
or incomplete KJ coverage of the dense-control path.

The implementation follows the released LightX2V `dynamic_sparse_attn` routing math: 128-query and
64-key blocks, mean-centred K, exact tail-aware block pooling, pooled QK scores, floor top-k at a
requested 15% keep ratio, and Sage2 block-sparse execution. It is an H3 ComfyUI adapter for that
released path, not the whole LightX2V inference runtime and not all sparse+linear branches described
by the general SLA paper.

One low-load real FL2VA probe completed on RTX 4060 Ti / sm89 with the current INT8 ConvRot base,
the exact SLA LoRA, 256×256×22, four `native_flow` calls, video shift 6 and audio shift 3. All 208
patches mapped and applied. Runtime Audit observed 4 model forwards, 50 main attention calls and 50
sparse kernel calls in each forward (200 total), zero dense fallback and zero kernel failures. At
sequence length 870, 2 of 14 key blocks were retained (14.2857%) and router workspace peaked near
0.309MiB; end-to-end execution took 46.77 seconds. This closes the mechanical LoRA/router/kernel
gate only. The upstream metadata names a BF16 FL2VA base, so the INT8 result remains an explicit
compatibility experiment. Same-input dense-control human A/B, 0.7MP BF16 profiling, listening, and
general 16GiB safety remain unclaimed.

The compatibility implementation passed 85 focused registration/schema/router/workflow tests and
the full 966-test CPU suite; four existing Triton deprecation warnings remain. Current installed
KJNodes source passed the composer's function-name/QKV/Sage helper/head-chunk structural contract.
Ruff, compileall, 110 JSON parses, `git diff --check`, and `comfy node validate` under UTF-8 passed.
Both dated workflows are present in the project source and synchronized user menu (108/108 JSONs).
No GPU generation, pressure test or quality/speed claim was added for the composer checkpoint.

### 2026-08-26 SLA compatibility repair

Two live ComfyUI installations rejected the default SLA workflow before model execution because
their otherwise compatible `PackedLayout.__init__` source hashes were not in the original exact
source whitelist. A repacked SLA filename was also rejected by the original fixed byte-size and
file-SHA gate, and Runtime Audit required exactly four forwards even when the user deliberately
connected an eight-step native-flow schedule.

The local repair keeps the same three node IDs, input order, defaults and public four-step
workflows. Core compatibility now requires the native H3 call signatures, an exact small
`patchify_video` ordering probe and an executable first/last-frame `PackedLayout` probe with target
audio/video tail ordering; source hashes remain in the report but are diagnostic only. SLA LoRAs
now require complete safetensors A/B pairs, H3 diffusion targets and complete mapping/application
to the loaded base; one filename, byte size and SHA are no longer a global runtime whitelist. The
historical LightX2V SHA remains recorded only as the artifact used for the original real probe.

The sigma contract accepts finite monotonic `native_flow` schedules from one through 64 NFE and
Runtime Audit derives its expected forward count from the connected schedule. The upstream
checkpoint's four-step, video-shift-6/audio-shift-3 route remains the only official reference;
eight steps and other schedules are explicitly reported as experimental compatibility, not an
upstream quality claim. A redundant built-in ComfyUI PyTorch or Comfy Kitchen attention override
can be recognized and replaced by the SLA owner; foreign attention owners remain fail-closed.
Focused tests cover unknown-but-semantic core hashes, non-pinned structural LoRA headers, built-in
attention replacement and eight-forward audit. No new GPU generation or perceptual claim is made
by this repair.

### 2026-08-26 SLA profile-router diagnosis

The later user-supplied file named `lightx2v_sla_fl2va_736x416_124f_00002-audio.mp4` does not contain
the geometry or duration advertised by its filename. FFprobe reports 704x416, exactly 22 frames and
0.916667 seconds; its embedded workflow also requests length 22. It therefore cannot establish a
failure that begins after the first second. A contact sheet still confirms a malformed visual frame,
but the container length and the image defect are separate facts.

One serial, non-stress control then used the same difficult distinct first/last images with the
corrected ordinary Alpha8 Turbo route, 736x416x124, eight native-flow model evaluations and 12/3
shifts. All 124 frames and the 32kHz stereo audio passed strict decode, but full-duration human review
rejected the result because it entered a persistent forced scene/scale transition after about one
second. Minimum observed free VRAM was 418MiB, so the project 512MiB safety gate also failed. Its audit
recorded zero SLA calls; this isolates the failure to incompatible FL2VA anchors / unsupported camera
transition rather than proving either success or failure of the SLA kernel.

Current LightX2V source confirms that its H3 SLA config stores five sigma grid points and executes
four model evaluations. It also validates the released BF16 checkpoint family and an FP8 inference
recipe; no reviewed source validates the local ComfyUI INT8 ConvRot base with the SLA LoRA/router.
The new append-only Profile Router therefore allows the exact 4-NFE/6/3/85-percent SLA profile only
for BF16 or LightX2V-FP8 evidence families and refuses INT8 with an actionable fallback message. Old
SLA nodes and their widget order remain unchanged for workflow compatibility and unsafe diagnostics.

## 1.43.0 append-only creator and compatibility helpers (2026-08-22)

Version 1.43.0 preserves the first 155 registered node IDs, their input order, defaults, and stable
sampling path. Five nodes are appended as IDs 156-160: an external MiniMax H3 BlockSwap parameter
bridge, LanPaint AV prepare/composite nodes, and local Prompt Rewriter 8B generate/unload nodes.
The BlockSwap and LanPaint implementations deliberately remain bridges to separately installed
upstreams and do not redistribute or silently emulate those projects. The prompt rewriter pins the
LightX2V 8B adapter/base contracts, defaults to unload after generation, and releases its own model
references on success, error, and OOM paths without globally unloading the user's H3 stack after
generation.

Six ComfyUI-native Quick Start blueprints cover T2VA, I2VA/FL2VA, Ref2VA, Audio Drive, Long Video,
and single-person repair. They contain existing node graphs with fewer exposed parameters and do not
replace the full dated workflows. Face detector paths now preserve a lexical alias under
`ComfyUI/models` while allowing that alias to resolve through a directory symlink or junction to
external storage; absolute paths and traversal remain rejected. Environment Audit and the strict
EAV/Sage composer now report or reject compute capability 12.x at 50,000 or more estimated packed
rows rather than treating import success as output-correctness evidence. The H3 row-mask fallback
implements the official 2x2 contract only when the newer core helper is absent, and Prompt Relay
admits the separately reviewed tokenizer hash while retaining its token-tail, byte, and presentation-
tag checks.

The release gate passed 951 CPU unit/schema/import tests with four pre-existing Triton deprecation
warnings. Ruff 0.14.8, compileall, 170 non-artifact JSON parses, version consistency, `git diff
--check`, and `comfy node validate` also passed. The initial Registry CLI attempt completed its
checks but crashed while printing a Unicode check mark through a GBK console; repeating under UTF-8
returned exit code zero. One earlier 16GB Prompt Rewriter generation completed but was slow and
truncated at the deliberately small 256-token probe cap. No three-cold/three-warm run, pressure test,
full BlockSwap or LanPaint H3 quality run, or universal 16GB claim was performed for this release.

## 1.42.0 automatic Motion Recovery validation (2026-08-22)

Motion Recovery now has seven append-only nodes. The analyzer defaults to
`auto_conservative_exp`, while `manual_ranges` and `report_only` remain explicit alternatives. The
new Auto Gate marks its repaired frame/audio inputs as ComfyUI lazy inputs. A deterministic poison-
branch runtime graph proved that an abstained plan never evaluated the connected pass-2 segment
node; the gate returned the original frame and audio objects.

A real calm Stock20 T2VA run at 736x416x124, 24fps and 20 NFE automatically failed the absolute
residual-motion gate, returned `status=abstained`, `second_pass_requested=false`, and
`baseline_object_passthrough=true`. Pass 1 and the routed result shared MP4 SHA-256
`5D01773935DA81F5313D3D5C02CDA7B7656696AE8A05D2A751677A3FAD4BD1F3`; decoded video and PCM
hashes were also identical. This proves actual pass-2 omission for that clip, not post-hoc selection
after executing both branches.

One real I2VA, FL2VA and independent-model Ref2VA route then completed at 736x416x124 with 20 pass-1
and 10 pass-2 calls. Default-safe I2VA, FL2VA and Ref2VA final SHA-256 values were respectively
`AA8A5A8321FE996D418A54AB14EFC43DA493AD0B76D014FFC526140D8833F713`,
`58A5F18D2C4B1B2BDB547F309591203E42452A8589D12AF1BC2FB0DC72F652DE`, and
`E522D4A8A771B41E7530714CEC578E8FD1D06127B73D07D51DB3BAE8C68F5881`. All recovered files were
124-frame, 24fps, 32kHz-stereo media and passed strict video/audio decoding.

The I2VA pass produced all three audio policies from the same pass-2 latent. Against pass 1,
`pass1_original` correlation was effectively 1.0, `pass2_recovered_exp` correlation was 0.4738,
and `blend_exp` correlation was 0.9839. Their complete-file SHA-256 values were
`AA8A5A83...13AC`, `94A29E49...13AC`, and `68A1F344...BB5E`. CPU multilingual ASR recovered the complete target Mandarin
sentence from every track. These measurements prove media integrity and speech-content retention;
they do not prove subjective audio non-inferiority. The user then completed full-sequence listening:
`pass1_original` sounded normal; `pass2_recovered_exp` suddenly became a distant voice in the middle
and then returned, so it failed and is retained for diagnostics only; `blend_exp` sounded normal for
this one clip at `pass1_mix=0.8`. That single blend pass is not generalized to other voices, prompts,
windows or mix values. Coarse whole-device minima for I2VA, FL2VA and Ref2VA were
approximately 47.8, 471.5 and 113.5MiB, all below the project's 512MiB gate. No pressure repeats
were run, and quality, experimental-audio, and general 16GiB-safety claims remain false.

## 1.40.0 EAV composition closeout (2026-08-22)

Three append-only composers occupy development node slots 146-148 while preserving the first 145
IDs and stable sampling. EAV + BlockCache authenticates the separately installed T8 CPU cache,
keeps its outer-sample lifecycle, and audits active full/hit forwards as 50/1 actually executed
blocks. EAV + STG owns the only post-CFG hook, applies the same FETA route to main and weak branches,
and audits the exact Stock20 main/weak sequence as 50 versus 50-minus-skipped measurements. EAV +
Long Video preserves the native scoped `extra_conds` layout owner, binds a fresh runtime to each
`segment_index/context_frames` pair, validates 5/22/39-frame motion offsets, and audits each segment
independently. Importable NOTE-equipped workflows were saved and synchronized to the user menu.

Only deterministic CPU/runtime contracts, native packed-layout probes, registration, JSON wiring,
and project/user SHA parity were run. At the user's request no high-resolution, cold/warm,
consecutive-task, near-limit VRAM, or long-chain pressure test was performed. These routes therefore
make no quality, seam, audio non-inferiority, acceleration, memory-saving, or general 16GiB claim.

## 1.39.2 EAV + Prompt Relay composer (2026-08-21)

The 145th append-only node resolves a real ownership conflict rather than weakening either
standalone node's fail-closed checks. Prompt Relay and EAV both require one H3 diffusion wrapper and
one `optimized_attention_override`; directly stacking them is therefore invalid. The composer
accepts only a current authenticated Prompt Relay MODEL with a complete binding and at least two
events, removes the standalone owner from a clone, validates the EAV task/layout contract, and
installs one combined wrapper. Within each attention call, Relay routing runs first and the FETA
gain is then applied only to target-video output rows. No additional model forward is introduced.

`disabled` returns the exact incoming Relay MODEL and disables only FETA, which provides a clean
Relay-only versus Relay+EAV single-variable control. This initial frontend route is restricted to
native Stock20 T2VA; unaudited Turbo tasks, reference layouts, ordinary LoRA, external Sage,
BlockCache, STG, Long Video, model-Hybrid artifacts, interior keyframes and denoise masks fail
closed. Deterministic tests cover binding tamper rejection, disabled identity, wrapper ownership,
operation order, target-video-only scaling, append-only registration and frontend wiring through
Runtime Audit. A later same-seed 736x416x124 Stock20 basic pair completed strict video/audio/combined
decode; whole-device headroom stayed below the 512MiB project floor and automatic metrics only
proved that the outputs differed. Perceptual review, listening, repeated cold/warm memory and general
16GiB safety remain incomplete, so no quality, audio, speed or memory claim is made.

## 1.39.1 learned two-pass workflow-set publication (2026-08-21)

The importable `13-latent-upscale` set now presents the same current contract in all five learned
two-pass generation templates: standard I2VA, native Mandarin speech, Hybrid `lock_source`, Hybrid
`remix_source=0.20`, and Hybrid `reference_only`. Each graph stores
`base_steps=8/coarse_steps=4/refine_steps=4`, so it executes four low-resolution and four
high-resolution joint-AV model calls. The fourth refine interval inserts the published high-noise
`0.8` sigma point; it is not an extra tail-detail step.

High-resolution Conditioning receives the learned upscaler's aligned width and height directly,
and explicitly opts into canvases above the former 1920x1088 reference area. The upscaler keeps its
32-pixel alignment, aspect safeguards, 4x scale limit, and risk report, but does not impose a hard
2MP execution ban. Native/remix/reference outputs save decoded pass-2 audio; `lock_source` saves the
Conditioning `mux_audio`. The canvas NOTE blocks state these different audio-ownership contracts.

This patch release does not add or reorder nodes, change a node schema/default, or alter stable
sampling mathematics. The five JSON files were already exercised by structural regression, while
the real 4+4 Mandarin native-speech route completed strict video/audio decode and intelligibility
checking. The older per-mode human review used the 4+3 schedule, so it remains bounded historical
evidence rather than a new 4+4 quality guarantee.

## 1.39.0 EAV + Strict Sage composer (2026-08-21)

The 144th append-only node owns both the H3 FETA route and the local
`sageattention.sageattn` HND backend. It exists because a third-party MiniMax H3 Sage patch that
replaces each block's full `Attention.forward` bypasses the optimized-attention observation point
used by the existing EAV adapter. The composer therefore refuses external attention/object/block
patches and calls Sage directly. Kernel errors fail closed; there is no silent PyTorch-attention
fallback. `disabled` still returns the exact original model without loading or checking Sage.

One T2VA probe used the same prompt and seed `2608217001` as the existing native-attention EAV
control, with 1152x640x124, 24fps and dual-clock Stock20. It completed 20 model forwards with
exactly 50 FETA measurements and 50 successful strict Sage calls per forward: 1000 of each, zero
kernel failures and zero fallbacks. Reported FETA gain min/mean/max was
`1.000000/1.000341/1.046687`, and the planned CFI workspace peaked at 31.585MiB.

The accepted output is H.264 1152x640, 124 frames at 24fps with finite AAC 32kHz stereo and a
14.667ms A/V duration difference. Three deterministic single-thread strict video, audio and
combined decode attempts all passed. Its SHA-256 is
`A9ABDA116404A319F4F453DD26EE792E0949656BF2885A098E7239D2F7832C65`. A contact-sheet review found
no black or corrupt sampled frame. Against the same-seed native-attention EAV apply output, video
SSIM was 0.8641, PSNR 26.1157dB, audio zero-lag correlation 0.9145 and Sage/native RMS ratio
1.0546. These values establish that the backend changes the joint AV result, not that it improves
quality or sound.

Full prompt execution took about 683.3 seconds including load, sampling, decode and save. Sparse
15-second polling observed about 730MiB minimum whole-device headroom, which is not a true peak and
clears the 512MiB project floor only narrowly. There is no matched hot/cold timing control, repeated
cold/warm matrix, human blind review or listening test. The node therefore remains Advanced EXP and
makes no claim of visual superiority, audio non-inferiority, acceleration, lower VRAM use or general
16GiB safety.

## 1.38.2 EAV reference-task 0.7MP validation (2026-08-21)

Ref2VA and task-Hybrid each completed one controlled native Stock20 pair at 1152x640x124 and
24fps. Every apply run executed 20 model forwards and exactly 50 H3 main-block measurements per
forward. Ref2VA reported `g=1.000000/1.000222/1.033226` and task-Hybrid reported
`1.000000/1.000256/1.037641`; both planned about 31.585MiB of bounded CFI workspace.

The accepted four files are H.264 with finite AAC 32kHz stereo and passed three deterministic
single-thread video, audio and combined strict-decode attempts. The first Ref2VA baseline and first
Hybrid apply containers each had one bad H.264 packet; cache-only re-encoding was accepted only
after history proved that every upstream model, conditioning, sampling and decode node was cached.
No diffusion trajectory was rerun for that packaging recovery.

Ref2VA apply/baseline wall times were 941.094/936.906 seconds with whole-device headroom about
417/1043MiB. Task-Hybrid apply/baseline wall times were 1062.594/990.406 seconds with about
2413/719MiB headroom. The different cache/load states forbid a general performance or memory
comparison, while Ref2VA apply falling below the 512MiB project floor directly denies a general
16GiB-safety claim. Automatic Ref2VA diagnostics were SSIM 0.5611, median Laplacian ratio 1.0097,
temporal-difference ratio 1.0102, audio cosine 0.8695 and RMS ratio 1.2253. Task-Hybrid diagnostics
were 0.7796, 1.0259, 0.9919, 0.9867 and 1.0460 respectively. These values measure difference, not
which result is perceptually better. A two-pair hash-traceable blind package exists locally; human
quality, reference-adherence and audio ratings remain pending.

The SPEED quality analyzer now pins FFmpeg strict decoding to one thread. This matches the project's
deterministic media gate and prevents thread-dependent H.264 decoder behavior from producing a false
failure. No runtime node, node schema, workflow, default or stable sampler mathematics changed.

## 1.38.1 EAV reference validation tooling (2026-08-21)

This patch adds a deterministic builder for controlled native Stock20 Ref2VA/task-Hybrid
`disabled` versus `apply_exp` API prompts and a separate hash-traceable anonymous A/B review
packager. The prompt builder fixes each pair to the same task-specific seed, 1152x640x124 canvas,
20-step dual-clock schedule, model stack, reference image and media output contract. The review
packager requires matching video/audio metadata, verifies copied media hashes, keeps the reveal key
outside the public HTML, and exports missing ratings as ties.

These additions are development and review infrastructure. They do not change any runtime node,
schema, default, stable sampling implementation, or saved workflow. The real 0.7MP Ref2VA and
task-Hybrid A/B matrix is still incomplete, so no visual-quality, audio non-inferiority, performance,
or general 16GiB-safety conclusion is added by this patch.

## 1.38.0 Enhance-A-Video Ref2VA / task-Hybrid composer (2026-08-21)

The 143rd append-only node enables only native Stock20 Ref2VA and task-Hybrid conditioning. It does
not modify the preceding EAV node, node IDs, stable sampler, or conditioning schema. At runtime it
reconstructs and validates the pinned H3 PackedLayout sequence—text, optional first/last condition,
reference image/audio blocks, target audio, then target video—and refuses any changed order or size.
FETA still measures target-video Q/K and directly scales only the target-video output slice.

Deterministic tests cover image, audio and video-audio reference sizes, Ref2VA/Hybrid classification,
malformed layouts, disabled identity, append-only registration and two NOTE-equipped importable
workflows. Prompt Relay, ordinary LoRA, Sage, BlockCache, STG, Long Video, model-Hybrid artifacts,
interior keyframes and denoise masks remain fail-closed. No real 0.7MP reference-task A/B has yet
passed, so this release makes no claim of visual improvement, audio non-inferiority or 16GiB safety.

## 1.37.1 learned two-pass 4+4 and high-resolution opt-in (2026-08-21)

The learned I2VA two-pass default now uses four low-resolution and four high-resolution Euler
intervals. The high-resolution schedule is
`0.9035, 0.8, 0.6316, 0.3158, 0`; because Euler executes one model call per interval, this is eight
joint video/audio Transformer calls in total instead of the former seven-call 4+3 graph. Saved
workflows that explicitly store three or five refine calls continue to load with their prior values.

A real corrected-Alpha8 I2VA run used a 736x416x124 first pass, learned 2x video-latent upscale,
1472x832x124 second pass, shifts 12/3, and the Mandarin sentence
“你在干嘛呢，我在这里呀，看看效果如何”. The resulting 124-frame 24fps file contains finite
32kHz stereo audio and passed strict video/audio decode. Local multilingual ASR differed only on
“呀/啊” after normalization (CER 1/16); this is one-material intelligibility evidence, not a
universal voice-quality or lip-sync claim.

The learned latent upscaler no longer treats the official 1920x1088 reference area as a hard 2MP
execution cap. It preserves 32-pixel alignment, maximum 4x scale and aspect-ratio safeguards while
reporting above-reference-area memory risk. Stable Conditioning remains fail-closed by default;
as of v1.52.2, that fail-closed behavior applies to real tensor/layout contracts rather than canvas
area. The legacy high-resolution flag remains at the schema tail for old-workflow compatibility but
larger 32-aligned canvases are warning-only regardless of its saved value.

## 1.37.0 Enhance-A-Video / FETA Advanced experiment (2026-08-21)

Two append-only Advanced nodes adapt the temporal cross-frame intensity equation from
Enhance-A-Video (`arXiv:2502.07508v3`) to MiniMax H3 full-3D packed attention. The implementation
uses target-video Q/K only, computes exact temporal CFI in bounded spatial chunks, delegates the
actual attention calculation to the already selected backend, and directly scales only the fresh
target-video output slice. Target audio is never directly scaled, but later joint-AV blocks can
still alter it indirectly.

Controlled 1152x640x124 pairs cover Stock20 T2VA, I2VA, FL2VA and L2VA plus strict corrected-Alpha8
Turbo8 T2VA. Every accepted output is 124 frames at 24fps with finite 32kHz stereo audio and passed
three strict FFmpeg decode attempts. Active Stock20 routes observed 20x50 block measurements and the
Turbo8 route observed 8x50. Removing an unnecessary full packed-output clone produced decode-exact
equivalence on the repeated I2VA seed, but its whole-device headroom still fell to about 271MiB.

The release conclusion is deliberately bounded: mechanical and media contracts pass, while stable
visual superiority, audio non-inferiority and general 16GiB safety remain unproven. Ref2VA/Hybrid,
Prompt Relay, Sage object patches, BlockCache, STG, Long Video, ordinary LoRA, interior keyframes and
denoise masks remain fail-closed until explicit composers receive separate controlled validation.

## 1.36.2 learned two-pass partial audio-clock repair (2026-08-21)

The author's current I2VA workflow carries pass-1 audio around the learned 3D video upscaler, then
feeds the reunited AV latent into a second unmasked sampler. That second stage is the owner of the
final native audio; the pass-1 audio is not a finished asset. Human listening invalidated this
project's preceding zero-mask recommendation: its latent audit passed, but the decoded sound was
clearly wrong.

The remaining custom-sampler mismatch occurred before its first pass-2 model call. Comfy's generic
KSAMPLER initialized every packed value using the partial video sigma. For the published I2VA start
of video sigma `0.9035` with shifts 12/3, the audio stream must instead start near sigma `0.701`.
`sampling.py` now recovers the exact scaled-noise term from KSAMPLER's actual `noise` and
`latent_image`, then rebuilds only the audio slice as
`sigma_audio * scaled_noise + (1 - sigma_audio) * audio_latent`. It leaves video, random noise,
schedule, model, dtype/device policy and NFE unchanged. Full sigma-1 starts remain identical, and a
dedicated partial-start zero-mask test proves the explicit lock route still reaches its latent.

Two real 736x416x124 to 1472x832x124, 4+3-call outputs used the same prompt, seed and model. Native
`ModelSamplingAV + Euler` prompt `19f64127-5185-436f-9e7b-b9e44ffd9ab8` completed in about 469.09s;
repaired custom dual-clock prompt `baaf7b1f-f70b-45c6-9ec2-0ed5febfc57a` completed in about 471.35s.
Both are 124-frame 24fps video with 32kHz stereo audio, and the user confirmed both sound normal.
Decoded PCM correlation between them is `0.94909455` with RMSE about `0.06363`; their near-equal
runtimes do not establish a speed advantage. Native output SHA-256 is
`EEE4DF987566CA885F30F8EE23E70D0255CB0CDE86EF9CB458AF82FC8871B651`; repaired custom output
SHA-256 is `49F4495A70C80AEC18CD83D6FC8B8C264988C9307ECBC9FCBD9F18EBFA48DD34`.
Video-only, audio-only and combined FFmpeg strict decodes passed for both files; each has a
5.166667s video stream and 5.152s audio stream, a 14.667ms difference below one 24fps frame.

The shipped native graph now uses `second_pass_audio_source=legacy_policy`, omits the output-node
Audio Audit, and connects pass-2 output directly to AV Decode. The append-only Audio Audit node and
explicit source/strength fields remain registered for intentional locks, but they are no longer a
native recommendation. This initial result covered one native I2VA material only. The controlled
four-mode speech validation below subsequently closed `lock_source`, `remix_source=0.20`,
`reference_only`, and native speech review for one shared fixture. Broader materials, trained
phoneme-sync measurement, general quality and universal 16GB safety remain unverified. Final gates
passed 841 project tests, changed-scope Ruff, compileall, every non-artifact JSON parse,
project/user workflow SHA parity, strict media decode checks and `git diff --check`.

### Controlled four-mode speech and lip-review closure (2026-08-21)

One front-facing image and the same licensed LibriSpeech utterance were used to exercise all four
audio modes through the repaired learned two-pass route. The exact 5.152s, 32kHz stereo source has
SHA-256 `9F43B22327A871CC11B8689507568CB2FCC9F9BA31D769FEF08AE47574E09C74` and says
"All the time he was talking to me, his angry little eyes were following Lake." Every run used seed
`2608215001`, 736x416x124 pass 1, learned 2x video-latent upscale, 1472x832x124 pass 2, shifts 12/3,
and the same published 4+3-call schedule. The three source-audio modes used Hybrid conditioning;
native used I2VA without an audio input.

| Mode | Prompt ID | Runtime | Output SHA-256 | Output-audio contract |
|---|---|---:|---|---|
| `lock_source` | `da1e8f81-5c22-462d-912f-b4ceb58d2572` | 478.30s | `E151D72AEA192D391E5F62285AF19C910476AB016416F888A5C0BDEA00575E53` | Conditioning `mux_audio`; source waveform retained |
| `remix_source=0.20` | `fe150edf-6d86-473a-b521-f2ce7f68e2a2` | 520.98s | `FAC8B9F0A76C2B8E291FF5A7B74BF34754B2FA32F872044700C849E41EA4C59D` | generated AV Decode audio |
| `reference_only` | `f249d33f-4a22-4008-820c-0040dc08b204` | 461.91s | `47B647360E09A502516EDA6E23545774FA5069BE8E6E1B6F32437AE3BC53A5CF` | blank target audio regenerated with `<Audio 1>` reference |
| `native` | `76a19361-10c5-431a-af3b-8d0809be1943` | 458.06s | `9D268CD152558A2FC9E811E0847DCBD6D2B0B1BA1CC30EFF2B4F647925FC1DC7` | generated AV Decode audio; no source input |

All four outputs contain 124 HEVC frames at 1472x832 and 24fps plus 5.152s AAC at 32kHz stereo;
video-only, audio-only and combined strict decodes passed. Relative to the decoded source,
`lock_source` had correlation `0.9997696` at zero lag, confirming the direct-source contract.
`remix_source=0.20` had correlation `0.8705903` at about -1ms and reduced above-8kHz energy from
`0.01858%` to `0.00772%`, so even a low denoise strength is not a transparent bypass.
`reference_only` had correlation `0.8492765` at about -1.03ms, but its code path starts from blank
target audio and regenerates it; similarity does not make it equivalent to `remix_source=1`.
Native correlation to the unrelated source waveform was `0.02549`. Its decoded RMS was about
`-10.98dBFS` and sample peak about `+2.50dBFS`, so the bounded review passed but downstream level or
true-peak limiting remains prudent.

Checksum-verified faster-whisper-small.en recognized the complete sentence in every output.
The source, lock, remix and reference files shared the source recording's `Lake`/`Link` ambiguity;
native matched the written 15-word target with WER 0. A 68-landmark proxy detected the face in
124/124 frames for all four outputs. The three source-audio modes had pairwise mouth-aperture
correlations from `0.9446` to `0.9728`, but raw mouth-aperture/audio-envelope correlation is not a
trained SyncNet metric and was not treated as proof of phoneme alignment.

The user then watched and listened to all four complete outputs and reported no issues, explicitly
approving the review. This is the decisive subjective pass for this bounded fixture. It verifies
that all four modes are usable through the repaired learned two-pass chain on this one image,
utterance, seed, model and reviewer. It does not establish comparative superiority, speaker-identity
preservation across voices, trained phoneme-sync accuracy, multilingual behavior, repeated-memory
safety, cross-GPU behavior, or a universal 16GB guarantee.

## 1.36.1 learned two-pass audio lock — default withdrawn (2026-08-20)

`MiniMaxH3TwoPassLatentReconcileT8Advanced` now keeps its legacy `audio_policy` route while adding
independent optional `second_pass_audio_source` and `second_pass_audio_strength` controls. The
shipped learned I2VA graph explicitly selects `first_pass` and `0.0`, so pass 1 owns native audio
generation/remix and pass 2 retains the first-pass audio stream under a nested zero mask while the
joint AV Transformer refines video against it.

The append-only 140th node, `MiniMaxH3TwoPassAudioAuditT8Advanced`, checks the actual pass-2 input
mask and compares audio latents before/after sampling. It permits at most `1e-5` float32 sampler
roundoff, fails closed above that threshold, and then replaces the sampled audio with the exact
pass-2 input audio before AV decode. Existing 139 node IDs and ordering are unchanged; stable
`sampling.py` and EXP multi-rate sampling mathematics are unchanged.

One real prompt `14b75733-7eb0-4bcb-bc2b-210eecd996bb` ran 736x416x124 pass 1, learned 2x video
latent upscale, and 1472x832x124 pass 2 with shift 12/3 and the published 4+3-call schedule. The raw
pass-2 audio difference was max-abs `1.1920928955078125e-7`, RMSE
`1.0963865371138581e-8`; the audit then reported `locked_audio_replaced_exact`. Total execution was
475.21 seconds and observed whole-device use reached approximately 15,753MiB. The final H.265/AAC
file SHA-256 is `B63D11C709356E5939F83FF16788EFD565C677D667918FA99F4F6CCBCC452326`;
decoded float32 PCM SHA-256 is
`A7C292AC999A13A2DEBF739CD409CC05CA773F679A372F0AC387A82DE9E5ACC3`. Video-only, audio-only and
combined strict FFmpeg decodes passed. Media is 1472x832, 124 frames at 24fps with 32kHz stereo;
the 14.667ms A/V duration difference is below one 24fps frame. Audio contained no NaN/Inf/denormal
values and peaked near -4.16dB.

The gate passed 839 project tests, focused Ruff, compileall, frontend/API workflow contract checks,
append-only 140-node registration and `git diff --check`. It proved only that a zero-mask latent was
preserved. Subsequent complete listening found the sound wrong, so v1.36.2 supersedes the default;
the hashes and metrics above remain as negative historical evidence, not a quality claim.

## 1.36.0 Prompt Relay release (2026-08-20)

Seven append-only Prompt Relay Advanced nodes provide authenticated multi-event plans, H3 target-video
query routing, an opt-in joint-AV route, chainable Event and Studio Packet bridges, model-free Preview,
and bounded resource estimates. Two more append-only nodes project one global Relay plan into the
existing Long Video windows. Existing Conditioning and sampler schemas remain unchanged, and stable
`sampling.py` retains SHA-256
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

The release includes eleven generation templates plus one model-free plan/estimate preview. Controlled
same-input, same-seed, same-NFE baseline/Relay pairs cover T2VA, I2VA, FL2VA, L2VA, Ref2VA image,
Hybrid, paired reference video plus matching soundtrack, and standalone reference audio. All sixteen
outputs strictly decoded as 736x416, 124-frame, 24fps video with 32kHz stereo audio and showed no
black/frozen transition or clipped PCM sample. A single reviewer judged the paired reference-video
results approximately equal and the standalone-reference-audio Relay result somewhat better in prompt
adherence. This is not evidence of universal quality, semantic-audio, speed, or 16GB safety.

Final release gates passed: 829 pytest cases, Ruff, compileall, 142 non-artifact JSON parses, 139-node
append-only registration, 85/85 project/user workflow relative-path SHA parity, `git diff --check`, and
the stable sampler hash above.

## Unreleased prompt-tag compatibility hotfix (2026-08-20)

The reported failure occurred before CLIP encoding in `prepare_prompt()`: the local validator
treated every plain `Image N`, `Video N`, or `Audio N` phrase as an intentional reference token and
raised whenever the ordinal was not connected. The compatibility path now keeps plain numbered
media prose as prose when zero same-type reference items exist, maps zero-based ordinals to the
official one-based form, and maps a stale positive ordinal only when exactly one same-type item is
connected. All such changes are recorded as warnings in the Conditioning report.

Strict fail-closed behavior remains for an explicitly bracketed tag with no corresponding media and
for an out-of-range ordinal when multiple same-type media items are connected. In non-strict mode an
unresolved explicit tag is demoted to prose instead of being passed to Qwen as a dangling media tag.
The focused prompt/Conditioning/still-image suite passed 25 tests. The complete project suite passed
746 tests with four existing Triton deprecation warnings; changed-scope Ruff and `git diff --check`
also passed. No node schema, input order, packed Conditioning payload, or sampler file changed.

## 1.35.1 learned H3 multiplier workflow hotfix (2026-08-19)

The current upstream workflow was pinned at
`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler@64fc9d4c7e2c03e8c61d6886182e3309365a1962`.
It uses native H3 shift 12/3, a Comfy `simple` eight-step schedule split after four low-resolution
calls, and raw high-resolution video sigmas `0.9035, 0.6316, 0.3158, 0`. It does not reinterpret
those values as a shift-6 schedule. The upstream 3D node also defaults to multiplier 2.0.

The project example now has one authoritative size control: `scale_by=2.0` on the learned upscaler.
Its actual aligned `width` and `height` outputs are connected directly to the high-resolution H3
Conditioning node. Changing the multiplier therefore cannot leave a stale second canvas behind.
Existing saved workflows keep their serialized values and node input order; stable nodes and
`sampling.py` are unchanged.

The first attempted 1472x832 run exposed a separate invalid LoRA conversion. The plain
`minimax_h3_fl2v_turbo_4step_v0.1_comfyui.safetensors` has SHA-256
`BE232BED4B6E2A808E53481056D2EB1DFEC064113FAAC9FF68C28D43D5261A07`, omits the PEFT alpha
normalization required by the source adapter and applies approximately 16x excessive update. Its
whole-frame melt is retained only as failure evidence. The accepted graph instead uses
`LoraLoaderBypassModelOnly` with
`minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors`, SHA-256
`35A4465A7911C8917C8BA3CFD2991270184474DC2FA9122BBDECAFF15D7C1B39`, at strength 1.0.

The corrected real I2VA graph ran 736x416x124 low resolution, multiplier 2.0 learned resize and
1472x832x124 high resolution with seven total joint AV forwards. It completed in 626.969 seconds;
whole-device peak use was about 15170.3MiB. The final H.265 file has 124 frames at 24fps plus 32kHz
stereo audio, passed separate `-xerror -err_detect explode` video and audio decodes, and has SHA-256
`06BEE97DAC218956FBB83288397878481211A6BA626EEA599667C897500924E0`. Eight sampled frames no
longer exhibit the rejected LoRA route's whole-frame melt. This proves the corrected sizing,
LoRA/schedule mechanics and media contract for one case; it is not a universal quality or 16GB claim.

## 1.34.0 learned H3 latent two-pass route (2026-08-19)

Three append-only Advanced nodes implement a clean-room H3-specific route: exact 322-tensor
checkpoint validation and 3D video-latent upscale, high-resolution AV/template reconciliation, and
a base-flow 4+4 sigma split derived from the MODEL video/audio shifts. The first 125 node IDs,
stable interpolation node, old workflows, and `sampling.py` remain unchanged.

The installed FP16 checkpoint is 690,592,672 bytes with SHA-256
`043E5A48E161610EF6C3EA974645220354D06FA618ABCA15F76D084812EB55C2`. A synthetic CUDA probe
expanded `[1,24,7,26,46]` to finite `[1,24,7,40,70]`, preserved the exact joint-audio object and
released the selected upscaler. A full I2VA run then completed low 736x416x124 4-step sampling,
learned upscale, rebuilt 1120x640 Conditioning, strict reconcile, high 4-step sampling and native AV
decode. The low/high denoising stages took approximately 52.9/184.5 seconds.

VideoHelperSuite H.264 exposed two independent delivery failures on this machine: metadata-enabled
muxing could read the intermediate before its `moov` atom was finalized, and metadata-disabled H.264
contained one decoder-fatal frame. The shipped example therefore uses H.265 with metadata disabled.
That final output passed fatal decoder-error checks at 1120x640, 124 frames, 24fps, 32kHz stereo and
less than one-frame A/V duration difference; SHA-256 is
`53F10C8CFB6EC0584679F33101EB6A5687491FED4089E646FE2A2675F53F5430`.

This validates the mechanical route, not a quality claim. No same-NFE full-resolution perceptual
control, Ref2VA/Hybrid/Long Video matrix, continuous peak trace or cross-GPU result exists yet.

## 1.33.1 H3 SPEED formal calibration and controlled denial (2026-08-19)

The user-provided `h3_speed_blind_review.json` was preserved byte-for-byte in the ignored local
evidence directory. Both copies have SHA-256
`BA47FA124C9FD0BF51E625DBB0D7F170FA795D0E2BD9654161A16A664B201365`. The frozen reveal maps
T2VA A and FL2VA A to SPEED, while Ref2VA B is SPEED. After reveal, all three overall, motion and
audio votes prefer the full-resolution baseline; Ref2VA reference adherence also prefers baseline.
The reviewer additionally marked FL2VA SPEED A and Ref2VA SPEED B as visibly broken.

This is a decisive rejection of those three fixed schedules, not proof that every possible SPEED
profile must fail. One reviewer and one case per route cannot isolate causality. Execution reports
showed one common hand-set treatment: 14 of 20 NFE at half resolution, transition threshold 0.85,
then only six full-resolution NFE. Because this threshold did not come from an H3 dataset fit, it is
the strongest shared mechanism suspect, but is not recorded as a proven cause.

The next implementation step therefore did not expand the failed default. Five append-only
Advanced nodes provide an H3 dataset calibration path:

- Dataset Accumulate stores only one CPU float64 HxW summed spatial-power grid plus hashes and
  provenance; it rejects repeated batches, exact repeated clip spectra and task/model/VAE/grid
  mismatches.
- Dataset File persists that sufficient statistic as one atomic safetensors file, with explicit
  write/overwrite confirmation and no source video or source latent storage. Load uses the complete
  file SHA as its ComfyUI fingerprint, while explicit Save is never cached.
- Model/VAE Fingerprint streams full-file SHA-256 values without loading a second GPU model.
- Dataset Finalize fits the aggregate mean once. Fewer than 100 actual unique clips or a failed R²
  threshold can only yield `research_probe_only`, never a validated delta-optimal profile.
- Calibration Window strictly resamples to 24fps/17n+5 and uses aspect-preserving center-cover;
  short clips fail and source geometry is never stretched. The legacy Source Media Window is unchanged.

The isolated example is
`2026-08-19_H3_SPEED_Spectrum_Dataset_Calibration_Advanced_EXP.json`. This closes the mechanical
calibration contract only. The paper's corpus method uses independent natural videos encoded through
the target video VAE; it does not require first generating 100 outputs with the diffusion model. This
project keeps a conservative formal minimum of 100 reviewed natural-video windows for the exact H3
video VAE/profile contract.

That formal accumulation is now complete. One hundred reviewed windows from pinned
Vchitect/Vchitect_T2V_DataVerse revision `e068be25f4d06a837992a1e9096fd00105c83f2c` passed exact pre-trim, strict decoding, hash and
near-duplicate screening, and a 10x10 human content-diversity review before sequential encoding by
the same H3 video VAE. The final sufficient-statistics file has SHA-256
`C219A739BDD9C17EDAA53EF1022CCDB6985B72B237AF2A44DE19AB34047BF43E`, contains 100 independent
entries on video-latent grid `[24,37,26,46]`, and fits A=29.96418670445687,
beta=2.3183720623777164, R²=0.9951511913433466. The strict finalizer therefore correctly marks
the exact task/model/VAE/grid profile as `validated_for_delta_optimal`. This validates the corpus and
fit contract only, not a generation claim.

Only one calibrated T2VA comparison was then run, as predeclared: 736x416x124, Stock20/Euler,
identical prompt, seed, model, CLIP, VAEs, noise and 20 total NFE. The full-resolution baseline took
243.203s with 12504.6MiB whole-device peak. The fitted plan selected 384x224 for 13 NFE and 736x416
for 7 NFE; calibrated SPEED took 248.688s, peaked at 16175.8MiB and left only about 203.7MiB
headroom. It was approximately 2.26% slower, added about 3671.2MiB peak usage, failed the 512MiB
headroom gate, and was classified `fits_with_thrashing`. Its original H.264 also failed 3/3 strict
decodes because of one corrupt decoded frame. The original is retained as failure evidence; a separate
re-encoded derivative passed strict decode solely to permit optional blind review and does not erase
the source failure. The implementation is therefore denied for stable acceleration, memory safety,
quality and audio/reference non-inferiority; no wider multimodal calibration matrix is authorized.

An isolated `127.0.0.1:8197` probe then encoded two strict-FFmpeg-clean 736x416x124 sources through
the same H3 video VAE and persisted a two-entry dataset across separate prompts. After a one-entry
Load had been cached, the second atomic overwrite changed the complete-file fingerprint; the identical
Load graph executed again and returned two entries instead of stale state. The 10,856-byte clean probe
has SHA-256 `80EE5756323AA6F8F0ED80A465C5EF5C812009EDEBDFE86F522FD649930080B4` and fitted
A=23.3507002, beta=2.0766619, R²=0.9946004, but correctly remains `research_probe_only` because
2 is below 100. A separate portrait source proved center-cover reporting and no anisotropic stretch,
but strict decoding exposed a CABAC error, so it was not counted in the clean dataset.

`tools/curate_h3_speed_spectrum_sources.py` now provides a read-only pre-encoding source gate. It
inspects ffprobe metadata, target-window strict decoding, full-file hashes, decoded-window hashes and
a 16x16/3fps temporal average-hash heuristic. Exact duplicates are rejected; heuristic near-duplicates
and filenames associated with comparisons, repairs, caches or experimental SPEED outputs require
manual review and are never silently declared independent. It never moves or deletes media.

The signature-mode scan of all 188 files under `ComfyUI/output/MiniMaxH3` produced 3 provisional
candidates, 55 manual-review files and 130 rejected files. Eighty-nine files passed strict target-window
decoding; nine failed it. Rejections overlap and include 69 clips too short for the 124-frame window,
62 sources requiring spatial upscaling, 29 exact decoded-window duplicates, four exact file duplicates,
nine strict-decode failures and five inspection failures. The ignored local report has SHA-256
`AE8FEAC8D70F9A38A8FD65AE2FC7E71A4847BE3947AC32B99DB28BFA8D5B865C`. These counts establish that
the existing output directory cannot satisfy the 100 independent-clip gate. They do not validate the
three provisional clips as independent or representative. A wider strict scan across five H3-named
output roots found 1,275 files: 3 provisional, 244 manual-review and 1,028 rejected. Overlapping
reasons included 574 short clips, 460 exact-file duplicates, 221 required upscales, 185 exact decoded
window duplicates, 22 strict decode failures and 13 inspection failures; 442 files passed strict
decoding. Its ignored report SHA-256 is
`2A7D54548D34009228D2516D06F3F4E9D180F6199E20C9B4FCB9F6A8BBD85582`.

Nine provenance-bound I2VA Stock20 controls were then selected from the frozen motion-quality
generation spec: three distinct first-frame images times three distinct seeds, with only the control
arm retained. All nine sources passed strict decoding and were accumulated sequentially through the
same H3 video VAE. The final dataset contract is `[9,24,37,26,46]`; its aggregate fit is
A=29.5579671, beta=2.3662343, R²=0.9971147. Finalize correctly returns
`research_probe_only`, `validated_for_delta_optimal=false` and `enough_unique_clips=false` because
9 is below 100. The differing two-clip T2VA and nine-clip I2VA fits are evidence that task families
must not silently share one spectrum profile; neither small probe establishes a production value.

The curation tool can now parse embedded ComfyUI prompt metadata into a privacy-minimized execution
contract. It emits hashes and model/task/VAE/LoRA/sampler/modifier/seed/content signatures, never raw
prompt or workflow text. A provenance-required scan of the same 1,275 files found 452 complete H3
contracts, 51 partial contracts and 768 files without a prompt tag; only 8 remained provisional,
456 required review and 811 were mechanically rejected. Exact contract grouping still exposed mostly
derived MultiKeyframe, sigma and repair matrices, not 100 diverse independent Stock clips. The local
report SHA-256 is `D089706EB19528F305F4899C1170D6ED2F2AB86E3CFB9669A0D39DDB037B5C96`.

A separate strict signature scan covered all 358 video files under `ComfyUI/input`. It retained 19
provisional candidates, left 15 for manual review and rejected 324. Overlapping reasons were 219
required upscales, 217 clips shorter than the exact window, 62 exact-file duplicates, 11 inspection
failures and 8 strict-decode failures. The report SHA-256 is
`EA5C1A80F34DA88E50CDD11B7784F39D8D917F8A2EF9B6EF7BE63FA0EDD9B5DC`.
`tools/build_h3_speed_spectrum_manifest.py` consumes only this signature schema, confines every source
to the declared input root, requires complete file hashes and refuses any formal threshold below 100.

The resulting 19-entry local-video-corpus proxy manifest has SHA-256
`52C04B72762A5EC1169B21BFB393C07D1205C8D6242D495529A5066FB9906922`; all 19 clips encoded through
the same H3 video VAE and appended successfully. The run report SHA-256 is
`CC7D8A83156AB7D2904E2982E62112FA245F20DADFB58F1AA1926E01AA4DB49E`, and the persisted dataset SHA-256
is `6750766764A8D45FA67EEF6295E2AB8261798B169515B815AED80949DAEF42B0`. Offline aggregate finalization
returned A=26.5381849, beta=2.1534428 and R²=0.9955738 on latent contract `[19,24,37,26,46]`; the
profile report SHA-256 is `9701DBA26EEFE44FCECC93A54DF92CD1B3AFF30F4E42DEC6D1F6FC5BA08B33EF`.
This is deliberately still `research_probe_only`: the 19 inputs are a local video-corpus proxy, not
100 independent natural-video windows with a formal diversity/provenance review. Compared with the
two-clip
T2VA probe, A and beta changed by about +13.65% and +3.70%, reinforcing the refusal to promote a
small-sample fit or run another quality A/B from it.

`delta_optimal` plans now preserve the fitted H3 video-latent C/T/H/W contract and compare it with
the requested final canvas and runtime aligned frame count. A spatial or temporal grid mismatch fails
before sampling; task/model/VAE equality alone is no longer treated as sufficient binding.

The T2VA frontend workflow now reproduces the exact formal dataset load/finalize, full model/VAE
fingerprints and `delta_optimal` execution graph used by the controlled run. Three visible notes state
the failed speed, peak-memory, original-decode and blind-review gates. All other SPEED task workflows
are explicitly historical mechanical examples because they lack route-specific formal profiles. The
ten project files and ten installed ComfyUI user-menu copies have exact SHA-256 parity.

The final v1.33.1 source gate passes 718 project tests, changed-scope Ruff, compileall, 125-node
append-only registration and the unchanged stable `sampling.py` SHA-256 against ComfyUI
`187eda8ef5e588c6a5765cad53e482765edae052`. Full-project Ruff still reports the pre-existing
`tests/test_sampling.py:180` E721 under the old bundled Ruff; it is outside this change set.

## 1.32.1 H3 SPEED controlled performance and decoder-error gate (2026-08-19)

No node, schema, default or stable sampler changed. Two reusable tools now build and analyze one
FL2VA remix and one Ref2VA-image full-resolution Stock20 control from the frozen SPEED prompts.
The contract checker requires identical model, CLIP, VAEs, media, conditioning fields, seed,
20 NFE, shifts, trim and modality-stable AV noise; only the staged spatial SPEED treatment differs.
The existing T2VA pair uses the same contract.

| Route | Full-resolution | SPEED | Speedup | SPEED minus baseline peak |
|---|---:|---:|---:|---:|
| T2VA 1056x608x124 | 683.953s | 313.859s | 2.179x | -77.0MiB |
| FL2VA remix_source 1024x576x124 | 706.610s | 319.829s | 2.209x | +33.9MiB |
| Ref2VA image 1024x576x124 | 619.281s | 269.344s | 2.299x | -162.5MiB |

These results validate a real end-to-end speedup only for the three exact local profiles. They do
not validate a universal speed factor. Peak VRAM did not consistently improve and every SPEED run
remained below the 512MiB minimum-headroom publication gate, so the memory-safe claim remains false.

The earlier three-pass decoder helper was insufficient because FFmpeg can print H.264 decode errors
while returning success. It now adds `-xerror -err_detect explode`. The stronger check rejected the
older T2VA SPEED bitstream, while its baseline and all FL2VA/Ref2VA files passed. T2VA SPEED was rerun
with the exact same model, prompt, seed, 20 NFE and plan; the clean v7 file passed the strengthened
check three times. All final files contain exactly 124 frames at 24fps, finite 32kHz stereo audio and
A/V duration within one frame.

The anonymous review is now complete: all three fixed SPEED routes lost to baseline, and two routes
were explicitly described as visibly broken. The proxy metrics remain diagnostics only and did not
override the human verdict.

The source release gate passed 664 tests, full Ruff, compileall, 126 non-artifact JSON parses,
`git diff --check`, the unchanged stable `sampling.py` SHA-256 and SHA parity for all 70 project/user
frontend workflows.

## 1.32.0 H3 SPEED multimodal/reference/Turbo8 GPU gate (2026-08-19)

The stable node inventory remains 120 and `sampling.py` is unchanged. One execution-scope enum value
was appended after the two existing SPEED values: `turbo8_t2va_research_exp`. It requires media-free
T2VA, native audio, exactly eight steps, shifts 12/3 and a weight-patched MODEL. Runtime reports the
presence of patches but deliberately keeps LoRA identity unverified because patch tensors do not carry
a trustworthy source filename. Existing strict Stock20 and multimodal scope values retain their order.

Seven controlled RTX 4060 Ti 16GB runs used 1024x576, 124 frames and 24fps. Stock20 used a two-stage
14+6 split; Turbo8 preserved total NFE as 6+2. All outputs contained exactly 124 decodable frames,
finite 32kHz stereo audio, stream-duration error within one frame and passed 3/3 full FFmpeg decodes.

| Route | Runtime | Peak VRAM | Minimum headroom | Mechanical result |
|---|---:|---:|---:|---|
| I2VA lock_source | 298.859s | 15924.2MiB | 455.3MiB | pass; first-frame corr 0.9985, locked AAC audio corr 0.9831 |
| FL2VA remix_source | 319.829s | 15934.3MiB | 445.2MiB | pass; first/last decoded corr 0.9984/0.9972 |
| L2VA native | 278.797s | 16107.8MiB | 271.7MiB | pass; last-frame decoded corr 0.9976 |
| Ref2VA image | 269.344s | 16043.0MiB | 336.5MiB | pass; perceptual reference adherence not scored |
| Ref2VA 2s video + numbered soundtrack | 416.797s | 15990.4MiB | 389.1MiB | pass; fixed reference rows materially increased runtime |
| Hybrid first-frame + image/audio refs | 303.250s | 15927.3MiB | 452.2MiB | pass; keyframe and refs survived both stage rebuilds |
| T2VA Turbo8, 208 LoRA patches | 149.578s | 16257.8MiB | 121.7MiB | pass with `fits_with_thrashing` telemetry classification |

These are mechanical executions, not speedup or perceptual winners. No route passed the 512MiB
16GB publication gate. Reference identity/style/action/audio adherence, remix/native audio quality,
extra speech, event/lip synchronization and same-input full-resolution speed/quality comparisons remain
pending. Unit gates now also reject Transformer wrappers/callbacks/patches, DiT replacements,
post-CFG/model wrappers and LongVideo/MultiKeyframe scoped MODEL patches. Six dated frontend workflows
carry three Markdown notes each; user-facing source/reference filenames are explicit replace-me values.
The final release gate passed 655 project tests, full Ruff, compileall, 70 workflow JSON parses, an
idempotent live-schema rescan, the stable `sampling.py` SHA-256 guard and SHA parity for all 70 project/user
menu workflows.

## 1.31.0 H3 Detail Mixer Advanced source gate (2026-08-18)

One append-only node was added after the existing 118 IDs. The five independent detail nodes,
stable sampler, old schemas, defaults and serialized workflows were not changed. The mixer reuses
their already-gated implementations instead of duplicating the sigma or wrapper mathematics:

- tail subdivision optionally changes the integrator schedule;
- smooth model-time bias optionally wraps the shared AV model time;
- H3 STG optionally adds one weak skipped-block prediction on active calls;
- joint AV rectified-flow Restart optionally appends a second re-noised trajectory.

All four toggles default false. The report distinguishes base, tail, restart and actual integrator
NFE from STG weak forwards, model-time-biased calls and total planned joint-AV Transformer forwards.
The known conflict and mask checks remain fail closed, including pre-existing post-CFG functions,
model wrappers, H3 block replacements, fractional video masks and partial or frozen audio Restart.

Decoded Temporal Detail is intentionally outside the sampler because it consumes IMAGE after AV
Decode. The new frontend workflow wires final audio directly from AV Decode, includes four Markdown
notes and enables only the conservative Tail + Bias + STG candidate; Restart remains off. This is an
Experimental composition contract, not evidence of a perceptual winner, audio non-inferiority or a
universal memory-safe tier.

Final source gates passed 616 project tests, changed-scope Ruff, compileall, strict parsing and
project/user SHA-256 parity for all 64 frontend workflows, `git diff --check`, the stable
`sampling.py` SHA-256 guard and a whitelist-only ComfyUI quick startup.

One real GPU mechanical probe then ran on the local RTX 4060 Ti 16GB using non-pruned FL2VA INT8,
Qwen NVFP4, both official VAEs, 256x256x22 I2VA and the same source image/prompt family as the
red-Hanfu examples. The mixer used two base steps, one tail subdivision, Bias and STG, with Restart
disabled. Prompt `87b8c224-514c-46cf-af79-09da785a820d` completed sampling plus both VAE decoders in
39.38 seconds and SaveImage returned exactly 22 readable 256x256 PNG frames. This closes real-device
composition plumbing only; the deliberately tiny/low-step run cannot support perceptual, speed or
memory-tier claims for the published 1152x640x124 workflow.

## 1.30.1 frontend workflow order compatibility hotfix (2026-08-18)

The defect was in API-to-frontend serialization, not H3 inference: API dictionary order was written
as positional frontend `inputs/widgets_values`, while ComfyUI restores widgets from the node schema
order. This could shift prompt, dimensions, frame count, enums and booleans and display `NaN`.
No stable node input, output, default, registry prefix or sampler equation was changed.

The converter now force-serializes the complete live `object_info` order, including omitted optional
sockets before later links. The standalone repair tool is conservative for existing files and also
checks connected-slot positions. It repaired 40 project workflows and the corresponding 40 installed
user copies. Pre-repair files are preserved under `artifacts/workflow-order-repair-20260818` and
`artifacts/workflow-order-repair-pass2-20260818`; a second live-schema scan reported zero changes.

Final gates: 610 project tests passed; Ruff, compileall, strict parsing of 63 project plus 60 user
frontend JSON files, `git diff --check`, and the stable `sampling.py` SHA-256 guard passed.

## 1.30.0 H3 SPEED Advanced source gate (2026-08-18)

Four Advanced nodes were appended after the existing 114 IDs. The implementation is a clean-room
adaptation of `howardhx/speed@ca7801c9` and paper v3, not a copy of the H3 WIP plugin. It implements
the official power-law transition equations, orthonormal DCT coefficient expansion, kappa state
rescale and aligned sigma. H3 sampling is whole-chain and stage-specific: AV latents, keyframes,
references, tokenization and PackedLayout are rebuilt at every multiple-of-32 canvas.
Media-free strict T2VA is the safe special case: its Qwen text conditioning is encoded once and reused,
while only the correctly sized empty AV latent is rebuilt at later stages. Spatial multimodal conditions
continue through full per-stage resize/VAE/token/layout rebuilding.

CPU/static evidence covers official-equation values, two- and three-stage exact NFE conservation,
aspect-preserving canvas resolution, DCT comparison with SciPy's type-II orthonormal convention,
deterministic high-frequency noise, radial profile gates and segmented AV-state transport. Video
alone receives spatial DCT coefficients. Audio remains in the shared Transformer and uses an
explicit target-anchored public-flow reindex; that H3 extension is not claimed by the SPEED paper.
The Harvester cannot promote a one-clip probe by merely declaring a larger evidence count: validated
dataset status requires at least 100 actual latent batch entries, explicit independent-dataset provenance,
checkpoint/VAE fingerprints and the configured fit threshold. Tensor shape verifies count, not statistical
independence, so the latter remains an auditable dataset assertion rather than an inferred fact.
Kappa uses the official requested scale ratio; the slightly different multiple-of-32 grid ratio is
reported separately and never substituted into Eq.5/Eq.6. Delta-optimal plans also bind task family
and recorded checkpoint/VAE fingerprints to the runtime Stage Source and fail closed on mismatches.

No real ComfyUI H3 generation was run in this source gate. Consequently the execution report keeps
`quality_validated`, `speedup_validated`, `vram_safe_16gb`, `audio_noninferiority_validated` and
`gpu_generated` false. Strict execution permits only T2VA/native audio at exactly 20 steps. I2VA, FL2VA, L2VA,
Ref2VA and Hybrid mechanics require explicit `multimodal_research_exp` and remain pending GPU tests.
The final source gate passed 606 project tests (four existing Triton deprecation warnings), full Ruff,
compileall, all 63 frontend workflow JSON parses, `git diff --check`, stable-sampler hash verification,
and a CPU whitelist-only ComfyUI quick start.

## 1.29.0 H3 tail/detail Advanced routes (2026-08-18)

Five nodes were appended after the previous 109-node inventory. The stable `sampling.py` source,
existing node IDs, schemas, defaults and serialized workflows were not modified:

- final-interval dual-clock subdivision (`extra_tail_steps=3`, 11 joint AV forwards);
- smooth shared-Transformer model-time bias (8 NFE, unchanged integrator sigmas);
- true joint audio/video rectified-flow endpoint restart (`video sigma=0.15`, 3 restart forwards);
- H3 skip-double-block spatio-temporal guidance (block 25, progress 0.25 to 0.85);
- decoded-frame motion-gated luma detail enhancement (no audio tensor input or mutation).

The fixed real-generation case used `10A.jpg`, 1152x640 (737,280 pixels), 124 frames at 24fps,
seed `2608172801`, the non-pruned FL2VA INT8 checkpoint, Qwen NVFP4, official video/audio VAEs,
Turbo LoRA, video shift 12, audio shift 3 and the same night-time fast-spinning red-Hanfu prompt.
All five new candidates completed. Their final synchronized files are H.264/AAC, contain 124 frames
and finite 32kHz stereo audio, and each passed three complete FFmpeg `-xerror -threads 1` decodes.
Every candidate has 5.1667s of video and 5.152s of audio; the -14.67ms stream-duration difference is
within one 24fps frame.

| Route | Observed execution | Final SHA-256 |
|---|---:|---|
| tail+3 | 574.94s | `1FB196B01B997AD9B5EFBF1D5D1C25E349D2467ECDACDCED0C74908BA3284230` |
| model-time bias | about 412s generation; cached re-save 4.02s | `3B969FA1600A6616E7064E9FA682CE6404D69E7ADC6B12AA00EAB1199EC5C001` |
| joint AV RF Restart+3 | 558.22s | `A093AB192912B090D4D05AF1329756A616F4590E889C35FCA7D2CFB1532DEBAE` |
| H3 STG | 655.66s | `7610B69D71F8212473450C899EB2EF9C0AFE5CD819B8C6565351A997A5718A39` |
| temporal detail | 423.66s | `DE27F74664A14F51D62ABC0922984B02BB70252DEDC075D59263944DC8B2ADFF` |

The model-time-bias generation itself completed, but its first VideoHelperSuite intermediate MP4 was
missing the `moov` atom and failed during mux. The corrupt intermediate was deleted and the same cached
decoded output was saved successfully. This is retained as a save-path incident, not silently counted as
a first-attempt end-to-end pass and not attributed to the model-time mathematics.

The reused upstream eight-step comparison file contains the isolated corrupt frame the user had already
accepted. OpenCV still returns 124 frames, but repeat strict decode fails, so it is explicitly marked as
an accepted historical comparison defect and is not included in the five new-file decode pass count.
The proxy high-frequency, motion and audio-level measurements are anomaly screens only. They do not prove
face quality, cloth detail, action adherence or sound non-inferiority. The randomized six-way complete-video
review is in `artifacts/h3-detail-routes-v1/blind/blind_review.html`; no perceptual winner is declared before
the user's full visual and audio review.

All mechanisms remain Experimental. H3 predicts audio and video in one Transformer, so forced audio
freezing was not made a hard constraint. Tail subdivision, model-time bias, Restart and STG can all change
the joint sound prediction and require listening. Only the decoded-frame detail node is mechanically
audio-independent, but its separately generated comparison audio is not used as proof of bit identity.

The final source gate passed 590 tests (four existing Triton deprecation warnings), changed-scope Ruff,
compileall, 115 non-artifact JSON parses and `git diff --check`. Live ComfyUI `object_info` exposed all five
new node IDs with their expected outputs. Stable `sampling.py` remained byte-identical at SHA-256
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

A final red-team pass found and closed two pre-release contract defects. Restart now rejects fractional
video masks and any locked or partial audio mask at both setup and sampler runtime; binary conditioned
video rows may remain fixed only while the complete audio latent participates. STG now rechecks the actual
runtime `patches_replace` map, so a downstream Block Cache/Activation Chunk cannot silently overwrite its
selected block. It also rejects non-H3 models and nonzero shared-AV global-standard-deviation rescale.
Additional gates reject positive model-time bias that can reverse model-visible time, process temporal
detail in bounded chunks with one-frame halos, select aspect-first non-shrinking 32-aligned geometry and
enforce a default 2.1MP output budget. These are fail-closed safety/contract changes; they do not turn the
single red-Hanfu comparison into a general quality guarantee.

## 1.27.0 native SAM3.1 multi-person Face Refine (2026-08-17)

Six append-only Advanced nodes implement in-memory reference profiles, a two/three-person cast, native
SAM3.1 shot-local tracking, CPU SFace suggestions plus manual assignment, one-character repair jobs and
review-gated sequential compositing. A class-and-callable probe verifies the current ComfyUI
`SAM31Tracker.track_video_with_detection` contract. The source tree's stable `sampling.py` SHA-256 remains
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

Real probes on ComfyUI `0.33.0@7fe8a61385` established the following limited facts:

- A 22-frame group shot retained three distinct color tracks while `maximum_people=3` deliberately omitted
  a fourth person. This proves the cap and short-shot track mechanics, not character identity.
- A 240x416x22 two-person clip produced separate Alice and Bob repair plans. YuNet localized each face inside
  its assigned SAM mask and the CPU SFace guard accepted both. A separate automatic-assignment run mapped
  `0:0 -> Alice` and `0:1 -> Bob` without manual JSON.
- Selective SAM cleanup reduced loaded Comfy models from two to one and whole-device use by about 2240MiB;
  another model remained loaded and `global_unload_called=false`.
- The two H3 MANUAL512 branches ran sequentially with seeds 42/43, Ref2VA pruned INT8, Qwen NVFP4, official
  VAEs and the reviewed 0.75 FL2V Turbo LoRA route. The final chained composite passed strict FFmpeg decode as
  240x416, 22 frames, 24fps H.264. SHA-256 is
  `C74000515CFED4DB8A7D6E1DCD428F4AF379D3CEA89A432C3AE5EEC806F818E2`.
- Coarse polling observed approximately 489MiB free in the complete two-person run; an earlier one-branch run
  observed 450MiB. Both are below the 512MiB project gate, so this is an operational pass only.
- A separate four-person shot was trimmed to 240x416x22. `person with a visible face` selected the three
  repairable visible faces and omitted the back-facing person. Reviewed mappings bound `0:0`, `0:1` and `0:2`
  to three authorized references; all three SAM-constrained YuNet plans and SFace guards completed.
- The first full three-person attempt completed two H3 branches, then correctly stopped when the compositor
  found an apparent outside-mask change. Root cause was a producer/consumer threshold mismatch: Parity Stitch
  defines exterior as `alpha == 0`, whereas the compositor used `alpha <= 1e-6`. The consumer now uses
  `alpha > 0`, rejects non-finite/out-of-range masks and has a sub-micro-alpha regression test. Genuine exterior
  changes and overlapping person masks remain fail-closed.
- The corrected three-person cold prompt `0c37c0b3-e910-405f-9b3f-0a159c048b9e` completed all three sequential
  branches in 95.78 seconds. Its H.264/AAC output is 240x416, 22 frames, 24fps and 0.917 seconds; strict video
  and audio decoding pass. File SHA-256 is
  `C3CCB956397AC7497E8241DAB97D057ABAFFC20C625945662DE2608917B4DC42`. Source and output decoded PCM share
  SHA-256 `3645A04B3F853F324732FFB9779EE1C95B01F6E5F68C6A07968ECBEDAAD552C1`.
- Coarse whole-device polling reached about 375MiB free after the three-person run. This is below the 512MiB
  gate and does not establish a universal 16GiB profile despite successful completion without OOM.
- A longer 608x448x73 / 3.042-second follow-up showed that the broad visible-face prompt could spend the third
  slot on a back-facing person. `front-facing person with a visible face` selected the intended three visible
  faces throughout seven track previews, so the regenerated examples now use that stricter prompt.
- The three roughly 50-100px source faces produced 73-frame MANUAL512 jobs and all seeds 42/43/44 H3 branches
  completed sequentially. The first final composite attempt stopped because the third feathered mask overlapped
  50,621 already applied pixels. A reviewed rerun changed only the final policy to `keep_old_exp`; all generation
  nodes were cached, earlier pixels won in the overlap, and the final output completed.
- The resulting 608x448 H.264/AAC file contains exactly 73 frames at 24fps and 3.042 seconds. Strict video/audio
  decoding passes; SHA-256 is `AB26FC42A0FD9EFA5DA32877100554F1487165DEF2498BCC0495DD7638F656BB`,
  and source/output decoded PCM MD5 are both `4c7905d4a36f6f9c456b7e074b52707e`. Coarse polling observed about
  426MiB free at the lowest sampled point, again below the 512MiB promotion gate.
- Full-frame and zoomed source/result comparisons were generated. Five labeled zoomed samples show modest local
  changes without sampled catastrophic face collapse or track exchange, but no blind panel has established broad
  perceptual restoration. This remains an EXP candidate rather than a quality guarantee.
- A later backward-compatible crop-scale update kept `legacy_crop_factor` as the node default and added explicit
  `target_face_px` for manual canvases. The regenerated two/three-person examples target about 300px faces in a
  512 crop and use the project Turbo 8-step baseline. Reports record effective crop factor, achieved face-height
  range and source-boundary-limited frames.
- A real 512x384x73 two-person probe completed both sequential branches at approximately 300px crop-space face
  height. At small-face strength 0.8, raw median Laplacian ratios were 0.9584/0.9720 and did not establish added
  detail. Raising only that strength to 1.0 raised the raw ratios to approximately 1.06/1.01, but dark/blonde
  motion-correlation medians fell to approximately 0.39/0.82 and full-frame paste sharpness remained about 0.85.
  The stronger route is therefore rejected as an automatic preset: it creates high frequency at the cost of
  temporal/identity risk. The source itself was a 2x enlargement of a 256x192 crop, so crop-space scale cannot be
  interpreted as native source detail.
- A separate native 1920x1408x69, 24fps two-person side-profile clip isolated the intended use case: facial
  texture was already clear while structure was reported as damaged. SAM3.1 tracked 69 frames; the first legal
  56-frame H3 window used two clear single-person references, MANUAL512, 300px target faces,
  `relative_to_clip 0.8/0.35`, Ref2VA pruned, Turbo LoRA 0.75 and sequential 8-step branches. The remaining
  13 frames stayed untreated as an internal control tail. The output retained the full 1920x1408 canvas and
  original audio. Because Windows FFmpeg 7.1 again showed nondeterministic multi-thread H.264 decode errors,
  the final media was re-encoded with one x264 thread and passed three complete decode checks.
- The user watched the complete clear-source result and accepted it as a real facial-structure repair. The user
  also confirmed the operative limitation: a blurred source tends to remain stylistically blurred, so this node
  must not be advertised as video sharpening, deblurring or super-resolution. This closes one clear-source
  two-person subjective acceptance case only; broad cross-source quality and identity guarantees remain open.
- The single-, two- and three-person frontend workflows now contain embedded Markdown instructions. They record
  the MANUAL512, relative-to-clip 0.8/0.35, 21/51 smoothing, face-only 24/24 stitch, Turbo 8-step, clear-reference,
  per-shot identity-map and sequential review recommendations.

No cross-shot re-entry matrix, anime identity calibration, broad perceptual panel,
three-cold/three-warm staircase or universal 16GiB claim is closed. The three-person fixture used approximately
17px source faces, reviewed manual mappings, a deliberately relaxed 0.10 SFace guard and one
`largest_face_exp` reference; it proves bounded mechanics, not production identity thresholds. The provided
two- and three-person examples no longer expose or enforce `rights_confirmed`; `accept_candidate=false` remains
the non-destructive review gate. Each frontend workflow now places four parameter/caution NOTE blocks beside the
relevant graph sections in addition to its overview. The complete checkpoint passed 563 project tests, changed-scope
Ruff, compileall, 108 non-artifact JSON documents
and `git diff --check`.

Version 1.27.2 closes two workflow-input failures without changing any stable sampler or existing node ID. The
black-haired clear single-person reference produced three YuNet detections because two low-confidence boxes landed
on hair/clothing; `dominant_face_auto` selected the real face with a 2.0976 area ratio and 0.4676 confidence margin,
while the blonde reference retained its single detection. Ambiguous similarly sized faces still fail closed. The
actual two-person source contains 69 frames, four fewer than the 73-frame H3 request. Repair Job now repeats the
last source frame only for those four model-context positions and Composite trims them, preserving all 69 original
timeline frames and the untouched source audio. Shortfalls above 16 frames still reject. Both canonical workflows
contain five colored Markdown NOTE nodes; a separately named local v1.27.2 copy avoids confusing an already-open
browser canvas, which ComfyUI does not update when the JSON file changes on disk.

One additional full native preprocessing attempt was not counted as a pass: while running the real 69-frame source
through the native SAM3.1 branch, `python.exe` exited without a Python traceback. Windows Application Error logged
`msvcrt.dll` access violation `0xc0000005`, followed by `ntdll.dll` `0xc0000008`. ComfyUI was restarted and the
updated live schemas were rechecked. The direct real-reference tests, one-shot 0..68 scene analysis and synthetic
69-to-73 plan/composite contract pass, but this native crash remains separate evidence rather than being attributed
to the new Python boundary logic.

## 1.26.0 author-parity Face Refine correction (2026-08-17)

The source audit found four material differences in the new Parity implementation rather than a model
quality mystery. The T8 detector passed RGB ndarray input to Ultralytics while the fixed upstream passed
BGR; the T8 face mask followed the literal detector position inside an edge-clamped crop while upstream
kept the smoothed face rectangle centred; T8 denoise used actual smoothed detector height while upstream
used `crop height / crop_factor`; and T8 matched colour in crop space while upstream matched after warp in
source coordinates. T8 also duplicated one pixel frame before VAE encoding, whereas the upstream path
encoded the 89 real frames directly.

An offline same-frame probe using the fixed face YOLO and identical manual-512/crop-2.5 parameters first
measured mean crop absolute error 0.020292 and a maximum 7.912-source-pixel crop-size difference. Reversing
only RGB to BGR reduced mean crop error to 0.00000677 and maximum crop-size difference to 0.001861px. After
the centred mask, crop-derived denoise and source-coordinate stitch corrections, maximum face-rectangle
difference was 0.002169 canvas pixels, maximum denoise-curve difference was 0.00001423, and a full
synthetic upstream-versus-T8 stitch comparison had mean/max absolute error 0.00000117/0.00053048.

The corrected live graph then succeeded without pixel-tail duplication under prompt
`1ed411fa-9b91-45c4-801d-7f45b3597fe5`. Total execution time was 112.48 seconds. The resulting
`face_refine_t8_author_parity_v2_seed42_00001_.mp4` has SHA-256
`0DD8C79F95B01647F3BF345B6503C83A5860BE99BA66D8D72114BD274E9A0884`; strict decode reports 89 H.264
frames at 320x320/24fps plus 32kHz stereo AAC. Decoded audio MD5 is the same
`26d40526bd022d7237ba183bd8777966` as source, selected author target and prior T8 output. Full-video SSIM
against the target is 0.967059, improving on the prior T8 result's 0.955273. Coarse whole-device polling
observed 15,823MiB used of 16,380MiB, leaving approximately 557MiB. Per-frame SSIM spans
0.943776-0.990705 across all 89 frames; the five minimum frames were separately rendered and show no
new sampled-frame collapse. The user then watched the complete author-target/T8-v2 side-by-side and judged
the two results equally good. This closes subjective non-inferiority only for the fixed fixture, seed and model
stack. The run passes only the fixed mechanical 512MiB observation; it does not establish cross-input quality,
identity preservation or universal memory safety.

## 1.25.0 MANUAL512 REL Face Refine baseline checkpoint (2026-08-17)

The first 100 node IDs and all previous defaults remain unchanged. One Advanced node is appended:
`MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced`. It accepts the Parity candidate only when the
hash-bound Plan, latent-injection report, per-frame denoise report and stitch report prove `manual_512`, crop factor 2.5, 21/51
smoothing, `relative_to_clip`, requested 0.8/0.35 strengths, replacement video mask, locked all-zero audio
mask and face-only 24/24 stitch with no fallback frames. It also rejects nonfinite candidate pixels and any
plan whose minimum crop-space face height is below 200px. Passing returns the identical IMAGE tensor; the
report explicitly keeps `quality_guaranteed=false`, `identity_verified=false`, `automatic_accept=false` and
`universal_16gb_safe=false`.

The fixed local 89-frame, 320x320, 24fps fixture resolved source faces of 105-195px into approximately
205-312px faces in the 512 canvas, with crop magnification 1.60-1.95x. The selected `relative_to_clip`
output passed strict single-thread H.264 decode and has SHA-256
`19EA5844643B962F6FD197E34705861916D69F7EA70F3E00A2DF022D6A017399`. In the six-way full-video review,
the user selected this result over source, author target, auto320 absolute, manual416 absolute and manual512
absolute. Source-similarity proxies had preferred manual512 absolute; that disagreement is recorded rather
than used to override the human review. The relative run completed on the local 16GiB GPU, but sampled
minimum headroom was only 161MiB, below the 512MiB safety gate.

The reviewed run used Ref2VA pruned INT8, Qwen3-VL NVFP4, official video/audio VAEs, the alpha8 T8-converted
FL2V Turbo LoRA at 0.75, two identity references, locked source audio, face YOLO, and
`er_sde + simple + 4 steps + denoise 0.45 + seed 42`. Although the fixture filename described 90 frames,
Comfy returned 89 decoded IMAGE frames. The implementation now records `89 -> 90` explicitly: latent
injection duplicates the final crop once, H3 operates on the legal 90-frame grid, and stitch discards exactly
that one generated tail before returning the original 89-frame timeline. The baseline guard verifies the
alignment policy and counts from all reports; arbitrary source trimming or hidden VAE temporal fitting fails.

The API and frontend examples now use the selected settings and connect Stitch directly through the new
mechanical guard. The 1.24.0 quality gate remains registered for old workflows and conservative rollback
experiments but is no longer the recommended output route.

After loading the 1.25.0 code in the live Comfy process, the recommended API completed through the package's
own Plan/Latent/Denoise/Stitch/Guard chain (prompt `57741215-c23b-4a9b-87b7-7288ce175ff1`) in 107.41 seconds.
The saved candidate is strict-decode clean at 89 frames, 320x320, 24fps and 32kHz stereo; SHA-256 is
`B91BBE09C2AF4266EDD2975760A13749A0DB819054BE6C8118E144F0D4AF3097`. Its decoded audio MD5 exactly
matches source and the selected upstream-mechanism candidate (`26d40526bd022d7237ba183bd8777966`), while
video SSIM to that selected candidate is 0.955273. Representative frames are visually close, but a proxy cannot
grant equal-or-better subjective quality; the generated side-by-side is retained locally for user review.

## 1.24.0 Face Refine parity and source-fallback quality-gate checkpoint (2026-08-17)

Version 1.23.0 appended four isolated Advanced nodes after the unchanged 95-node prefix. Focused tests verify
reflected Gaussian smoothing, the 21/51 plan defaults, source reference crop output, strict crop-video
latent injection with the same audio tensor and mask object, per-frame video denoise with an all-zero
audio mask, multi-shot refusal and bit-exact pixels outside the stitch mask. The stock sampler chain is
deliberately external: `er_sde`, `simple`, four steps and base denoise 0.45, with the upstream example's
0.75 FL2V Turbo strength and seed 42 recorded in the example workflow. Version 1.24.0 leaves those first
99 IDs unchanged and appends a conservative source-fallback gate. It validates source/plan identity,
requires a Parity mask with bit-exact pixels outside it, measures source-relative global face SSIM, RGB
delta, Laplacian ratio and residual temporal jitter, filters isolated one-frame accepts, and tapers only
accepted continuous runs. Its report explicitly denies identity validation, quality validation and
automatic acceptance.

Three real 736x416x124, 24fps, four-step candidates were then executed on ComfyUI
`0.33.0@7fe8a61385`. The upstream-strength FL candidate and an otherwise identical all-50 Hybrid
candidate both showed obvious facial distortion and were rejected; their face-region SSIM means were
0.46179/0.45737. Reducing only the per-frame strengths from 0.8/0.35 to 0.45/0.15 raised face-region
SSIM to 0.76044 and motion correlation to 0.69200, but its median candidate/source face-Laplacian ratio
was only 0.50535 and the automatic report remained `quality_promotion=false`. The first run took
36.8 seconds and an observed sampling snapshot left about 433MiB whole-device headroom, below the
512MiB conservative gate. The low-strength output is therefore a less-destructive review candidate,
not a promoted restoration preset. Decoded final audio correlation to the original mux source was
0.99319 after codec round trips.

The fixed upstream commit's original `H3FaceTrackCrop`, `H3InjectVideoLatent`, `H3PerFrameDenoise` and
`H3FaceStitch` nodes were then executed in an isolated ComfyUI server against the same source and local
model stack. The exact face YOLO detected 116/124 frames and interpolated eight, with no ambiguity event;
`auto_capped_768` resolved to a real 256x256 crop because the largest smoothed crop was approximately
247px. Thus the 256 canvas is upstream behavior rather than a T8 forced resize. Upstream 0.8/0.35 measured
face SSIM 0.49077, median Laplacian ratio 0.74321 and motion correlation 0.42102, with visible ghost faces.
Changing only to 0.45/0.15 measured 0.77776/0.56297/0.70423 and remained softer than source. These runs
use the pinned upstream code but not the author's unrecoverable original GGUF, embedded prompt, full refs
or isolated vocals, so they do not prove the upstream project fails generally.

The real default T8 high-strength candidate then passed through the new quality gate. An encoded mask of
the accepted-change output had zero non-black frames out of 124, proving every generated face frame was
rejected. The gated output versus a separate source-only Comfy pass measured full-frame SSIM 0.999885,
face SSIM 0.996810 and face RGB MAE 0.000471 after separate H.264 encodes. Unit tests additionally prove
that the zero-accept path is source-tensor bit exact, a synthetic structurally close sharpness gain can
pass a continuous run, and any candidate change outside the Parity mask fails closed. This contains the
known ghost-face regression but does not establish a real restorative gain.

The complete project gate passes 548 tests, full Ruff, compileall, 104 non-artifact JSON documents,
workflow bidirectional-link validation, live object-info loading and the stable sampling hash guard.
Stable `sampling.py` remains SHA-256
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

## 1.22.1 Face Refine extended validation checkpoint (2026-08-16)

No runtime node contract changed. Reproducibility tools were added for the isolated Face Refine Advanced
route, including strict reveal analysis for exported blind-review JSON. The pinned YuNet integration completed
a non-official fixed-threshold IoU-0.5 evaluation
over all 3226 WIDER FACE validation images and 39123 valid faces. Threshold 0.35 measured precision/recall/
F1 of 0.6223/0.6499/0.6358; threshold 0.60 measured 0.8610/0.5694/0.6855. Under-16px recall dropped from
0.4360 to 0.3225 at the higher threshold, so no product default was changed from overall F1 alone.

The aspect-safe full-H3 matrix used FL2VA pruned INT8, Qwen3-VL NVFP4, both H3 VAEs, 736x416x124,
512px face crops and 12 low-denoise steps. Three cold and three warm executions all completed; minimum
whole-device headroom was 717.6/922.1MiB and post-private spread was 3.2/79.1MiB. One 736x416x362,
384px-crop cold run completed in 295.235 seconds with 1176.2MiB total-minus-used headroom and
41311.6MiB peak process private memory. Exact media contracts passed, but one 362 run is not a repeat matrix.

A real 416x736x124 group probe had multiple detections in 113 frames and no sampled within-shot switch,
while a separate 362-frame hard negative tracked a lamp and a cartoon billiard-ball figure. The tracker has
no appearance/identity model. A five-scenario controlled crossing matrix then demonstrated a sustained A-to-B
swap after only three occluded target frames. Six real candidate source-similarity proxies reported face SSIM
mean 0.4791-0.5392, candidate/source Laplacian median ratio 0.5493-0.6835 and motion-difference correlation
mean 0.3581-0.4071; none is an identity metric or perceptual non-inferiority proof. One reviewer completed all
six randomized source-versus-candidate pairs before reveal. Source won overall and identity 6-0; all six motion
preferences were ties. Source averaged 5/5 and candidate 1/5 for identity, expression/mouth, temporal stability,
seam and naturalness, while both averaged 5/5 for motion. Every note reported candidate facial distortion and
repeated face jitter. This is sufficient to reject the six current candidates for the fixed source/settings, but
one reviewer, one repeated source and six candidate runs cannot satisfy the preregistered five-reviewer panel or
support a broader causal/general claim. The checkpoint therefore closes bounded detector and memory mechanics
and rejects the current candidate set; identity-safe multi-person tracking, visual-quality promotion, automatic
acceptance and universal 16GiB safety remain denied. The full project gate is 532 tests.

## 1.17.0 Hybrid patch-stack compatibility checkpoint (2026-08-12)

One isolated Advanced node was appended after the previous 61-node prefix. The node returns the
exact input MODEL object and defaults to `report_only`. Its optional strict mode blocks proven
Hybrid identity/offset-set failures, Hybrid-before-LoRA order violations, later selected-AdaLN
overlap, partial or foreign Block Cache/Sage patches, Long Video/MultiKeyframe patch or Conditioning
mismatch, and configured current-VRAM/host-commit gate failures. It recognizes stable dual-clock/
native AV, EXP multi-rate, stock and unknown sampling routes without modifying sampling math.

The Hybrid Loader now attaches only the immutable policy-application provenance needed downstream:
schema/fingerprint/mode, whether the policy was applied, cleanup scope, reserve target, ComfyUI and
AIMDO setter routes, gate results, and `memory_safe_claim=false`. Full before/after telemetry is not
retained on MODEL. Clones inherit the attachment through ComfyUI's native ModelPatcher contract.

Deterministic validation covered:

- all 100 canonical Hybrid operations for the 25–49 video+audio recipe;
- nonselected attention/MLP LoRA acceptance, selected AdaLN overlap, wrong order, missing and
  duplicate set entries;
- complete and incomplete H3 Block Cache markers, full/partial/foreign Sage attention patches;
- Long Video and MultiKeyframe MODEL/Conditioning pairing and mutual exclusion;
- policy missing/report-only/applied states, low whole-device VRAM and low host commit;
- stock, stable dual-clock and EXP multi-rate sampling identification;
- API and frontend routing through Audit before BasicGuider, all link directions, and the unchanged
  previous 61-node prefix.

The complete project suite reports **339 passed**. Ruff, compileall, all example JSON parsing,
`git diff --check`, and an isolated CPU ComfyUI whitelist import passed. A live isolated server
reported 62 plugin nodes; `/object_info/MiniMaxH3HybridCompatibilityAuditT8Advanced` confirmed the
five required inputs, optional Conditioning, defaults, three outputs, category and EXP status. The
new frontend workflow was visible through `/userdata`.

A real RTX 4060 Ti 16 GiB integration probe then used the exact FL2VA pruned INT8 base, existing
27.69 MiB Hybrid artifact, 4 GiB T8 VRAM policy, KJ MiniMax H3 SageAttention, T8 H3 Block Cache,
stable dual-clock setup, Qwen3-VL NVFP4 and both H3 VAEs at 256×256, 22 frames and one joint step.
Strict audit mode remained in the MODEL path before BasicGuider and the prompt completed in 64.91
seconds, saving all 22 frames. Logs proved the direct AIMDO 4 GiB route and Block Cache `0/1` with a
5.4 MiB CPU cache. Coarse polling observed 825.46 MiB minimum whole-device free VRAM. This is a
mechanical integration probe, not a quality comparison or a universal 16 GiB safety result.

A second, higher-scope matrix on 2026-08-13 kept the same Hybrid/Sage/Block Cache/4 GiB/strict-audit
stack but ran 736x416, 124 frames and Stock20. Three fresh ComfyUI processes and three subsequent
same-process warm prompts all completed. Block Cache reported 6/20 hits and a 117.7 MiB CPU cache
on every run. Cold headroom was 992.80/806.70/766.38 MiB; warm headroom was
953.45/1004.89/791.33 MiB. Warm baselines were 15157.41/15060.25/15143.19 MiB, with only an
82.94 MiB maximum positive consecutive movement. Six 124-frame PNG sequences and six 32 kHz stereo
FLAC files were byte-identical for the fixed seed. This passes the exact local mechanical and
repeatability gate, but no same-current-stack Cache OFF run or perceptual comparison was performed;
quality, generalized speed benefit, other workloads/GPUs and universal memory safety remain open.

The same exact stack was then run with Block Cache removed: three cold and three warm OFF prompts,
plus three interleaved warm ON prompts in the third process. OFF/ON mean full-workflow times were
169.93/129.19 seconds (23.98% saving); sampler times were 146.24/105.31 seconds (27.99% saving).
All ON runs again cached 6/20 forwards. OFF and ON were each internally exact-repeatable, and the
new ON decoded pixels/PCM exactly matched the prior ON matrix. OFF versus ON was not bit-exact:
mean/minimum-frame SSIM was 0.8432/0.7577, uint8 MAE 10.37, audio correlation 0.9207 and audio
SNR 7.99 dB. One OFF cold run left only 239.40 MiB, below the 512 MiB gate. The exact local
performance benefit is therefore verified, while perceptual non-inferiority, multi-material/seed
quality and general 16 GiB safety remain open; a blinded A/B package was generated locally.

A further three-material/two-seed extension completed ten additional prompts. Five valid warm
performance pairs, including the earlier warm portrait seed, showed 22.05-28.47% end-to-end
saving (24.35% mean) and 27.94-33.05% sampler saving (29.07% mean), with 6-7/20 hits. Reversing
ON/OFF execution order did not remove the benefit. Across six quality pairs, mean video SSIM was
0.7020 with a 0.5192-0.9373 pair range and 0.4774 minimum frame; audio correlation averaged
0.9329 with a 0.8792-0.9806 range. No pair was bit-exact. The new matrix's minimum headroom was
1333.64 MiB, but the earlier cold OFF minimum remains 239.40 MiB. One human reviewer scored the
randomized package before reveal: six video ties and six audio ties. All B clips were perceived
as slightly lighter, but B represented Cache OFF in the first three pairs and Cache ON in the
last three; decoded YAVG/SATAVG checks also found no treatment-consistent B-side direction.
This passes only an exact-profile single-reviewer subjective smoke screen. The wider automated
differences than threshold 0.08, limited panel and prior cold memory result still deny lossless,
statistical non-inferiority, universal-threshold, quality-validated, or 16 GiB safety claims.

A conservative-threshold follow-up tested 0.08/0.10 on the two most divergent pairs, then ran
0.08 across all three materials and two seeds. Threshold 0.10 produced non-monotonic superhero
audio degradation and was not promoted. All six 0.08 runs succeeded with 3-4/20 hits. Five valid
warm comparisons saved 9.05-14.38% end-to-end (12.35% mean) and 12.20-18.39% sampler time
(15.10% mean). Video SSIM averaged 0.8598 (pair range 0.6013-0.9840; minimum frame 0.5294), and
audio correlation averaged 0.9635 (range 0.8883-0.9927). Both proxies improved over 0.12 in all
six pairs, while the difficult superhero case remained materially different. Minimum warm
headroom was 2434.52 MiB, but the prior 239.40 MiB cold OFF observation remains the safety limit.
A verified randomized OFF-versus-0.08 six-pair package was scored by one human reviewer before
reveal. Video produced one threshold-0.08 win, five ties and no Cache-OFF win; the reviewer saw a
slight difference in real-person material and no discernible difference in animated material.
Audio produced five ties and one low-confidence Cache-OFF win. That sole audio win occurred in
superhero seed 2, whose two tracks were already measured as effectively silent, so it is not
robust evidence that Cache OFF sounds better. No overall preference was inferred because the
reviewer did not explicitly score it.

The earlier anonymous model-side sparse-frame review recorded six low-confidence visual ties,
no gross sampled-frame collapse and zero clipping in all 12 tracks, but did not listen. Together,
these results pass only an exact-profile, single-reviewer subjective smoke screen. They do not
prove statistical perceptual non-inferiority or losslessness. No defaults changed; 0.08 remains
an opt-in quality-first candidate rather than a universal recommendation.

The node therefore always reports `quality_validated=false` and `memory_safe_claim=false`. VBAR
pages weights but does not bound activations, attention workspaces, VAE/CLIP, CUDA context, pinned
memory, other processes, drivers, or host commit. Detailed usage and issue codes are in
`docs/HYBRID_COMPATIBILITY_AUDIT.md`.

## 1.16.0 Hybrid artifact maintenance checkpoint (2026-08-12)

One isolated Advanced output node was appended after the previous 60-node prefix. Its default
`inspect_only` path performs no writes. Every mutating action requires explicit confirmation and a
positive operation epoch. The implementation derives one exact content-addressed path from a fully
verified Hybrid plan and can quarantine/restore a complete artifact pair, recover an interrupted
pair move, or quarantine sufficiently old build residue. It never scans or deletes source diffusion
checkpoints and does not unload MODEL cache or release VRAM.

Transactions use same-volume atomic moves plus fsynced atomic journals containing exact source and
recycle paths, byte sizes, SHA-256 values, phase, and moved count. Symbolic links, path escape,
noncanonical manifests, malformed phase/count pairs, incomplete artifact-pair journals, duplicates,
and hash/size changes fail closed. A Windows subprocess was killed after the artifact moved but before
the sidecar moved; explicit recovery archived the aged dead-owner lock and restored a valid pair.
Windows PID liveness now uses process handles and exit codes instead of `os.kill(pid, 0)`.

The frontend and API examples ship in safe inspection mode. The dedicated contract and limitations
are recorded in `docs/HYBRID_ARTIFACT_MAINTENANCE.md`.
All 327 project tests, Ruff, compileall, 67 example-JSON parses, and `git diff --check` pass. The
stable `sampling.py` SHA-256 remains
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

## 1.14.0 Advanced hybrid-model checkpoint (2026-08-12)

Three isolated Advanced nodes were appended after the previous 56-node prefix: exact pair
inspection, content-addressed small-artifact construction, and a stock-loader-based Hybrid MODEL.
The stable Conditioning, dual-clock and multi-rate samplers, Long Video, MultiKeyframe, and all old
node schemas were not modified.

The exact local FL2VA/Ref2VA pruned pair passed full SHA-256 and 932-tensor contract validation. The
default blocks-25-to-49 video+audio artifact is 29,030,400 bytes with 100 offset-set operations and
SHA-256 `fcb4cdcc5dbb9742af6163654da7246783dbffb045c6dd32d99a6c42772cd0ab`.
Its curve-table relative fit error is `4.9343e-5`, and its worst saved effective-modulation
reconstruction error is `2.3021e-5`.

Real DynamicVRAM loading retained `ModelPatcherDynamic`, the stock cached factory, clone,
non-dynamic delegate, and same-device deepclone behavior. A 256x256/22-frame/one-step real H3 chain
completed with 50 patched tensor keys and 100 patch entries. One controlled 736x416/124-frame/
Stock20 pilot then completed FL2VA, Hybrid, and stock Ref2VA at whole-device peaks
12932.8/12684.9/12883.2 MiB. This single run does not establish a material memory benefit or a
quality winner.

Follow-up development added Conditioning-aware minimal modality matching, a resumable sequential
matrix tool, local-only optional ASR/InsightFace/WavLM metrics, blind-review packages, and dedicated
audio-only plus visual-and-audio workflows. Fifteen Stock20 run records across visual, audio-only,
and mixed references all completed. In the one mixed-reference seed, Hybrid face/WavLM cosine was
0.523/0.868 versus FL2VA 0.449/0.467 and Ref2VA 0.443/0.945; all normalized ASR word streams matched
the target. These are research signals, not universal identity thresholds. The later precision
matrix reached only 41.34 MiB minimum whole-device headroom, so it supersedes the earlier pilot for
the safety decision and explicitly denies a 16 GB `memory_safe` label. Full evidence is in
`docs/HYBRID_MODEL_ADVANCED_VALIDATION.md`.

All 305 project tests, Ruff, compileall, 63 example-JSON parses, and `git diff --check` pass. An
isolated whitelist server imports the plugin in 0.0 seconds, exposes all 59 T8 nodes, and reports
the three new nodes under `T8/MiniMax H3/Models/Experimental`. All three Hybrid frontend examples
were copied byte-for-byte to `user/default/workflows/MiniMax H3 T8/`.

## 10Eros curve-pruned Turbo LoRA checkpoint (2026-08-10)

An independent Experimental acceleration LoRA was generated for the exact fine-tuned curve-pruned
checkpoint `10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors`. No ComfyUI node, stable sampler,
workflow contract, or input model was modified. The result is a LoRA, not a fused 40GB checkpoint.

The immutable inputs were locked and re-hashed after publication:

| Role | Bytes | SHA-256 |
|---|---:|---|
| 10Eros pruned BF16 main | 40,225,724,112 | `f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062` |
| existing ComfyUI Standard Turbo LoRA | 779,858,752 | `35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b` |
| full FL2VA time-embedder reference | 34,038,892,334 | `7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5` |

Only the reference checkpoint's four FP32 `time_embedder` tensors were read. Its quantized
Transformer weights were not used. The target's `adaln_t_table [1025,8]` has raw SHA-256
`ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e` and is bit-identical to
the installed official pruned FL2VA table.

Static target inspection found 208/259 directly compatible attention/MLP adapters and 51
incompatible AdaLN adapters whose source A is `[16,2688]` while the pruned target consumes eight
curve coordinates. `tools/convert_minimax_h3_turbo_for_pruned_curve.py` therefore retained the 208
direct adapters byte-for-byte and fit each AdaLN response over `t_j=j/1024` with
`pinv([adaln_t_table,1])`. It stored `[16,8]` A as BF16, retained B exactly, and stored the mandatory
output-space intercept as FP32 `.diff_b=B@c`. A no-intercept conversion is explicitly rejected:
prior read-only measurement showed that it loses roughly 94%-99.8% of the AdaLN Turbo response.

The converter has no force/overwrite option. It validates a same-directory unique `.partial`, uses
non-overwriting publication, and re-hashes all three inputs after both outputs are complete. Five
focused tests cover native four-step times, exact curve interpolation, the required affine
intercept, output-space error algebra, and overwrite refusal.

Generated outputs:

| Output | Bytes | Tensors | SHA-256 |
|---|---:|---:|---|
| `minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001.safetensors` | 794,888,696 | 569 | `6c2f38d45dfa3fc282a48de3171b6946a5e6d46e13f832c43b93734f6d12edf5` |
| `minimax_h3_turbo_4step_10ErosMax_test4_pruned_core208_ablation_v001.safetensors` | 620,286,192 | 416 | `1af864dcc864d998b342472cc53248cfbdc58b6c54294220e8b2e650600c63ff` |

Independent current-ComfyUI parsing resolved the curve output as 259 `WeightAdapterBase` objects
plus 51 regular bias-diff patches and consumed 569/569 source keys. Every target weight and bias
shape matched the 10Eros checkpoint. The core ablation resolved 208 adapters and consumed 416/416
keys. The 416 direct tensors and all 51 projected-module B tensors were `torch.equal` to the source.
No `.partial` remained. Stored projection errors were:

| Metric | Relative error |
|---|---:|
| 1025-point aggregate | `9.57255e-5` |
| native four-step aggregate, including visual/audio conditioning times | `1.58861e-4` |
| worst native-four module | `5.85406e-4` |

A live same-prompt/same-seed smoke then used ComfyUI 0.31.0 at `cbbc9dab1`, an RTX 4060 Ti 16GB,
the 40GB BF16 10Eros main, NVFP4 Qwen3-VL, both H3 VAEs, bypass LoRA strength 1.0, 256x256,
124 frames, T2VA, four-step `dual_clock_euler + native_flow`, shifts 12/3, and seed `2608104101`.
The running server had SageAttention enabled. Both curve and core208 prompts completed successfully:

| Treatment | Runtime | Video | Decoded audio | Local faster-whisper |
|---|---:|---|---|---|
| curveproj1025 | 357.84s | 124 frames; six-frame screen coherent | stereo 32kHz; 162,816 samples; RMS 0.01879; peak 0.15245; finite; 0% clipping | `Today, the rain finally stopped.` |
| core208 | 288.58s | 124 frames; six-frame screen coherent | stereo 32kHz; 162,816 samples; RMS 0.01222; peak 0.12440; finite; 0% clipping | `Today the rain finally stopped` |

The pair's full-video RGB MAD was 5.615/255 and decoded-audio correlation was 0.719. This proves
that the projected AdaLN route is active rather than behaviorally equivalent to dropping the 51
modules. It does not prove that either treatment is perceptually superior. Sparse polling observed
up to 14,869MiB whole-device use, but this was not a peak-telemetry harness.

This checkpoint passes structural conversion, current ComfyUI consumption, one real four-step AV
execution, finite/non-clipped audio, intelligible target speech, and a six-frame no-melt screen. It
does **not** pass the planned 20-step anchor, four-step no-LoRA negative control, Stock-attention
control, high-resolution matrix, held-out prompts/seeds, blind listening, identity, or lip-sync
gates. The LoRA remains Experimental and exact-checkpoint-specific. Detailed sidecars are next to
the local generated outputs and are intentionally not tracked by Git.

## 1.12.0 experimental dialogue-safe audio checkpoint

Three nodes were appended after the prior 51-node inventory. No old node ID, order, default,
schema prefix, or execution path changed:

- `MiniMaxH3DialogueBoundaryAnalyzerT8` performs read-only local CPU faster-whisper analysis. It
  returns a boundary only for exactly one contiguous exact normalized target sequence; zero or
  multiple exact spans are rejected. Signal energy after the boundary is reported as activity,
  not classified as speech.
- `MiniMaxH3DialogueSafeMasterT8` requires an upstream accepted boolean and independent stems. It
  places verified speech on an exact sample timeline and preserves full-duration music, ambience,
  and SFX after speech ends. Strict is the default fit policy; loop/pad/trim only occur when the
  user explicitly selects and receives a report for them.
- `MiniMaxH3TimedAudioBedLockT8` is a two-pass H3 helper. It encodes an independent dialogue-free
  background bed, preserves the input video stream/mask, and applies a 40Hz audio mask with a
  default zero-denoise tail. Existing audio masks are caps, so the node never increases generation
  freedom in an already constrained interval.

Model-free boundary, mixing, mask, immutability, strict-fit, explicit-fit, and registration tests
passed. A real FL2VA pruned INT8, Qwen3-VL NVFP4, H3 Audio VAE, 256x256, 124-frame, stable
dual-clock four-step forward also passed. The standard 124-frame Audio Window encoded to 206 audio
latent steps against the AV clock's 207, so strict mode deliberately failed and `fit_reported`
explicitly zero-padded one step. Comparing saved audio latents before and after sampling at the
2.0-second/step-80 lock boundary produced:

| Region | Mean absolute change | Maximum absolute change |
|---|---:|---:|
| editable head, steps 0–79 | 0.502230 | 2.402396 |
| locked tail, steps 80–206 | 1.81e-8 | 2.38e-7 |

The locked tail therefore remained within `1e-6` absolute tolerance across four sampler steps,
while the head materially changed. This establishes the latent-mask endpoint mechanically, not
decoded perceptual quality. The same Audio VAE decoded both latents to 165,600 samples at 32kHz.
The first 100ms after the 2.0-second boundary still had a 0.34177 maximum difference, and decoder
influence decayed through roughly 2.3 seconds; later 100ms windows were approximately `3.97e-4` or
lower in maximum difference. A latent boundary is therefore not a sample-exact audio cut and is
not advertised as seamless.

The read-only analyzer was also run against two prior real Joint-dialogue failures using the local
multilingual small model. When unwanted words interrupted the expected sequence, it returned
`target_not_found`. When 17 unwanted units preceded one contiguous exact target, it returned
7.00–9.72 seconds with `clean_exact=false`; it did not auto-trim or accept the mixed result.

Automatic source separation remains deliberately absent. The installed `audio_separator` Python
package has no selected local model, and common vocal/music separators are not evidence of safe
target-dialogue removal from a master containing intentional singing, music, ambience, and SFX.
Synthetic known-stem leakage/damage tests, real H3 mixes, and listening gates are required before
any separator can become an opt-in experiment. No speech-stop, mouth-stop, seamless-tail,
source-separation, or 16GiB `memory_safe` claim is made.

## 1.9.0 experimental visual-reference strength checkpoint

`MiniMaxH3VisualReferenceStrengthEXPT8` was added as node 36, after all 35 existing nodes. It is a
post-conditioning node that calls `node_helpers.conditioning_set_values()` with
`minimax_visual_cond_noise_aug`; it does not receive or patch MODEL, latent, VAE, sampler,
scheduler, sigmas, shifts, or steps. It rejects missing visual conditions and audio-only refs,
allows keyframes with an explicit global-scope warning, and reports values at or below 0.950 as
aggressive. The current H3 core maps this field to `visual_cond_noise_aug` and uses 0.999 when the
field is absent.

The live matrix used ComfyUI `0.31.0` at
`cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`, Windows, an RTX 4060 Ti 16GiB, DynamicVRAM, the full
`minimax_h3_ref2va_int8_convrot.safetensors`, Qwen3-VL NVFP4 CLIP, FP16 video VAE, FP32 audio VAE,
and no LoRA. Every treatment fixed one reference image, prompt, seed `2608102201`, 736x416,
124 frames, 20 steps, `dual_clock_euler + native_flow`, and shifts 12/3.

| Treatment | Runtime | Whole-device peak | Minimum free | Result |
|---|---:|---:|---:|---|
| no post node | 254.375s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.999 | 303.984s | 16,344.617MiB | 34.883MiB | success |
| explicit 0.995 | 276.812s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.990 | 276.563s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.980 | 264.703s | 16,337.598MiB | 41.902MiB | success |
| explicit 0.950 | 378.860s; repeat 482.731s | 16,017.422MiB on repeat | 362.078MiB | both success |

The timing variation includes dynamic-loading stalls and is not attributed to the scalar itself;
the graph retains 20 DiT steps and the new node adds no model call. All observed free margins are
below the project's 512MiB safety gate, so this matrix does not establish a 16GiB memory-safe tier.

The critical compatibility gate passed exactly. Decoded no-node and explicit-0.999 video both
contained 124 RGB frames and 113,897,472 bytes with identical SHA-256
`59b0cff4408b2656f7c42c9e2c5430649e25b8899047fe7d54cf45e69b5763df`; their decoded float32 audio
both contained 325,632 samples and had identical SHA-256
`82796fa54b165f0ad9c86bf00777d47f68d11637328c567bb991e1f05ec8477f`. Both maximum absolute errors
were zero. Two explicit-0.950 runs were also decoded-video and decoded-audio identical, with maximum
absolute errors zero. This proves deterministic routing for these fixed controls; it is not a
cross-hardware bitwise guarantee.

Whole-video objective proxies were computed over all 124 decoded frames:

| Strength | Mean RGB MAD vs 0.999 | Temporal MAD | Temporal gray SSIM | Face high-pass std |
|---:|---:|---:|---:|---:|
| 0.999 | 0.0000 | 4.9202 | 0.89021 | 8.1772 |
| 0.995 | 24.8277 | 3.6117 | 0.91423 | 8.1577 |
| 0.990 | 25.5275 | 4.1108 | 0.90307 | 8.1552 |
| 0.980 | 13.5682 | 4.9253 | 0.88860 | 8.2397 |
| 0.950 | 33.5909 | 3.4078 | 0.92725 | 7.6752 |

MAD/SSIM only measure change, and Haar-face plus high-pass/Laplacian values are composition and
sharpness proxies, not identity or skin-realism scores. Higher temporal SSIM can also mean less
motion. Manual first/middle/last-frame review found that 0.995 through 0.950 changed pose,
expression, motion trajectory, and/or composition; 0.950 introduced the largest background and
framing shift. Its full-frame edge variance rose while its face high-pass proxy fell. This one
case therefore proves that the control is effective, but it does not establish a winning
"de-wax" value or monotonic visual improvement.

Regression evidence:

- 188 project tests passed; Ruff reported no findings;
- isolated live `/object_info` exposed the exact two inputs, 0.999/0..1/0.001 numeric contract,
  two named outputs, EXP flag, and category;
- the API and ComfyUI 0.4 frontend examples passed structural/link checks and the frontend workflow
  passed live object-info validation;
- `sampling.py` SHA-256 remains
  `111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`;
- `conditioning.py`, `sampling_multirate_exp.py`, and `still_image.py` also remain byte-for-byte
  unchanged, with SHA-256 `E15D95454FFD60076FFADECA5C205B9608AE225606ED955A09AAD95F0212C9E4`,
  `BADCFA055938FF2AB0E0B8BD8C2FD789B6FAB33CC312F891E5226E8419BD4D5F`, and
  `B154E3E154FD4DB1927A7E52BE96AC05EA827BE0A8CE6B5C2A27529016B23CE8`.

Local detailed telemetry, decoded-equivalence files, objective metrics, and the contact sheet are
under `artifacts/ref2va-visual-strength-check/`. That directory is intentionally excluded from Git.
Remaining quality work is a representative multi-reference, multi-seed, image/video/keyframe and
human-preference matrix; until then the node remains Experimental and must not be described as a
Ref2VA oiliness fix.

## 1.7.0 explicit background executor checkpoint

Implemented locally on 2026-08-09 as two nodes appended after the prior 23. Stable sampling
math and every old node position remain unchanged.

- `Background Start` executes before model work, captures the complete API prompt only in
  process memory, associates it with the current ComfyUI prompt ID, and starts history
  monitoring so errors before the terminal node are observable;
- `Auto Accept & Continue` is the explicit output boundary. In background mode it validates and
  accepts one candidate, requests the configured release policy, validates one copied prompt,
  and queues exactly one next segment. Its safe default remains non-mutating `review_only`;
- persistent `background_job.json` contains lifecycle state, counters, accepted paths, and a
  prompt SHA-256, but not the prompt body. The error serializer allowlists node/error/traceback
  fields and excludes ComfyUI `current_inputs/current_outputs`, media tensors, and prompt text;
- controls are status, pause-after-current, resume, and prompt-ID-targeted cancel. Pause retains
  the accepted manifest. Cancel does not wipe unrelated ComfyUI queue items;
- retries reuse the exact prompt and never silently change size, frames, context, seed, sampler,
  scheduler, or steps. The default is one additional attempt; exhaustion leaves `failed`;
- release policies are `keep_loaded`, `clear_execution_cache`, and `unload_all_models`.
  The middle option explicitly sets `unload_models=false` plus `free_memory=true`; the strong
  option is correctly described as a global ComfyUI model unload, not an H3-only operation.
  The selected policy is requested after every durable acceptance, including pause and final;
- copied prompts are sanitized of ComfyUI runtime `is_changed` fingerprints before queueing.
  This prevents `keep_loaded` from treating an advanced segment as a whole-graph cache hit;
- final composition may run automatically. Any failure after the manifest acceptance boundary
  stops without retrying a prompt whose Orchestrator would already resolve to a later segment.

Validation evidence:

- 139 project unit/structure tests and Ruff pass; the original 23 IDs keep their order and the
  two background IDs are appended;
- isolated `--quick-test-for-ci` import succeeds; live `/object_info` reports both schemas and
  the status/control routes respond with expected 200/409 semantics;
- a model-free live graph accepted and composed two segments under two prompt IDs;
- another graph paused while segment 0 was running, reached `paused` after accepting exactly one
  segment, resumed through a new prompt ID, and completed segment 1;
- a targeted running-prompt cancel signalled an interrupt, ended with error history, accepted
  zero segments, and created no manifest;
- a deterministic upstream error produced exactly two history records with `max_retries=1`,
  retained the same prompt/settings, then stopped in `failed`. This probe exposed that raw ComfyUI
  errors contain complete `current_inputs`; the allowlist fix above was added before release;
- the first live resume route test exposed a same-event-loop wait and returned a 60-second timeout.
  Route controls were moved through `asyncio.to_thread`; the complete pause/resume test then
  passed in 5.7 seconds with an immediate `running` resume response.

A final real-model mechanical probe used non-pruned FL2VA INT8, Standard Turbo EMA LoRA,
Qwen3-VL NVFP4, both H3 VAEs, 256x256, a 124-frame window, 22-frame AV context, one sampling
step, DynamicVRAM with 2.0GiB headroom, and `unload_all_models` between segments. Both distinct
prompt IDs completed successfully. Manifest revision 2 contains 124+20=144 contiguous frames
and 192,000 absolute audio samples; the automatically composed H.264/AAC file reports 24fps,
144 frames, and exact 6.000-second video, audio, and container streams. Runtime was 79.41 seconds;
whole-device polling observed 7,129MiB baseline and 14,254MiB peak. The final SHA-256 is
`e24acdc57996ae15a15a1590f3066c738f3ce39ba1949ffd5b42f2743e75eb7b`.

This one-step 256x256 result validates executor mechanics and strong-release reload only. It does
not establish four-step perceptual quality, high-resolution memory safety, cross-GPU behavior,
multi-reference safety, or any universal no-OOM guarantee. A separate, bounded crash-recovery
probe is recorded below.

The crash-recovery probe hard-killed the owned ComfyUI process after segment 0 was durably
accepted at manifest revision 1 while the next prompt was active. After restart, status converted
the stale persisted active state to `detached`, retained `accepted_count=1`, and returned
`recovery_action=queue_workflow_once`. Requeueing the workflow once created a new job linked to
the old job and generated segment 1 only. Segment 0's candidate ID, MP4 SHA-256, AV-tail tensor
SHA-256, and file modification time were unchanged. Revision 2 produced a 256x256, 144-frame file
whose video, audio, and container durations are exactly 6.000 seconds. The post-lease repeat's
native-v2 recovery whole-device peak was 13,537.02MiB and the final SHA-256 is
`5a2d59d69c8ff56549a76a0d274d8ce61c194bb1ebac2298f8e2803ba21461d8`.

The same probe then queued two independent real H3 background chains. The single ComfyUI prompt
queue interleaved them as `A0, B0, A1, B1`; both reached revision 2 with isolated jobs, prompts,
parent chains, roots, manifests, and final files. The native-v2 repeat peaked at 13,511.44MiB and
no OOM occurred.

The local multi-process gate now uses an OS-owned `manifest.lock.v2`. Two processes competing for
one chain/index/revision produced exactly one accepted revision and one pre-copy rejection; four
processes retained all 100 of 100 protected updates; and a killed lock owner was replaced within
two seconds. A chain-wide background lease rejected a second ComfyUI process before generation,
then allowed a third process to attach with `previous_job_id` after the first was killed. Tests
also prove live legacy locks are respected, dead legacy residue remains in place but does not
block v2, unknown schemas cannot roll back to an older backup, same-schema optional metadata is
preserved through a later commit, invalid primaries do not poison valid backups, unreadable
auxiliary state is quarantined, and newer background state is not overwritten.

Accepted manifests and background states now use explicit schema-2 format markers. Fixture tests
prove read-only in-memory normalization of schema 1, next-write atomic manifest upgrade with the
raw schema-1 primary preserved as backup, schema-1 background reattachment to schema 2, recovery
through a schema-1 backup, and fail-closed handling of unknown future schemas. An existing real H3
schema-1 chain was read without changing either raw file hash, while a new hard-kill plus dual-chain
probe wrote and retained native schema 2. In that repeat, all 13 crash assertions and all 9
isolation assertions passed.

An acceptance-transaction audit added eight deterministic fault-injection cases. Exact
idempotent re-accept now repairs missing/corrupt accepted assets from a hash-verified candidate
without changing manifest revision. Reusing a canonical candidate id for different bytes, or
reusing an invalidated id to collide with an archived path, is rejected before any overwrite.
Context-copy failure and failure after the backup write but before primary replacement leave the
old manifest authoritative and are recoverable by retrying the same candidate. A missing primary
now loads a valid backup even under `allow_new=True`; unknown-schema or corrupt backups fail closed
instead of resetting the chain to empty.

The accepted-media/manifest split and backup/primary split were then tested with actual
subprocess termination on Windows/NTFS. The worker held `manifest.lock.v2`; the parent killed it
after the accepted MP4 copy but before context/manifest, and after the revision-1 backup write but
before revision-2 primary replacement. Three independent rounds exercised both breakpoints, for
6/6 successful recoveries. The OS lock released automatically, the pre-kill manifest state stayed
authoritative, and retrying the same candidate completed in under two seconds with the expected
hashes and revisions. These were small-media transaction tests, not live H3 CUDA kills or power-loss
tests.

This closes the tested post-acceptance hard-kill boundary, same-queue multi-chain isolation, local
schema-1 to schema-2 migration contract, and same-host Windows/NTFS same-chain ownership/manifest
serialization. It does not validate a complete upgrade/downgrade matrix across independently
released plugin/ComfyUI builds, network/shared-filesystem locking, arbitrary-instruction recovery,
simultaneous GPU execution, or multi-GPU parallelism.

The representative four-step gate was subsequently run with non-pruned FL2VA INT8, Standard
Turbo LoRA, 736x416, a 124-frame render window, 22-frame AV context, DynamicVRAM headroom 2GiB,
and global `unload_all_models` between segments. A 60-second request produced 14 distinct prompt
IDs and every history record ended in success without retry or OOM. Accepted manifest revision
14 is exactly `124 + 12*102 + 92 = 1440` frames and 1,920,000 audio samples. Every accepted media
and continuation-context hash matches the manifest, and the candidate-parent chain is contiguous.
The final H.264/AAC file has 736x416, 24fps, 1440 decoded frames, and exact 60.000-second video,
audio, and container streams. Its SHA-256 is
`cb3bdf5bae847c6f0fe708d991bd35a736fdc012e44cb21ef612d7c0f2f83ed0`.

Half-second `/system_stats` polling measured 1,229.63MiB baseline, 12,823.13MiB whole-device
peak, 3,556.37MiB minimum free margin, and 3,496.30MiB PyTorch peak. The running maximum reached
12,784.24MiB at segment 3, increased only 38.89MiB at segment 9, then remained flat through
segment 14; this one sequence shows no staircase leak. Total runtime was 1,478.83 seconds.
The same prompt/seed's segment-0 MP4 is bit-identical to the six-second preflight; context file
metadata differs, while `video_tail` and `audio_tail` tensor payloads remain bit-identical.

Across 13 visual seams, MAD median/max was 0.04696/0.07420 and SSIM median/min was
0.73139/0.64589. Only two seam MAD values were locally highest; inspection of all contact pairs
found no obvious scene cut, identity reappearance, or camera reversal in this walking shot.
Audio half-second level change had -0.42dB median and 13.75dB maximum absolute value. The 5ms
bridge reduced median single-sample jump from 0.08247 in source segment contacts to 0.00222 in
the final encode, about 96.16%, but does not solve level, speech, music, timbre, or lip-sync
continuity. Therefore this is a passed local fixed-profile/single-case gate, not a universal
`memory_safe`, seamless, or never-OOM result. Later evidence closes fixed-prompt cross-base-seed
cold mechanical repeatability only; same-seed whole-chain warm repeats, different materials, other GPUs,
high resolutions/references, actual cross-version migration, network/shared-filesystem locking,
and true GPU parallelism remain open; the separate post-acceptance hard-kill, local multiprocess,
and same-queue dual-chain probes above do not close those broader gates.

The 60-second run also exposed that the original release call was only reached when another
prompt was queued, so pause/final states retained model VRAM. The state machine was changed to
request the selected policy immediately after every durable acceptance and before branching to
continue, pause, or final. A real 256x256 one-step/two-segment follow-up completed with
`last_release_policy=unload_all_models`; whole-device use fell from about 8,124MiB at completion
to 1,230MiB within 15 seconds without calling `/free`, a 6,894MiB drop. Unit coverage also locks
pause/final release and stops without queueing if a release request fails after acceptance.

A controlled release-policy matrix then used the same FL2VA INT8 checkpoint, Standard four-step
Turbo LoRA, prompt, 736x416 canvas, 124-frame window, 22-frame AV context, and two-segment
six-second plan. Three paired seeds were run for each policy in three fresh-process cold trials.
For warm trials, each policy received one unmeasured same-process primer followed by the same
three measured seeds. All 21 chains succeeded without retry or OOM; all 18 measured chains reached
manifest revision 2.

| Policy | Cold runtime mean (SD) | Warm runtime mean (SD) | Cold/warm peak mean | Cold/warm post-15s mean |
|---|---:|---:|---:|---:|
| `keep_loaded` | 170.89s (2.24) | 153.10s (0.12) | 13,449.95 / 13,467.30MiB | 8,083.22 / 7,987.22MiB |
| `clear_execution_cache` | 188.28s (0.80) | 185.08s (0.42) | 13,434.52 / 13,408.94MiB | 1,229.63 / 1,229.63MiB |
| `unload_all_models` | 189.08s (0.68) | 197.13s (3.07) | 13,421.52 / 13,384.03MiB | 1,229.63 / 1,229.63MiB |

For each seed, segment-0 video, segment-0 accepted AV-tail tensor payload, segment-1 video, and
final H.264/AAC file hashes were identical across all six policy/temperature conditions. Every
paired whole-device peak difference was below the existing 128MiB material-difference threshold;
the roughly 1GiB strong-unload peak advantage in the earlier single run did not repeat.

Relative to `clear_execution_cache`, `keep_loaded` was 17.39 seconds faster cold and 31.97 seconds
faster warm on average, but retained 6,853.59/6,757.59MiB more device memory after 15 seconds.
Global unload was only 0.80 seconds slower cold but 12.05 seconds slower warm than the default,
with no material peak advantage and broader side effects. The default therefore remains
`clear_execution_cache`; `keep_loaded` is an explicit throughput-for-residency tradeoff.

The first `keep_loaded` attempt had exposed that current ComfyUI mutates prompt nodes with runtime
`is_changed` fingerprints: replaying those values cached the whole second prompt and prevented the
manifest from advancing. Sanitizing the snapshot fixed the issue. Across the complete matrix,
keep-loaded cached only unchanged loader nodes 1-5; orchestration, sampling, saving, and terminal
acceptance reran. The other policies had no cached nodes. These results remain specific to the
local GPU/model/profile and do not establish cross-GPU or universal no-OOM behavior.

## 1.6.0 total-duration orchestration and resume checkpoint

Implemented locally on 2026-08-08 without changing the stable sampler:

- `MiniMaxH3LongVideoOrchestratorT8` registers after the prior 22 nodes and emits only
  plan/state values; it does not own a MODEL or retain historical IMAGE/AUDIO tensors;
- total duration is quantized once to an exact 24fps frame count, then split across a
  fixed legal `17n+5` render window. The default 60-second/124-window/22-context plan is
  14 segments with effective frame counts `124 + 12*102 + 92 = 1440`;
- the final short tail still renders through the same fixed 124-frame H3 window and is
  trimmed only as final output, preserving the bounded per-segment sequence contract;
- global and per-segment prompt/seed/note values are supported, with fixed, incrementing,
  and deterministic chain/segment hash seed policies;
- steps, video/audio shifts, sampler, and scheduler are single-source Orchestrator values.
  They drive the stable sampler and a machine-generated candidate `sampling_summary`; accepted
  chains reject a changed summary before another segment is sampled;
- the contiguous accepted-manifest length selects the first unaccepted segment. Accepted
  fps, frame count, absolute timeline, and final identity are validated against the plan;
  incompatible settings fail instead of silently resuming the wrong chain;
- after the final segment is accepted, the node returns full progress and a ComfyUI
  `block_execution` reason, preventing an accidental extra sampling pass;
- the frontend and API graphs are respectively
  `examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json` and
  `tests/fixtures/api/long_video_auto_resume_api.json`.

Validation evidence:

- 104 project unit/structure tests pass, including duration quantization, fixed-window
  planning, final-tail trimming, prompt/seed overrides, deterministic hash seeds,
  manifest resume, changed-plan rejection, and complete-chain execution blocking;
- Ruff and `git diff --check` pass;
- stable `sampling.py` remains SHA-256
  `111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`;
- ComfyUI `--quick-test-for-ci` imports the plugin successfully in isolation;
- a temporary isolated live server registers 23 T8 nodes, exposes the Orchestrator with
  13 required inputs and 22 outputs, and lists the installed auto-resume frontend
  workflow through `/userdata`.

A real execution probe then ran the new API graph in the user's normal DynamicVRAM environment:

- non-pruned FL2VA INT8, Standard Turbo LoRA, NVFP4 H3 CLIP, both H3 VAEs, 736x416,
  a 124-frame internal window, one sampling step, and a one-second target;
- the joint H3 sampling, video/audio decode, exact final trim, and candidate save completed;
- the candidate descriptor contains 24 frames, absolute audio samples `[0, 32000)`, final=true,
  and no continuation context. PyAV reports 24fps/24 frames and exactly 1.0-second video,
  audio, and container streams;
- accepting the candidate produced manifest revision 1. Re-queueing the complete graph returned
  success with no downstream outputs and did not create another candidate, proving the final
  execution block in live ComfyUI;
- accepted-file composition produced a 736x416, 24fps, 24-frame MP4 whose video, audio, and
  container durations are all exactly 1.0 seconds.

The first controlled attempt used `--novram` and completed H3 sampling but failed at audio VAE
decode with a CUDA-input/CPU-filter device mismatch. A separate core-only graph using
`EmptyMiniMaxH3LatentAV -> VAEDecodeAudio` reproduced the same traceback at
`comfy/ldm/minimax/audio_vae.py:102`, without any T8 node in the decode path. This isolates the
failure to current ComfyUI H3 Audio VAE dynamic buffer handling under `--novram`; the normal
DynamicVRAM route succeeds. No plugin workaround was added without an independent correctness
and memory proof.

During the one-step probe, the temporary API mutation changed the sampler step count but initially
left the example's manually entered description at its four-step default. This exposed a genuine
provenance weakness rather than a sampling failure. The post-probe hardening removes that duplicate
entry from the auto-resume workflows: Orchestrator now emits steps/shifts/sampler/scheduler to the
sampler and emits the corresponding machine-generated summary to Candidate Save. Targeted schema,
API, frontend-link, resume-conflict, and completion tests pass for this revised route. A second
real-model run changed only Orchestrator `steps` to 1; the actual sampler ran 1/1 step and the
candidate recorded `1-step dual_clock_euler/native_flow shift12/3`. Acceptance, completion
blocking, and exact one-second composition succeeded again, so the provenance fix is covered by
runtime evidence rather than structure tests alone.

A second real probe covered four-step multi-segment auto-resume with the same non-pruned FL2VA
INT8/Standard Turbo/NVFP4 CLIP/dual-VAE stack, 736x416, a fixed 124-frame window, 22-frame AV
context, and a six-second target. Segment 0 produced 124 frames; after acceptance, the same API
graph resumed at segment 1 with the accepted parent/context and produced a 20-frame final tail.
The revision-2 manifest covers exactly 144 frames. A further queue returned success with zero
output nodes and no extra candidate. Both unbridged and 5ms-bridge compositions report 24fps,
144 frames, and exact 6.000-second video/audio/container streams.

The bridge reduced the post-AAC single-sample boundary jump from 0.000178755 to 0.000035433,
about 80.2%, but the adjacent audio-window level changed by about 33.30dB. Video contact-sheet
inspection found no obvious identity/composition cut, while boundary MAD and SSIM discontinuity
both ranked at the 100th percentile among 16 nearby intra-segment transitions and flow ranked at
87.5%. Device peaks were about 15,461.4 and 16,181.5MiB; the second leaves only about 198MiB and
fails the 512MiB safety gate even with `--vram-headroom 0.5`. Background auto-queue,
pause/cancel, automatic retry/model release, the multi-material 8-16 segment quality matrix,
blind listening, and VRAM safety tiers remain open.

### Four-step 60-second / 14-segment runtime checkpoint

A third real probe used the same FL2VA INT8/Standard Turbo/NVFP4 CLIP/dual-VAE stack,
736x416 canvas, 124-frame render window, 22-frame AV context, four-step
`dual_clock_euler/native_flow shift12/3`, and DynamicVRAM. It ran all segments in one warm
ComfyUI process without an explicit `/free` call:

- all 14 candidates were generated and accepted in parent/context order; the manifest reached
  revision 14 with effective segment frames `124 + 12*102 + 92 = 1440`, timeline end frame
  1440, audio end sample 1,920,000 at 32kHz, and a final segment;
- unbridged and 5ms cosine-bridge compositions both report 24fps/1440 frames and exactly
  60.000-second video, audio, and container streams;
- the first uncached full re-queue after final acceptance returned the expected ComfyUI
  `ExecutionBlocked` terminal status with an empty traceback and execution stopped at the
  Orchestrator. A later cached re-queue reported success while executing only the review node.
  Neither path sampled or created a fifteenth candidate;
- measured device peaks ranged from 15,479.95 to 16,228.17MiB. The descriptive warm-run peak
  slope was +28.03MiB/segment, the baseline slope was negative rather than a monotonic staircase,
  and the peak range was 716.83MiB. This one run does not show a cumulative VRAM leak, but five
  segments left less than 512MiB and segment 12 left only 151.33MiB. It fails the proposed 16GB
  safety gate; 0.25-second polling can also miss shorter spikes;
- generation time summed to about 1,336.8 seconds for the 14 segments, excluding orchestration,
  acceptance, composition, and offline analysis.

All 13 video boundaries were decoded and compared. Median/max boundary MAD were
0.01618/0.01906; median/min SSIM were 0.96374/0.92868; median/max mean optical flow were
0.2603/0.4748. The worst values occurred at the 11-to-12 seam. Contact-sheet inspection did not
show a hard subject/background cut at the worst two seams, while the segment-middle timeline
showed gradual subject-appearance and exposure drift. Pixel and flow metrics are not identity
verification.

Audio evidence rejects a seamless or lossless claim. Adjacent half-second level change had a
median of -9.51dB and a maximum absolute value of 40.83dB at the 12-to-13 seam. The above-8kHz
energy ratio fell from -32.14dB in segment 0 to -68.44dB in segment 13, a -36.30dB change that is
consistent with substantial recursive high-frequency loss. Descriptive adjacent-window NCC had
a median of 0.206 and median absolute best lag of 65.16ms; these are not overlap-reconstruction
scores. The cosine bridge reduced the median post-AAC single-sample boundary jump by 97.23%,
but it cannot repair level, spectrum, speaker identity, speech semantics, or lip sync.

This checkpoint proves one exact, resumable 60-second execution and bounded peak behavior in one
warm process. It does not satisfy the multi-material/multi-seed quality matrix, cold/warm repeat
matrix, blind listening, ASR/speaker/lip-sync evaluation, or a publishable 16GB safety tier.

### 5-frame versus 22-frame context pilot

Two controlled 736x416 probes used the same prompt, base/incremented seeds, model, Standard
Turbo LoRA, four-step sampler, render window, duration, DynamicVRAM arguments, and no explicit
`/free` between segments. Only context length and isolation labels differed. Segment 0 was
bit-identical across the 5-frame, historical 22-frame, and control 22-frame chains. The historical
and control 22-frame segment-1 MP4s were also bit-identical. Nevertheless, the same 22-frame
segment-1 output had measured peaks of about 16,181.55 and 15,194.31MiB, a 987.24MiB range.
This directly demonstrates that a single polled peak cannot be attributed to context length.

In the matched no-`/free` pair, the 5-frame segment-1 peak was 15,330.47MiB and the 22-frame
control was 15,194.31MiB; normalizing each to its own context-free segment 0 still left 5 frames
about 66.94MiB higher. The 5-frame segment was 8.30 seconds (8.77%) faster. Video seam metrics
were mixed and visually both contacts remained continuous in this low-motion sample. The
5-frame audio level drop was 43.65dB versus 33.29dB for 22 frames, while its descriptive NCC/lag
were better; neither route supports a seamless-audio claim.

A second pair raised the canvas to 1056x608 (642,048 pixels) and used 0.10-second VRAM polling.
Both two-segment chains completed and all four bridge/unbridged assemblies were exactly
24fps/144 frames with 6.000-second video, audio, and container streams. Segment 0 was again
bit-identical. The 5/22-frame segment-1 peaks were 15,205.91/15,341.03MiB with free margins
1,173.59/1,038.47MiB. After normalizing against each pair's segment-0 peak, the context delta
differed by only 11.44MiB. The 5-frame segment was 30.28 seconds (12.86%) faster, but the
22-frame route had lower video MAD (0.01477 versus 0.01612), higher SSIM (0.95616 versus
0.95035), a smaller audio level change (-2.39 versus +7.01dB), and higher descriptive NCC
(0.406 versus 0.111) in this single sample.

The planned alternating-order repeat gate was then completed at both resolutions. Each matrix
used a fixed accepted segment-0 baseline per context, three paired seeds, three isolated cold
runs per context, a same-process primer plus three warm runs per context, and 0.10-second polling.
Within each resolution, segment-0 MP4s and AV tail tensors were bit-identical; all six matching
context+seed cold/warm outputs were also bit-identical.

At 736x416, cold absolute peak means were 15,279.5MiB for 5 frames and 15,224.0MiB for 22 frames.
Paired `22-5` differences were +96.6/-78.3/-184.9MiB, so the device-peak direction was not
repeatable. Sampler PyTorch-pool means were 3,189.9/3,495.3MiB, a repeatable local reduction of
about 305MiB for 5 frames. Cold runtime means were 86.53/93.08 seconds; warm means were
69.27/78.01 seconds. The warm minimum margin was 97.6MiB and 5/6 runs were below 512MiB.
The three-seed 22-frame mean video MAD/SSIM and audio level/NCC were better at this resolution.

At 1056x608, cold absolute peak means were 15,739.0/15,724.2MiB, but paired `22-5` differences
were -752.0/+10.8/+696.7MiB. Sampler pool means were 5,753.4/6,381.2MiB, a repeatable local
reduction of about 628MiB for 5 frames. Cold runtime means were 200.29/230.38 seconds; warm means
were 187.89/218.40 seconds. All six warm runs were below the 512MiB gate and the minimum margin
was only 33.6MiB. Five frames had lower mean MAD/higher SSIM, which may include reduced motion;
22 frames had materially better audio level/NCC, while one seed showed a clear front-to-profile
video boundary change.

The 39-frame gate then reused the 736x416 accepted baseline and the same three paired seeds. Three
isolated cold starts and three same-process post-primer warm trials all completed without OOM;
matching cold/warm candidate MP4s were bit-identical, and the 5/22/39 segment-0 MP4 plus video/audio
tail tensors were identical. The 39-frame cold/warm sampler-pool means were 3,799.32/3,798.63MiB,
repeatably about 303-304MiB above 22 frames. Cold/warm runtime means were 101.65/87.38 seconds.
The warm process reached only 77.35MiB free and 3/3 measured trials failed the 512MiB gate.

Quality did not improve monotonically. The three-seed 39-frame mean video MAD/SSIM was
0.08801/0.68411 versus 0.00839/0.95501 at 22 frames. Manual contact-sheet inspection found one
continuous low-motion boundary, one visible pose/framing jump, and one severe identity/shot change;
the latter two were visibly worse than their 5/22-frame controls. Audio NCC increased to 0.626 on
average, but mean absolute level change was 10.27dB, so the audio evidence also does not establish
a universal quality tier.

Decision: keep 5 frames as `fast_context_5_experimental`, not `memory_safe`. Its runtime and
sampler-pool savings are real, but the end-to-end device peak is dominated by DynamicVRAM model
residency/conditioning state and changes direction between paired runs. Twenty-two frames remains
the current balanced default candidate. Thirty-nine frames is downgraded to
`context_39_high_risk_experimental`, not a quality or safety tier. The 1056x608/39-frame treatment
was not forced because its 22-frame warm control already failed the 512MiB gate in all six trials
and reached only 33.6MiB free. This is a predefined safety-gate denial, not a claim that the
unexecuted treatment must OOM. The controlled Block Cache/Sage/DynamicVRAM headroom gate was
subsequently completed below; multi-material dialogue/fast-motion/rhythmic-audio quality tests
and the 1056x608/39-frame treatment remain open.

## 2026-08-09 local memory-policy gate

The controlled matrix fixed the model, four-step sampling, 124-frame render window, 22-frame AV
context, prompt, paired seeds, and DynamicVRAM headroom 2.0GiB. Stock and Sage each completed
three cold and three warm trials at both 736x416 and 1056x608; every trial retained more than
512MiB, and matching strategy+seed cold/warm outputs were bit-identical. Default Block Cache
hit 0 of 4 forwards, held about 117.7MiB of CPU cache, and cannot remove the mandatory first
full forward, so it was rejected as the default OOM treatment.

Sage reduced runtime, but at equal headroom its whole-device peak was higher than Stock by an
average of about 1833.92MiB at 736x416 and 1411.86MiB at 1056x608 in the warm trials. It also
changed the AV output. Manual inspection of the 1056x608 sheets found material camera distance,
framing, pose, or trajectory divergence in two of three seeds. Sage is therefore only a
high-risk approximate speed experiment, not the default memory profile.

The conservative Stock+headroom-2.0 policy then completed a real uninterrupted 60-second chain:
14/14 accepted segments, manifest revision 14, exactly 1440 timeline frames, 1,920,000 audio
samples, and two 736x416 24fps H.264/AAC assemblies whose video, audio, and container durations
are all 60.000 seconds. No restart or explicit `/free` occurred between segments. Device peaks
ranged from 12,829.44 to 13,640.09MiB with a 13,137.67MiB median and 2739.41MiB minimum free
margin. The warm first-to-last difference was +182.81MiB and descriptive OLS slope was
+26.31MiB/segment, but the sequence was non-monotonic rather than a staircase leak.

Against the prior same-prompt/same-seed Stock headroom-0.5 chain, all 14 segment MP4 SHA-256
values and all 13 continuation `video_tail`/`audio_tail` tensor payloads were identical. The
context container hashes differ because chain/model metadata differs. Median device peak fell
by about 2635MiB while total generation time increased by about 1.63%. This establishes a
**validated local conservative profile** for the exact RTX 4060 Ti 16GiB, FL2VA INT8,
Standard four-step LoRA, 736x416, 124-frame, 22-context and plugin contract. It is not a general
`memory_safe` tier or never-OOM guarantee; other GPUs, higher resolutions/reference counts,
desktop VRAM pressure, and background queue/release behavior require separate gates.

### Three-base-seed 60-second cold-start follow-up

The fixed Stock+DynamicVRAM-headroom-2.0 contract was repeated with base seeds `2608082000`,
`2608083101`, and `2608083202`. Each chain used a separate ComfyUI process while prompt, FL2VA
INT8 model, Standard four-step LoRA, 736x416 canvas, 124-frame render window, 22-frame AV context,
and sampling settings remained unchanged. The first run preserves its real schema-1 manifest as
read-only migration evidence; the two new runs created native schema-2 manifests.

All 42/42 segments completed once with no OOM, retry, or candidate reuse. Independent analysis
recomputed and matched every candidate/accepted MP4 and context SHA-256, verified every parent ID
and manifest revision, and confirmed exact 1440-frame/1,920,000-sample timelines, final completion
blocking, and six 60.000-second assemblies. The three per-chain maximum peaks were 13,640.09,
13,414.01, and 13,426.72MiB; the aggregate minimum free margin was 2739.41MiB, no segment fell
below 512MiB, and all three peak sequences were non-monotonic. Total generation time per chain
ranged from 1334.251 to 1358.531 seconds. This passes the fixed local profile's cross-base-seed
cold-start mechanical and memory gate. It does not establish same-seed whole-chain warm
repeatability or a general `memory_safe` tier.

Visual quality did not pass the long-term identity gate. Worst-seam contact sheets remained
locally continuous without an obvious hard cut, yet all three 14-segment middle-frame timelines
accumulated facial-age and identity drift; seed `2608083101` changed most severely. Across runs,
median seam MAD was 0.01525-0.01651 and median seam SSIM was 0.91555-0.96374, demonstrating why
local seam metrics alone cannot establish long-term identity preservation. Maximum adjacent
half-second audio level gaps were 23.59-48.06dB, descriptive NCC medians were 0.127-0.206,
median absolute lags were 64.97-81.00ms, and the final segment's above-8kHz energy ratio was
9.66-36.30dB below the first. The 5ms bridge reduced median post-AAC single-sample jumps by
94.93%-97.33%, but cannot repair level, timbre, semantics, speaker identity, lip sync, or
recursive high-frequency loss.

The aggregate report is
`artifacts/long-video-generation-check/stock-headroom2-60s-multiseed/analysis/REPORT.md`. Its
remaining gates are same-seed whole-chain warm repeats; different prompts/materials; dialogue
ASR/speaker/lip-sync and blind listening; fast motion and rhythmic music; other GPUs, higher
resolutions/reference counts, and desktop-load profiles. The 0.10-second `/system_stats` polling
may also miss shorter peaks, and adjacent-window NCC is descriptive rather than overlap
reconstruction correlation.

### Default-off persistent first-frame identity-reference checkpoint

Source audit of the failing three-seed timelines found that the recommended background workflow
has no reference image connected. Even when a user connects `first_frame`, legacy continuation
behavior deliberately ignores it after segment 0 because previous-tail motion keyframes own the
target head. The chain therefore has motion continuity but no direct observation of the original
appearance after the first segment.

Long Video Conditioning now appends the optional `first_frame_reuse` input without changing the
existing input/widget prefix. `segment0_only` remains the default and preserves the old path.
`persistent_identity_reference` requires a connected first frame, retains its exact segment-0
keyframe role, and on continuation segments prepends the same image as non-timeline `<Picture 1>`
before any user reference images. The local MiniMax H3 payload patch already supports motion
keyframes, image references, and the marked continuation-audio window together, so no global
ComfyUI patch or new model is introduced. Explicit FL2VA plus the reference still fails closed;
users must select `auto` or `Hybrid`. Persistent plus user reference images are capped at nine.

Automated coverage verifies legacy first-frame ignore behavior, segment-0 behavior, persistent
reference ordering/media mapping, motion+image+audio payload coexistence, missing-image rejection,
reference-limit rejection, explicit-FL2VA rejection, and the appended schema default. This is a
structurally isolated implementation. Each continuation adds an image-reference block and a VAE
encode, so VRAM/runtime can increase and motion may be overconstrained.

A first same-source/prompt/base-seed two-segment real A/B ran in independent isolated ComfyUI
processes with FL2VA INT8, Standard four-step LoRA, Qwen3-VL NVFP4, both H3 VAEs, 736x416,
124/22 frames, Stock plus native DynamicVRAM headroom 2 GiB, and `unload_all_models`. Both sides
completed 2/2 prompts without OOM, retry, or cache reuse. The accepted segment-0 MP4 SHA-256 and AV
context tensor hash matched exactly, while segment 1 changed only when the persistent reference was
enabled. Both final streams decode to exact 144-frame/6.000-second video and 6.000-second 32 kHz
audio. That first quality probe was invalid for identity because the subject turned fully away
before frame 123; its single paired +607.43 MiB whole-device delta is retained as raw evidence but
is not treated as a fixed persistent-reference cost.

The replacement face-visible matrix used the same source identity and low-motion prompt with base
seeds `2608096001/2/3`. Six independent cold chains completed 12/12 prompts without OOM, retry, or
cache reuse. Every pair retained byte-identical accepted segment-0 MP4/context hashes and changed
only segment 1; all six final files again decode to exact 144-frame/6.000-second video and
6.000-second 32 kHz audio. InsightFace buffalo_l selected the largest face above a 0.8% frame-area
floor and detected the primary face on all 60 continuation frames in both modes. Pooled
persistent-minus-legacy source-embedding cosine mean/median were +0.0424474/+0.0268420, and the
persistent side was higher on 45/60 frames. Seed means were -0.0020399/+0.0699808/+0.0594012:
two seeds improved clearly and one was approximately neutral/slightly negative. Descriptive median
absolute age-estimator error improved by 3.0/3.5/1.5 years, but estimated age is not ground truth.
Frames inside one generated continuation are correlated, so 45/60 is not an independent-sample
significance test.

Whole-device paired peak deltas were -336.46/+294.12/-43.20 MiB and changed sign. They are dominated
by model-load/poll timing and do not establish a fixed peak overhead, despite the structurally added
reference block and VAE encode. PyTorch peak deltas were +122.29/+120.61/+877.19 MiB, runtime deltas
were +5.03/+3.61/+21.17 seconds, and every pair returned to the same 1231.63 MiB whole-device usage
after 15 seconds. Contact sheets show no obvious hard cut or new face artifact. Because the prompt
intentionally allowed only subtle movement, this matrix alone cannot rule out motion suppression.

The follow-up motion-rich matrix used the same source and runtime controls, base seeds
`2608097001/2/3`, and a prompt requiring continuous football tosses and wide arm movement across the
boundary. Six independent cold chains again completed 12/12 prompts without OOM, retry, or cache
reuse. Every pair retained byte-identical accepted segment-0 MP4/context hashes and changed only
segment 1; all six final files remained exact 144-frame/6.000-second video and 6.000-second 32 kHz
audio. Legacy primary-face detection passed 59/60 continuation frames and persistent passed 60/60;
the missing legacy frame occurred during vigorous motion/occlusion and was not imputed.

Across the 59 paired detected frames, persistent-minus-legacy source-embedding cosine mean/median
were +0.0269638/+0.0196597 and 38/59 frames were higher. Seed means were
-0.0254073/+0.0292898/+0.0796427: one negative and two positive. Descriptive median absolute age
error improved by 1.5/3.0/-0.5 years. Persistent/legacy temporal-MAD ratios were
0.9658/1.1970/1.0030, flow-P90 ratios 0.8405/1.3747/1.0159, and normalized face-center path ratios
1.1084/1.2221/1.1541. Visual contact sheets show active ball and arm motion in both modes. This is
evidence against an obvious systematic freeze in this short probe, not proof of equivalent action
freedom, prompt adherence, or preference.

Persistent boundary MAD was slightly higher in all three seeds, although contact sheets show no
obvious hard cut. Whole-device peak deltas were +549.93/-378.29/-169.03 MiB and therefore do not
establish a fixed device-level cost. Torch peak deltas were consistently positive at
+122.29/+133.10/+120.60 MiB; runtime deltas were +15.08/-15.14/+8.89 seconds, and all pairs returned
to equal post-15-second device occupancy. The persistent path remains default-off and must not be
called identity locking, action preserving, or memory-safe. At that checkpoint the probes authorized
one controlled intermediate 8-16-segment A/B, while the three-seed 60-second matrix remained
conditional on its identity, motion, audio, and VRAM result.

The intermediate gate then ran as one matched 32-second/eight-segment A/B with base seed
`2608097101`, the same source/prompt/model/sampling controls, independent cold processes, and global
`unload_all_models` after each accepted segment. Both modes completed 8/8 prompts without OOM,
retry, or cache reuse. Both manifests reached revision 8; segment 0 MP4/context tensors remained
byte-identical, all seven continuation MP4 hashes differed, and both final files decode to exact
768-frame/32.000-second video plus 32.000-second 32 kHz audio.

Persistent-minus-legacy whole-device peak was +84.54 MiB, Torch peak +122.29 MiB, and runtime
+61.91 seconds. Persistent minimum free VRAM was 3321.63 MiB and the post-15-second occupancy delta
was 0 MiB. This fixed-case mechanical and 512 MiB margin screen passed; it is not a general memory
safety result.

Ten evenly spaced frames per accepted segment were screened with the same InsightFace model.
Across 60 paired detected continuation samples, persistent-minus-legacy cosine mean/median was
+0.1008925/+0.0844391 and 53/60 samples were higher. Persistent won 6/7 continuation-segment
medians. The relative result is positive, but the depth target failed: legacy medians fell from
0.5482 to 0.1092, while persistent fell from 0.6134 to 0.1336. The source, both eight-frame
timelines, metric-worst seams, and eight-frame strips for segments 5-7 were inspected; both modes
visibly drift away from the source face. Relative improvement therefore does not establish long-term
identity preservation.

Persistent/legacy temporal-MAD ratios by continuation were
1.002/1.031/1.180/1.105/0.841/0.746/0.834 and flow-P90 ratios
1.118/0.945/0.924/0.899/0.823/0.590/0.903. Late strips show active football and arm motion in both
modes, including the 0.590-ratio segment, so this is not a literal freeze; it remains a material
motion-amplitude/trajectory warning. Video seam MAD median/max was 0.04871/0.08773 legacy versus
0.06791/0.08125 persistent. No metric-worst contact shows an obvious scene cut.

Audio maximum absolute adjacent half-second level gap was 4.10/3.93 dB, descriptive NCC median
0.250/0.267, and first-to-last above-8 kHz energy change -11.23/-4.73 dB for legacy/persistent.
This relative screen shows no material persistent-side regression in the one case; it includes no
blind listening, ASR, speaker, or lip-sync test.

**Decision:** the eight-segment intermediate gate fails because identity still collapses with depth
and a late motion warning remains. The predeclared gate therefore denies the three-seed 60-second
persistent/legacy matrix. `persistent_identity_reference` stays Experimental and default-off; the
identity-conditioning strategy must be redesigned before more long-chain generation. Raw metrics,
contacts, and the final report are in
`artifacts/long-video-generation-check/identity-anchor-intermediate-8segment-ab-seed2608097101/analysis/`.

### Dedicated identity crop and scene-plus-identity redesign

The redesign appends `persistent_identity_image` and `persistent_identity_strategy` after the old
input schema. Segment 0 always retains the exact original `first_frame`; the dedicated image is
continuation-only. `single_reference` prefers the crop and falls back to the full first frame, while
`scene_plus_identity` emits both as separate image references and fails closed if the dedicated
image is absent. The implementation reports the exact reference sources/count, includes the real
count in the nine-image limit, and leaves the default `segment0_only` path and old widget/API prefix
unchanged. Automated coverage includes segment-0 invariance, crop preference, inactive-policy
ignore behavior, legacy fallback, dual-reference ordering, count reporting, and missing-crop
rejection.

The motion-rich crop-only matrix reused base seeds `2608097001/2/3` and all fixed model, sampling,
canvas, context, cold-process, and release controls. Three new crop chains completed 6/6 prompts
without OOM, retry, or cache reuse; accepted segment-0 MP4 and AV-tail hashes matched both existing
baselines. Crop-minus-legacy pooled InsightFace cosine mean/median was +0.0827160/+0.0984536, with
54/59 paired detections higher and all three seed medians positive. Crop-minus-full-scene was
+0.0557522/+0.0990135 on 47/59 frames, but seed `2608097003` regressed in mean/median. Crop-only is
therefore useful evidence, not a universally dominant strategy.

The scene-plus-crop strategy then completed another three independent two-segment chains with 6/6
successful prompts and matching segment-0 hashes. Relative to legacy, pooled mean/median was
+0.1127789/+0.1162798 and 56/59 paired frames were higher; seed medians were all positive. Relative
to the full-scene strategy, pooled mean/median was +0.0845382/+0.0945541 and 52/60 frames were
higher, again with a positive median in every seed. All per-seed absolute medians were within 0.02
of the better single-reference result. Manual three-way/four-way contacts showed active football
and arm motion without a new hard cut, freeze, or obvious face artifact. These seeds selected the
strategy, so this is a development gate rather than unbiased generalization evidence.

One independent scene-plus-crop 32-second/eight-segment chain used base seed `2608097101`. It
completed 8/8 distinct prompts once without OOM, retry, or cache reuse; manifest revision 8 contains
124/102/102/102/102/102/102/32 output frames and the final H.264/AAC streams decode to exactly
768 frames and 32.000 seconds. Segment 0 video and AV-tail hashes match both old baselines. Whole
device peak/minimum-free/Torch peak were 12473.43/3906.07/3747.61 MiB, runtime was 961.84 seconds,
and post-15-second occupancy returned to 1231.63 MiB. This is one fixed local profile, not a general
memory-safety or no-leak result.

Ten evenly spaced samples per segment gave scene-plus-crop continuation medians
0.6989496/0.6388768/0.6441070/0.6092270/0.7367208/0.6013633/0.5735967. Its last/first ratio is
0.8206553, versus 0.1991358 legacy and 0.2178784 full-scene. Pooled scene-plus-crop minus legacy
mean/median was +0.4294471/+0.4666961 on 58 paired detections (57 higher); minus full-scene was
+0.3377138/+0.3580239 on 63 (59 higher). It won all seven continuation medians versus both
baselines. Timeline review confirms materially stronger identity retention through the last segment.

The predeclared composite gate nevertheless did not pass. Scene-plus-crop/legacy temporal-MAD
ratios were 0.986/0.770/1.178/0.982/0.647/1.003/0.867 and flow-P90 ratios
1.073/0.546/0.875/0.897/0.538/0.839/0.828. The 0.70 floor therefore fails in two flow segments and
one MAD segment. Two-fps strips for the flagged segments show ongoing football, arm, and pose motion,
so there is no literal freeze, but the lower action amplitude/trajectory cannot be dismissed. Video
seam MAD median/max was 0.04459/0.07521; audio maximum half-second level gap was 3.59 dB and
descriptive NCC median 0.390. Timeline and metric-worst seam review found no obvious new hard cut.

**Decision:** the redesign clears the fixed-case mechanical, identity-depth, relative-audio, and
512 MiB VRAM screens, but fails the predeclared relative-motion floor. It remains Experimental and
default-off. Do not run or claim the three-seed 60-second matrix yet. The next gate is a genuinely
unseen multi-seed/multi-source 32-second replication plus a motion-regression investigation; only a
pass may authorize a bounded 60-second matrix. Raw evidence is in
`artifacts/long-video-generation-check/identity-anchor-crop-motion-rich-multiseed-analysis/`,
`artifacts/long-video-generation-check/identity-anchor-scene-plus-crop-motion-rich-multiseed-analysis/`,
and
`artifacts/long-video-generation-check/identity-anchor-scene-plus-crop-intermediate-8segment-seed2608097101/analysis/`.

The fixed-cadence motion-regression experiment then appended the optional
`persistent_identity_interval`, with default `1` preserving the existing every-continuation
behavior and old workflow/API prefix. Interval `2` injects on continuation segments 1/3/5/7 and
uses bounded motion/audio context alone on 2/4/6. One matched development chain completed all eight
prompts and exact 32-second media without OOM, retry, or cache use. Runtime was 825.92 seconds;
whole-device peak/minimum free/Torch peak were 12744.43/3635.07/3735.12 MiB, and post-15 occupancy
returned to 1231.63 MiB. Its identity continuation last/first ratio was 0.52509, below the
every-segment strategy's 0.82066. Relative flow-P90 fell to 0.648/0.457/0.652 in three later
continuations, including both injected and skipped segments. Fixed alternating injection therefore
does not establish a causal or repeatable release of motion constraint. Interval 1 remains the
workflow/default behavior; larger values are an Experimental research control, not a recommended
quality mode.

Two genuinely new source/prompt/base-seed chains then used interval 1, 736x416, 124/22 frames,
four-step Stock plus DynamicVRAM h2, and `unload_all_models`. Each completed 8/8 prompts once,
without OOM, retry, or cached nodes, and produced exact 768-frame/32-second A/V:

- The qipao fan-dance/drum case ran 872.95 seconds with peak/minimum-free/Torch values
  12170.76/4208.74/3726.08 MiB and post-15 1231.63 MiB. Manual timeline and motion-strip review
  retained the same woman, qipao, courtyard, and active fan choreography; the metric-worst seam did
  not show a hard cut. Face sampling was sparse during motion, but the continuation last/first
  median ratio was 0.93654. Audio maximum half-second level gap was 2.68 dB and final-minus-first
  above-8-kHz energy was -3.41 dB. The prompt requested a steady 120 BPM, while Librosa's
  half/double-aware descriptive estimate was about 104.17 BPM; strict rhythm adherence therefore
  did not pass, and no listening result is inferred from beat tracking.
- The two-woman dialogue case ran 862.41 seconds with peak/minimum-free/Torch values
  12227.57/4151.93/3738.38 MiB and the same 1231.63 MiB post-15 occupancy. Two-source InsightFace
  assignment found both people in every one of 80 sampled frames and both remain visible at the
  end. Manual review nevertheless rejects seamless framing: segment 6 to 7 jumps from full-body to
  a close two-shot. A checksum-verified `Systran/faster-whisper-small.en` CPU model recognized both
  requested phrases with best-window word error rate 0 throughout the 32 seconds, but the speech
  mostly repeats those phrases and is not natural long-form conversation. A cropped-face
  mouth-aperture/audio-envelope proxy reached only 0.042/-0.008 correlation with 85.4/27.1 percent
  track coverage. This is not SyncNet and cannot establish lip sync; human viewing/listening and a
  trained audio-visual metric remain open gates.

The current user target is typical approximately 30-second creation, so this cycle treats 32
seconds as the complete-chain gate. The arbitrary-duration and existing 60-second capability remain
implemented, but no additional 60-second rendering is required. These results support local
32-second mechanical/memory stability for the fixed profile; they do not close rhythm, framing seam,
lip-sync, blind-review, high-resolution, or cross-GPU gates.

The final regression for this checkpoint is 150 passed tests with four third-party Triton
deprecation warnings and no project failure. Ruff passes for the project and all local analysis
scripts, 21 non-artifact JSON files parse, `git diff --check` passes, isolated ComfyUI whitelist
import succeeds against `cbbc9dab1`, and stable `sampling.py` remains SHA-256
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`. Raw telemetry,
metrics, and contact sheets remain under the local excluded
`artifacts/long-video-generation-check/` tree.

## 1.5.0 accepted-state and file-composition checkpoint

Implemented locally on 2026-08-08, following the 1.4.0 P1 generation run:

- four additional Long Video Experimental nodes register after the original 18;
- candidate MP4/context/descriptor writes are atomic and do not mutate accepted state;
- acceptance verifies SHA-256, uses a same-chain manifest lock and atomic replacement,
  retains one valid prior manifest revision, and is idempotent for the same candidate;
- every continuation candidate records its accepted parent candidate ID. A stale parent,
  a gap, or a final segment followed by another segment is rejected before commit;
- intentional replacement of segment N retains invalidated history and removes N and all
  later dependent segments from the active accepted chain without deleting their files;
- composition verifies every accepted file and all absolute frame/sample boundaries, then
  re-encodes with memory bounded to one video frame plus one segment PCM buffer;
- the duration-preserving cosine bridge changes samples only at the start of the new segment,
  decays to zero over a configurable window, and never shortens the total timeline;
- seven synthetic MP4 tests cover candidate/review/accept, accepted-context loading, stale/gap/revision
  rejection, replacement invalidation, backup fallback, exact sample accounting, streaming
  composition, and the bridge invariant.

The prior real 124/102/102-frame H3 outputs were then ingested through an isolated accepted
manifest and composed with bridge disabled/enabled. Both outputs contain 328 frames with a
13.6667-second video stream and a 13.667-second AAC stream. Before the final AAC encode, the
two value jumps changed from about 0.04164/0.04124 to zero. After decoding the final AAC,
the boundary jumps changed from about 0.04226/0.03509 to 0.00434/0.00704, reductions of
about 89.7% and 79.9%. The local report is under
`artifacts/long-video-generation-check/delivery-real-check-20260808-174353/REPORT.json`.

This is evidence for lower zero-order amplitude discontinuity, not a listening test or proof
of phase/semantic continuity. H.264/AAC composition is a re-encode, not lossless concatenation.
Human-reviewed total-duration resume is now implemented in 1.6.0. Later checkpoints above add
the 14-segment run, background auto-queue/retry/release, one bound local memory profile, and its
fixed-prompt three-base-seed cold follow-up. The multi-material long-chain quality matrix,
same-seed whole-chain warm repeats, and general/cross-configuration VRAM tiers remain open.

## 1.4.0 experimental long-video continuation checkpoint

Validated on 2026-08-08 against ComfyUI `a464ac335`:

- four new nodes register after the unchanged original 14-node list, under
  `T8/MiniMax H3/Long Video/Experimental`;
- no global `PackedLayout` or `MiniMaxH3.extra_conds` monkey patch is installed;
  the long-video node clones MODEL and adds one local `extra_conds` object patch;
- state loading selects exactly segment N-1; saving keeps at most a 39-frame
  video/audio latent tail on CPU, validates tensor hashes and metadata, and uses
  same-directory atomic replacement;
- 5/22/39 frame math maps to 2/7/12 H3 video latent steps;
- direct video latent tails avoid a full previous-video IMAGE load and video-VAE
  decode/re-encode, but do not remove their current-segment Transformer rows;
- non-final segments preserve the sampled endpoint; a final exact trim disables
  the next checkpoint to prevent continuation from hidden frames;
- the long-video examples use core `CreateVideo -> SaveVideo`, not VHS
  `apad + -shortest`, so the tested MP4 audio streams match their video timelines;
- all 84 tests and Ruff passed at the original P1 checkpoint.

The initial structural probe used one-step sampling and no LoRA. A later real
four-step run used the non-pruned FL2VA INT8 model, Standard Turbo LoRA, NVFP4
H3 CLIP, both H3 VAEs, 736x416, 124-frame windows, 22-frame AV context, and
DynamicVRAM. Three direct-latent segments completed without OOM and produced
124/102/102 video frames with matching audio-stream durations. Their measured
device peaks were approximately 15,998/15,881/16,135MiB, all below the planned
512MiB free-margin gate. In a same-source A/B, one decoded last frame was clearly
worse than both 22-frame routes; video-VAE re-encoding showed no convincing
quality win over direct sampler latent and was slower. Audio sample-boundary jumps
remained near the top of local differences. At this 1.4.0 checkpoint there was no 8-16 segment
degradation run or controlled VRAM safety matrix, so no `memory_safe`, lossless,
arbitrary-length, seamless, or no-OOM claim is made.

## Artifacts

| File | Size | SHA-256 |
|---|---:|---|
| `minimax_h3_turbo_4步加速.safetensors` | 779,849,991 | `9344cd958f8d354da03dd00b7d462933eb5d0cbf11e56a25d8e9911bb971160e` |
| `minimax_h3_turbo_4步加速_comfyui.safetensors` | 779,858,752 | `35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b` |
| `minimax_h3_turbo_4步加速ema.safetensors` | 779,849,991 | `8a1265e81e5368ab0e52cbb990aee3cb59b28b91fdfa415ef8dbabf81aef890e` |
| `minimax_h3_turbo_4步加速ema_comfyui.safetensors` | 779,858,752 | `b07ab477437c6a525dfdaf11107722aad609975ac172f3b577a7a87b228ff7b3` |

## Checks passed

1. Both sources contain exactly 259 paired LoRA modules / 518 BF16 tensors:
   50 main transformer blocks, two token-refiner blocks, and the final AdaLN.
2. Every expected name and exact H3 shape was checked before conversion.
3. Every output key has the `diffusion_model.` prefix required by ComfyUI's
   generic diffusion-model LoRA mapping.
4. All 518 output tensors were read back and compared with their source tensor
   using exact `torch.equal`; no tensor value changed.
5. Both outputs were passed through current `comfy.lora.model_lora_keys_unet()`
   and `comfy.lora.load_lora()`:
   - target stems found: 259
   - adapters parsed: 259 `LoRAAdapter` objects
   - source tensors consumed: 518/518
   - unloaded-key warnings: 0
6. A representative AdaLN adapter was executed through ComfyUI's
   `BypassForwardHook`; its output exactly equaled `B(A(x))`.
7. Official Comfy-Org checkpoint headers were inspected for FL2VA and REF2VA,
   BF16 and INT8 ConvRot, pruned and non-pruned variants:
   - non-pruned base: 259/259 adapter module shapes match
   - pruned base: 208/259 match; all 51 AdaLN inputs are 8 instead of 2688

## Scope limit

An end-to-end video render was not run because no MiniMax-H3 base diffusion
model, text encoder, or VAEs were present in the supplied directory. The
conversion itself, ComfyUI key resolution, adapter parsing, and bypass math were
validated. For a render test, use the non-pruned base files listed in
`README_ComfyUI.md`; using a pruned base would not be a full or safe test of
these LoRAs.

## Dual-clock sampler validation

`minimax-h3-audio-T8` 1.2.0 was installed and validated against the user's
ComfyUI `0.30.0` tree at commit `6f7cd7fce`:

- the four-step video sigma grid is exactly
  `[1, 36/37, 12/13, 4/5, 0]`;
- mapping the same base times to audio shift 3 gives
  `[1, 0.9, 0.75, 0.5, 0]`;
- a synthetic joint H3 velocity test integrates both streams to their exact
  Euler endpoints on their own clocks;
- audio denoise-mask 0 retains ComfyUI's flat-clock inpaint endpoint behavior;
- all 40 plugin tests and Ruff checks pass;
- a CUDA tensor/device regression passes on an NVIDIA GeForce RTX 4060 Ti;
- ComfyUI `--quick-test-for-ci` loads the installed custom node successfully.

No full H3 render was run as part of this local sampler test, because the base
model stack was not placed in this workspace. The user-provided installation
can now run the included `tests/fixtures/api/dual_clock_4step_api.json` workflow after its
placeholder model names are replaced.

## Experimental multi-rate sampler validation

The new `MiniMaxH3MultiRateSamplerEXPT8` is isolated in
`nodes_multirate_exp.py` and `sampling_multirate_exp.py`. The stable
`sampling.py` remained byte-for-byte unchanged with SHA-256
`26A3E6BAB2DEBB1519570D28165F682968F97FE828E3AA1541C834B190705CDB`.

Validated properties:

- 4/8 uses microstep counts `[2, 2, 2, 2]` and 4/10 uses
  `[2, 3, 2, 3]`;
- both schedules preserve the exact four video macro boundaries of the stable
  4-step sigma grid;
- `audio_steps` exactly equals the number of complete joint H3 model calls;
- video commits only one frozen-derivative Euler update per macro interval;
- audio is integrated on its shift-3 clock, while denoise-mask 0 still follows
  ComfyUI's flat inpaint clock and lands on the locked endpoint;
- the installed plugin passes 40 tests, Ruff, and ComfyUI whitelist import;
- a real CUDA 4/10 synthetic integration test passes on the NVIDIA GeForce
  RTX 4060 Ti with exactly 10 model calls.

The whitelist startup also reported an existing lock on `user/comfyui.db`
because another ComfyUI process was running; the custom node itself imported
successfully in 0.0 seconds. No full H3 render was run by this automated test,
so 4/8 versus 4/10 perceptual audio quality should be compared in the supplied
workflow using identical seed, prompt, and inputs.

## Frontend workflow validation

Three complete ComfyUI 0.4 frontend workflows were added for stable 4/4, EXP
4/8, and EXP 4/10. Each contains 12 nodes and 18 links, uses the installed
non-pruned H3 INT8 base, NVFP4 H3 text encoder, both H3 VAEs, EMA Turbo LoRA,
and `LoraLoaderBypassModelOnly`. Every node type, input type, and output type was
checked against the live ComfyUI `/object_info` endpoint; all links were also
checked bidirectionally in the plugin test suite. Copies were installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/`.

A fourth ComfyUI 0.4 frontend workflow,
`2026-08-07_H3_Still_Edit_22Frames_EXP.json`, covers the experimental Ref2VA still-image
path. It uses the locally available pruned Ref2VA INT8 checkpoint without Turbo
LoRA, the H3 text encoder and video VAE, a 512x512/22-frame/20-step setup, Still
Preflight reporting, middle-frame Still Decode, and PNG output. Twenty-two
frames are on the native `17k+5` grid and map to video latent T=7 and audio
latent T=37, but remain below the approximate 124-frame training range.
The installed copy was listed by the live `/userdata` endpoint; all 13 nodes,
19 links, and serialized input/output types produced zero contract errors
against an isolated current-code `/object_info` server.

## VRAM validation harness

Added `tools/validate_h3_vram.py` as a diagnostic-only harness. It does not modify the stable or
experimental sampler implementations. The tool can inspect API prompts, build a controlled stock
Euler versus dual-clock pair, submit runs through the native ComfyUI API, correlate `/system_stats`
VRAM samples with WebSocket node/progress events, preserve OOM tracebacks, and reject comparisons
whose non-sampling controls differ.

Validated locally against the running ComfyUI `0.30.0` server at commit `2eb609766`:

- live `/system_stats` inspection identified comfy-aimdo `0.4.13`;
- the startup log supplied explicit `DynamicVRAM support detected and enabled` evidence;
- a lightweight API prompt completed through the WebSocket collector and produced node/progress
  events plus baseline samples;
- static analysis identified the stable 4-step setup and an intentionally constructed 12-step
  mismatch; it also resolves the Orchestrator's literal steps/shifts/sampler/scheduler output links
  in the auto-resume API instead of reporting a false `steps=None` mismatch;
- unit tests cover API/frontend format detection, DynamicVRAM evidence, A/B rewiring, telemetry
  peak attribution, and controlled-input comparison.

### Real H3 VRAM checkpoint (2026-08-07)

After the model stack became available, the harness was run against the user's known-working
frontend workflow translated to the equivalent API graph. The active path used the non-pruned
FL2VA INT8 ConvRot model, SageAttention patch, Standard bypass Turbo LoRA, H3 text encoder and H3
video/audio VAEs. The muted reference-image node was correctly excluded from execution.

The reported stress scale was reproduced with `0.6M`, 15 seconds aligned to 362 frames, no preview,
and a 2,037.5 MiB pre-run device baseline:

| Treatment | Steps | Status | Duration | Device peak | PyTorch peak | Peak node |
|---|---:|---|---:|---:|---:|---|
| stock Euler + stock scheduler | 4 | success | 1,210.9 s | 16,213.5 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |
| T8 dual clock | 4 | success | 1,631.4 s | 16,182.2 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |
| T8 dual clock stress run | 12 | success | 3,280.2 s | 16,245.5 MiB | 14,573.5 MiB | `SamplerCustomAdvanced` |

The generated 4-step pair retained identical non-sampling controls. Its comparison verdict was
`no_material_peak_difference` at a 128 MiB threshold: dual-clock minus stock peak was -31.3 MiB,
and their measured PyTorch peaks were exactly equal. This run therefore does **not** support the
hypothesis that `MiniMaxH3DualClockSamplerT8` bypasses DynamicVRAM/VBAR and causes a material model
residency increase. Both paths are nevertheless extremely close to the 16 GiB device limit, so
small differences in other CUDA users, previews, allocator fragmentation, model cache state, or
workflow wiring can still decide whether an individual run OOMs.

This is one warm-cache A/B sequence on one RTX 4060 Ti 16 GiB environment, not a universal proof.
A cold-start, order-swapped repeat and the affected user's exact API-format official/modified pair
remain the next tests before considering a production sampler change. The 4-step stock control is
for memory attribution only; its audio integration is not numerically equivalent to dual-clock H3.

## ComfyUI FLOW_AV compatibility regression (2026-08-07)

ComfyUI commit `bdcb886a4` introduced `ModelType.FLOW_AV` / `ModelSamplingAV`, required
`model_sampling.audio_scale`, and changed MiniMax H3 from slope-scaled audio velocity to raw audio
velocity. Commit `a464ac335` is the validation HEAD. A property-only workaround would remove the
`AttributeError` but would retain the wrong audio integration math, so version 1.3.1 detects the
active H3 base-model protocol and selects the matching update rule. Its custom samplers expose a
neutral `audio_scale=1.0` because they already own the separate audio clock.

Validation evidence:

- all 63 Audio T8 tests pass, including legacy/current constant-velocity endpoints, mask and
  callback behavior, exact current `MiniMaxH3.audio_scale()` access, stable setup, and EXP setup;
- Ruff passes for Audio T8 and the companion H3 Block Cache project;
- a whitelist cold start imports Audio T8, H3 Block Cache, and H3 Prompt Enhancer together;
- live `/object_info` exposes stable, EXP, conditioning, still-image, Block Cache, and Prompt
  Enhancer nodes;
- real FL2VA INT8 / Qwen3-VL / H3 VAE probes at 512x512, 22 frames and one step completed both
  stable and EXP sampling; the deliberate core `SaveLatent` sink then failed because ComfyUI's
  `SaveLatent` does not support packed `NestedTensor`, after sampler execution had completed;
- a real one-step H3 forward with Block Cache attached also completed, reporting `cached 0/1` and
  a 19.1 MiB CPU cache before the same deliberate post-sampling sink error;
- all 14 Block Cache tests cover current raw audio velocity and simulated legacy slope-scaled
  velocity; all 74 Prompt Enhancer tests pass. The disabled EasyCache directory and RH H3 directory
  contain no active sampling implementation.

## Version 1.3.2 media, VAE, and 2.0MP regression (2026-08-07)

Three independent issues were reproduced and fixed without changing either stable or experimental
sampling mathematics:

- VideoHelperSuite returns its audio as a lazy `Mapping`, not necessarily a concrete `dict`.
  The shared audio validator now accepts the mapping protocol while preserving the same waveform,
  sample-rate, rank, and finite-value checks. A live `VHS_LoadVideo` output from `1.mp4` was connected
  directly to `ref_video_audios.ref_video_audio_0`; conditioning completed and mapped the media as
  `Video 1` plus `Audio 1`.
- Current ComfyUI initializes a generic `audio_sample_rate` attribute on both H3 VAE wrappers, so
  attribute presence cannot distinguish video from audio VAEs. Preflight now identifies the native
  H3 VAE contract from the underlying class or the latent geometry (`24/3D` video, `32/2D` audio).
  Live main and still-image preflights both classified the installed video VAE as `video`; the main
  preflight also classified the installed audio VAE as `audio` and returned `ready=true`.
- The accepted canvas-area envelope was raised from 1,032,192 pixels to 2,088,960 pixels, with
  `1920x1088` accepted exactly and larger test input `1952x1088` rejected. Canvases above the old
  0.98M threshold remain allowed but produce a high-VRAM warning.

Validation evidence:

- 65 project tests pass and Ruff reports no findings;
- isolated ComfyUI whitelist import succeeds against ComfyUI `0.30.0` at `a464ac335`;
- a live `1920x1088`, 22-frame, one-step stable dual-clock run completed a real joint H3 forward
  using the FL2VA INT8 ConvRot model, Qwen3-VL NVFP4 encoder, and both native H3 VAEs;
- that run completed in 30.4 seconds in the then-warm process, and coarse `/system_stats` polling
  observed a minimum of about 1,212 MiB free VRAM on the RTX 4060 Ti 16GB.

The real-model probe stopped at the generated joint latent and did not decode or assess perceptual
quality. It proves that the new boundary is executable for this short one-step case, not that a
2.0MP 124- or 362-frame workflow will fit every 16GB environment. Resolution, frame count, steps,
reference-media size, previews, allocator state, and other loaded models can still determine OOM.

## Version 1.3.3 selectable sampler/scheduler regression (2026-08-08)

The stable `MiniMaxH3DualClockSamplerT8` now appends two optional controls after the existing
`steps`, `shift_video`, and `shift_audio` widgets. `dual_clock_euler + native_flow` remains the
default and executes the same explicit dual-clock sampler and shifted-uniform sigma construction as
the previous five-argument setup. Existing API prompts may omit both new inputs.

Alternative sampler execution is deliberately separated from the custom default. When current
ComfyUI exposes `ModelSamplingAV`, a selected built-in sampler receives a newly patched native
FLOW_AV sampling object with coherent video/audio shifts and audio carry scale. Legacy H3 builds keep
the explicit T8 Euler default but do not expose built-in sampler alternatives. Alternative schedulers
use `comfy.samplers.calculate_sigmas`; changing that time grid is supported plumbing, not a claim of
better Turbo quality.

Validation evidence:

- all 71 project tests pass and Ruff reports no findings;
- implicit defaults and explicit `dual_clock_euler + native_flow` produce identical sampling type,
  sampler function, and sigma tensors;
- current-protocol built-in Euler setup produces native `ModelSamplingAV` with `audio_scale=4.0`
  for shifts 12/3; a simulated legacy protocol rejects that path with a clear FLOW_AV error;
- a non-default `normal` scheduler matches current ComfyUI's scheduler output while retaining the
  explicit T8 Euler audio protocol;
- the supplied eight-step frontend workflow retains its original `[8, 12, 3]` widget array;
- an isolated whitelist import succeeds, and isolated `/object_info` reports the original five
  inputs in required order followed by optional `sampler_name` and `scheduler`, defaulting to
  `dual_clock_euler` and `native_flow`.

No full perceptual H3 comparison across the additional sampler/scheduler matrix was run for this
change. The regression proves routing, backward compatibility, and protocol selection; users should
compare alternative numerical methods against the preserved default with controlled seeds before
adopting them for production.

## Version 1.15.1 Advanced VRAM Policy real 16GiB matrix (2026-08-12)

The new strict `make-policy-pair` route kept the FL2VA pruned INT8 base, 27.69MiB Hybrid artifact,
Qwen3-VL NVFP4 encoder, both H3 VAEs, reference image, 736x416 canvas, 124 frames, Stock20 sampler,
seed and output chain fixed. DynamicVRAM was proven by the startup log on ComfyUI `cbbc9dab1` with
comfy-aimdo 0.4.13.

| Treatment | Peak | Minimum headroom | Result |
|---|---:|---:|---|
| No policy, cold | 16,337.621 MiB | 41.879 MiB | success, 512MiB gate failed |
| Fixed 2GiB, cold | 16,036.414 MiB | 343.086 MiB | effective, gate failed |
| Fixed 3GiB, cold | 15,850.672 MiB | 528.828 MiB | marginal pass only |
| Fixed 4GiB, worst of three cold | 15,351.383 MiB | 1,028.117 MiB | pass |
| Fixed 4GiB, worst of three warm | 14,978.085 MiB | 1,401.415 MiB | pass |

All nine measured runs succeeded. The three warm baselines had at most a 12.25MiB positive
consecutive increase, below the 256MiB staircase threshold, and the warm peak range was
198.635MiB. All three warm outputs passed the 736x416, 124-frame, 24fps and 32kHz stereo media
contracts. Decoded same-seed video and PCM hashes were identical across no-policy, 2GiB, 3GiB and
4GiB treatments, so the memory policy did not alter the generated numerical result.

The post-matrix host snapshot retained 209.651GiB commit headroom from a 252.834GiB limit. Version
1.15.1 therefore changes the fixed-policy widget suggestion and supplied workflow from 2GiB plus
global cleanup to 4GiB without global cleanup. This is a conservative starting point only for the
exact validated 16GiB workflow. It is not a universal `memory_safe` or `never_oom` tier: 0.6M/362
frames, 1080p, long video, speech, other GPUs, concurrent CUDA processes, pinning and lower host
commit remain unvalidated. Detailed evidence is in `docs/VRAM_POLICY_ADVANCED_VALIDATION.md`; local
machine-readable records are under the ignored `artifacts/vram-policy-validation/` directory.

## Version 1.18.0-1.18.2 recommended-route implementation and bounded validation (2026-08-13/14)

Version 1.18.0 preserves the preceding 62 node registrations, schemas, defaults and stable sampler
math, then appends 24 Advanced/Experimental nodes. The added routes are isolated environment
auditing, H3 MLP activation chunking, bounded Qwen visual-reference prefix caching, Studio planning
and repair execution, reviewed Context IR, file-level Reel Delivery, scheduled drive-audio latent
injection, AV decode safety, and same-process trajectory checkpoints.

The safety defaults are intentional: audits and optimization patches default to report-only,
external IR upload requires a separate explicit confirmation, Reel composition and repair
acceptance default false, scheduled audio injection defaults to bypass, AV decode defaults to
preflight-only, and trajectory persistence defaults false. The stable `sampling.py` SHA-256
remained `111da5e52b28f2424f57b36f88db63e3ea02b538a8cdfdea1c8ad2f122ad7bb5`.

### Real bounded probes

- The installed 32B Qwen3-VL NVFP4 encoder completed same-reference/new-prompt cache hits, two-entry
  LRU eviction and a 64MiB oversize refusal. The repeat matrix then ran three fresh-process cold
  pairs and three same-process warm pairs. Every cache-hit arm was faster: paired mean elapsed time
  changed by -11.97% cold and -11.01% warm, while warm process private memory showed no upward
  staircase. The outputs were not bit-exact. Across the six paired details, decoded video SSIM mean
  was about 0.9819 and the minimum was 0.92460; one warm audio pair fell to correlation 0.23234.
  Whole-device minimum headroom was only 75.63MiB cold and 168.08MiB warm, so the 512MiB safety gate
  and perceptual non-inferiority gate both failed. This closes the exact repeated timing probe only;
  it does not establish a lossless cache, fixed speedup, VRAM optimization or 16GiB-safe profile.
- During this checkpoint ComfyUI advanced to `v0.32.0-15@86aedfd9`. Its Llama/Qwen implementation
  added merged QKV/MLP support, fixed KV, layer prefetch and an in-place residual output. The original
  prefix contract therefore remained fail-closed until re-audited. Production now wraps prefix and
  suffix execution in an explicit no-grad inference context and includes the directly invoked
  `TransformerBlock.forward` in the exact source contract; the previous `cbbc9dab1` hashes remain
  accepted. A current-core tiny-Llama tuple-KV prefix/suffix forward again matched the full causal
  forward within `2e-6`.
- A real current-core single-process control then ran conditioning-only OFF/HIT primes followed by
  full 512x512x22 one-step OFF/HIT H3 branches using the installed 32B NVFP4 encoder, FL2VA pruned
  INT8 and both H3 VAEs. HIT reported one hit after one miss from a 108.283MiB entry. Full elapsed
  time changed from 13.297s OFF to 9.375s HIT. Both outputs contained 22 frames and finite 32kHz
  stereo audio; video SSIM mean/minimum was 0.951217/0.924603 and audio correlation was 0.956522.
  Whole-device OFF/HIT headroom was only 116.998/337.583MiB. This closes current-core mechanical
  compatibility only and preserves the non-bit-exact, non-lossless and non-16GiB-safe decisions.
- ComfyUI next advanced to `v0.32.0-16@ddbaa8752`, moving MiniMax projection-format detection before
  model construction while leaving the Qwen forward path used by the cache unchanged. The expanded
  exact-source contract, tiny-Llama tuple-KV equivalence probe, 447-test full project regression,
  Ruff, compileall, JSON/workflow checks and isolated plugin import passed. A native 48-frame,
  2-second video-reference full A/V control then recorded one real hit after one miss from a
  110.744MiB entry. OFF/HIT elapsed time was 25.266/15.578s; all 22 output frames and finite 32kHz
  stereo audio were present. Video SSIM mean/minimum was 0.950934/0.944633 and audio correlation was
  0.953029. OFF/HIT whole-device headroom was only 344.340/338.833MiB, so the 512MiB gate failed.
  This is current-build mechanical compatibility, not losslessness, perceptual non-inferiority,
  VRAM optimization or a 16GiB-safe profile.
- A two-image reference, 256x256x22, one-step full A/V pair closed the short multi-reference
  mechanical route. Both fresh-process cold and same-process warm HIT arms reported one real hit
  after one miss with a 60.7043MiB entry. Elapsed time changed by -6.04% cold and -6.87% warm. The
  paired outputs were non-exact: decoded video SSIM mean/minimum were 0.918691/0.911305 and audio
  correlation was 0.959562 with 10.63dB SNR. Minimum whole-device headroom was 311.85MiB, so this
  route failed the 512MiB gate and does not establish perceptual non-inferiority.
- A native ComfyUI `LoadVideo` plus `GetVideoComponents` route read a real 512x512, 48-frame,
  2-second, 24fps source and exercised Ref2VA video-reference prefix reuse. Conditioning-only cold
  and warm HIT arms reported real hits from a 110.7443MiB entry and changed elapsed time by
  -9.54%/-10.87%. A full one-process OFF/HIT A/V pair changed elapsed time by -13.81% and peak device
  use by -166.31MiB, but left only 145.15/311.46MiB OFF/HIT headroom. Its 22 decoded frames were
  non-exact with SSIM mean/minimum 0.950934/0.944633; finite 32kHz stereo audio correlation was
  0.953029 with 9.91dB SNR. This establishes the exact short video-reference mechanism only, not a
  fixed performance benefit, perceptual equivalence or a 16GiB-safe configuration.
- The follow-up two-image multi-material matrix used portrait/mechanical, mechanical/city-character
  and city-character/portrait combinations with two seeds each in one warm isolated process. All
  6/6 pairs completed their media contracts, recorded one real hit after one miss, and had a faster
  HIT arm. Mean elapsed-time change was -11.09%, ranging from -26.46% to -5.38%. Paired video SSIM
  averaged 0.931408 across cases, with pair means from 0.859847 to 0.978290 and a minimum frame of
  0.853100. Audio correlation averaged 0.977138, ranging from 0.966095 to 0.984150. Post-pair process
  private memory showed at most a 59.91MiB positive step, below the 256MiB staircase threshold, but
  whole-device headroom fell to 111.93MiB. The one-step contact sheet is visibly too degraded for a
  perceptual-quality decision; this closes automated multi-material repetition, not human
  non-inferiority, useful-profile quality, fixed speedup or 16GiB safety.
- The useful-profile follow-up replaced the one-step final branches with Stock20 while reducing each
  prime to the Qwen conditioning/statistics chain. One seed from each material combination produced
  3/3 real hits and all HIT arms were faster, with a mean elapsed-time change of -5.00% and range
  -6.22% to -3.62%. Full diffusion amplified the cached-versus-uncached numerical difference: video
  SSIM averaged 0.822721 across the three pairs, pair means ranged from 0.678955 to 0.907300 and the
  minimum frame was 0.605222. Audio correlation averaged 0.718805 and ranged from 0.260251 to
  0.989405. Whole-device minimum headroom was 190.68MiB. This does not pass automated quality or
  512MiB safety gates; a blinded package is required for subjective interpretation, but cannot
  convert these data into a lossless, generally non-inferior or 16GiB-safe claim.
- H3 MLP activation chunking completed one real 256x256x22, one-step native-versus-chunked smoke;
  all 22 decoded PNG frames and decoded PCM were identical. A controlled FL2VA pruned INT8,
  736x416x124, one-step A/B then rejected it as a memory optimization for this backend. Cold
  baseline/chunk256 measured 39.188/39.875s and 16049.30/16337.62MiB device peak (chunking was
  +288.32MiB); warm baseline/chunk256 measured 32.109/31.437s and 16006.74/16016.04MiB absolute
  peak, with controlled peak deltas differing by only -22.88MiB, below the 128MiB material gate.
  All four runs left less than 512MiB. The detected TensorWise INT8 path fuses SwiGLU, so the
  theoretical full-fc1 activation proxy does not apply. No 362-frame extension is justified for
  an INT8 memory-saving claim; other precision/backend research remains separate and unverified.
- Two real 256x256x22, one-step H3 Studio shots were selected and bound into the non-destructive
  repair path. A subsequent isolated 14-segment/60-second accepted chain retained all 27 accepted
  media/context assets and the original manifest through six forced process exits: after accepted
  copy, during accepted copy, after primary replacement, after backup, during audio composition and
  after replacement. All six retries recovered. A real H3 segment-7 replacement produced 102 frames
  and 136,000 samples; overlay and rollback both composed to exactly 1,440 frames and 1,920,000
  samples. This did not prove seamless middle replacement: incoming adjacent-frame SSIM was
  0.940863 versus base 0.944997, but outgoing SSIM fell to 0.803963 versus base 0.932967 and the
  repaired audio boundary gap was 19.396dB because the next segment still derives from the original
  context. The first crash matrix also left recoverable orphan temporary files at a half-written
  accepted copy and a mid-audio compose.
- Accept and compose now clean only atomic temporaries matching the exact accepted destination or
  assembled output while holding their OS operation locks; no recursive project-wide temp scan is
  used. The isolated real 14-segment fixture was rebuilt and all six process-kill points repeated.
  All expected exit codes and durable markers passed, the original manifest plus 27 accepted assets
  remained unchanged, and every retry succeeded. The half-copy and mid-audio temporary lists were
  non-empty before retry, explicitly reported by the node as removed, and empty afterward. This
  closes transaction recovery and crash-clean behavior for those named same-host Windows/NTFS
  boundaries, but the executor is still not cascade-continuity-safe.
- Reel Delivery completed a real two-clip, three-audio-event composition with 42 requested frames
  and 84,000 requested 48kHz samples. A second execution reused verified video/audio phases.
- The 1.18.1 Reel scale probe completed exactly 1,800 seconds from 50 independently addressed
  36-second clip paths and four dialogue/music/ambience/SFX events. The decoded output contained
  43,200 video frames and 86,400,000 48kHz samples; source hashes stayed unchanged, and a repeat
  compose reused both verified phase files. Peak process-tree working set was 870.53MiB and peak
  private bytes were 2,880.78MiB; these are one-run observations, not a cross-platform bound. The
  50 paths were hardlinks to one 64x64 H.264 fixture, so this closes the mechanical timeline-scale
  gate but not codec/content diversity or high-resolution throughput.
- A separate Windows/NTFS codec-diversity probe composed three distinct 128x96, exact-24fps synthetic
  sources in one reel: H.264/AAC MP4, HEVC/MP3 MKV and VP9/Opus WebM. Four file lanes used PCM WAV,
  FLAC, Opus and AAC at 32/44.1/48kHz input rates. The output matched the 132-frame plan, covered the
  264,000 requested 48kHz samples and reported an exact 5.500-second audio stream. All source hashes
  stayed unchanged; the second execution reused both verified phases, returned the same path and kept
  the output SHA-256 stable. AAC decoding exposed 192 padding samples while stream duration remained
  exact, confirming that the stream-time contract—not raw decoder padding—is the valid one-sample
  assertion. PyAV emitted a non-fatal Opus packet-header diagnostic while still decoding all 96,000
  source samples. This closes local synthetic codec/container mechanics.
- A real-H3 delivery probe exposed a distinct final-mux defect. Its three 256x256x22 H3 sources planned
  58 frames and 116,000 samples. The phase WAV was exactly 116,000 samples and the phase video exactly
  58 frames, but the default MP4 movie timescale of 1000 represented the AAC stream as 115,968 samples,
  32 samples short. Removing `-t`, using `-shortest` or padding did not repair the logical stream time;
  ALAC proved the PCM boundary itself was intact. Setting the MP4 movie timescale to
  `LCM(24fps, sample_rate)` produced exact logical boundaries for 32kHz (77,333/77,333), 44.1kHz
  (106,575/106,575) and 48kHz (116,000/116,000), while the corresponding default-timescale deltas were
  +11, -29 and -32 samples. Production now applies that timescale and validates final video-frame and
  audio-sample durations from the temporary container before atomic replacement.
- The original three-source real-H3 probe then passed unchanged at exact 58 frames, 116,000 logical
  AAC samples and 2.4166667 seconds. A higher-resolution follow-up used three distinct Stock20 H3
  outputs (portrait, mechanical dragon and city superhero), each 736x416x124 at 24fps with audio.
  Twelve-frame transitions produced exactly 348 frames, 696,000 logical 48kHz samples and 14.5 seconds;
  the maximum transition-tail estimate was 10.51MiB. Both real-H3 probes preserved source hashes,
  reused verified phases and produced stable repeated output hashes. Decoded AAC still contains normal
  encoder padding, so the claim is exact logical stream duration rather than lossless PCM identity.
  This closes local real-H3 and 736x416 delivery mechanics, not 1080p or non-Windows behavior.
- A derived-1080p probe then placed those three real-H3 736x416 clips into 1920x1088 canvases without
  claiming native H3 1080p generation. The fixture step itself exposed a local FFmpeg 7.1.1/libx264
  frame-thread assertion (`analyse.c:1317`) and was made deterministic with single-threaded video
  scaling followed by a separate audio stream-copy mux. More importantly, Reel's PyAV/libx264
  auto-threaded video stage first native-crashed with a 9.9MiB orphan temporary, then three apparent
  successes all contained strict-decoder H.264 reference-frame/CABAC errors and non-repeating hashes.
  Counting frames alone would therefore have produced a false pass.
- Reel now selects one x264 thread only when width*height is at least 2,000,000 pixels; lower-resolution
  delivery retains automatic threading. High-resolution phase and final-mux temporary files must
  each pass a single-thread FFmpeg full-stream decode using `-xerror` and `err_detect=explode` before
  atomic replacement. The exact `ffmpeg_single_thread_xerror_v2` policy is persisted in phase state,
  and an older high-resolution phase lacking that value is invalidated and re-encoded. Three
  1920x1088 projects then passed 3/3: every phase and final stream completed strict single-threaded
  full-frame decoding with no error; all three phase SHA-256 values were identical and all three final
  SHA-256 values were identical. Each reel contained exact 348 frames, 696,000 logical samples and
  14.5 seconds with a 71.72MiB transition-tail estimate. Re-running a pre-fix cached project upgraded
  `auto` to `1`, then the v1 boolean marker to the v2 policy, and reproduced the same phase/final
  hashes. Two local FFmpeg 7.1 Windows builds showed nondeterministic errors under automatic decoding
  threads even for byte-identical high-resolution files, while both builds in single-thread mode and
  PyAV 16/libavcodec 62 passed repeated full-frame decoding. This closes the explicit local
  single-thread derived-1080p delivery contract, not native H3 1080p generation quality, arbitrary
  player/decoder behavior or cross-platform support.
- A separate Ubuntu 24.04.4 WSL2 probe ran the production Reel module under Linux
  `6.18.33.2-microsoft-standard-WSL2` on ext4 `/tmp`, with Python 3.12.3, PyAV 18.1.0 and the Linux
  John Van Sickle FFmpeg 7.0.2 static build. Two 128x96 H.264/AAC clips and one FLAC lane composed to
  exact 66 frames and 132,000 logical 48kHz samples. The resumed run reported both phases reused,
  kept their hashes and mtimes unchanged, and reproduced the final SHA-256
  `a50eab77ea13b17e4d9f046615ccb9fa5e35c14f12b4093e0c7b297a1b5ac2c8`. Source hashes stayed
  unchanged and no matching temporary file remained. A real second POSIX `flock` contender timed
  out while another process held the project lock, then the same lock was reacquired after that
  process terminated. Local evidence is
  `artifacts/reel-delivery-linux-wsl/reel_linux_wsl_ext4_summary.json`. This closes one WSL2
  Linux/POSIX low-resolution mechanical path, not native bare-metal Linux, macOS, high-resolution
  Linux, arbitrary FFmpeg builds or cross-GPU execution.
- External-kill probes found and fixed two production defects: a Windows FFmpeg handle could make
  temporary cleanup mask the primary failure, and an interrupted audio mix could leave an
  unvalidated official stage file. Audio now validates a temporary WAV before atomic replacement;
  bounded retry preserves the primary error; one OS advisory lock serializes a project; and the next
  run removes only matching orphan temporaries before reusing hash-verified phases. Recovery passed
  after killing the audio FFmpeg child, final-mux FFmpeg child and parent Python process. Each resumed
  output matched the baseline delivery SHA-256 and left no matching temporary file.
- Regular AV decode completed with the installed H3 video/audio VAEs at 128x128x22 in about 2.4
  seconds, emitting 22 PNG frames and finite 32kHz audio. Source and behavior inspection then showed
  that current H3 regular decode internally tiles above 256 pixels, while its public decode_tiled
  entrypoint aliases regular decode and ignores explicit tile controls. A validation-only spatial
  coordinate patch kept the existing temporal chunk contract and replaced only tile-local spatial
  coordinates with full-canvas dimensions and offsets. Its 256x256 one-tile control was bit-exact,
  but all three 736x416 eight-tile source-reconstruction cases regressed: mean source SSIM changed by
  -0.0828, mean PSNR by -1.141dB, and neither x nor y seam ratio improved in any case. The direct
  global-coordinate substitution is therefore rejected and was not added to product code. The
  Advanced report continues to classify tiled H3 decode as high-risk; a different trained or
  architecture-aware remedy would require a new controlled validation.
- The original 256x256x22 probe exposed a non-bit-exact resume reconstruction. Trajectory v2 therefore
  uses a dedicated cloned sampling object whose noise/inverse-noise scaling directly transports the
  saved internal `x_sigma`; Load emits the required `resume_noise`, and second-stage DisableNoise is no
  longer valid. On RTX 4060 Ti 16GiB with FL2VA pruned INT8, Qwen NVFP4, both H3 VAEs and stable four-step
  dual-clock Euler, 736x416x124 and 256x256x362 full-versus-2+2 final video/audio latents were bit-exact
  (`max_abs=0`). Shapes were `[1,24,37,26,46]`/`[1,32,2,207]` and
  `[1,24,107,16,16]`/`[1,32,2,603]` respectively.
- The 124-frame repeat gate used three new-process cold cycles and three same-process warm cycles. All
  18 full/split/resume prompts succeeded, and all six paired final checkpoints had identical SHA-256.
  Warm full-run peaks were 15450.58/15430.43/15358.13MiB with no upward staircase; the entire repeat
  matrix minimum headroom was 587.15MiB. The 362 full run peaked at 15858.99MiB and left only 520.51MiB,
  just 8.51MiB above the 512MiB gate. This passes the exact local numerical/repeatability gate but not a
  universal 16GiB or higher-resolution safety claim.
- The 124/362 final checkpoint files were about 4.10/2.66MiB because the lower-resolution 362-frame
  spatial latent is smaller; frame count alone does not determine disk cost. Across the three warm
  pairs, full averaged about 70.75s and split+resume about 72.30s, so no throughput benefit is claimed.
  Same-process MODEL/SAMPLER identity, full-schedule matching and empty `patches_replace` remain hard
  requirements; restart resume, wrapper stacks, other samplers and cross-GPU behavior remain refused or
  unverified. Local raw evidence is summarized in ignored artifact
  `artifacts/trajectory-validation/trajectory_validation_summary_v2.json`.
- The same implementation was then rechecked on the current ComfyUI
  `v0.32.0-16@ddbaa8752874c275290d054ee4fddd6e004f5fdf`. Full, split and resume at both
  124 and 362 frames completed 6/6 real prompts. Full and resumed checkpoints were byte-identical in
  each pair: `88A5A71D...FFA1BD` for 124 and `863C80A3...D47A7` for 362. Full-run whole-device
  headroom was 749.019MiB at 736x416x124 and 548.502MiB at 256x256x362. The latter is only
  36.502MiB above the 512MiB project gate and does not establish a higher-resolution or universal
  16GiB safety tier. Raw evidence is in ignored artifact
  `artifacts/trajectory-validation/trajectory_current_core_matrix.json`.
- Stable dual-clock legacy compatibility was exercised against an isolated official ComfyUI
  `0.30.0` snapshot at commit `563b98eefbe643a4cd510ee7f0b43e79880d5a3f`, before native
  `ModelSamplingAV`. The legacy and current `cbbc9dab1` processes used the same plugin worktree,
  FL2VA pruned INT8, Qwen NVFP4, both H3 VAEs, 256x256x22 one-step workflow and seed. Both completed;
  all 22 PNG frames were byte-identical. Both audio outputs were finite 32kHz stereo with 29,600
  samples; PCM correlation was 0.999688, SNR 36.12dB and maximum difference one int16 LSB. This proves
  the stable `dual_clock_euler` legacy velocity branch on that exact H3-era build, not every older
  ComfyUI release or every Advanced route.
- A separate current-plugin import/schema probe then attached `1.18.2@c7f5080` to that exact legacy
  snapshot and queried `/object_info`. The process logged the plugin import with no traceback; all
  86 expected plugin node IDs and all 24 appended Advanced node schemas were present. This closes
  import and schema construction. Raw evidence is
  in ignored artifact `artifacts/legacy-comfy-validation/legacy_v1182_import_summary.json`.
- Trajectory Advanced was then exercised rather than inferred from its schema. On the same legacy
  snapshot, a 256x256x22 four-step full prompt, two-step split/save prompt and load/resume prompt all
  completed on the RTX 4060 Ti. Full and resumed checkpoint SHA-256 were both
  `F4EB6674...93C326D`. Full/split/resume durations were 18.688/6.641/10.297 seconds and their
  whole-device headroom was only 100.136/99.823/94.153MiB. This is a numerical compatibility pass
  for one short Advanced route and an explicit failure of the 512MiB memory-safety gate; it is not
  evidence for other Advanced nodes or universal legacy support. Raw evidence is in ignored artifact
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_trajectory_summary.json`.
- A CPU-only old-core pass then executed four read-only/model-free/no-write API graphs: Environment Audit
  (1/1 node), Studio/Prompt/Repair planning (7/7 dependency-forced nodes) and local Context IR
  validation/compiler (2/2 nodes), plus Reel plan/no-write compose (2/2 nodes). All 4/4 prompts
  completed without traceback, external upload or media mutation.
  Raw evidence is in ignored artifact
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_model_free_summary.json`.
- Qwen Prefix Cache first loaded the real NVFP4 encoder on the legacy core. Both report-only and
  `memory_lru_exp` wrapper-install paths plus Stats completed. A follow-up then encoded the same
  native 48-frame reference-video prefix twice in one process and recorded exactly one miss, one hit,
  one 110.744MiB CPU entry and no disk writes. OFF/HIT full A/V prompts both succeeded at
  256x256x22/one step; HIT took 19.281 seconds versus 24.657 seconds (-21.80%) and reduced observed
  peak by 434.238MiB in this ordering. The output was explicitly non-exact: 22-frame mean/min SSIM
  was 0.950934/0.944633 and 32kHz stereo audio correlation was 0.953028. The four-prompt minimum
  headroom was only 334.508MiB, so the 512MiB safety gate failed. This proves one short old-core real
  HIT route, not losslessness, repeatability, broad material quality or 16GiB safety. Raw evidence is
  in ignored artifacts `artifacts/legacy-comfy-validation/legacy_v1182_advanced_qwen_contract_summary.json`
  and `artifacts/legacy-comfy-validation/legacy_v1182_advanced_qwen_real_hit_summary.json`.
- AV Decode Safety was exercised in `decode_regular`, not only preflight: a 256x256x22 one-step H3
  latent produced 22 PNG frames and an audio file. Whole-device headroom was only 69.588MiB. The same
  graph with Activation Chunk `apply_exp` then stopped at that node before sampling with the intended
  `unknown ComfyUI H3 source contract` error. This is a verified compatibility pass for AV Decode
  Safety and a verified fail-closed incompatibility for Activation Chunk; its source hash was not
  weakened to force a run. Raw evidence is in ignored artifact
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_activation_decode_summary.json`.
- The four Repair execution nodes were then run against an already persisted real 14-segment chain.
  Bind and Stage read the verified plan/assets, Accept remained false, and Compose produced only a new
  base-rollback validation render. The source manifest SHA-256 remained
  `4C2EEC42...5EC037E` and all 27 accepted assets remained unchanged. This proves the bounded
  accept-off/rollback path, not accepted-overlay mutation or crash recovery on the old core. Raw
  evidence is in ignored artifact
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_repair_execution_summary.json`.
- A separate isolated two-segment fixture then exercised the old-core write path with
  `accept_repair=true` and `repair_overlay` composition. All five helper/Bind/Stage/Accept/Compose
  nodes executed, replacement index 1 and unselected index 0 were represented in the overlay, and
  the base manifest plus all original accepted-asset hashes remained unchanged. The composed file
  decoded 44 video frames. Its AAC stream duration was 58,688 samples versus 58,667 logical samples
  (+21), while the decoder returned 59,392 samples including codec padding; this is not labeled
  sample-exact. It proves transaction mechanics on a synthetic fixture, not real-H3 repair quality or
  old-core crash recovery. Raw evidence is in ignored artifact
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_repair_accept_summary.json`.
- Scheduled Audio Injection first ran in its default `report_only` mode on a real
  256x256x22 one-step H3 chain. It emitted 22 PNG frames and one audio file, but peaked at
  16,363.815MiB with only 15.685MiB whole-device headroom. The actual `scheduled_injection` apply
  route was then exercised on the same bounded profile and also emitted 22 frames plus audio; it
  peaked at 16,282.165MiB with 97.335MiB headroom. Both are execution-contract passes and strong
  failures of the 512MiB safety gate. The apply pass does not prove that injection suppresses
  unwanted speech; the controlled current-core A/B below already rejected that quality claim. Raw
  evidence is in ignored artifacts
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_scheduled_audio_summary.json` and
  `artifacts/legacy-comfy-validation/legacy_v1182_advanced_scheduled_audio_apply_summary.json`.
  Together, all 24 newly appended Advanced IDs now have route-specific execution or explicit
  fail-closed evidence on this exact legacy core. The evidence is deliberately bounded: Qwen covers
  one short reference-video hit, Activation refused application, Repair acceptance used a synthetic
  fixture, and Scheduled Audio retains a negative quality result. This is not blanket compatibility for all
  modes, old releases or 16GiB.

### Negative scheduled-audio result

The scheduled drive-audio route was tested rather than presumed effective. A controlled
256x256x124, four-step FL2VA/Turbo run detected extra speech at approximately 2.26 seconds in the
baseline. Full-strength scheduled injection still detected extra speech from approximately 2.10
seconds. Therefore the feature remains default-off EXP and cannot be described as an unwanted
speech suppressor. It acts on the complete supplied audio latent and cannot isolate dialogue from
music, ambience or effects.

### Regression and remaining release gates

- Environment Audit now exposes cumulative process RSS/private/pagefile, page-fault and I/O
  counters, pinned-memory state and optional NVML temperature, power, clocks and thermal
  throttling. A point-in-time audit deliberately classifies this as
  `fits_current_snapshot_thrashing_unmeasured`; `validate_h3_vram.py run` resolves a local
  ComfyUI listener PID, takes before/after deltas and screens `fits`, `fits_with_thrashing`,
  `unsafe` or `unknown`. The 64GiB read threshold is a conservative screen, not a disk benchmark.

- 447 project tests passed.
- Python compileall passed.
- 91 tracked API/frontend JSON files parsed, including 45 frontend workflows.
- the append-only registration contract contains 86 nodes;
- the Studio Timeline frontend passed Node syntax checking and renders untrusted labels with DOM
  `textContent`, not `innerHTML`.
- Ruff passed using the available standalone executable. The embedded Python wrapper remains broken,
  so verification deliberately did not rely on that wrapper.

Remaining gates include activation behavior on non-fused precision/backends (the current INT8
memory-optimization hypothesis is rejected), repeated multi-material Qwen video-reference behavior,
Stock20 cache blind interpretation and multi-GPU behavior, repair-cascade continuity,
Reel bare-metal Linux/macOS and high-resolution non-Windows filesystem/FFmpeg behavior, an alternative
architecture-aware tiled-VAE remedy, and external-provider quality/privacy deployment. Trajectory
v2 has closed the exact local 124/362 numerical, disk-budget and 124-frame repeat gates, but wrapper
owners, restart resume, alternate samplers, higher-resolution 362-frame use, cross-GPU and universal
16GiB safety remain refused or unverified. The stable sampler now has one exact older-ComfyUI real
probe, current 1.18.2 import/schema construction passed there, and one short Trajectory Advanced GPU
route is numerically exact but fails the memory headroom gate. The remaining new Advanced IDs now also
have bounded route evidence: twelve model-free/no-write nodes executed, Qwen completed a real video-
reference cache hit and A/V pair, AV Decode Safety decoded real media, Repair completed real-chain
accept-off rollback plus isolated-fixture accept-overlay composition, Scheduled Audio completed both report-only and apply generation, and Activation
Chunk explicitly rejected the old source contract.
This closes node-ID route coverage only; broader modes and behavior on other older releases remain
unknown.
The registered Ubuntu 22.04 distribution still lacks its `ext4.vhdx`, while the Ubuntu 24.04.4
distribution intermittently failed its normal user systemd session with `E_UNEXPECTED`. Running the
isolated probe as root after terminating the WSL instance succeeded on the existing filesystem and is
counted only as the bounded WSL2/Linux result above; no distribution was rebuilt and Docker remained
unused.
This machine exposes only one RTX 4060 Ti, so genuine cross-GPU
execution cannot be certified locally. Cross-GPU execution was explicitly removed from the current
validation scope on 2026-08-14; it remains an unverified future profile and is not implied by this
closure. Version
1.18.2 therefore keeps `memory_safe_claim=false` and `quality_guarantee=false`.

## v1.19.0 motion-quality P0 mechanical checkpoint (2026-08-14)

Two append-only Experimental nodes were added after the previous 86-node registry:
`MiniMaxH3AVSigmaTailSubdivisionT8Advanced` and `MiniMaxH3MotionQualityAuditT8Advanced`.
The stable `sampling.py` implementation was not edited. Future Turbo dual-clock quality,
compatibility, performance and memory tests use eight video and eight audio steps; four-step results
and workflows remain only as historical and compatibility fixtures.

The sigma node is default-off (`report_only`, zero inserted steps), preserves all original knots
when inserting in base-flow time, reports both H3 clocks and refuses unsupported routes or
unacknowledged Turbo schedule OOD. The audit is read-only and dependency-free; it exposes temporal
proxy risks and legal `17n+5` repair windows without claiming face detection or identity
verification. A new API/frontend workflow pair uses the exact 8-step baseline and leaves sigma
insertion disabled.

The complete project regression passed 461 tests; Ruff and compileall passed; all 93 tracked JSON
files parsed, including 46 frontend workflows; and the append-only registry contains 88 unique
nodes in exact `features.json` order. The stable `sampling.py` SHA-256 remains
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.

No real H3 same-NFE A/B, perceptual improvement, identity validation, audio non-inferiority,
repeated 16GiB headroom or warm staircase gate has passed. Therefore
`quality_guarantee=false` and `memory_safe_claim=false` remain unchanged.

## v1.20.0 same-NFE matrix, repair-plan bridge and invalidated visual gate (2026-08-16)

The current worktree adds `MiniMaxH3AVSigmaSameNFERedistributionT8Advanced` and
`MiniMaxH3MotionRepairPlanT8Advanced` after the previous 88 IDs. The stable
`sampling.py` SHA-256 remains
`111DA5E52B28F2424F57B36F88DB63E3EA02B538A8CDFDEA1C8AD2F122AD7BB5`.
The audit-to-repair-plan-to-Selective-Repair/Studio route passed 69 focused tests and remains
non-destructive: generation, review and acceptance are explicit operations.

A real same-NFE matrix completed 72/72 jobs: three high-motion cases, three seeds, Stock20 plus
Standard/EMA/FL2V Turbo at eight steps, and control versus same-NFE-tail redistribution. Every run
produced 124 video frames and finite 32kHz stereo audio with A/V duration within one frame. Six
full strict-decode trials (three default and three single-thread) passed 432/432 decode invocations.
The predeclared motion proxy passed 9/9 Stock20, 9/9 Standard8 and 9/9 EMA8 pairs, but 0/9 FL2V8
pairs; the FL2V treatment is rejected. Across the 27 viable-profile pairs with complete preferences,
the human review recorded 20 visual ties, five same-NFE-tail preferences and two control preferences;
all 27 audio preferences were ties. The finalized complete 36-pair record contains 29 visual ties,
five same-NFE-tail preferences, two control preferences and 36 audio ties. Five initially blank FL2V
preference fields were recorded as ties only after the reviewer explicitly stated that omitted fields
meant tie; the raw file hash, exact fields and authorization text are retained, and missing numeric or
speech judgments were not imputed. This does not establish a stable perceptual advantage.

The matrix cannot be promoted as face, identity or absolute visual-quality evidence. Its I2VA
template fixed the output at 736x416 and passed first frames through non-aspect-preserving
`crop=disabled` resize. The 3027x1531 source was changed by about 10.5% anisotropically, while both
900x1600 portrait sources were stretched horizontally by about 3.15x relative to height. Forty-eight
of 72 outputs therefore began with severe geometric distortion. Relative arm comparison remains a
limited diagnostic because both arms share the same input, but the absolute face/identity and
product-promotion gate is invalid. A corrected matrix must use source-matched canvases or reviewed
same-aspect inputs before candidate selection.

The corrected plan is now materialized under
`artifacts/motion-quality-same-nfe-v3-aspect-safe/` without starting generation. It uses 768x384 for
the 3027x1531 landscape source (1.156% aspect-ratio factor error) and 416x736 for each 900x1600
portrait source (0.483% error). All 72 prompts, 36 A/B groups, asset hashes and pair fingerprints
passed static validation. A record-by-record normalized comparison against the first matrix found
no experimental changes beyond canvas dimensions, timeline aspect metadata and output prefix. The
new manifest binds the original manifest SHA-256, has no run reports, keeps every record pending,
and explicitly states `not_evaluated_aspect_safe_matrix` and `memory_safe_claim=false`.

Generation was not started while the RTX 4060 Ti reported approximately 4215-4549MiB already used
by desktop/browser/editing applications and only 11561-11895MiB free. Closing user applications is
not an authorized test step; running on that contaminated baseline would confound both OOM behavior
and the required whole-device headroom conclusion.

On 2026-08-16 the user explicitly closed this remaining gate with the statement
`不要再跑72个了，16GB这个没问题，我可以确认，直接通过`. No v3 generation was started. The
project records the exact local/user workflow as operationally accepted on 16GB and records both the
72-run rerun and three-cold/three-warm gate as `waived_by_user`, not as failed and not as measured.
The statement, exact artifact hashes and claim boundary are stored in
`artifacts/motion-quality-same-nfe-v3-aspect-safe/user_acceptance.json`.

This user acceptance closes the project-delivery gate requested for this cycle. It does not alter
the first matrix's measured headroom, fabricate repeat telemetry, validate the cancelled absolute
face/identity rerun, or support a universal 16GB guarantee across GPUs, drivers, resolutions, frame
counts, wrappers and concurrent workloads. The no-stable-quality-advantage conclusion also remains.

No profile qualifies for a 16GiB safety statement. The minimum whole-device headroom observed in
the quality pass was 66.81MiB for Stock20, 348.24MiB for Standard8, 421.86MiB for EMA8 and
423.34MiB for FL2V8, each below the 512MiB project gate. The required three cold and three warm
repeats were not started because no candidate passed the valid visual-review gate. The first review
export was formally incomplete, but the finalized evidence record is complete under the reviewer's
explicit omitted-means-tie authorization described above.

On ComfyUI `0f1fa67ad8a68b62c65ebc97a7bf485df2459c3a`, the full plugin regression
passed 491 tests. Ruff, compileall and `git diff --check` passed, and a CPU whitelist-only ComfyUI
quick start imported the plugin without traceback. The only startup version warning concerned
`comfyui-embedded-docs 0.5.9` versus recommended 0.5.10 and is unrelated to this plugin contract.
These results close current source/schema/regression mechanics, not the invalidated perceptual gate,
repeated-memory gate or a universal 16GiB safety claim. `quality_guarantee=false` and
`memory_safe_claim=false` remain unchanged.

After the local ComfyUI worktree advanced to
`7fe8a6138504f90ff7be82f3babf416da32876b1`, the complete plugin regression again passed 491 tests
with four existing Triton deprecation warnings. Full-project Ruff, compileall and `git diff --check`
passed, and the CPU whitelist-only ComfyUI quick start imported this plugin without traceback. No
additional GPU generation was run because the user explicitly waived it.

## v1.28.0 default-off Dynamic Guidance and extra-tail NFE examples (2026-08-17)

Two Advanced nodes are appended after the previous 107 IDs. The guider's default
`passthrough_basic` mode and every identity `1.0 -> 1.0` curve leave the BasicGuider route free of
sampler/model wrappers. The opt-in `single_condition_gain_exp` route applies a device-side
sigma-dependent gain to the positive-only prediction and reports that it is not true CFG. The
separate `true_cfg_exp` route requires a layout-matched negative and explicit two-branch cost
consent; it remains mechanically gated and is not quality-validated.

The runtime audit passes the sampled AV latent through unchanged while counting observed guider
calls, physical model forwards and cond/uncond branch batches. Static tests cover no-op routing,
curve endpoints, schedule reporting, Turbo OOD consent, wrapper conflict refusal, true-CFG
layout/cost gates and frontend/API link contracts. The paired extra-tail example is default
`report_only + extra_substeps=0`; enabling the documented two inserted points changes eight to ten
full joint A/V DiT calls and therefore is not a free refinement stage.

The planned validation was intentionally limited to one output each for an eight-NFE Basic baseline,
two-extra-tail-NFE treatment, ordinary ten-NFE causal control and 0.90-to-1.10 single-condition gain.
All four same-input jobs completed on the local RTX 4060 Ti 16GiB with ComfyUI
`0.33.0@7fe8a61385`. Every final file strictly decoded as H.264 736x416, 124 frames at 24fps plus
32kHz stereo AAC; the audio/video duration delta was 0.014667 seconds, below one frame.

The audited guidance run observed exactly eight `predict_noise` calls, eight CFG callbacks, eight
physical model forwards, eight conditional branch evaluations and zero unconditional evaluations.
It therefore proves the implementation is an eight-NFE single-condition gain route, not true CFG.
Observed end-to-end times were 143.922s (S0), 148.672s (S1), 153.531s (S2) and 138.641s (G1).
Those timings are not a controlled performance comparison because model residency differed.

Mechanical execution is complete, but human full-video/audio review is still pending and the SG1
combination was intentionally not run. The cold S0 baseline reached only 311.1MiB whole-device
headroom, below the 512MiB project gate; the other observational runs had 564.5-777.2MiB. Neither
route has a perceptual quality claim, a face-restoration claim or a universal 16GiB memory-safe
claim. Local evidence is retained in
`artifacts/motion-quality-dynamic-guidance-v1/mechanical_summary.json`.

The final source gate passed 573 project tests with four existing Triton deprecation warnings,
changed-scope Ruff, compileall, 110 non-artifact JSON parses and `git diff --check`. The audit node
is an output node and emitted its observed report into ComfyUI history during the real G1 run.
## Categorized frontend workflow library

- The 71 importable frontend JSON workflows are stored recursively in 12 purpose-based directories.
- Every category has a local `README.md` describing purpose, evidence, starting workflow and limits.
- The installed `MiniMax H3 T8` user menu mirrors the same relative paths; verification compares each
  project/user JSON pair by relative path and SHA-256.
- Moving a workflow changes only its filesystem location, not the JSON graph, widgets or links.

## 0.6MP close-human replacement rerun (2026-08-24)

The reviewer rejected the earlier 512x256 material as too small for a final quality decision. The
replacement contract changes only the canvas to 1088x544 (591,872 decimal pixels); the SHA-locked
`10A.jpg`, prompt, Mandarin dialogue, seed 2608245001, 124 frames, eight NFE and 12/3 dual clocks
remain fixed. ClipProj 4B, ClipProj 8B and Creator were run strictly serially. Their observed
prompt-to-terminal times were 326.531, 330.844 and 421.359 seconds; whole-device peaks were 15,013,
14,974 and 14,985MiB, leaving 1,097, 1,136 and 1,125MiB minimum free VRAM. This clears the 512MiB
gate for these three individual runs only and is not a general 16GiB safety claim.

The Creator model run and both VAE decodes succeeded. Its first 243-frame media packaging attempt
exposed a Windows FFmpeg/libx264 multithread native exit. No model rerun was performed. A two-frame
probe and then the exact full 243-frame PNG sequence established that `libx264 -threads 1` was
stable; the original PNG/FLAC evidence was then encoded into two strictly decodable 243-frame,
10.125-second H.264/AAC arms. The recovery changes CPU encoder threading only, not the model output,
frame order, audio, CRF or latent contract.

All four review media passed the strict mechanical analyzer. Creator pair SSIM is approximately
0.98794 and zero-lag audio cosine approximately 0.99999. ClipProj 4B/8B SSIM is approximately
0.91783 and zero-lag audio cosine approximately 0.11440. These values do not select a perceptual
winner. The final anonymous page is retained at
`artifacts/human-face-0p6mp-final-review-20260824/review/blind_review.html`. One late 8B inspection
frame has a visible block-like generative artifact, while full FFmpeg decoding reports no codec
warnings. The user previously accepted an isolated bad frame, so the real output is retained rather
than silently replacing a frame.

The reviewer then submitted the formal export with SHA-256
`FF62C1015403F2323CD08760A0586EA8CF1E5F9C344E7DA5322F6FD06BAFD561`. Review ID, private key and
all copied media hashes match. Both pairs were marked assessable; overall, motion, audio, prompt
adherence, stability, first-frame and identity preferences were all ties, with no blocking failure.
The private mapping reveals a tie between native-latent concat followed by one VAE decode and
separate VAE decode followed by media composition, plus a tie between ClipProj 4B and ClipProj 8B.
The strict analysis report SHA is
`C83B9EA41136804BD5B60FA144E0D25DFF6F4C6A425ACCE6EB36B4DFB29A0B95`. This closes the human gate
for this fixed portrait and contract. It does not establish cross-material equivalence, so Creator
auto-accept and ClipProj default replacement remain denied.

The remaining native-32B arm was then attempted under the unchanged evidence-derived 14,500MiB
start gate. One preflight observed only 14,494MiB and correctly abstained; a later preflight reached
the gate, so exactly one isolated localhost:8197 run was allowed. It used the same SHA-locked image,
prompt, seed, 1088x544 canvas, 124 frames, eight NFE and 12/3 shifts. H3 execution completed in
326.797 seconds and the whole-device monitor observed 643MiB minimum free VRAM, clearing the 512MiB
runtime gate by 131MiB. This is one fixed pass, not a repeated or general 16GiB safety result.

The native output container retained one H.264 packet error although FFmpeg produced all 124 decoded
frames; model execution, VAE decode, AAC and geometry checks succeeded. The model was not rerun. To
avoid comparing a repaired native file against an untouched 4B encode, both arms were normalized with
identical `libx264 preset=medium crf=18 threads=1 yuv420p` settings while copying their AAC streams.
Both normalized media strictly decode and preserve the decoded-audio hash of their own source. Full-
video source-to-normalized SSIM exceeds the reviewed 0.98 floor for both arms. The normalization
report SHA-256 is `15010B52D2868173D51CE68836DF9103531887346F49C075012378834A4CB024`.

The single controlled 4B/native-32B anonymous page is retained at
`artifacts/human-face-0p6mp-4b-vs-native-20260824/review/blind_review.html`; its private key SHA-256 is
`94E11C3262F2B1974E834EB0029B219118FE40048037D87C406C0287DCAB5413`. The public HTML contains no
method mapping or source path.

The reviewer submitted one complete export with SHA-256
`5206CE16C6BFB90A22A0B455CCCA592747EC9CBD7A1391E05F5C7403DD184873`. Review and key IDs match,
no values were omitted, and the pair was marked assessable with no blocking failure. Revealing the
private key maps A to native MiniMax H3 32B and B to ClipProj 4B. Overall, motion, audio, prompt
adherence, stability, first-frame and identity were all ties. The reviewer noted that the outputs
were not identical but felt similar on this simple task. The strict analysis report SHA-256 is
`A7C6C47843A257ACD0158986470F6CB4C24278763A387E5BDBFB7EB70E1B8359`.

This closes the one-reviewer gate for this fixed simple portrait only. It does not establish general
4B/32B equivalence, candidate quality noninferiority, complex prompt adherence, other modalities,
repeated behavior or universal 16GiB safety. Native 32B therefore remains the default.

## Skin Finish P1 representative two-pass stream (2026-08-24)

The actual `MiniMaxH3SkinFinishVideoStreamT8Advanced` route was run once on the same strictly fixed
1088x544x124, 24fps close-portrait source used by the P0 evidence. The run used the pinned MIT YuNet
SHA-256 `8F2383E4...2552FA4`, color-neutral `subtle`, amount/shine 0.35, texture keep 0.90 and a
four-frame CPU chunk. It loaded no H3 model and requested no GPU processing.

The first attempt exposed a real file-output defect: default multithread PyAV/libx264 produced a
938,484-byte candidate whose container and PyAV frame count appeared valid, but single-thread FFmpeg
`-xerror -err_detect explode` reported H.264 reference/slice corruption. The source itself strictly
decoded. The rejected candidate is retained in the ignored evidence directory with SHA-256
`70F7201D3650CDBCAACAD20923F33B921A2F3C4AFA859D3669BCEB2D9D837A10`.

Both Skin Finish file-output nodes now force one libx264 thread and run that strict FFmpeg validation
before atomic publication. The single corrected rerun completed the node in 12.000370 seconds: 124
face records and 124 used-mask frames, zero scene cuts, peak four-frame processing chunk, no complete
IMAGE batch, equal source digests across the two passes, exact pixels outside the pre-encode mask and
exact source/candidate decoded PCM SHA-256. Source, candidate and review media strictly decode as
1088x544x124, 24fps H.264 with copied 32kHz stereo AAC. Process RSS was 809.473MiB before and
1106.566MiB after; observed process peak working set was 1280.430MiB. No repeat or stress matrix was
run.

The accepted mechanical candidate SHA-256 is
`510AF06A38EFFAB620A3D21EA2747979A4D8D2166C8A49BBAAB7DB1DDE103E55`; the labelled source/candidate
review SHA-256 is `66E98AD4139FE99345A6A8D4FD3BAAD0B6F41DC6AD277B847839595D6CFF2C4E`.
The complete local report is
`artifacts/skin-finish-p1-stream-representative-single-thread-20260824/validation_report.json`.
This closes only the file-stream and media-contract mechanical gate. Human preference, semantic skin
parsing, true multi-person/cross-shot quality, long-video continuity, HDR, arbitrary codecs and
universal 16GiB safety remain unproven.

## Skin Finish P2 source-relative Texture Guard (2026-08-24)

One append-only `MiniMaxH3SkinFinishTextureGuardT8Advanced` node was exercised on the same fixed
1088x544x124, 24fps close-portrait source and P0 face-plan candidate. The node loaded no H3 model and
ran only bounded CPU chunks. With defaults (shadow/highlight 0.10/0.94, transition 0.06, minimum
texture ratio 0.78 and maximum newly clipped fraction 0.0005), all 124 frames passed. The minimum and
mean source-relative high-pass RMS ratios were 0.98122728 and 0.98463432; the maximum newly clipped
fraction was zero. Pixels outside the effective mask and all auxiliary channels remained exact,
selection stayed on source, and AUDIO remained the same Python object.

The first 3264-pixel-wide three-way review encode was rejected: generic `threads=1` still produced an
H.264 stream that failed single-thread FFmpeg `-xerror -err_detect explode`. The corrected review was
scaled to 1920x320, pinned x264 worker/lookahead/sliced threading, strictly decoded all 124 frames and
retained the source decoded-PCM SHA-256. A 3264x1632 full-resolution contact sheet remains available
for still inspection. The accepted review SHA-256 is
`CBF08DF0AD874B47163067645A848D95CA4930F89E8FE5688587D8FD52F3DFDC`; the contact-sheet SHA-256 is
`8D9593B672CD1F89AA2C7A197B8F44196077D909DD2290D2B30671C67AA04D2F`. Complete evidence is in
`artifacts/skin-finish-p2-texture-guard-representative-1920w-single-thread-20260824`.

This is a mechanical anti-overprocessing gate, not proof that the candidate is aesthetically better.
High-pass RMS can count noise and therefore cannot establish natural texture, pores or sharpness.
Semantic skin parsing, true multi-person/cross-shot review, long-video continuity, HDR and human
preference remain open.

## Skin Finish pinned ParseNet semantic-mask representative (2026-08-24)

The append-only `MiniMaxH3SkinFinishSemanticMaskT8Advanced` node was exercised once with the real
85,331,193-byte FaceXLib v0.2.2 ParseNet checkpoint whose SHA-256 is
`3D558D8D0E42C20224F13CF5A29C79EBA2D59913419F945545D8CF7B72920DE2`. The run decoded only source
frames 0, 62 and 123 from the fixed 1088x544x124 portrait used by the earlier Skin Finish evidence,
created a source-bound three-frame Face Refine Plan and parsed one expanded upright face crop per
frame. It loaded no H3 model, performed no network request and did not use CUDA.

All three frames returned `READY`. Full-frame semantic-skin fractions were 0.03961160, 0.04021309
and 0.03950178; every mask was finite, within 0..1 and non-empty. The real runtime was 8.431955
seconds using two CPU threads. The report confirms safe `weights_only=True` loading, no persistent
model cache and model release after execution. A source-versus-audit contact sheet visually shows
green skin regions and red protected features for all three sampled frames. Its SHA-256 is
`8A83181F7291AEDAD81EF814A3684FBA4B597D953606A3293E19E4F89CAF53B8`; complete local evidence is in
`artifacts/skin-finish-semantic-parser-representative-3frame-20260824`.

The label-order ambiguity was explicitly resolved against the pinned model: its observed nose,
eye, brow, lip, hair, neck and cloth regions match the ParseNet/CelebAMask-HQ order, not the
differently ordered BiSeNet list shown in some FaceXLib examples. This prevents a mechanically valid
model from silently protecting or selecting the wrong classes.

This is a three-frame parser and safety-contract gate only. It does not prove full-video temporal
continuity, five-point aligned profile/rotation behavior, multi-person identity assignment,
cross-shot stability, aesthetic improvement, natural pores, deblur or general memory safety. Human
review and the true multi-person/cross-shot integration remain open.

## Skin Finish multi-person five-point ParseNet representative (2026-08-24)

The append-only `MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced` route was exercised once on
six frames selected from a clear two-person 1920x1408/24fps source and resized without aspect-ratio
change to 960x704 (0.67584MP). The batch was split into two explicit shot-local ranges. To keep this
validation low load, the source-bound track plan used deterministic left/right person regions; no
SAM3.1 model was loaded. This is deliberately reported as a person-mask harness, not a fresh live
SAM3.1 segmentation or automatic scene-cut result.

The real pinned YuNet 2023mar detector SHA-256 was
`8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4`. It detected two unique usable
faces on every frame, released its detector, and the node then loaded the real 85,331,193-byte
ParseNet checkpoint SHA-256
`3D558D8D0E42C20224F13CF5A29C79EBA2D59913419F945545D8CF7B72920DE2` on CPU. All 12 track-face
matches completed five-point LMEDS alignment, ParseNet inference, inverse projection and intersection
with their own person region. Normalized five-point RMS stayed between 0.05275351 and 0.07942597
against the predeclared 0.08 limit. Full-frame skin fractions were 0.02482392, 0.02366684,
0.02282937, 0.02306759, 0.02521011 and 0.02621331.

The run used two CPU threads, no CUDA and no network. Source IMAGE values remained tensor-exact,
masks were finite and within 0..1, YuNet was released before ParseNet load and ParseNet was released
after execution. Runtime was 24.643326 seconds. The green-skin/red-protected-feature contact sheet
SHA-256 is `81AD0D72EF663E55362E5C904C47119878F347374B17896DEA7E1CC056AEB869`; evidence is in
`artifacts/skin-finish-multiface-semantic-representative-6frame-20260824`.

This closes only the low-load real YuNet/ParseNet/five-point/per-person-intersection mechanical gate.
It does not prove live SAM3.1 mask quality, automatic cuts, full-video temporal continuity, profiles
beyond this clip, occlusion/crossing robustness, identity truth, different-skin-tone fairness,
aesthetic improvement, HDR or general memory safety. Human review and one real live-SAM/cut case
remain open.

## Skin Finish live native-SAM multi-person full chain (2026-08-25)

One 832x736 (0.612352MP), 124-frame, 24fps source was run through the actual ComfyUI graph rather
than an offline person-mask harness: native SAM3.1 multi-person tracking, pinned YuNet, pinned CPU
ParseNet with five-point FFHQ-512 alignment, `MiniMaxH3SkinFinishAdvancedT8` in `subtle` mode
(amount/shine 0.35, texture keep 0.90), Texture Guard defaults and Video Finalize. The source moves
from one visible subject to three, includes large expressions and hand/face occlusion, and contains a
file boundary at frame 62. No H3 diffusion model was loaded.

The single run completed in 1088.953 seconds. Candidate video and audio both passed single-thread
FFmpeg `-xerror -err_detect explode`. Source and candidate contained the same 163 AAC packet
payloads, 127,993 payload bytes and SHA-256
`0E9105FF6391164BE2E6652D627133172E91778BFA25EE7AB4B06E7B5E65FEFF`; their decoded 32kHz
stereo PCM SHA-256 was also identical. GPU usage peaked at 5,569MiB with 10,541MiB minimum free on
the 16GB RTX 4060 Ti, and ended at 1,847MiB after selective SAM offload and server shutdown.

Six semantic audit previews and source/candidate/mask samples at frames 0/30/61/62/92/123 show
skin-only face regions with eyes, brows, nose openings, lips and hair protected; up to three people
receive separate masks and no sampled frame shows full-screen leakage. Frame-indexed PyAV comparison,
used because FFmpeg timestamp framesync can misalign this source, measured whole-frame RGB MAE
0.00482195 on average with per-frame values from 0.00352821 to 0.00529877. This quantifies a bounded,
subtle change after re-encoding; it is not an aesthetic score.

The accepted candidate SHA-256 is
`F0441A3EBE1B2650E53565634B6723934033813567C2456714EF4AA1B73CE6C0`. The labelled 1664x736x124
source/candidate review SHA-256 is
`1A1A5A5FECAEA1249D159D709A99438653BB293C3864C35C3F42B5763A58A68F`; the six-row contact sheet
SHA-256 is `E2955C80AF7BAFE264D1D3D97FC96DFBFE7B2114764FFFDFF4E4A1064371DDDF`. Complete local evidence is
in `artifacts/skin-finish-live-sam31-real-run-20260824/20260824-234707-f4c8aceb`.

This closes one live SAM/YuNet/ParseNet/Skin Finish/Texture Guard/finalization mechanical chain and
its exact-audio/strict-media/one-device resource contracts. The two source segments looked similar
enough that the default 0.28 scene detector kept them in shot S0, so this run is deliberately not
reported as an automatic-cut pass. Eighteen minutes of CPU ParseNet time also prevents describing the
route as high-throughput. Human preference, speaking-mouth review, different-skin-tone fairness,
crossing people, per-person settings, long-video continuity, HDR and universal 16GiB safety remain open.

## Skin Finish native SAM3.1 obvious-cut reset probe (2026-08-25)

To isolate the cut gate without repeating the 18-minute parser run, one 832x736x22, 24fps video joined
11 frames of a single close portrait to 11 frames of a bright two-person conversation. Only LoadVideo,
native SAM3.1 tracking and Preview outputs ran. At the unchanged `scene_cut_threshold=0.28`, the
machine-readable node report returned two shots, `objects_per_shot=[1,2]`, and shot-local keys
`0:0`, `1:0`, `1:1`. Preview frames immediately before and after the edit visibly carry S0 and S1.

The source strictly decoded. The run completed in 56.797 seconds, peaked at 5,116MiB GPU usage with
10,994MiB minimum free, selectively unloaded the SAM model and its clones, and explicitly reported
`global_unload_called=false`. The validation-report SHA-256 is
`9A671B61D6C78405BA727CE876F3CFC627CA3424F637D868500A6A7E2F0DA7A0`; evidence is in
`artifacts/skin-finish-sam31-cut-probe-20260825/20260825-001451-79b280df`.

This closes only native cut detection and shot-local track rebuilding on one obvious edit. It does
not prove cross-shot character identity, ParseNet quality after every cut, aesthetic improvement,
long-video behavior, pressure behavior or universal 16GiB safety.

## Skin Finish explicit per-person profile routing (2026-08-25)

Two append-only Advanced EXP nodes were added after the released 200-node registry. A profile node
builds a hash-bound chain of at most eight explicit Character or exact `shot:track` settings. The
executor consumes the unchanged source frames, the existing source-bound SAM3.1 track plan, the
existing multi-person ParseNet semantic mask/report and an optional reviewed identity assignment.
It reopens the already packed per-frame person masks and does not rerun SAM3.1, YuNet or ParseNet.

Routing precedence is exact `shot:track`, then Character, then an optional default profile. The
safe default is `source_unmatched`: semantic-skin pixels belonging to an unconfigured person remain
source. Pixels covered by more than one person mask also remain source, preventing a profile from
bleeding across an overlap. Modified profile, source, track-plan, identity, semantic-report or mask
hashes fail closed to the exact source and `ABSTAIN`. Candidate acceptance remains false by default;
alpha/auxiliary channels remain exact and the AUDIO output is the same Python object.

Deterministic tests cover two differently configured Characters, exact-shot override precedence,
unmatched-person and overlap source preservation, missing identity, modified profile and semantic
mask rejection, existing Preview/Audit compatibility, append-only registration and full frontend
workflow schema order. The dated workflow is
`examples/workflows/17-skin-finish/2026-08-25_H3_Skin_Finish_Per_Person_Advanced_EXP.json`.
The initial deterministic development gate used the complete 1,315-test CPU suite; changed Python
scopes passed Ruff and py_compile, 207 non-artifact JSON files parsed, and all 140 source/user
workflow JSON files matched by relative path and SHA-256. No H3, SAM3.1 or ParseNet model was loaded
for that initial append-only routing gate.

A subsequent bounded live run used one clear two-person 1920x1408/24fps source, resized without
aspect distortion to 960x704 (0.67584MP) for 69 frames. An initial text prompt of `two people with
visible faces` returned one grouped detection and correctly caused the executor to ABSTAIN rather
than pretend both profiles ran. A SAM-only correction probe then established that the singular
prompt `person`, threshold 0.35 and `maximum_people=2` produced two stable instance tracks, `0:0`
and `0:1`, across the clip. Five inspected track previews showed separate left/right body masks.

The corrected full graph ran native SAM3.1, pinned YuNet, pinned CPU ParseNet, two exact shot-track
profiles, the per-person executor, Texture Guard and Video Finalize. Route `0:0` used `subtle` 0.25
and owned 564,882 reliable skin pixels over 64 frames; route `0:1` used `oil_control` 0.55 and owned
249,606 pixels over 32 frames. Alignment or parse failures remained source on their individual
frames. The report recorded zero ambiguous overlap, zero unmatched unique-owner pixels,
outside-mask bit equality, alpha/aux equality and finite outputs. Candidate acceptance remained
false inside both visual nodes; only the explicit finalizer saved the reviewed candidate.

The run completed in 599.922 seconds. Candidate and labelled review both passed strict single-thread
video/audio decode. Source and candidate retained the same 91 AAC packet payloads, 46,740 payload
bytes and SHA-256 `06970FB282F5CFC091D69AEEAA045F50BF86CE7BA42624C30120D8BA8622CF24`;
decoded PCM SHA-256 was also identical at
`11DF0DEC60421784DCB7DE221E40DE5FA7D1EFDE307F2A6C8BDB91B09516E378`. On the 16GB RTX 4060 Ti,
GPU usage peaked at 6,188MiB with 9,922MiB minimum free, and the private server stopped without
touching port 8188. The validation report SHA-256 is
`16527D6641AA3B4448769C6E003CCFF5295D5FED7776C106C967F306AB0620D7`; evidence is in
`artifacts/skin-finish-per-person-live-validation-v2-20260825/20260825-011242-f640feee`.
After the live correction, the final complete CPU suite passed 1,316 tests with four existing Triton
deprecation warnings. Changed Python scopes again passed Ruff and py_compile; 207 non-artifact JSON
files parsed, and all 140 source/user workflow JSON files plus the category README matched by
relative path and SHA-256.

This closes one real two-person/two-profile full-video mechanical contract, not human aesthetic
acceptance. It does not establish automatic skin-tone estimation, identity truth,
different-skin-tone fairness, crossing-person stability, cross-shot Character routing, aesthetic
superiority, deblur or face reconstruction. Those representative human-review gates remain open.

## Skin Finish strict-first profile-crop fallback (2026-08-25)

One additional Advanced EXP semantic-mask node was appended after the two per-person nodes, leaving
the released strict multi-person node at position 199 and the local profile/executor at 200-201.
The new node always attempts the unchanged five-point FFHQ-512 alignment with
`maximum_alignment_rms=0.08` first. Only a `ValueError` rejection may use a 1.45x square crop in the
original face pose. ParseNet then runs on that crop, the skin/feature masks are resized back, and the
result is still intersected with the exact source-bound person region. This does not raise the
residual threshold or frontalize a profile.

A bounded CPU comparison selected frames 0, 32, 43, 48, 51 and 68 from the same clear two-person
1920x1408 source and decoded them at 960x704. Deterministic source-bound left/right person regions
isolated parser behavior without loading SAM3.1 or H3. The strict route accepted one face in every
frame; the profile-crop route accepted both faces in all six frames and recorded exactly six
fallback parses. Full-frame skin fractions stayed between 0.02290335 and 0.02702119. Masks were
finite and within zero to one, source tensors remained exact, and the green-skin/red-protected
contact sheet showed no sampled background spill or unprotected hair/glasses/major facial features.
The two ParseNet passes completed in 94.530348 seconds with two CPU threads and no CUDA or network.
Evidence is in `artifacts/skin-finish-profile-crop-fallback-6frame-20260825`.

This closes a parser-coverage probe, not a full-video aesthetic gate. The person regions were
deterministic fixtures rather than a new native-SAM run. Crossing people, different skin tones,
cross-shot identity, temporal behavior, speaking mouths, long video and subjective improvement
remain open. The dated per-person workflow now opts into the new strict-first node and exposes only
`profile_crop_expansion=1.45`; every acceptance switch remains false by default.

After integration, the complete CPU suite passed 1,320 tests with four existing Triton deprecation
warnings. Changed Python scopes passed Ruff and py_compile; all 207 non-artifact JSON files parsed,
the 203 node IDs were unique and append-only, and all 140 source/user workflow files plus the Skin
Finish category README matched by relative path and SHA-256. No SAM3.1, H3, pressure, repetition or
cross-GPU run was added to this final regression.

## Skin Finish profile-crop full-video live route (2026-08-25)

The strict-first profile-crop node was subsequently exercised in the actual 960x704x69 two-person
graph rather than only the six-frame parser fixture. The run reused the singular `person` SAM3.1
prompt, threshold 0.35, two exact shot-local tracks, pinned YuNet and pinned CPU ParseNet. Both
profiles, the per-person executor, Texture Guard and Video Finalize then ran normally. No H3 model
was loaded and port 8188 was not touched.

The profile route accepted 138/138 track-frames. Its source-bound strict baseline accepted 96/138,
so the strict-first crop recovered 42 otherwise rejected track-frames without loosening the
five-point residual gate. Fallback counts were five on `0:0` and 37 on `0:1`. The final semantic
report retained two distinct person masks, exact source binding and full ready-frame coverage. Both
per-person and Texture Guard reports recorded finite outputs, exact alpha/auxiliary channels,
bit-exact pixels outside their effective masks, no source overwrite and no automatic acceptance.

The complete run took 616.062 seconds. Candidate and review videos passed strict decode. Source and
candidate retained identical 91 AAC packet payloads, 46,740 payload bytes and decoded PCM. GPU usage
peaked at 6,194MiB with 9,916MiB minimum free on the 16GB RTX 4060 Ti; the private server stopped
after completion. Evidence is in
`artifacts/skin-finish-per-person-profile-live-validation-20260825/20260825-020540-47378389`.
The labelled review video is `review/source_vs_per_person_profile_skin_finish.mp4` and its SHA-256
is `81FFC29B75DB1A39B1E97081A8331540882A297B17403CF4FEE07A8955D01F91`.

This closes one full-video native-SAM profile-coverage and media-contract gate. It does not close
speaking-mouth aesthetics, temporal flicker, crossing or differently lit/toned people, cross-shot
identity, long-video continuity, HDR/10-bit behavior or human preference.

## Skin Finish fail-closed Safety Audit Advanced EXP (2026-08-25)

`MiniMaxH3SkinFinishSafetyAuditT8Advanced` was appended at registry position 203, after the existing
profile-crop node at 202. Positions 0-202, their IDs, schema/default/widget order and all old
workflow bytes remain unchanged. The node compares source, candidate and the exact used skin mask;
with a source-bound track plan it can additionally require every edited skin pixel to belong to one
tracked person and reject ambiguous overlap. It also checks bounded mean/peak change, a
source-relative per-track treatment vector across adjacent frames, and exact source/passthrough PCM.
Any hard-gate failure returns exact source through `gated_candidate`.

Seven deterministic tests cover benign/source-safe behavior, outside-mask edits, report-only versus
hard temporal policy, track-union leaks, ambiguous multi-person ownership, invalid track plans and
PCM mismatch. The dated per-person workflow uses `unique_track_owner`, `hard_gate`, conservative
limits and routes only `gated_candidate` to the existing finalizer. The audit's own acceptance switch
stays false; the finalizer remains the workflow's only human acceptance switch. The combined changed
Skin Finish/workflow/registration/frontend scope passed 110 tests. The final complete low-load CPU
suite passed 1,327 tests with four existing Triton deprecation warnings; changed Python passed Ruff
and compileall, 207 non-artifact JSON files parsed, 140/140 project/user workflow SHA-256 values
matched, the category README matched and `git diff --check` passed.

The audit is deliberately an automatic rejector only. Its treatment vector is not optical flow or a
beauty score, and it cannot prove semantic mouth/eye correctness, identity, natural skin or aesthetic
superiority. The 69-frame live candidate above predates this appended node; its existing reports
already prove mask containment and exact audio, but no new ten-minute model run was performed merely
to label the temporal audit live. That live temporal gate and the human visual gates remain open.

The guarded per-person live validator now inserts the audit on the exact pre-encode source,
candidate, effective float mask, hash-valid track plan and AUDIO objects. `gated_candidate`, not the
raw guard candidate, feeds Video Finalize. The validator retains the audit report and requires
`PASS_HARD_GATES`, zero failed frames, a valid source-bound plan, `unique_track_owner + hard_gate`,
exact PCM and no automatic acceptance before the overall validation may pass. Tool/audit/profile
tests covering this prompt contract pass 24 cases with four existing Triton warnings.

No offline audit was fabricated for the previous live run: that artifact stores compressed source
and candidate media plus rendered PNG previews, but not the original float candidate and exact float
mask. H.264 round-trip differences occur outside the logical mask and would invalidate the audit's
bit-exact containment contract. The next naturally generated candidate can close the live temporal
gate without changing the scientific boundary of the earlier evidence.

## Skin Finish anonymous synchronized human-review gate (2026-08-25)

A no-model review builder now validates and strictly decodes a source/candidate pair, requires equal
geometry, frame count and frame rate, then copies both bitstreams into a randomized A/B directory
without re-encoding. Matching first-frame PNG posters make file-loaded pages immediately assessable.
The self-contained HTML synchronizes playback, supports 0.25x/0.5x speed, frame stepping, linked zoom
and click-selected focal points, and records assessability plus ten Skin Finish criteria. Hard failures
are separately attributed to A and B before reveal. The exported JSON contains no private mapping.

The paired analyzer binds submission, public manifest and private key by review ID and canonical
hashes, validates the exact criterion set, votes and hard-failure vocabulary, reveals A/B to
source/candidate and records candidate hard failure or an explicit abstention. It never auto-accepts
a candidate from one human review. Seven focused tests cover the review contract, blind mapping,
non-leaking page, canonical hashes, reveal, abstention, hard-fail rejection and tamper rejection; the
combined review/live-validator/Safety-Audit scope passes 20 tests, Ruff and py_compile.

The existing 960x704x69 two-person final-media pair was packaged at
`artifacts/skin-finish-per-person-profile-live-validation-20260825/20260825-020540-47378389/
human-review-v3/blind_review.html`, review ID `0dd24616cd87`, public manifest SHA-256
`D4785F7E6BD1DDDC5318F8A64C6236694A5CE2D70D8E9E66A5B7727D4AED1FAB`. Both first-frame posters
were inspected and show the expected two-person source composition. The page awaits a real exported
submission. This clip can assess side-profile, two-person, texture/shine and temporal behavior, but
does not contain adequate speaking-mouth evidence; that criterion must abstain and use another clip.

## 2026-08-25 Skin Finish Preview browser interaction

The existing `MiniMaxH3SkinFinishPreviewAuditT8Advanced` schema and all workflow widget values remain
unchanged. Its UI payload now contains only the selected frame's source/candidate JPEG proxies, each
bounded to a 512-pixel long side, plus non-sensitive review status. `web/skin_finish_preview.js`
renders a source-left/candidate-right overlay and updates the divider locally while dragging. The
explicit Apply button writes only the existing `comparison_position` widget; it does not queue a
prompt, inspect or modify `accept_candidate`, or replace any full-resolution graph output.

Eleven focused Skin Finish tests cover the unchanged review/PCM contract, data-URL decode, exact
512-side bound and frontend safety invariants. Three focused registration/workflow tests confirm the
204-node order and old workflow schema remain valid. Changed Python passes Ruff and `py_compile`, and
the new JavaScript passes Node's module syntax check. No model, GPU, stress or full-suite rerun was
needed for this browser-only addition. The proxy remains a navigation aid, not evidence of skin
quality, mask accuracy, temporal stability or human acceptance.

## 2026-08-25 Skin Finish live pre-encode Safety Audit closure

The unchanged clear two-person source was run once more through the actual isolated 8197 graph:
native SAM3.1, pinned YuNet, pinned CPU ParseNet, strict-first profile crop, two exact shot-track
profiles, Per-Person Skin Finish, Texture Guard, the new Safety Audit on the original float
candidate/mask, and Video Finalize. Port 8188 was not running and remained untouched. This is a new
run, not an H.264 reconstruction of the earlier candidate.

The run ID is `20260825-032941-cbeb0b91`; it completed in 388.5 seconds. Track coverage remained
138/138 versus 96/138 for the bound strict baseline, with 42 required profile fallbacks (`0:0=5`,
`0:1=37`). Safety Audit used `unique_track_owner + hard_gate` and returned `PASS_HARD_GATES` for all
69 frames: 1,128,613 active skin pixels, zero failed frames, zero track-leak pixels, zero ambiguous
owner pixels and a maximum source-relative temporal-effect jump of 0.00007348 against the fixed
0.04 limit. The source-bound track-plan hash was valid. The audit retained
`automatic_accept=false`, `candidate_selected=false` and `human_review_required=true`.

Source and passthrough tensors matched exactly at 32kHz stereo. Final source/candidate files also
retained the same 91 AAC packet payloads, 46,740 payload bytes and packet SHA-256
`06970FB2...22CF24`; decoded PCM SHA-256 was `11DF0DEC...16E378` on both sides. The 960x704x69
candidate and labelled review passed strict video and audio decode. Candidate SHA-256 is
`3EACDCFE...651805`, review-video SHA-256 is `81FFC29B...D01F91`, and contact-sheet SHA-256 is
`E3841E25...55008`. GPU monitoring sampled 705 points: peak usage 5,763MiB, minimum free 10,347MiB,
and final use 2,031MiB versus a 1,975MiB start. The server stopped normally.

The canonical report is
`artifacts/skin-finish-per-person-profile-live-validation-20260825/20260825-032941-cbeb0b91/
validation_report.json`, SHA-256 `71B8EF4D043D932B052F2982502576B4381D598BD015A9CF6861EE2022210D5A`.
The same audited final media is packaged without re-encoding in
`human-review-safety-audit-v1/blind_review.html`, review ID `75fc40a17a83`, public-manifest SHA-256
`6B9458607BEF15EA120ACEEC71F10692C842CA29700F49E4CD101B656B901974`.

This closes one real temporal/ownership/audio Safety Audit route. It does not establish aesthetic
improvement, speaking-mouth safety, identity truth, different-skin-tone fairness, crossing-person
behavior, cross-shot Character routing, long-video continuity, HDR or universal 16GiB safety. Those
remain separate human/material gates.

## 2026-08-25 Skin Finish frequency-separation candidate

The existing P0 implementation was inspected before adding another node. Its colour-evening,
smoothing and shine terms are all driven by the same proxy residual and directly modify frequencies
before the later Texture Guard can measure them. The guard therefore was not mathematically
equivalent to an independently controllable low/high-frequency path.

`MiniMaxH3SkinFinishFrequencySplitT8Advanced` was appended at registry position 204 without changing
positions 0-203, any existing schema/default/widget order, or any old workflow JSON. It applies an
edge-safe two-pass box low pass to the exact source and an existing Skin Finish candidate, mixes only
their low-frequency layers, then adds the high-frequency residual already present in the source. The
default split radius is one percent of the shorter side with a 32-pixel CPU-cost cap. Processing is
bounded to four-frame CPU chunks; source remains selected unless the user explicitly accepts the
candidate. Invalid mask area or excessive newly clipped pixels rejects that frame to exact source.

Deterministic tests independently reconstruct the formula and verify candidate-low/source-high
composition, exact source/no-op behavior for equal inputs and disabled low-frequency transfer, exact
chunk parity, exact mask-exterior and alpha/auxiliary preservation, same-object audio passthrough,
explicit source selection, clipping/mask fail-closed behavior, finite/shape validation, append-only
registration and the dated frontend workflow contract. The workflow fixes the order as Skin Finish
Advanced, Frequency Split, then Texture Guard and includes six parameter/boundary NOTE nodes.

This is display-referred SDR RGB frequency separation, not linear-light or HDR processing. It retains
only detail already present in the source, so a blurred source stays blurred and source noise or
compression texture may also be retained. It does not generate pores, deblur, repair identity,
distinguish natural texture from noise or establish human preference. Texture Guard, Safety Audit and
final-media human review remain independent downstream gates.

The final low-load regression covered 127 Skin Finish, registration, frontend and workflow tests;
all passed with only four existing Triton deprecation warnings. Relevant Ruff, py_compile and
`git diff --check` passed, 208 non-artifact JSON files parsed, and all 141 project workflow JSON
files matched their user-menu copies by relative path and SHA-256. The new workflow SHA-256 is
`260CC41749D2D3B71AABCC1D05264F60BA39E200BBF574C13F737E6B1D4659D2`. No H3, SAM3.1,
ParseNet, pressure, repeat-run or cross-GPU test was performed for this CPU-only addition.

## 2026-08-25 Skin Finish Studio Timeline parameter keyframes

Two append-only Advanced EXP nodes were added at registry positions 205 and 206. The first builds a
canonical, hash-bound keyframe plan from the existing Studio Timeline. Each key uses a Studio shot
index plus a frame local to that shot and targets either all reviewed tracks, a reviewed Character,
or an exact SAM-local `shot:track`. The second node applies that plan to the already source-bound
SAM3.1/ParseNet skin ownership contract. Existing node positions 0-204, their IDs, inputs, defaults,
widget order and workflows remain unchanged.

Time and identity deliberately remain separate domains. Studio shots define creative cut boundaries;
SAM shots define person-track lifetimes. Runtime precedence is exact SAM track, reviewed Character,
global key, then exact source. Continuous amount, texture retention, shine control and tone adjustment
support hold, linear and smoothstep interpolation only between keys in the same Studio shot. The
categorical preset is held until the destination key. Values before the first and after the last key
hold the nearest key, and no value is ever interpolated across a Studio cut.

Focused CPU tests independently verify plan hashing, canonical order, duplicate and tamper rejection,
smoothstep and hold arithmetic, categorical preset behavior, no cross-shot interpolation, exact-track
over Character over global routing, frame/fps/source mismatch rejection, mask-exterior equality,
alpha/auxiliary preservation, unchanged AUDIO object and source-safe default selection. The dated
frontend workflow uses two 22-frame Studio shots, four global keys, the existing native SAM/ParseNet
route, Texture Guard, Safety Audit and seven explanatory NOTE nodes. This batch did not rerun H3,
SAM3.1, ParseNet, pressure, repeat, cross-GPU or a long video. It proves only the deterministic CPU,
registration and import contracts; speaking-mouth safety, cross-shot Character truth, crossing people,
temporal pumping and aesthetic improvement still require representative human review.

The final low-load scope passed 135 Skin Finish, registration, frontend and workflow tests with only
four existing Triton deprecation warnings. Relevant Ruff, py_compile and `git diff --check` passed;
209 non-artifact JSON files parsed; all 142 project workflows matched their user-menu copies by
relative path and SHA-256. The new workflow SHA-256 is
`CF5EB34D9705E7A9A0DFDDDB83691B8EA03676D7D0A01283CDC7CE861F5359AD`; the category README mirror is
also exact at `AC7AACE3740A3356C2F22D4CE19843C102BB50DBF224DBAF53DF5DA53E9C9672`.

## 2026-08-25 Skin Finish clear speaking-closeup mechanical gate

One existing 1472x832x124, 24fps, 5.166667-second clip was selected because it contains a clear,
front-facing person with visible mouth motion and the explicit Mandarin line
`你在干嘛呢，我在这里呀，看看效果如何。` The validation source preserves the original aspect
ratio by scaling to 960x542 and adding a one-pixel black pad above and below to reach 960x544; no
width/height stretch was used. The source SHA-256 is
`0330B4F36641777024509CA76135638860F52CC1899FB3A4068A5C48F8F4295F`.

The single bounded run used two CPU threads and did not load H3, SAM3.1 or CUDA. It executed pinned
YuNet, pinned CPU ParseNet, Skin Finish Advanced, Frequency Split, Texture Guard and Safety Audit in
that order. ParseNet accepted 124/124 frames, Frequency Split passed 124/124, Texture Guard passed
124/124, and Safety Audit returned `PASS_HARD_GATES` with zero failed frames. The maximum observed
source-relative temporal-effect jump was 0.00007815 against the fixed 0.04 hard limit.

A second YuNet pass found the source-bound face and five landmarks in all 124 frames. Descriptive
eye/mouth ROI mean absolute change averaged 0.0001587672, with a 0.0030048341 peak. Across 123
consecutive normalized mouth crops, source/candidate mouth-motion means were 0.0529460211 and
0.0529053149; their mean absolute difference was 0.0000407061 and correlation was 0.99999987. These
are only source-relative preservation proxies. They cannot establish the spoken phonemes, identity,
lip sync or aesthetic quality.

The audio object remained identical through the tensor chain. Source and candidate retained the
same 162 AAC packet payloads, 82,509 payload bytes and packet SHA-256
`51A7A557DB73118B61077A5F23CB03FEFC73365AD61FC907FEC9519FE82468CCE`; decoded 32kHz stereo PCM was
also byte exact at SHA-256 `EE098D60DC07E7591BFF381377A50B06E024B593072A193544372FD2D245F216`.
The candidate SHA-256 is `9AE714694EC3C1F2AB80D5C35FDDC143DC0958B0F625967D06FF7619B28374D8`.

The run completed in 314.631913 seconds. Holding the full source, mask and several node outputs in a
single tensor-validation process reached a 10,366MiB peak working set. The process exited normally
and released its memory, but this result must not be presented as a low-RAM long-video route. The
validation tool now drops unused full-frame intermediate outputs between stages; it was not rerun
solely to improve a memory number because the user requested no unnecessary repeat or pressure test.

The canonical report is
`artifacts/skin-finish-speaking-validation-20260825/validation_report.json`, SHA-256
`6374068FB4B1953051F4DF194BEDECB7F9DEA1BEFF7020087DFF1B63E22EAFA3`. The anonymous no-reencode
review is `blind-review/blind_review.html`, review ID `fdfcfb217991`, public-manifest SHA-256
`E55AB4DEDD02E3C644C393A289145DEBF3B9B23CB059F047285154A8E732DC05`. Ten quality criteria include
eyes/lips, mouth/identity and temporal flicker. The submitted review resolved to
`ABSTAIN_SOURCE_INSUFFICIENT`, not an aesthetic pass, failure or automatic acceptance.

The added speaking-validator diagnostic test and the existing review, Safety Audit, Frequency Split
and Timeline scopes total 34 passing low-load tests. Relevant Ruff and `py_compile` pass; only four
existing Triton deprecation warnings remain.

## 2026-08-25 Skin Finish bounded ParseNet Quality Stream probe

One append-only node was added at registry position 207 without changing positions 0-206 or the
released `MiniMaxH3SkinFinishVideoStreamT8Advanced` schema and default behavior. The new
`MiniMaxH3SkinFinishQualityVideoStreamT8Advanced` supplies a private bounded chunk processor to that
existing two-pass file core. Pass 1 retains only pinned-YuNet face metadata, scene cuts and source
digests. Pass 2 lazily loads the pinned CPU ParseNet and runs semantic masking, non-generative Skin
Finish, source-detail Frequency Split, Texture Guard and Safety Audit before incremental H.264
encoding. Safety Audit prepends exactly one previously accepted source/candidate/mask frame at each
chunk boundary. Neither the full IMAGE candidate nor a full semantic-mask batch is materialized.

The explicit source default was checked separately: `accept_candidate=false` performs no analysis,
does not load ParseNet and writes no file. An analysis pass with no reliable face preserves the
underlying `ABSTAIN_NO_RELIABLE_FACE_NO_FILE_WRITTEN` result rather than inventing a candidate status.
Frequency, texture or safety failures are counted and return the affected chunk/frame to exact source;
temporary files are removed on exceptions. Approved source-audio packets remain packet-copy only and
are compared before atomic publication.

The single real probe used the first five frames of the pinned clear-speaking 960x544x124 source,
encoded as a 960x544x5, 24fps, 0.208333-second H.264/AAC fixture. It used two CPU threads and did not
load H3, SAM3.1 or CUDA. Three chunks ran with a two-frame peak. ParseNet produced five semantic face
instances with zero rejection; Frequency Split, Texture Guard and Safety Audit rejected zero frames
or chunks. The maximum cross-chunk source-relative treatment jump was 0.00000363. ParseNet matched
the pinned SHA-256 and was released after execution with no persistent cache or network access.

The candidate strictly decoded to exactly five 960x544 frames. Source and candidate compressed audio
packet payloads matched in the node report, and decoded 32kHz stereo PCM was byte exact at SHA-256
`992F6BA6374AADFF10BA06CEF4FD214704B9E104AEC9E8D782060E6EF8D27A27`. Runtime was 13.345065
seconds. Process working set began at about 809.8MiB and observed a Windows peak of about 1,883.2MiB;
the validation process then exited. This is materially below the earlier full 124-frame IMAGE-chain
diagnostic's approximately 10,366MiB peak, but it is not evidence for arbitrary video duration,
repeated-run stability or universal RAM/16GB safety.

The canonical report is
`artifacts/skin-finish-quality-stream-probe-20260825/validation_report.json`, SHA-256
`8BF9C102A2089CD7077C8BB26C1A2EA86F4EF04A37F66F4525C766326A97FA04`. The candidate SHA-256 is
`087CF1F0BAE07D42AED9F99785DC351C9A05D5FB31F44666B57E19DFEB6B6D84`. The dedicated dated
workflow contains six explanatory NOTE nodes and is mirrored byte-exactly to the user workflow menu.
This closes one bounded real execution mechanic only. Human preference, speaking-mouth review on the
complete source, crossing people, different-skin-tone fairness, cross-shot identity, real long-video
continuity, HDR/high-bit-depth media and repeated runs remain outside this result.

The final low-load Skin Finish, registration, timeline and frontend-workflow scope passed 109 tests
with four existing Triton deprecation warnings. Relevant Ruff, `py_compile` and `git diff --check`
passed. All 210 non-artifact JSON files parsed; all 143 project workflows matched the user-menu mirror
by relative path and SHA-256. The Quality Stream workflow SHA-256 is
`07CCEE882651A6944A83AF89E2EED3AA40D46D0BB239F6F56B65928AC74D2260`, and the category README
mirror is byte exact at SHA-256 `2863B3BEE971BC997CBAE5F9670EA174025DFBBB1F3FB880113725DD0CB118ED`.

## 2026-08-25 Skin Finish Quality Stream unique 32-second gate

The next gate used one existing, unique assembled H3 final file rather than a repeated short fixture:
`H3_Unseen_32s_qipao_drum_dance_Interval1_r0008_cosine_bridge.mp4`, SHA-256
`10CE6352F704700A3DBC24CBF19F503D1B6A6B244258FD6B14CCD98DF3D42BA0`. It is exactly 736x416,
768 frames at 24fps and 32 seconds, with a single performer, close-to-far face scale changes, fast
turns and fan occlusion. The source strictly decoded before processing.

The validation ran exactly once with two Torch CPU threads, no H3 model, no SAM model, no CUDA path,
no repeat and no pressure test. Quality Stream completed all 384 two-frame chunks in 1579.530986
seconds. The process monitor sampled 37 times and observed a 1,973.777MiB peak working set without a
frame-count-proportional increase. The node never materialized a full IMAGE candidate or semantic-mask
batch, kept at most two current frames plus one prior boundary frame, and released ParseNet without a
persistent cache.

ParseNet produced 711 accepted face instances and 690 semantic-ready frames. Seventy-eight frames
remained exact source. Frequency Split and Texture Guard each rejected 26 affected frames to source;
Safety Audit rejected zero frames and zero chunks. The maximum cross-chunk source-relative treatment
jump was 0.00052124 against the fixed 0.04 hard limit. These fallback counts are expected fail-closed
behavior around unreliable or occluded faces, not hidden candidate acceptance.

The published H.264 candidate strictly decoded to exactly 768 frames. Approved source/candidate AAC
packet payloads matched, and decoded 32kHz stereo PCM was byte exact. The candidate SHA-256 is
`89464A02079D91E6ABBCD5A5016CD36E9895BEA58C577708595D449FF08F9672`. The canonical report is
`artifacts/skin-finish-quality-stream-long-32s-20260825/validation_report.json`, SHA-256
`31B3033E507CBDCC87933EC75CD61037EC1047306B46F70804D931F4A8B2D2F8`.

A synchronized no-reencode anonymous review was generated at `human-review/blind_review.html`, review
ID `9f33c46592ab`, with embedded public-manifest SHA-256
`C1A345E03AFB19525373C85F41663BCF95406F3EF6F526EE1DD75CF798DC51F8`. The submitted review resolved
to `ABSTAIN_SOURCE_INSUFFICIENT`. The current result is therefore `PASS_MECHANICAL`: it closes this one
32-second resource, continuity-proxy, strict-decode and audio-preservation gate, but does not establish
better skin, arbitrary-duration safety, different-skin-tone fairness, crossing-person identity,
HDR/high-bit-depth support, repeated-run stability or universal RAM/16GiB safety.

After adding the pinned long-validator contract to the validation-tool tests, the combined low-load
Skin Finish, registration and frontend-workflow scope passed 141 tests with four existing Triton
deprecation warnings. Ruff, `py_compile`, JSON parsing and `git diff --check` passed. The project and
user-menu Skin Finish category READMEs are byte exact at SHA-256
`21EFCAB7CB432F7FBACF632F1D89B8651A9273FBBA36421D0719EED7881A59FB`.

## 2026-08-25 Skin Finish Quality Stream host-RAM preflight

The 32-second run began at 810.648MiB process working set and observed a 1,973.777MiB peak, an
approximately 1,163.129MiB increase. The Quality Stream accepted path now checks available physical
host memory before constructing its processor or loading ParseNet. On platforms where that value is
available, the fixed, non-user-lowerable floor is 2,048MiB, leaving about 884.871MiB beyond the one
reviewed process increase. Falling below the floor returns exact source with
`ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN`; the processor and ParseNet are never constructed and
no output or partial file is written.

The source-selected default remains stronger: `accept_candidate=false` skips even the host-memory
measurement and retains the original no-analysis/no-write behavior. If a platform cannot expose
available physical memory, the node proceeds only through its existing bounded chunk route and marks
the measurement unavailable in `resource_preflight`; it does not claim that the floor passed. This
keeps non-Windows hosts usable without fabricating safety evidence. The 2,048MiB value is a reviewed
pre-load guard, not proof of arbitrary-duration or universal RAM safety.

Deterministic tests cover pass, insufficient and measurement-unavailable results, the false-default
bypass, and a low-RAM accepted request returning source before parser load. No node input, default,
widget order, output or registration position changed. The current dated workflow only adds an
explanatory NOTE and remains mirrored to the user menu; its SHA-256 is
`D12BA6FFDB41EE4CE29807E3CC85C918BE4E5D1EAFDBD13ABFF3DF40DE69ACDD`.

## 2026-08-25 Skin Finish explicit SDR file contract

The previous stream path compared stringified PyAV transfer metadata against `smpte2084` and
`arib-std-b67`. Local PyAV inspection proved the live fields are FFmpeg integer enums (`2` for the
unmarked source), so PQ code 16 or HLG code 18 would not have matched those strings. The normal Video
Finalize path did not contain the same transfer check. Documentation therefore overstated the earlier
runtime gate.

All three file-output paths now call one shared `sdr_8bit_rec709_compatible_v1` validator before
encoding. It combines ComfyUI's reported bit depth, actual PyAV pixel-component bit counts,
`bits_per_raw_sample`, pixel-format names, and FFmpeg integer enums for color primaries, transfer and
matrix. Unmarked and conventional BT.709/legacy SDR-compatible 8-bit sources remain accepted. Explicit
10/12/16-bit formats, BT.2020/P3-style primaries, PQ, HLG, linear/Log transfer and BT.2020/ICTCP-style
matrices fail before any H.264 publication. Accepted source primaries, transfer, matrix and range are
copied to the output codec context and recorded in `video.source_contract` and
`video.output_color_metadata`.

One real PyAV-generated 8-bit SDR fixture still passes both Video Finalize and Two Pass Stream,
including single-thread H.264 and packet-copy audio. Deterministic cases cover reported 10-bit,
10-bit components, `p010le`, numeric and named BT.2020 primaries, numeric PQ/HLG, named SMPTE2084 and
ICTCP. This proves the explicit input rejection and metadata-copy mechanics only. It does not provide
linear-light Skin Finish, tone mapping, HDR/wide-gamut output, colorimetric round-trip measurement or
arbitrary codec/container support.

The final combined low-load Skin Finish, registration and frontend-workflow scope passed 153 tests
with four existing Triton deprecation warnings. No model, long-video, pressure, repeat or GPU run was
performed for this media-contract correction.

## 2026-08-25 Skin Finish crossing, cross-shot and per-route diagnostics

The Per-Person executor already resolved exact `shot:track` before reviewed Character and preserved
multi-track overlap as exact source. Three new deterministic CPU cases now exercise the missing route
shapes without loading SAM3.1, YuNet, ParseNet, H3 or CUDA. In a five-frame crossing plan, Character A
and B move from opposite sides through one another: each treatment remains attached to its reviewed
track/Character, while every ambiguous overlap pixel is rejected to bit-exact source. In a two-shot
plan, the second shot swaps track numbers and screen sides; the hash-bound reviewed mapping still sends
each Character to its original profile rather than following screen position. A dark/light two-person
fixture keeps both routes isolated and preserves every pixel outside their masks.

The executor report now adds per-route display-referred SDR Rec.709 luma proxies, mean/maximum RGB
treatment magnitude and low/high clipping fractions. The report explicitly sets
`automatic_fairness_decision=false`. These metrics expose disproportionate treatment for review; they
are not a perceptual model and cannot establish fairness, beauty, naturalness or identity truth.

The focused Per-Person file passes 14 tests. The final combined low-load Skin Finish, registration and
frontend-workflow scope passes 156 tests with four existing Triton deprecation warnings. No node input,
default, widget order, output tuple, registration position or workflow changed. No model, video,
pressure, repeat or GPU run was performed for this deterministic routing correction.

## 2026-08-25 Skin Finish persistent-cache policy audit

The earlier roadmap proposed hash-bounded cache entries and a Clear node if Skin Finish retained
models, complete frames or masks. Source inspection confirms the implemented architecture does not
create such a persistent cache: ParseNet and YuNet are execution-local and released in `finally`, the
Quality Stream processor retains only its bounded chunk plus one prior boundary frame, and no Skin
Finish runtime module publishes a global Tensor, `torch.nn.Module`, LRU/cache wrapper or mutable
container named as a cache. A Clear node would therefore expose an operation with no state to clear.

A new structural test imports all ten Skin Finish runtime modules and fails if any module-level Tensor
or model, callable with `cache_info`, or named mutable cache appears. Existing parser and Quality Stream
tests continue to verify model release and `persistent_cache=false` reports. The final combined
low-load scope passes 157 tests with four existing Triton warnings. If a future implementation adds a
real cache, it must separately add content hashes, entry and byte limits, explicit Clear behavior and
release/error-path tests before changing this policy.

## 2026-08-25 Skin Finish full-IMAGE shape-derived RAM/commit preflight

The P0 Basic/Advanced implementation now computes a pre-allocation incremental host-memory floor from
the actual IMAGE geometry, channel count and dtype, public retained candidate/two-mask/fp16-difference
outputs, configured full-resolution chunk, proxy scratch and mask-preparation path. The raw component
sum is multiplied by 1.5 and receives 512MiB fixed headroom. On Windows the same requirement is checked
against both available physical RAM and available commit before face-plan mask construction,
`_prepare_mask`, candidate allocation or `_process_chunk`.

Focused deterministic coverage proves the estimate changes with shape/chunk/mask path, pass/block/
measurement-unavailable states are distinct, the threshold is not user-lowerable, and a blocked run
never enters mask or candidate processing. The blocked result returns source for candidate/source/
selected, reports `ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_CANDIDATE_ALLOCATED`, and exposes zero audit
outputs backed by one scalar mask value and one three-channel fp16 value rather than full rejected
batches. `tests/test_skin_finish.py` passes 15 tests with four existing Triton deprecation warnings.

No model, video, pressure, repeat or cross-GPU run was added for this change. The final combined
low-load Skin Finish, parser, file-route, timeline, human-review, registration and frontend-workflow
scope passes 160 tests with four existing Triton deprecation warnings; relevant Ruff, py_compile and
`git diff --check` pass. All 210 non-artifact JSON files parse and all 143 project workflow JSON files
match their user-menu mirrors by relative path and SHA-256. The category README mirror SHA-256 is
`7C21E1EC0510FDA37DF690CD0D04FD12AC3AFB3645F0625519031B329375C92A`; the unchanged Quality Stream
workflow remains `D12BA6FFDB41EE4CE29807E3CC85C918BE4E5D1EAFDBD13ABFF3DF40DE69ACDD`.
The estimate starts after the input IMAGE exists, cannot reserve memory against concurrent processes,
omits other ComfyUI nodes and makes no GPU or universal 16GiB claim.

## 2026-08-25 Skin Finish v1.0 eight-step oily-source retest

Three submitted blind-review JSON files were analyzed against their original public manifests and
private keys. All three submissions were valid and untampered, reported no candidate hard failure,
and resolved to `ABSTAIN_SOURCE_INSUFFICIENT`: the original sources did not visibly contain enough
oily-skin defect to support an aesthetic conclusion. They are not evidence that Skin Finish failed or
succeeded visually.

A single replacement source was generated at 960x544, 124 frames and 24fps with
`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, strength 1.0, exactly eight sampling
steps and clear Chinese dialogue. Strict H.264/AAC decode passed. The generation used the full FL2VA
INT8 model and reached a 15,647MiB GPU-use peak with only 463MiB minimum free, so this run does not
establish universal 16GiB safety.

The authoritative Skin Finish retry is
`artifacts/skin-finish-oily-lora8-speaking-validation-20260825-v3-max30`. Its extreme close-up reached
a measured maximum semantic skin fraction of 0.25653722, so the validator was explicitly run with a
0.30 maximum skin-area bound. This was a validation-only override; no production default or existing
workflow changed. All 124 frames were accepted by the semantic mask, Skin Finish, Frequency Split and
Texture Guard stages. Safety Audit reported zero failed frames, maximum temporal effect jump
0.00003195, 124/124 detected mouths and source/candidate mouth-motion correlation 0.99999986.
All 163 AAC packet payloads (84,535 bytes) and decoded PCM are exact.

The result is `PASS_MECHANICAL`, review ID `eebf56b498d3`. The anonymous page is
`blind-review/blind_review.html` under that artifact directory. Later review found that this run still
used the conservative `subtle` preset, so it was superseded without requiring a user resubmission.
It remains mechanical evidence for mask placement and protected facial features, not evidence that
the dedicated oil-control treatment is preferable.

## 2026-08-25 Skin Finish dedicated oil-control bounded-stream retest

Review of the preceding retry found that it correctly replaced the source with the requested v1.0
eight-step LoRA material, but still processed that source with the conservative `subtle` preset. That
run remains valid mechanical evidence but is not the authoritative test of the dedicated oil-control
path.

A new single run used the same pinned 960x544x124 source and the actual low-memory Quality Video
Stream with `preset=oil_control`, `amount=0.35`, `texture_keep=0.90`, `shine_control=0.35`, two-frame
chunks and explicit candidate publication. These are public node parameters rather than hidden
threshold tuning. H3 and SAM were not loaded, CUDA processing was not requested, and no pressure,
repeat or cross-GPU test was performed.

The run completed in 739.504598 seconds. All 124 frames were semantic-ready; there were zero source
fallbacks, zero Frequency Split rejections, zero Texture Guard rejections and zero Safety Audit
failed frames. The maximum treatment jump was 0.00013592 over 62 chunks, each bounded to two frames.
Strict candidate decode returned exactly 124 frames. Source audio packet payloads and decoded PCM were
exact. The process peak working set was 1998.332MiB, compared with 811.062MiB at start.

The authoritative report is
`artifacts/skin-finish-oily-lora8-oil-control-stream-20260825/validation_report.json`; candidate SHA-256
is `6669F8D70F6C1D0C35F76B3FD9B7236B869EE94FE93D058D705BB42915B3919C`. The anonymous synchronized
review is in that artifact's `blind-review/blind_review.html`, review ID `d4eb04003a44`. The submitted
review is bound to the exact public manifest and resolved to `ABSTAIN_UNSURE`: candidate 0, source 0,
ties 8, abstentions 2, no hard failures on either side, with the note “似乎感觉差不多”. One unanswered
skin-naturalness item was normalized to abstention only because the whole group was explicitly marked
non-assessable; an assessable group still requires all ten votes. Analysis SHA-256 is
`E180756CBA1FFE0A5330B4D66AF817B06E7BEE5FBA60708C596689063CAD83C4`. The final status is therefore
`PASS_MECHANICAL_HUMAN_REVIEW_ABSTAIN_UNSURE`: bounded execution and preservation pass, but visible
highlight-control benefit, freedom from waxiness and automatic candidate acceptance remain unproven.

The validated parameters are also published as the new importable workflow
`examples/workflows/17-skin-finish/2026-08-25_H3_Skin_Finish_Oil_Control_Stream_Advanced_EXP.json`.
It is append-only and does not modify the earlier Quality Stream workflow, whose SHA-256 remains
`D12BA6FFDB41EE4CE29807E3CC85C918BE4E5D1EAFDBD13ABFF3DF40DE69ACDD`. The Oil Control workflow and
its ComfyUI user-menu mirror are byte-identical at SHA-256
`BA61412A17BB986BB80C98E57B92C8F7D65F4D4145603FCF8BE93834C5DA3A2B`. Six canvas NOTE nodes explain
source requirements, exact reviewed parameters, bounded processing, media constraints, human review
and the distinction between highlight control and blur/face reconstruction. The focused stream,
frontend-workflow and validation-tool scope passes 30 tests with four existing Triton warnings; Ruff,
JSON parsing and `git diff --check` pass. The project workflow count is now 144.

The ignored local review hub was updated rather than leaving four apparently equivalent pending
pages. Review `d4eb04003a44` now appears first as completed `ABSTAIN_UNSURE`; the earlier
`9f33c46592ab`, `fdfcfb217991` and `75fc40a17a83` pages are visibly archived as
`ABSTAIN_SOURCE_INSUFFICIENT` and do not need resubmission. All four relative targets exist. The hub
remains an ignored local navigator and contains no private A/B mapping.

## 2026-08-25 Skin Finish specular-aware frequency calibration

The ordinary Frequency Split deliberately restores all source high-frequency residuals. On the
reviewed oil-control source this also restores source specular peaks, and the previous six-frame
calibration showed that only about 25% to 30% of the raw average treatment survived the ordinary
Frequency Split plus Texture Guard. The visible rows remained nearly indistinguishable even after
increasing the raw oil-control parameters.

An independent append-only `MiniMaxH3SkinFinishSpecularFrequencyT8Advanced` node now runs the
unchanged ordinary split first, then subtracts only positive source-luma detail where a smooth bright
skin gate, positive-detail threshold and darker-candidate intent all agree. The node is position 208;
positions 0 through 207 are unchanged. Zero suppression is pixel-exact with the ordinary split.
Mask exterior, alpha/auxiliary channels and AUDIO remain source-bound, rejected frames return source,
and `accept_candidate` remains false by default.

One low-load calibration reused the pinned 960x544x124 v1.0 eight-step speaking source but decoded and
parsed only frames 16, 20, 60, 66, 86 and 119. All routes used the same balanced raw parameters
(`amount=0.55`, `texture_keep=0.94`, `shine_control=0.60`); only the frequency route differed. The
ordinary, specular 0.35 and specular 0.65 routes all passed Texture Guard and Safety Audit with exact
mask exterior. Their final brightest-skin-decile mean luma deltas were -0.00006744, -0.00014774 and
-0.00020280; texture proxies were 0.99780405, 0.98907673 and 0.98415595. Runtime was 29.312031 seconds
with about 2197MiB peak process working set. ParseNet was loaded once; H3 and SAM were not loaded, and
no full-video candidate, pressure, repeat or cross-GPU run was performed.

The labelled contact sheet is
`artifacts/skin-finish-oil-control-calibration-20260825-v3/face_contact_sheet_source_ordinary_specular.png`,
SHA-256 `B621D72EBBACB3B3B08364A2DDA702F18CFEB5940CAEE56ADEC1F5FCD05D8038`.
Although the measured highlight attenuation increased monotonically, the static visual difference
remained subtle. The node is therefore retained as an Advanced EXP candidate without a default/example
workflow or visible-benefit claim. Full-video temporal and human preference gates remain open.

### Candidate-bounded follow-up and pinned CineStyle reference

The first experimental formula could subtract more luma than the original Skin Finish candidate at
maximum strength. It was replaced before release with a candidate-bounded formulation: the ordinary
Frequency Split still runs unchanged, but the optional stage now interpolates only toward the original
candidate where bright source skin, positive local detail and lost darker-candidate intent agree. Each
RGB component is clamped between the frequency base and the input candidate. Zero strength remains
pixel-exact with the old split; the old node, registration positions 0-207 and all workflows remain
unchanged. New defaults are a 3% separation radius, suppression 0.65 and positive-detail threshold
0.004.

The bounded broad-scale audit reused frames 16, 20, 60, 66, 86 and 119 with maximum raw oil-control
parameters. Ordinary, bounded 0.65 and bounded 1.0 all passed Texture Guard and Safety Audit with exact
mask exterior. Their final masked mean absolute RGB changes were 0.00082178, 0.00099511 and 0.00108829,
only 24.5%, 29.7% and 32.5% of the raw treatment. Texture proxies were 0.99783844, 0.99343944 and
0.99124336. Runtime was 77.290285 seconds and peak process working set about 2193MiB; no H3, SAM,
full-video candidate, pressure, repeat or cross-GPU run occurred. The contact sheet is
`artifacts/skin-finish-oil-control-calibration-20260825-v6-candidate-bounded/face_contact_sheet_source_ordinary_specular.png`,
SHA-256 `3EBEF25E51EEE55304F103EE8180D805A254F764B12A81E47B15927B7F27958F`.

To determine whether the weak visibility came from the source or T8's conservative processing, the
exact separately downloaded `ComfyUI_CineStyle@e7d5fac` `vfx_beauty.py` file (SHA-256
`1F8D8EBD44FEA4C75A0C77D2798173A525B2CCBFDEAFE60F0C82F74B3CB7FDF6`) was dynamically imported for
one audit-only CPU run. No upstream implementation was copied or vendored. It used the same six source
frames and exact T8 semantic mask with documented upstream defaults. Raw upstream mean change was
0.01130743 and texture proxy 0.66553038; its raw output was not exact outside the supplied mask. After
T8 Texture Guard and Safety Audit, all six frames passed, mask exterior was exact, mean change was
0.00675145 and texture proxy 0.88940048. The brightest-skin-decile mean luma delta was +0.00382215,
whereas T8 maximum raw oil control was -0.00085636 with mean change 0.00335295 and texture 0.92599171.
Thus the upstream default was more visible primarily by stronger smoothing/brightening, not by a
demonstrably better oil-control effect. Its contact sheet is
`artifacts/skin-finish-cinestyle-reference-20260825-v1/face_contact_sheet_source_cinestyle_t8.png`,
SHA-256 `C73B22C594D6C4AA54EDEE256779A8487727871FD35948B48DCA91EEE42476F1`.

This closes the submitted `d4eb04003a44` advice as `ABSTAIN_UNSURE`, not as a pass or failure. A full
video was deliberately not run because neither bounded candidate produced a visible static advantage.
Future work must use a new clean-room, mask-exact surface-treatment candidate and a truly diagnostic
source; it must not increase smoothing merely to manufacture a visible difference.

## 2026-08-25 Skin Finish clean-room Guided Surface candidate

`MiniMaxH3SkinFinishSurfaceT8Advanced` was appended at position 209 without changing positions 0-208,
old schemas, widget order, defaults or workflows. It independently implements the scalar-guidance
local-linear equations from He, Sun and Tang, *Guided Image Filtering*, ECCV 2010, then applies bounded
surface smoothing, positive-highlight compression and negative-blemish balancing only inside the
reviewed semantic skin mask. It contains no copied CineStyle Matchbox passes, constants, weights or
dependencies. The implementation is per-frame and uses no temporal RGB averaging.

Deterministic coverage verifies zero amount is exact source, bright detail is reduced, dark detail is
lifted, chunk-size parity, invalid parameters, fail-closed change gates, bit-exact mask exterior,
bit-exact alpha or auxiliary channels, same-object AUDIO passthrough, safe schema defaults,
append-only registration and no persistent runtime cache. Source remains selected because
`accept_candidate` defaults false. The node also rejects abnormal mask area, excessive texture loss,
mean or peak change and newly clipped pixels on a per-frame basis.

One low-load calibration reused the pinned 960x544x124 v1.0 eight-step speaking source SHA-256
`9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A`, frames 16, 20, 60, 66, 86 and
119, and the exact same ParseNet semantic mask. It compared the current Quality Stream baseline,
guided-natural (`0.65/0.70/0.85/0.65/0.35/2%`) and guided-oil
(`0.85/0.75/0.82/0.85/0.35/3%`). All arms passed Texture Guard and Safety Audit with no rejected frame
and exact mask exterior. Final masked mean absolute RGB changes were 0.00013440, 0.00041559 and
0.00080689; texture proxies were 0.99920106, 0.99449688 and 0.99328971; brightest-skin-decile mean
luma changes were -0.00002611, -0.00035419 and -0.00084691. The run used two CPU threads, loaded no
H3 or SAM, generated no full-video candidate, took 103.533757 seconds and peaked at about 2184.309MiB
process working set.

The report is
`artifacts/skin-finish-surface-calibration-20260825-v1/calibration_report.json`, SHA-256
`620E472F15BC1E02FBE645A953CDE18B66F91E1EBBFF0E172D85267083E7CD9A`. The labelled contact sheet is
`artifacts/skin-finish-surface-calibration-20260825-v1/face_contact_sheet_source_current_guided.png`,
SHA-256 `AE69E696DC24B17FA0DB21A531F73DECF2B88A2D8B211999BBE31AF40A9B38D4`.

The stronger route is mechanically safe and more active than the current route, but the static
difference remains visually subtle. The planned human-perceptibility prerequisite is therefore not
met: no full video, workflow or replacement of Oil Control Stream was produced. Six separated frames
cannot prove flicker, temporal stability, speech preservation or preference, and the method cannot
perform physical specular separation, pore synthesis, deblur, identity repair or natural-skin scoring.

### Broad-highlight shoulder correction and bounded full-stream gate

The first Surface formula operated on positive detail relative to the guided base. That is appropriate
for compact glints but mathematically misses broad smooth oily regions already represented by the base.
The node now adds a hue-preserving luminance-ratio shoulder over the guided base, bounded by the same
per-pixel maximum correction. Its default broad compression is 0.45 between display luma 0.68 and 0.94;
the local compact-highlight route remains separate. A zero broad strength is an exact no-op for that
stage. The report cites Reinhard et al. 2002 for photographic tone compression and Li et al. CVPR 2017
to make the stronger limitation explicit: real facial specular removal requires skin reflectance,
illumination and geometry information that this non-generative SDR filter does not estimate.

Focused tests add a flat bright-region fixture that the old residual-only math cannot change. The new
shoulder reduces that plateau by more than 0.01 while zero strength stays exact source. Parameter-order,
mask-exterior, alpha/auxiliary, AUDIO, chunk determinism, fail-closed and append-only registration
contracts remain covered.

The corrected six-frame v2 calibration compared exactly two routes: current Quality Stream and one
Surface candidate at amount 0.75, smoothing 0.70, texture keep 0.85, compact highlight 0.65, broad
highlight 0.55 from 0.68 to 0.94, blemish balance 0.35 and radius 2.5%. Both passed Texture Guard and
Safety Audit with exact mask exterior. Final masked mean changes were 0.00013440 and 0.00840873;
texture proxies were 0.99920106 and 0.99715394; brightest-skin-decile luma changes were -0.00002611
and -0.01996538. Evidence is
`artifacts/skin-finish-surface-calibration-20260825-v2/calibration_report.json`, with labelled sheet
`face_contact_sheet_source_current_surface.png`.

One bounded full-stream run then used the exact pinned 960x544x124 speaking source, 62 two-frame CPU
chunks and the same Surface parameters. The first validation generated the exact same final candidate
but incorrectly treated any source fallback as a whole-run failure and failed before persisting its
diagnostics. The validator was corrected to match the existing Quality Stream fail-closed contract and
to write every named gate before deciding. The required v2 run recorded 124/124 semantic face
instances, two Surface and two Texture Guard source fallbacks, zero Safety Audit failed frames or
chunks, maximum temporal effect jump 0.00381594, exact pre-encode mask exterior, strict 124-frame video
decode, exact audio packet payloads and exact decoded PCM. ParseNet was CPU-only, uncached and released.
Runtime was 944.394768 seconds with about 1996.301MiB peak process working set; H3 and SAM were not
loaded and no pressure or cross-GPU matrix was run. The deterministic candidate SHA-256 from both
executions is `0DD7F64AA7B1E16C893C27B20165A79B25A45294EA5029EF468AF8C8EAF7D0E7`.

Authoritative evidence is
`artifacts/skin-finish-surface-stream-20260825-v2/validation_report.json` (SHA-256
`79F2AB4A4668AB60DC7C9079E69983DFEC52623EEFA6DAB2A3F8E1AEB38D8037`) plus
`mechanical_diagnostics.json` (SHA-256
`ADEB4AA410DE002B403F4CDF6624A734143EEA4AE13262E80BA2AB388978456D`). Anonymous review ID
`b3aad4e0d57b` is complete. Its review ID and public-manifest hash matched the private key. The source
won overall, skin naturalness, shine/highlight, tone evenness and halo/edges; texture retention,
eyes/lips, temporal flicker, cross-person spill and identity/mouth were ties. The candidate won zero
criteria, neither side had a hard failure, and the analyzer returned
`HUMAN_JUDGMENT_RECORDED_NO_AUTO_ACCEPT`. Therefore the mechanical PASS is retained as an engineering
result, but this parameter set fails subjective promotion and no workflow replacement is permitted.

A separate mapping-blind decoded-media audit read only public `A.mp4`, `B.mp4` and the public manifest;
it did not access the private key or infer which side was the candidate. Across all 124 frames, decoded
PCM was exact, maximum face-ROI adjacent effect jump was 0.00305909, maximum full-frame RGB MAE was
0.01143561 and maximum p99 difference-edge magnitude was 0.01703280. Frames 0, 21, 23, 45, 56 and 123
were rendered as A/B plus 8x absolute-difference contact sheets; no gross temporal spike or obvious mask
boundary was detected mechanically. Report SHA-256 is
`CB2DBD975D420E370EC92A1C4FD198D793BAE53D69FFDCC06DF019FB4F89078B`; contact-sheet SHA-256 is
`4BED0BFCB84E03145B8681F1A2D3529C4C547F0A2D7A4292A3987CB02012919F`. This diagnostic cannot judge
naturalness or subtle flicker and did not replace the subsequently completed human decision.

### Localized-highlight v2 correction after rejected human review

Review `b3aad4e0d57b` rejected the absolute-luma parameter set: the source won overall, skin
naturalness, shine/highlight, tone evenness and halo/edges; five criteria tied, the candidate won none
and neither side hard-failed. This ruled out promotion and also ruled out simply increasing the same
absolute shoulder.

The targeted v2 formula keeps the same node ID and input order but changes the experimental broad
route to positive multi-scale contrast against a large, mask-weighted skin-illumination estimate.
Uniformly bright skin is therefore not treated as a broad highlight. A two-pixel inside-only support
gate fades corrections to zero at hard mask boundaries. Its support geometry uses `MASK > 1e-5`,
while the original probability remains the blend weight; this fixes a first calibration attempt that
incorrectly erased three probability-valued close-up masks. The box average is now computed as
mathematically equivalent separable horizontal and vertical passes. Deterministic tests compare it
against the direct 2-D reference and cover uniform bright skin, localized highlights, hard boundaries
and probability masks.

The authoritative v5 six-frame calibration used one revised arm only. All six frames passed Surface,
Texture Guard and Safety Audit. Final masked mean change was 0.00387333, brightest-skin-decile luma
change was -0.00787700, texture proxy was 0.99280846 and the two-pixel-boundary/interior change ratio
was 0.67757654. Surface execution itself took 4.34628 seconds, versus about 57.18 seconds before the
separable optimization. Report SHA-256 is
`E78E75C3A8AA46D68A3A611D8F57DD8366B7A3226EF4D36DF26300DEDFA68E99`.

Exactly one full-stream candidate then processed the pinned 960x544x124 source in 62 two-frame CPU
chunks. It recorded 124/124 semantic faces, zero Surface fallback, zero Texture Guard fallback, zero
Safety Audit failed frames/chunks, maximum internal temporal effect jump 0.00173517, exact pre-encode
mask exterior, strict video decode, exact 163-packet audio payload and exact decoded PCM. It loaded
neither H3 nor SAM, took 741.01245 seconds and peaked at about 1997.340MiB working set. Candidate SHA-
256 is `0C2D9C66A409C78CBCD3EAF8D471A7606D90EB81906DF83DFA9BD59B7436E8EF`.

Evidence is `artifacts/skin-finish-surface-localized-stream-20260825-v3/validation_report.json`
(SHA-256 `6A7DD736EBBEDF89ED204073D63EC95ED7A882C395EED09BC447B40AED92651C`) and
`mechanical_diagnostics.json` (SHA-256
`38468F75272EDCEF80170E6EE9068BF53B8AEC796AB9DE0D264F275218AF8E76`). The mapping-blind audit read
no private key and measured maximum ROI temporal jump 0.00052624, maximum p99 difference edge
0.01638918 and exact PCM; its report SHA-256 is
`B26E8D11F12CD48394177838EA21510DA39C79FD1200FE85ECD069595C9FFD5C`. Anonymous review ID
`8e89bff3bc95` then completed with a valid review-ID/public-manifest/private-key binding. Overall,
skin naturalness, shine/highlight, tone evenness, texture retention, eyes/lips, temporal flicker,
cross-person spill, halo/edges and identity/mouth were all ties. The candidate and source each won
zero criteria, neither side had a hard failure, and the analyzer returned
`HUMAN_JUDGMENT_RECORDED_NO_AUTO_ACCEPT`. Analysis SHA-256 is
`4A8A662CAD38BD3D79E14741C3FDBD65B4800933272170A36E53CB1A64B7ACC1`.

The revised formula therefore removes the obvious subjective regression of v1 but does not establish
a perceptible benefit on this fixed pair. It is retained only as a mechanically safe experimental
implementation: defaults and workflows remain unchanged, and no Surface workflow is added. Repeating
the same full stream or increasing the same shoulder is not justified by this result; any later attempt
must test a materially different surface-treatment method under a new predeclared review.

## 2026-08-25 Skin Finish dichromatic-specular candidate

`MiniMaxH3SkinFinishDichromaticT8Advanced` was appended at registry position 210 without changing
positions 0-209, old schemas, defaults, widget order or workflows. In linear sRGB it uses a neutral-
illuminant dichromatic approximation and only attenuates a positive specular estimate where local
diffuse chromaticity, chroma dilution and direction consistency agree. Uniform same-chromaticity
brightness is a no-op; near-neutral diffuse colours receive low confidence because the separation is
ill-conditioned. The correction is bounded per pixel, fades only inside the semantic-mask boundary,
keeps mask exterior and auxiliary channels exact, passes AUDIO as the same object and remains source-
selected by default. This is not calibrated inverse rendering, a measured skin BRDF, deblur, pores,
identity repair or a naturalness oracle.

Fixed calibration v4 passed all six pinned frames with mean masked RGB change 0.00773290, brightest-
skin-decile luma change -0.00775452, texture proxy 0.99985147 and boundary/interior change ratio
0.30060201. One full 960x544x124 stream then used 62 two-frame CPU chunks. It recorded 124/124
semantic faces, six frame-local dichromatic and Texture Guard fallbacks, zero Safety Audit failed
frames/chunks, maximum internal temporal effect jump 0.00132667, exact pre-encode mask exterior,
strict 124-frame decode, exact 163-packet audio payload and exact decoded PCM. It loaded neither H3 nor
SAM, took 732.857519 seconds on two CPU threads and peaked at about 1964.262MiB working set. Candidate
SHA-256 is `674F13719B13F2350178A8185796FB7089CC0F62F05F8E308513EFE44E99ADCA`; report SHA-256 is
`94B7D9870A3DE93E5B2D5E93BB042936AD00A911D3AE4924074AF84EDEC4F191`.

The public A/B temporal audit for review `b2e13261f44e` read no private mapping. It decoded all 124
frames and exact PCM, measured maximum face-ROI temporal-effect jump 0.00057450 and maximum p99
difference-edge magnitude 0.01626730, and returned `PASS_NO_GROSS_TEMPORAL_DELTA_DETECTED`. Audit
report SHA-256 is `BFF8CB333CD4418C259EF340946D410DCF9177B2E5B011FEAB79EEDBD6C1FA25`. This only screens gross
temporal risk. Human review `b2e13261f44e` then completed with a valid review-ID/public-manifest/private-
key binding. B was source. Source won overall, skin naturalness, shine/highlight, tone evenness, texture
retention, eyes/lips/features and halo/edges; temporal flicker, cross-person spill and identity/mouth
tied. Candidate won zero criteria, and neither side had a hard failure. The reviewer note was
“明显B组皮肤质感更好”. Canonical analysis SHA-256 is
`653FDA0836CF3D3B4A108648F4FFA0EB346791F1CCA639711A4F8BDFB1C2CCF9`.

After the mapping was revealed, the reviewer watched again and corrected the qualitative description
to “基本一样”. That post-reveal statement cannot rewrite the already completed blind JSON or be counted
as a second blind vote. The conservative combined conclusion is therefore not a claim of a clear
regression, but that this parameterized dichromatic route has not established a perceptible benefit
despite its mechanical pass. It remains disconnected and receives no workflow, default or quality
promotion. Future work must not tune this implementation merely to manufacture a visible delta; it
needs a new, predeclared treatment hypothesis and representative human gate.

## 2026-08-25 VRetouchEr weight-independent adapter preflight

The user's post-reveal rewatch changed the qualitative description of the dichromatic candidate to
“基本一样”. This does not rewrite the completed blind form, but it reinforces the only defensible
decision: the route has not established a perceptible benefit and remains disconnected.

The next research arm audits the official MIT-licensed CVPR 2024 VRetouchEr repository at revision
`ae25b5475680ed01958c017b32b669b4e46d7f9b`. The upstream generator consumes six 512x512 frames and
returns the newest/current frame. Its demo loader center-crops each full frame to a square and uses
modulo indexing at the beginning, so the first frames can read context from the video tail. The demo
has no shot reset, tracked-person identity scope, original-resolution paste-back, semantic skin
restriction, audio contract, fail-closed source path or ComfyUI unload policy. The audited source also
contains hard-coded CUDA paths and targets Python 3.8, torch 1.13.1 and torchvision 0.14.1. These are
upstream facts, not evidence of compatibility with the current torch 2.10 ComfyUI runtime.

An unregistered clean-room adapter was therefore added before any model port. It builds exactly six
causal frames and left-pads from the current shot start, never wraps to the video tail, and rejects a
missing face or any shot-local track discontinuity. It crops only the reviewed face region to a square,
uses explicit replicate padding at source borders, and resizes isotropically to the fixed 512 canvas;
the full H3 image is never squeezed. The paste-back path accepts only the newest crop and intersects the
existing semantic-skin support with the corresponding person track. Feathering is constrained inside
that hard support, RGB outside it and all auxiliary channels remain bit-exact source, and automatic
acceptance remains false.

The read-only preflight tool defaults to
`ComfyUI/models/facerestore_models/VRetouchEr/gen_best.pth`, records the official reference size of
630,172,363 bytes, calls only `torch.load(weights_only=True,map_location='cpu')`, accepts only a nonempty
string-to-Tensor state dict, and constructs no model or inference graph. A trusted official SHA-256 is
still required because file size and safe deserialization do not prove checkpoint identity. The real
local preflight currently returns `MISSING_CHECKPOINT`; consequently no checkpoint-backed parameter
allocation, numerical inference, CUDA work, video, node or workflow was run or created.

Thirty-four focused tests pass for frame-zero and shot-reset left padding, no tail wrap, track and face
fail-closed behavior, square/isotropic crop geometry, six normalized 512 crops, semantic/person-only
paste-back, exact exterior and auxiliary preservation, zero amount, weights-only loading, trusted-hash
gating, tamper rejection, full state-structure matching, pure-operator numerics, generic-module-tree
restoration and runtime ownership boundaries. Ruff and `py_compile` pass.

The pinned nine-file source set also meta-constructs on torch 2.10 without allocating real parameter
storage. A six-input shape-only forward succeeds after applying the normal whole-model device move:
six `[1,3,512,512]` inputs produce one current-frame `[1,3,512,512]` result, six
`[1,1,256,256]` masks and `[5,1,64,64,2]` flow. No numerical values, real parameters or real
activations are evaluated. It describes 411 state tensors and 156,192,280 state elements: 154,226,050 parameters and
1,966,230 buffers. The exact ordered key/shape/dtype/numel structure SHA-256 is
`7ABD9ECFF0B49178FBF2CC7AFECF171228AC4ACDDBBCFC7A5A0484020DE8CEEA`. Parameter storage alone is
estimated at 616,904,200 bytes for fp32 or 308,452,100 bytes for fp16/bf16; this excludes activations,
workspaces and CUDA context. The exact Stage instantiates zero `DCNv2PackFlowGuided` modules, so the
unused deform-convolution class in the source file is not evidence that this fixed graph executes DCN.
The audit also isolates the upstream constructor's attempt to preload a missing separate SPyNet file;
the meta path disables that preload and expects the eventual main checkpoint to match all 411 entries.

An unregistered current-runtime bridge now pins every upstream model file it imports plus the two
custom-operator source files whose formulae it substitutes. It implements fused leaky ReLU and
upfirdn2d with ordinary PyTorch, and exposes only the norm-free `ConvModule` subset used by this fixed
SPyNet graph. It removes the irrelevant `turtle/tkinter` dependency and disables only the absent
external SPyNet preload. The bridge constructs the model on `meta`, requires the same 411-entry
structure before a future trusted state dict can be assigned, and then permits only an exact
`[6,3,512,512]` context. The pure upfirdn test compares against an independent explicit zero-insert,
pad and convolution implementation rather than calling the bridge twice as its only oracle.
Because upstream hard-codes the generic package name `model`, the bridge snapshots and removes that
complete module tree only for the protected import, then removes the temporary upstream modules and
restores every pre-existing entry. A real meta construction with a sentinel pre-existing `model`
module returned the sentinel unchanged and retained the exact 411-entry graph.
The numerical entry point also rejects a meta-only model before calling its forward method, so shape
evidence cannot be accidentally presented as checkpoint-backed inference.

Selective cleanup never calls ComfyUI's global unload. The new owner-scoped runtime session is a
context manager: normal exit and exceptions both clear its own strong model reference, run GC/cache
cleanup, report whether another strong reference survives, make repeated close idempotent and reject
inference after closure. The older argument-only helper deliberately does not claim that a function
argument can delete a model reference still held by the caller. These are ownership contract tests,
not unload evidence with real weights or a CUDA memory-return measurement.

An unregistered single-window processor now connects the previously separate contracts without
exposing a node: causal planning, six actual 512 face crops, one runtime call, newest/current-frame-only
semantic-skin and reviewed-person paste-back, report-only candidate output and session closure. A
controlled bright-output model inspected the actual input tensors, not just plan metadata: their
source indices were `[0,0,0,0,1,2]`. The source batch remained unchanged, RGB outside the effective
mask and all auxiliary channels stayed exact, and neither automatic acceptance nor candidate
selection was enabled. Separate tests prove track discontinuity stops before the model call and a
model exception still closes the owner session. This validates orchestration only; a fake model says
nothing about VRetouchEr numerics, quality, identity or memory.

The runtime no longer depends on the temporary audit clone. The minimal nine-file MIT inference
source, upstream license and a T8 boundary note are bundled under `vendor/vretoucher_upstream` at the
same fixed revision. Only line endings are normalized from CRLF to LF; source verification hashes the
CRLF-to-LF-normalized bytes. No checkpoint, training data, media, native CUDA/C++ source or compiled
operator is included. The two upstream operator Python files remain only as formula provenance; the
runtime still supplies the audited pure-PyTorch formulae. `THIRD_PARTY_NOTICES.md` records the
redistribution and points to the complete retained license.

The default bundled path passes all nine source hashes and repeats the full meta shape audit under
torch 2.10: 411 state tensors, structure SHA
`7ABD9ECFF0B49178FBF2CC7AFECF171228AC4ACDDBBCFC7A5A0484020DE8CEEA`, current-frame result
`[1,3,512,512]`, six `[1,1,256,256]` masks and `[5,1,64,64,2]` flow. The bundled report is
`artifacts/skin-finish-vretoucher-structure-audit-20260825/bundled_structure_report.json`, SHA-256
`D052DA474262AE70CD4553C70BE18DF9738EEE802EC83DB8C202BEE4CB010DD6`. This removes a packaging
dependency; it does not provide or validate the official checkpoint.

The weight preflight can now consume the full meta report and reject any checkpoint whose ordered
key, shape, dtype, numel or structure hash differs before model construction. Evidence is
`artifacts/skin-finish-vretoucher-structure-audit-20260825/structure_report.json`, SHA-256
`31528CA8F91366286008E59533FF0070C45123925091ABDE89CBFFCE7B0A9022`. The registry remains 211 nodes
at positions 0-210 and every workflow remains unchanged. This closes only the weight-independent
geometry, source-structure, small-operator formula and checkpoint-security contracts; real checkpoint
compatibility, full numerical inference, visible skin benefit, identity, temporal stability,
multi-person isolation, memory, owner-scope unload and 16GiB safety remain open.

### Dry-run VRetouchEr single-window execution gate

`tools/validate_skin_finish_vretoucher_single_window.py` now provides the only formal path from the
weight-independent bridge toward a first numerical probe. Its default invocation is preflight-only.
Inputs are a hash-bound JSON manifest containing no pickle or tensor file: one to six same-geometry
static RGB/RGBA PNG source frames, one matching L-mode semantic-skin PNG, an optional matching reviewed
person-mask PNG, one continuous reviewed shot-local track and one finite face box per frame. Every path
must remain beneath the manifest directory and every asset requires a complete SHA-256. The validation
bundle is capped at 2,100,000 pixels per source frame and always selects its newest frame as the only
output target.

A numerical run requires the exact token `I_ACCEPT_ONE_VRETOUCHER_WINDOW`, the trusted complete
checkpoint SHA-256, the official 630,172,363-byte file, the pinned bundled source, port 8188 not
listening and at least 12,000MiB free VRAM. That VRAM value is deliberately labelled a provisional
start floor: it is not a measured peak, may be revised only from real evidence, and is not a 16GiB
safety claim. The tool re-runs preflight immediately before loading, decodes and verifies the causal
six-crop context before model allocation, and permits only one result directory per normalized
manifest. A successful mechanical run can write only current-source, current-candidate and effective-
mask PNGs plus an atomic report; `automatic_accept`, `candidate_selected`, quality, identity, temporal
stability and 16GiB safety all remain false.

Seven new tests cover safe paths and PNG hashes, traversal, content drift and post-preflight drift, missing-checkpoint early
exit without importing or constructing the model, a single fake six-frame inference, exception-path
owner release, and non-implicit confirmation/resource floors. Together with the existing adapter,
weight, structure, runtime and pipeline files, all 41 VRetouchEr-focused tests pass. The real default
preflight found the manifest and official checkpoint absent and reported 10,864MiB free VRAM below the
provisional floor. It left `real_model_loaded=false`, `checkpoint_deserialized=false` and
`inference_executed=false`. No node, workflow, video, H3, SAM, real VRetouchEr weight or pressure run was
introduced.

### Hash-bound single-window bundle builder

The validator no longer requires a caller to hand-author paths and SHA-256 values. The separate
`tools/build_skin_finish_vretoucher_single_window_manifest.py` imports neither Torch nor ComfyUI and
defaults to inspection only. The caller supplies one to six reviewed static source PNGs, exactly one
reviewed face box for each, one L-mode semantic-skin mask, an optional L-mode person mask and one
shot-local track key. The builder rejects duplicate frames, mixed geometry/channel modes, over-2.1MP
frames, non-intersecting boxes, non-L or geometry-mismatched masks and an existing output directory.

Only explicit `--write-bundle` publishes data. It stages files on the destination volume, copies the
input bytes without transcoding, verifies every hash during the copy, writes `manifest.json` and
`build_report.json`, then renames the complete staging directory into a previously absent destination.
It does not overwrite an old bundle. The files are not made operating-system read-only; the contract
is instead hash-bound, so later edits are rejected by the formal validator.

Five tests prove inspection-only zero writes, byte-exact publication, round-trip acceptance by the
formal validator, face-box/geometry/mask rejection and non-overwrite behavior. The full VRetouchEr
focused scope is now 46 tests. This closes reproducible input packaging only and does not detect a
face, identify a person, judge mask semantics, load the checkpoint, execute inference or validate
quality, identity, time consistency, memory or 16GiB safety.

### Prepared real-input VRetouchEr single window

A real model-free input bundle was prepared from existing reviewed evidence rather than generating
new H3, SAM or ParseNet material. It uses source frames 6, 12 and 18 from shot 0 at 960x704, the same
reviewed left-person track `0:0`, one reviewed face box per frame, the existing current-frame semantic-
skin mask and a source-bound left-person region mask. The three-frame causal context is intentionally
left-padded by the adapter at inference time; it is not wrapped to the end of the source video.

The non-overwriting bundle is
`artifacts/skin-finish-vretoucher-input-bundle-20260825-shot0-left`. The literal `manifest.json`
SHA-256 is `550D642AA2F1F49A0A4AB3409465B7ABE76C220BCF40BABFF621280A3143F381`; the validator's normalized
manifest SHA-256 is `EADD382D46C450060C294903445B34250B18279D5860F91D267461B262CCE878`. The build report returned
`HASH_BOUND_SINGLE_WINDOW_BUNDLE_WRITTEN_NOT_EXECUTED` and explicitly records no Torch or ComfyUI
import, no model load, no inference, no automatic acceptance and no candidate selection.

The formal validator accepted the real input paths, hashes, static-PNG geometry, causal track, boxes
and masks, then stopped with `ABSTAIN_CHECKPOINT_MISSING`. Port 8188 was not listening; 9,951MiB free
VRAM was also below the provisional 12,000MiB start floor. `real_model_loaded`,
`checkpoint_deserialized` and `inference_executed` all remained false. The preflight report SHA-256 is
`8E371B80CBD5EA1052A4B036DED48A67A9270EA585A0D32A1FB974A0FB9DC039`. This closes only reproducible
real-input preparation. The left-person mask is a deterministic source-region harness, not identity
evidence, and no quality, model compatibility, memory peak, 16GiB safety or usable-node conclusion is
available until the official checkpoint is supplied and passes the separate gates.

### Real VRetouchEr checkpoint identity and CPU load gate

The checkpoint manually obtained from the official Baidu Netdisk release was moved to
`ComfyUI/models/facerestore_models/VRetouchEr/gen_best.pth`. It is exactly 630,172,363 bytes. SHA-256
before and after the cross-volume move was
`F008748623325FDDB9F8DE6523A8F00B3712CF5EA5CA4ED695EA6C9F03E9B733`, so the source copy was removed
only after the destination hash matched.

The read-only weight audit used `torch.load(weights_only=True,map_location=cpu)` and found 411 tensors:
406 float32 and five int64. Every ordered key, shape, dtype and numel matched the pinned report; the
state-structure SHA-256 is
`7ABD9ECFF0B49178FBF2CC7AFECF171228AC4ACDDBBCFC7A5A0484020DE8CEEA`. Status was
`EXACT_STRUCTURE_AND_TRUSTED_HASH_PASS_MODEL_NOT_LOADED`.

A separate one-process CPU fp32 gate then constructed the pinned bundled network, assigned the real
state dict with strict loading and immediately closed its owner-scoped session without a forward. It
completed loading in 1.871 seconds and increased process RSS by about 694.89MiB. The close report was
`VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED`, with no external model reference and no global ComfyUI
unload. The process then exited, so its allocator retention did not persist. This proves real checkpoint
compatibility with the audited bridge, not numerical output quality or GPU safety.

The formal single-window validator subsequently accepted the checkpoint byte size, pinned SHA, all
three reviewed input frames, masks, boxes and the normalized manifest. Port 8188 was quiet. It stopped
with `ABSTAIN_PROVISIONAL_FREE_VRAM_FLOOR`: 11,136MiB was free against the independent 12,000MiB
provisional start floor. Consequently `real_model_loaded`, `checkpoint_deserialized` and
`inference_executed` remained false in the validator process. The preflight report is
`artifacts/skin-finish-vretoucher-single-window-preflight-checkpoint-pass-20260825/latest_preflight.json`,
SHA-256 `202C816F078AC0F7F9D968E2D66356EDC73EF87D829AF3ED5B48995219F74A32`. No floor was lowered and no
CUDA pressure run was attempted.

### First numerical VRetouchEr window: FP16 rejected, BF16 mechanical pass

After free VRAM reached 15,113MiB, the validator ran one process at a time with all CPU thread pools
limited to one. The first FP16 attempt exposed a real compatibility issue rather than OOM: the upstream
FP32-oriented graph creates runtime Float tensors while the loaded parameters were Half, producing
`expected scalar type Float but found Half`. No candidate was accepted. The runtime now enters CUDA
autocast only for fp16/bfloat16 parameters, leaves CPU/fp32 unchanged, captures the full failure
traceback and rechecks the owner weak reference after exception frames are detached. Twenty-three
focused runtime/validator tests, Ruff and py_compile pass.

The corrected FP16 path completed the model forward but returned non-finite proposal values. The new
post-forward finite gate rejected them; no output candidate was written. This is evidence that FP16 is
not numerically valid for this checkpoint/runtime, not a reason to sanitize NaNs or weaken the gate.

One scientifically motivated BF16 retry then passed. It used CUDA autocast and returned a finite
`[3,512,512]` current-frame proposal, six `[1,1,256,256]` masks and `[5,1,64,64,2]` flow. The validator
wrote only source, candidate and effective-mask PNGs under
`artifacts/skin-finish-vretoucher-single-window-real-20260825-shot0-left-v3-bf16/run-eadd382d46c4`.
Automatic acceptance and candidate selection remained false. The owner report was
`VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED`, no external model reference remained, no global ComfyUI
model was unloaded, and free VRAM returned to 14,958MiB. Validation report SHA-256 is
`7D39ADE8C22EFCD5C708FAEF8434FEBCD5DB6B7059803C0975708FAAC70FA9EE`.

Every RGB value outside the effective mask is byte-exact. The mask contains 8,151 of 675,840 pixels
(1.2061% of the frame); 10.8944% of mask pixels changed. Inside-mask mean absolute change is only
0.08052/255 per channel, p95 is 1 and maximum is 5; whole-frame mean is 0.000971/255. The source face
was already clean, so this result establishes finite BF16 mechanics and bounded paste-back only. It
does not establish perceptible skin improvement, identity quality, temporal stability, multi-person
isolation, a usable node or 16GiB repeated-run safety. A later quality gate must use one predeclared
representative with visible imperfections; it must not tune this clean frame to manufacture a delta.

### One predeclared oily-skin VRetouchEr window: still no quality promotion

The follow-up used the existing 960x544, 124-frame oily-LoRA source rather than generating new H3
material. The previously reviewed contact sheet identified frame 66 as the strongest of its six sampled
forehead/nose highlight proxies (0.141). Frames 61 through 66 are one continuous detected track in shot
0. A single CPU-thread process decoded those six frames and ran the pinned ParseNet only on current frame
66. Its result exactly matched the earlier full-run record: `READY`, 91,708 selected pixels and skin-area
fraction 0.21754557. The semantic preview excludes hair, eyes, brows, nose, mouth and background.

The non-overwriting bundle is
`artifacts/skin-finish-vretoucher-input-bundle-20260825-oily-frame66`. Its manifest-file SHA-256 is
`D1788E175A3D665BFFB5C237DDDE5B516D377EB4AA7A17D04EE75736E1D8877F`; normalized-manifest SHA-256 is
`1BF5886FDB0E08FE3CAF23C32C83D91B8D9CC0FBD6B579434001D3003192BAA4`. The formal preflight saw
15,131MiB free VRAM, quiet port 8188 and the exact official checkpoint identity, so one BF16 process ran.
No FP16 retry, parameter grid, video generation, repeated run or concurrent process was used.

The result is under
`artifacts/skin-finish-vretoucher-single-window-real-20260825-oily-frame66-bf16/run-1bf5886fdb0e`.
Validation report SHA-256 is
`4AE2F89908961704BA045BA74BAC8E2D9FCBAC1346165247F8C4AE462FE0A688`. The model returned finite output,
the owner closed as `VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED`, no external reference remained, no
global ComfyUI model was unloaded, and candidate selection stayed false. All values outside the
effective mask are byte-exact source.

The effective mask contains 113,885 pixels. Inside it, mean absolute RGB change is 0.94667/255 per
channel, p95 is 3, p99 is 10 and maximum is 54; 75.6026% of mask pixels change. This is larger than the
clean-window delta, but the highlight evidence remains weak: masked luminance p99 changes only from
0.92836 to 0.92277, while the fraction above 0.85 changes from 20.2555% to 20.2415%. The labelled
source/candidate/difference/mask sheet remains visually subtle and does not establish preferred skin
quality or oil control. Therefore the 3-5 second gate remains closed and no node, workflow, README claim
or default is added.

### Official-path parity and controlled nose-mask ablation

The pinned upstream inference and training files were inspected without running another model. Official
test preprocessing opens RGB through PIL, applies `ToTensor`, resizes to 512 and normalizes all channels
with mean/std 0.5 before stacking six chronological frames. The saved newest-frame output is interpreted
in `[-1,1]`. The T8 bridge follows the same RGB channel order, range, temporal order and newest-frame
selection. A deterministic 512-square regression now compares the actual six bridge inputs with the
literal RGB-CHW `* 2 - 1` reference, including an asymmetric blue-channel sentinel; the exact focused
test passes (`1 passed in 0.12s`). This rules out the suspected BGR/RGB, normalization and frame-order
mismatches for that controlled input.

The official training route pairs source faces with FFHQR retouched targets and describes facial
imperfections/blemishes. It does not define generated specular-oil suppression as a task. This is the
most plausible task-domain explanation for the weak H3 result, not proof that every suitable real-face
input would fail.

The first oily-window evaluation also had a separate mask confound: the production ParseNet policy
protects the complete nose class, so the prominent nose highlight received exactly zero paste-back.
One CPU-only ParseNet audit built a research mask that kept normal skin at weight 1.0 and added only the
nose class at weight 0.35. It increased the nonzero mask from 113,885 to 127,943 pixels and covered
28,176 of the frame's 31,524 high-luma pixels, while still excluding eyes, brows, mouth, hair and
background. A hash-bound bundle was written at
`artifacts/skin-finish-vretoucher-input-bundle-20260825-oily-frame66-nose035`; manifest-file SHA-256 is
`3BE6DA6EC4A1A075D1C0FD564DCA6BAB2406CC72FC199C26ED5E9D2035CB2BFA` and normalized-manifest SHA-256 is
`3303AC7392C0EE3DC717EFB7FC2A2D8C30BAB00EF7C4EE32340ACF52BD64361E`.

Exactly one further serial BF16 window reused the same six source frames, checkpoint and model settings;
the only treatment change was that controlled mask. Its report is under
`artifacts/skin-finish-vretoucher-single-window-real-20260825-oily-frame66-nose035-bf16/run-3303ac7392c0`,
SHA-256 `E820AA747E36027556F4C73568D1F661E24F90B35A1DD876809C1BE85BDA65EA`.
The candidate remained finite, mask exterior stayed exact, selection stayed false and the owner again
closed as `VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED` with no external reference or global ComfyUI
unload. Inside the newly admitted nose class, mean absolute RGB change was only 0.15911/255, p95 1,
p99 2 and maximum 6. Nose luminance p99 moved from 0.97652 to 0.97152, while the fraction above 0.85
slightly increased from 37.4086% to 37.4167%. The three-way source/skin-only/skin-plus-nose sheet
SHA-256 is `D89AED4752DAEC04A871CF940CE7D1FE13E61061A5BEE05EE5961A29D61EA4CD`.

The ablation proves that the old safety mask excluded part of the intended target, but it still does
not establish visible oil control after that target is admitted. Increasing nose weight, running a
parameter grid or escalating to a 3-5 second video is therefore not scientifically justified. The
VRetouchEr route remains unregistered research and no existing node, workflow, README or default changed.

## SLA runtime compatibility hotfix (2026-08-26)

Two live ComfyUI H3 core revisions were previously rejected solely because their `PackedLayout`
source hashes were not in a fixed whitelist. The SLA adapter now retains source hashes only as
diagnostics and instead verifies required signatures, exact `patchify_video` ordering and an executable
FL2VA packed-layout contract. SLA LoRA admission likewise uses complete A/B pair structure plus full
mapping/application to the loaded H3 base, rather than one filename, byte size or whole-file SHA.
Native-flow 8 NFE is accepted and audited as an experimental schedule; the published LightX2V 4 NFE,
video shift 6 and audio shift 3 route remains the official reference.

After a real ComfyUI restart, the first live retry passed the old hash gate and exposed a separate
process-global compatibility fault from an installed old `ComfyUI-PainterNodes`: its marked
`PackedLayout` wrapper forwarded the removed `frame_count` keyword to the current native constructor.
A focused regression reproduced the exact `TypeError`. SLA now reuses the existing T8 Hybrid executable
compatibility probe, which removes that wrapper only when its enclosed native constructor independently
passes the current keyframe-plus-reference ordering contract. Unknown or unverifiable global wrappers
remain fail-closed. This changes no node ID, widget order, workflow graph or sampling formula.

### SLA persistent-tail-collapse quality investigation (local, human-rejected)

A later full-timeline review invalidated the earlier implication that successful sparse-kernel calls
were sufficient quality evidence. In a 736x416x124 FL2VA run, approximately the first second appeared
normal and the remaining frames stayed collapsed. The earlier fixed-85-percent route and the later
all-dense fallback with the same SLA LoRA were both rejected on full viewing. The dense rerun therefore
rules out hidden sparse-kernel execution as the sole cause, but it does not establish a normal ordinary-
Turbo control and cannot isolate SLA sparsity by itself. Container decoding and a single corrupt frame
were ruled out mechanically. The failed packed sequence was approximately 12,587 tokens; the published
1344x768x362 reference geometry is approximately 111,590 tokens.

The existing saved `apply_lightx2v_sla` value currently compiles an experimental `auto_safe_v1` plan. Below
50,000 packed tokens it performs dense attention on every model forward while keeping the same LoRA.
At or above the boundary it keeps the first and final denoising forwards dense and uses SLA sparse
attention only for the middle forwards. Sparse middle forwards preserve the learned video top-k and
add all packed text, two visual-condition and joint-audio key blocks before the target video. This does
not freeze audio or alter query rows; it can increase the actual retained ratio above 15 percent, so
the requested 85-percent sparsity is not a measured speedup claim. The exact released all-forward route
remains available only as `apply_lightx2v_sla_upstream_exact_exp`.

Runtime Audit derives dense/sparse ownership per forward, rejects any unplanned call, and samples actual
retention at layers 0, 25 and 49. Layer 0 additionally reports packed-segment and target-video quadrant
coverage. Missing legacy planning metadata fails toward dense quality instead of silently restoring
the unsafe all-sparse route. The internal patch schema is version 2; public node IDs, inputs, widget
order and both dated workflow graphs remain unchanged. Focused CPU tests validate short all-dense,
long dense/sparse/sparse/dense, exact all-sparse, KJ owner dispatch, prefix preservation and workflow
widget compatibility.

One guarded real rerun then reused the failed case's exact embedded API graph while changing only the
output prefix: FL2VA, 736x416x124, seed `2608224201`, four native-flow NFEs, video/audio shifts 6/3 and
the public `apply_lightx2v_sla` mode. It ran once in an isolated 8197 process with no parallel or stress
arm and stopped the server immediately afterward. The downstream output could only execute after the
Runtime Audit node; therefore a saved clip proves that its NFE, 50-block-per-forward, planned owner and
zero-kernel-failure gates did not throw. The initial probe utility failed only after generation because
the old History response did not retain the report string; the utility now captures the v3 WebSocket
`executed` payload and persists the phase before post-processing.

The mechanically valid container has 124 H.264 frames at 736x416/24fps and AAC stereo at 32kHz. Strict video,
audio and combined decodes passed. Its SHA-256 is
`DD3C6524C2FB60FECAB3462220C636B6E57EED4CFAD27CD0BE4E6DB0C1B492A2`; audio measured approximately
-10.0 LUFS integrated with -1.1dBFS true peak. CPU analysis found no black/white frames, frozen pairs,
container corruption or hidden sparse calls. The user then watched the complete clip and explicitly
rejected the result because only the first second appeared normal. That verdict invalidates the prior
sampled-frame implication that dense fallback repaired quality.

The test fixture itself also prevents a clean causal claim: its first frame is a street-level close
portrait while its final hard FL2VA frame is an elevated wide shot where the red subject is only a
small dot. The output starts the required scale/camera transition after roughly one second, so subject
shrinkage and SLA degradation cannot be separated on this fixture. It additionally uses an INT8
ConvRot base, 736x416 and 124 frames, whereas the upstream public configuration validates the SLA LoRA
together with 85-percent dynamic sparse attention, four model evaluations, 6/3 shifts, 768p FL2VA,
an FP8 DiT/VAE and 362 frames. Keeping the SLA-trained LoRA while replacing its sparse attention with
dense attention is not an upstream-validated ordinary-Turbo path.

The 50K boundary is therefore retained only as unreleased diagnostic code, not a quality guarantee.
The next single A/B must use matched-scale first/last images and compare ordinary FL2V four-step Turbo
LoRA plus dense/Sage against SLA LoRA plus the upstream-exact SLA route. If the SLA arm fails again on
the local INT8 short-video profile, that profile must ABSTAIN/fall back to an explicitly supplied
ordinary Turbo LoRA; it must not silently run the SLA LoRA dense or a four-step bare base. Evidence is
under `artifacts/sla-auto-safe-tail-validation-20260826/20260826-200927`.

Follow-up evidence separated the failed fixture from the SLA implementation. The Profile Router
transition rerun used the ordinary corrected-Alpha8 Turbo8 profile at 8 NFE and 12/3; its Runtime Audit
recorded zero main, sparse and dense-control SLA calls on all eight forwards. Therefore the user's
full-timeline rejection of that clip cannot be attributed to SLA attention. A same-frame control using
the explicit 4-NFE/6/3 upstream-exact experimental route completed 50 sparse calls on each of four
forwards and remained visually coherent across its 124-frame contact sheet. This is only an isolation
control: it still uses the local INT8 base and short low-resolution geometry, so it does not establish
upstream parity or consumer readiness.

The two source anchors were also measured after resizing to one 736x416 canvas. The close portrait and
aerial street view had a pixel correlation of `0.032`, edge IoU of `0.0828`, only three ORB ratio-test
matches and no solvable homography. In contrast, the same-frame control scores 1.0 correlation/edge IoU
with 644 ORB matches. These metrics do not define a universal creative limit, but they independently
confirm that the failed pair is not a modest same-scene camera transform. The original user-supplied
MP4 further embeds `length=22` and probes as 22 frames / 0.9167 seconds despite `124f` in its filename.

The video branch of T8 dual-clock Euler was compared algebraically with LightX2V's `training_euler`:
ComfyUI CONST returns `denoised = x - sigma * model_output`; H3 returns the negated native velocity;
`to_d` therefore reconstructs that native velocity and the update is exactly
`x + (sigma - sigma_next) * velocity`. No missing fifth model evaluation exists: LightX2V stores five
sigma grid points for four updates.

A low-load CUDA random-tensor probe then held the learned block map fixed and compared attention
kernels. At 15-percent retained keys, the high-precision Triton sparse kernel had approximately
`0.000302` RMSE against an independent selected-block FP32 reference; the installed quantized
`spas-sage-attn` Sage2 path measured approximately `0.00517` RMSE, about 17 times larger, although its
single-layer cosine similarity remained `0.99933`. This is a real candidate for a future precision
backend A/B across 50 layers and multiple steps, not proof that it caused the rejected transition: the
rejected Profile Router transition did not invoke this kernel at all.

The later same-scale blind export did not promote SLA either. The reviewer selected `unsure`, so the
formal analyzer correctly excluded the pair, but every retained preference field selected the ordinary
Turbo8 control and the SLA arm was marked as the blocking failure. The result therefore establishes no
SLA non-inferiority and denies automatic enablement; it is not valid evidence that the INT8 SLA route is
usable.

A code-path audit then found a previously uncontrolled quantization variable. The SLA loader used
`ModelPatcher.add_patches`; for a ComfyUI `QuantizedTensor`, model patching dequantizes, adds the LoRA and
calls `requantize_from_float`. LightX2V's published FP8 SLA configuration and its ordinary INT8 and INT8
ConvRot configurations instead all declare `lora_dynamic_apply=true`. The project now has an append-only,
non-default `sla_4step_int8_bypass_exp` profile that requires the observed INT8 Tensorwise ConvRot base
and uses ComfyUI model-only bypass injection, leaving the quantized base weight untouched. It retains the
same four model evaluations, 6/3 shifts and upstream-exact 85-percent sparse attention; only LoRA
application changes. Thirty-eight focused CPU tests pass. No real render has run yet because the observed
free VRAM was about 11.7GiB, below the existing 14.5GiB start gate, so this is a testable candidate rather
than a quality fix.

A header-only audit avoided allocating either model. The local base's block-0 QKV quantization record is
`int8_tensorwise` with `convrot=true` and group size 256. The SLA LoRA and the already executed corrected
Alpha8 Turbo bypass LoRA each contain 624 A/B/alpha tensors and exactly the same 208 target-module names.
This removes target-set mismatch as a likely cause of bypass-hook failure, but it does not replace live
208-hook injection, sampling, strict media decode or human review.

The live loader gate was then closed in a fresh CPU-only process. It loaded the real 31.70GiB INT8
ConvRot checkpoint and 1.82GiB SLA LoRA without CLIP, VAE, sampling or decode. In 2.625 seconds the
profile verified all 200 main INT8 ConvRot targets, eight unquantized token-refiner targets, 208/208 LoRA
mappings and 208/208 bypass hooks. The resulting ModelPatcher had no standard weight patches, contained
the `bypass_lora` injection and reported `base_weight_mutation=false`. Observed RSS deltas were about
4.72GiB for the dynamically loaded base and 5.58MiB for bypass binding. The independent process then
exited, releasing its mappings. Evidence is
`artifacts/sla-int8-bypass-loader-probe-20260826/report.json`. This proves loader mechanics only; the
full-duration render, strict media audit and human quality gate remain pending.

## Cross-feature runtime compatibility audit (2026-08-26)

The SLA failure was not isolated. A tracked-source audit found five other execution paths that used an
exact Python source hash as a compatibility decision: Activation Chunk, Qwen Prefix Cache, Prompt
Relay, Enhance-A-Video, its external T8 BlockCache composition, and Multi-Keyframe admission. The
current ComfyUI core at `b78cec879b9460d5cb25228a83a942fb78d2cd24` provided direct evidence of the
problem: `MiniMaxH3Model._forward` had a new source hash (`14bdf...`) while the executable H3 layout,
patch ordering and required signatures remained compatible, so Activation Chunk rejected the current
core before execution.

Six focused regressions first reproduced rejection after source-only-equivalent changes. The repaired
paths retain every source fingerprint in reports, but no longer treat it as a whitelist. Admission now
uses the minimum feature-specific executable contract:

- H3 core users verify required method signatures, native video patch ordering and live PackedLayout
  target-row placement.
- Qwen Prefix Cache additionally retains its exact token-prefix/suffix binding, native model identity
  and FixedKV numerical regression.
- Prompt Relay retains its MODEL/CONDITIONING binding and single attention-owner rules.
- EAV plus BlockCache requires exactly one cache outer wrapper, one diffusion wrapper, a CPU 50-block
  prototype and only block 0/49 replacements; implementation hashes are diagnostic.
- Multi-Keyframe probes the actual middle-frame position and accepts a semantically compatible global
  layout wrapper. It bypasses only the specifically marked obsolete Painter wrapper whose enclosed
  native constructor independently passes the same probe. Its copied per-keyframe forward remains
  guarded by required signature and structural markers.

A separate schema audit inspected all 211 registered nodes and found five optional inputs that were not
actually omittable at Python execution time. Speech Studio now accepts its optional `speech_guard`
dependency, and Context IR Provider/Compiler optional dialogue/transcript fields have defaults. A
permanent registration test now fails if any future optional input is absent from `execute` or lacks a
default. Modern frontend workflow tests were also corrected to inspect linked sockets by name instead
of historical widget-inclusive slot numbers.

The maintenance-tool audit found a second compatibility class that did not affect runtime sampling but
could make valid modern workflows appear stale or non-portable. ClipProj and Quick Start builders had
been locking raw JSON bytes and reconstructing widgets from the pre-ComfyUI-0.4 input-slot layout.
They now hash canonical JSON semantics, decode registered widget schemas, and rebuild deterministically.
Six Quick Start subgraphs remain semantically unchanged. Quick Face Repair additionally replaces four
machine-specific installed-model inventories with portable `COMBO` selectors; its public UUID, input and
output names, node types and execution graph remain unchanged. The ClipProj 4B workflow only updates its
recorded canonical source identity. These are maintenance metadata/portability changes, not sampler or
conditioning changes.

The follow-up audit removed the remaining user-model allowlists. ParseNet, SFace, VRetouchEr, Hybrid
base/overlay, SLA/PDD/Turbo LoRA and the learned latent-upscaler now report reference filenames, sizes,
hashes and roles only as diagnostics. Their actual framework loader, safe deserializer or strict
state-dict assignment is authoritative, so an incompatible model may fail naturally during execution.
Checkpoint/resume, Hybrid artifact/sidecar, manifests, accepted media and runtime state hashes still
prevent stale, cross-source or corrupted user data. Native H3 batch/channel/layout, 17n+5 frame grid,
24fps reference semantics, 32-pixel alignment and explicitly scoped scientific NFE profiles also remain
enforced. The 1920x1088 area is retained only as a warning/reference threshold and does not block a
larger canvas. These are model or integrity contracts, not source-version whitelists.

The current suite collects 1557 tests. One complete serial run finished with 1554 passes and one
Windows FFmpeg child-process failure; that exact Long Video fixture passed immediately in isolation.
After adding two final Hybrid identity regressions, the complete 46-test Hybrid scope passed. A second
serial full-suite attempt reached 69 percent without a Python assertion failure before Windows raised a
native access violation in the Skin Finish video-stream fixture; that exact fixture also passed in
isolation. Repeating the native-media suite again would be a stress test rather than useful evidence, so
it was not repeated. All 147 frontend workflow JSON files parse with required ComfyUI nodes/links
metadata. This audit is CPU/contract evidence only and makes no new visual-quality, audio, speed, VRAM
or general 16GiB claim.

## In-node Long Video with Prompt Relay and Enhance-A-Video (2026-08-27)

An append-only output node now projects one global Prompt Relay plan across all segment timelines and
creates a fresh Enhance-A-Video runtime for every segment. Long Video remains the only packed-layout
and extra-conditioning owner; one combined wrapper routes Prompt Relay before applying FETA to target
video rows. Existing Long Video, Prompt Relay and EAV nodes and workflows are unchanged.

The full project passes 1673 tests. One isolated serial 256x256, six-second Stock20 run used a
124-frame render window and 22-frame continuation context. Segment 0 delivered frames 0-124; segment 1
rendered 102-226 and delivered only 124-144. Both segments completed 20 model forwards, shared the
same global Relay-plan hash, used distinct projected-plan hashes and wrote verified effect-audit
sidecars before acceptance. The final 144-frame 24fps H.264 and finite stereo 32kHz AAC file strictly
decodes and has SHA-256
`691a32f9b9891a23435fbe5f5ddb3ff5f0ed2978e84b3a5929513e8e71ce4f55`. Sampled frames 118-129 show
no black frame, corruption or scene reset. Peak device use was about 14260MiB and minimum free VRAM
about 1850MiB. This low-resolution mechanical result does not establish visual improvement, audio
non-regression, arbitrary-duration stability or universal 16GiB safety.

## Canvas area warning-only hotfix (2026-08-27)

Version 1.52.2 removes the former 2,088,960-pixel execution gate from Conditioning, Source AV,
Long Video (including both in-node loops), Multi-Keyframe, Still Image, SPEED, Prompt Relay resource
estimation and Environment Audit. Positive dimensions and 32-pixel alignment remain mandatory. The
legacy Conditioning opt-in stays at the schema tail so old workflows keep their input ordering, but
its saved true/false value no longer changes admission. The reported 2,396,160-pixel case is covered by
the equivalent valid 32-aligned 2080x1152 contract. A serial focused CPU suite passed 217 tests; no
model generation, visual-quality test, memory-safety claim or stress test was performed.

## Prompt Relay public-tokenizer compatibility hotfix (2026-08-28)

Issue #13 reported that Prompt Relay rejected a CLIP value before sampling because its internal
`tokenizer.qwen3vl_32b.tokenizer` path was not visible. The reported ComfyUI commit and the locally
validated core contain the same native MiniMax H3 tokenizer source, so the fix does not assume that an
older core uses different tokenization and does not remove exact prompt-token binding.

The original direct native path is unchanged. If a CLIP proxy hides the internal object, version 1.52.3
uses its public `tokenize()` output, requires exactly one `qwen3vl_32b` batch, constructs ComfyUI's
bundled native `MiniMaxH3Tokenizer`, and compares every prompt token ID. Only an exact match may reuse
the native byte decoder for local-character-span binding. Missing public tokenization and any token
length or ID mismatch remain fail-closed with an actionable `Load CLIP / type=minimax` diagnostic, so
an arbitrary tokenizer cannot silently shift event timing.

The complete Prompt Relay scope passed 87 tests. Registration and frontend workflow compatibility
passed another 48 tests. Focused Ruff and `py_compile` checks passed. This is a CPU/token-contract
compatibility repair: no node ID, widget order, workflow graph, model patch, sampling formula, GPU
generation, visual/audio claim or memory-safety claim changed.

## Official MiniMax H3 core PR compatibility (2026-08-28)

Five open ComfyUI pull requests were audited against the installed H3 core. The plugin now appends an
H3 AV latent builder, a clone-scoped standard attention-hook bridge, and an instance-scoped forward
optimization that reduces the two visible sigma host synchronizations to one while caching text-token
tags only on the current payload. Existing PDD setup now prefers the official shape-changing
`set_weight`/`set_bias` path when an executable semantic probe passes and otherwise retains the tested
T8 fallback. Admission uses callable signatures, shapes and behavior; source hashes, ComfyUI versions,
model hashes and file sizes are diagnostic only.

The tiled-VAE global-coordinate proposal was not promoted. A serial decode used the installed fp16 H3
video VAE and one real final H3 latent at 736x416. Both outputs were finite, but the proposed global
coordinates introduced visibly stronger regular grid and stripe artifacts. Mean absolute pixel
difference from the local-coordinate control was `0.012596`, maximum difference was `0.341046`, and
the compatibility clone matched a direct core-style patch exactly (`max difference 0`). The registered
node therefore defaults to `report_only`, returns the original VAE object unchanged and exposes the
global-coordinate path only as an explicitly named experimental reproduction mode.

The four new node IDs are appended at positions 222-225; no prior node ID, widget order or saved
workflow was changed. The frontend example is under `examples/workflows/20-core-compatibility`.

## Classic-paper Advanced/EXP contract batch (2026-08-28)

Six creator-facing routes were appended at registration positions 236-243: RAFT motion audit and
reviewed-mask propagation, bbox-keyframe trajectory planning/rendering for H3 Fun Control,
RealBasicVSR temporal restoration, FreeNoise video-noise rescheduling for both in-node long-video
runners, a dual-clock AYS schedule contract, and CADS visual-reference annealing. The previous 236
node IDs and their input/widget order remain unchanged.

The implementation intentionally distinguishes faithful components from H3 adaptations. RAFT uses a
real torchvision RAFT backend. The RealBasicVSR architecture strictly loaded the local official-format
checkpoint with no missing or unexpected tensors and completed one two-frame 64x64 low-load CUDA
inference. FreeNoise implements deterministic temporal noise reuse/permutation but not the paper's
single-long-latent sliding temporal attention. AYS does not ship an SD/SDXL/SVD table as an H3 optimum;
its default is the original native-flow schedule and manual mode only validates an externally calibrated
base knot list. CADS follows the published condition-annealing formula but remains uncalibrated for H3
identity and endpoint adherence. Trajectory control creates explicit control frames for the existing Fun
Control path and does not copy TrailBlazer's U-Net attention mechanism.

The initial synthetic RAFT smoke was followed by four serial real-input effect reviews without a stress
matrix. RAFT Small processed 48 frames from a real 1152x640 H3 bird clip at a 640-pixel analysis side.
Five reviewed masks at frames 0/12/24/36/47 yielded nonzero propagated masks on 48/48 frames, mean
forward/backward flow confidence `0.826199` and mean final propagation confidence `0.567456`. The
side-by-side video tracks the subject usefully, but visible foreground holes and occasional floor/tail
inclusion confirm that RAFT is transport rather than identity segmentation; new anchors remain mandatory
after occlusion, re-entry or cuts.

The two-object trajectory renderer produced a strictly decodable 1152x640x124 control video and nonzero
mask on all 124 frames, with smoothstep paths crossing as authored. This proves the creator-facing plan and
control-video effect, not that H3 Fun Control follows it in a useful-resolution final render.

RealBasicVSR strictly loaded the local official-format checkpoint and processed 32 real H3 frames at
416x232. Strength `0.65` increased mean Laplacian variance from `273.0993` to `1059.5230` but visibly
created over-sharpening, bright edge halos and a plastic look. Strength `0.30` reached `531.8011` with mean
absolute change `0.008042`, p95 `0.031373` and materially milder ringing. The code, node and workflow
default were therefore changed from `0.65` to `0.30`; AUDIO remained the exact same Python object.

The natural H3 long-video boundary correctly abstained with mean RGB drift `0.001263`. A controlled
real-frame probe injected a fixed five-percent display-domain gain drift after a reviewed boundary. The
bounded candidate corrected 16 frames, reduced seam MAD from `0.021009` to `0.013025`, limited maximum
per-pixel change to `0.029761` and strictly decoded. The run exposed a real atomicity defect: a rejected
boundary frame could previously leave later transition frames modified. The implementation now stages the
whole transition and commits it only when the boundary frame is safe and the seam error improves; otherwise
the complete transition returns exact source.

No new H3 generation, multi-seed quality grid, stress test or general 16GiB claim was run for these four
reviews. Model files remain outside Git and are never downloaded at runtime; filename, byte size and
SHA-256 are diagnostic rather than execution allowlists. Each node releases only resources it owns and
never calls global ComfyUI model unload.

After correcting two serialized workflow output-link indices, all 166 frontend workflows parsed and
passed the compatibility suite. Six new project workflows were copied byte-for-byte to the ComfyUI user
menu, with matching SHA-256 values and a final mirror count of 166. Project regression was split to avoid
native-media concurrency pressure: 1,752 non-delivery tests and all 28 Long Video Delivery tests passed
serially (1,780 total). The two FFmpeg cases that had failed during an earlier concurrent full run also
passed separately. Ruff passed for all non-vendored source, `compileall`, JSON parsing and `git diff
--check` passed; the pinned VRetouchEr upstream vendor retains its pre-existing lint findings unchanged.

FreeInit and PAG remain rejected implementation gates, not missing deliverables. FreeInit needs a
validated MiniMax H3 flow-forward-noise and joint AV reinitialization contract. PAG needs one composer
that can isolate target-video rows in H3 packed self-attention, report both main and perturbed forwards,
and demonstrate acceptable audio behavior without conflicting with SLA, Prompt Relay, Enhance-A-Video
or STG ownership. Until those conditions are met, no same-name approximation is registered.

## Recent creator-node effect audit (2026-08-28)

The six-node delivery above was followed by bounded, serial, same-seed H3 effect checks. This addendum
separates **an observable effect** from **a general quality claim**; no stress matrix or multi-seed sweep
was run.

- **FreeNoise:** one 1152x640x144 two-segment Turbo4 long-video pair completed with the target-video
  noise rescheduled in both segments and target-audio noise unchanged. The bird path visibly changed.
  This proves temporal video-noise routing, not improved continuity.
- **CADS:** one 736x416x22 Stock20 disabled/apply pair completed. Reference annealing changed blink
  timing and local eye/detail structure, but the difference was subtle and no arm was consistently
  superior. The feature remains experimental and identity/reference adherence must be reviewed.
- **Trajectory + Fun Control:** one 736x416x22 Turbo8 baseline/controlled pair completed. The controlled
  arm followed the requested upward-middle/downward-right trajectory more strongly. The first attempt
  also exposed an actionable incompatibility: the local pruned/basis control model has an 8-value AdaLN
  input while the full base has 2688. The Apply node now compares live tensor dimensions before VAE
  work and explains the required pairing; it does not gate filenames, hashes or sizes. With
  `end_percent=0.85`, the last 15 percent is intentionally released, so endpoint locking requires 1.0.
- **Prompt Relay + Enhance-A-Video:** one 1152x640x22 Stock20 Relay-only/apply pair rendered the authored
  red-left, green-upper-center and blue-right windows in order. EAV raised mean Laplacian variance from
  `464.8855` to `596.7232` and slightly raised temporal change from `2.8356` to `2.9989`, proving a real
  visual effect. It was not audio-neutral: waveform correlation was `0.2080` and RMS became `2.1605x`
  the Relay-only arm. FETA directly scales target-video attention rows only, but later joint-AV layers
  can propagate that change into audio; important dialogue still requires A/B listening.
- **Dual-clock AYS schedule contract:** one 736x416x22 Turbo8 native/manual-knot pair completed. The
  uncalibrated manual sensitivity array raised mean Laplacian variance from `75.1691` to `107.0719`,
  kept temporal change similar (`13.4756` versus `13.3232`), and reduced audio RMS to `0.6976x` native
  with waveform correlation `0.9691`. This proves that imported knots affect the real dual-clock run;
  it is not a KLUB-optimized MiniMax H3 AYS schedule and is not shipped as a quality preset.

Ignored machine-readable reports and local comparison media are under
`artifacts/recent-feature-effect-audit-20260828/`. They are intentionally not packaged in Git. The
source-controlled workflow notes contain the model-pairing and audio-review boundaries established by
these checks.

## FlashVSR v1.1 low-load validation (2026-08-30)

The official `JunhaoZhuang/FlashVSR-v1.1` model folder was loaded in tiny, tiny-long and full modes
with the local ComfyUI Python, Torch 2.10/CUDA 13 runtime and a compatible `spas_sage_attn` wheel.
Only tiny mode performed real generation so the structural checks did not become a stress run. A direct CUDA probe exercised
`block_sparse_sage2_attn_cuda` with the generated split-K LCSA mask and returned a finite tensor with
the expected shape and dtype.

Three serial 2x restoration routes were then run at 192x128 input without concurrency or stress tests:

- `quality_locked`: 21 frames, full-frame/resident, fixed `2.0/3.0/11` budget;
- `balanced_dynamic_exp`: 45 frames, full-frame/resident, with an actual interior low-motion chunk
  reduced to `1.7/2.0/9` while boundary guards remained fixed;
- `memory_safe`: 21 frames, two same-seed feathered tiles and staged offload.

All final outputs contained 384x256 H.264 video and source audio, and strict FFmpeg joint decode
completed without errors. Core-node audio tests proved exact Python object identity; the validation
mux encoded source audio to AAC, so the files are not presented as PCM-bit-exact evidence. A same-seed
45-frame fixed/dynamic diagnostic measured SSIM `0.984505` and node-core elapsed time of `5.834` versus
`5.439` seconds. This is one tiny fixture only and is not a general speed, quality, VRAM or hardware
claim. The square-input diagnostic was discarded after the same distortion reproduced in the fixed
control, proving it was a validation resize error rather than a dynamic-budget effect.

## Subject-safe RGB composite v8 packaging (2026-09-01)

One append-only Advanced EXP node and one dated ComfyUI frontend workflow package the previously
reviewed `D0 + alpha * (T2 - D0)` post-process. The implementation requires matching D0/T2 frame
geometry and a reviewed per-frame alpha, preserves zero-alpha RGB exactly from D0, supports an
optional explicit protection mask, returns the same D0 audio object, and falls back to the complete
source on contract failure. It performs no automatic subject, face, text, camera or quality decision
and does not gate models by filename, hash, file size or pixel area.

Focused CPU tests covered exact ownership, protection, fallback, explicit mask broadcasting, audio
identity and append-only registration. Registration and all dated frontend-workflow compatibility
checks passed with 268 node schemas and 183 workflows. No H3 model or GPU render was run for this
packaging change; the quality boundary remains the two previously reviewed single-person samples,
both non-inferior ties without blockers.

## 2026-09-01 — MV Vocal Lock V2 clear-speech synchronization validation

The released v1.62.0 MV route was reopened because its only real sample contained music without
assessable speech or singing. Visual review of that file could not close the lip-sync gate. V2 is
append-only and leaves all three v1 nodes and their workflow unchanged.

V2 adds a required, timeline-aligned `vocal_lock_audio` input alongside `full_song`. The isolated
track owns H3 `lock_source` conditioning and candidate-preview audio; `full_song` owns only the one
final delivery mux. Both media signatures are included in the V2 resume contract. The V2 prompt
compiler follows the official Ref2VA section order (`subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`), defines
`<Subject 1>` from `<Picture 1>`, binds isolated `<Audio 1>` to `<Subject 1> (S1)`, and emits
`fully_preserved` / `fully_copy`. Vocal-active shots force a front or three-quarter medium close-up
with a fully visible, unobstructed mouth. Exact words appear only when explicitly supplied.

One bounded serial local run used
`h3_twopass_voice_5683_5p152s.flac`, whose exact English transcript was supplied to the compiler,
and `03_Blonde清晰人物参考图.png`. The graph used local Ref2VA INT8 plus the local Turbo4 LoRA,
eight sampling steps, `dual_clock_euler/native_flow`, shifts 6/3, seed 5683, 736x416 and 124 frames.
Exactly one local ComfyUI validation prompt was submitted by the external test harness; the product
nodes themselves called neither HTTP `/prompt` nor any remote service. It completed in 149.031
seconds with one accepted scene and state policy
`isolated_vocal_lock_conditioning_full_song_muxed_once`.

The final file is
`F:\AI-T8-video-onekey\ComfyUI\output\minimax_h3_t8_long_video\mv_vocal_lock_v2_clear_speech_5p152_20260901_r1\assembled\MiniMaxH3_MV_VocalLock_V2_ClearSpeech_master_audio.mp4`,
SHA-256 `68bdc57312d6e8d4211490d394a8be29353d15af7179b56badc99c6c6c3c4533`.
It contains 124/124 H.264 frames at 736x416 and 24fps (5.166667 seconds) plus finite 32kHz stereo
AAC. Strict video-only, audio-only, and combined FFmpeg `-xerror` decode all passed. Aligned source
versus final-audio APSNR was 161.084/161.086dB.

Temporal synchronization was evaluated with the MIT-licensed official
[`joonson/syncnet_python`](https://github.com/joonson/syncnet_python) implementation and its official
`syncnet_v2.model` (local SHA-256
`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`). The face-visible center crop
was converted to 224x224 at 25fps. The unmodified candidate measured AV offset 0 frames, minimum
distance 8.016, confidence 5.844. A predeclared negative control delayed the video by exactly 0.400
seconds (ten cloned 25fps opening frames) while preserving the audio; SyncNet measured exactly +10
frames, minimum distance 8.209, confidence 5.911. Therefore the tested candidate passes the
zero-offset plus shifted-control mechanical gate. The user then reviewed the normal-speed original
plus mouth-zoom aid and explicitly reported `口型通过`, closing the bounded lip-sync human gate for
candidate SHA-256 `68bdc57312d6e8d4211490d394a8be29353d15af7179b56badc99c6c6c3c4533`.

The same feedback asked why the area around the performer looked soft. This was not an intentional
Vocal Lock effect: the route contains no blur filter. The clear fixture reference is a side-profile
portrait with glasses, while the validation prompt requested a front/three-quarter performance plus
a restrained slow push-in at 736x416. The observed soft subject boundary is recorded as generated
reprojection/motion softness, separate from the lip-sync pass. Before release, the unreleased V2
recommended default was changed to a locked-off camera and its prompt now asks for sharp, temporally
stable hair, face, shoulder, clothing and silhouette edges without haloing, smearing or double
contours. Users are also advised to choose a sharp reference whose face direction matches the target
shot. This prompt/default refinement was not rerendered and is not claimed to guarantee sharpness.
The result does not prove that every generated mouth shape is linguistically correct, nor does it
establish universal H3 identity or visual quality.

After that unreleased prompt/default refinement, the five-file MV/registration/frontend scope passed
107 tests. The complete repository passed 1,937 tests with the same five existing Triton/PyTorch
warnings. Full Ruff, 598 Git-scope Python compilation checks, 253 JSON parses, `git diff --check`, a
zero-match credential scan, Comfy configuration/security validation, CPU whitelist import, and the
185/185 source-to-user workflow mirror gate passed. An isolated `1.63.0` package contained 492 entries,
included every required V2 runtime/workflow file, and contained no model weights, media, nested archive,
development tools/tests/docs, `artifacts`, `SKILL.md`, or `roadmap.md`. No H3/GPU sample was rerun for
the locked-camera prompt refinement.
