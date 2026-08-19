# LoRA and sampler verification report

This report records the historical LoRA, stable sampler, and multi-rate sampler
verification checkpoint. For the current plugin version, node inventory, and
Ref2VA still-image status, also read the project-root `README.md` and
`features.json`.

The current 1.18.2 checkpoint passed the full project regression and a real Qwen-cache/H3 generation
probe on 2026-08-14 against ComfyUI `v0.32.0-16` at
`ddbaa8752874c275290d054ee4fddd6e004f5fdf`. The immediately preceding compatibility checkpoint was
`v0.32.0-15@86aedfd943d36d485e5ed3cb9d962f21f73d1741`. The wider real-generation matrix below remains anchored
to `0.31.0@cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`. Historical LoRA conversion evidence was
originally recorded on 2026-08-06 against source commit
`563b98eefbe643a4cd510ee7f0b43e79880d5a3f`.

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
