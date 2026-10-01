# Veda Sparse for MiniMax H3 T2VA (experimental)

This opt-in route adds the [Miowtion Veda](https://github.com/veda-sparse/Miowtion) tile-score predictor to the native MiniMax H3 video attention path. The author modules are vendored under their MIT license at commit `3bc5e5ae53e8c0d4539f2bc872265164947d013a`; on Windows the tile mask is executed by PyTorch FlexAttention, **not** the author's FA4 kernel. The predictor weights are separate from the H3 base model and are not distributed with this node pack.

## Exact supported scope

### Updated official predictor and the 8-step pairing (local)

The official repository revision `9a1fd3a41b4a754a7886e64e82edbddf599fd1bd`
contains an updated **Veda tile-score predictor**, not an H3 diffusion checkpoint
or a replacement Turbo LoRA. Its file is 275,415,648 bytes, SHA-256
`2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a`.
The existing Bundle node can load it directly; no ComfyUI key conversion is
needed. For side-by-side installation here it is named
`minimax_h3_t2va_veda_8nfe_600step_preview_fp8_9a1fd3a.safetensors` in
`ComfyUI/models/veda_scorers`. The earlier filename, weights and workflows
are preserved. A filename alone does not identify which release was tested:
check the Bundle/Audit SHA.

Two additive [new-predictor canvas examples](../examples/workflows/63-veda-t2va/README.md)
copy the separately completed native Dense/Sparse canvases; only their display
titles change. Their entire executable inputs/edges match the actual successful
Queue prompts after the unchanged frontend task-label serializer, not just an
API-only reconstruction. They retain explicit eight steps and five-second AV
delivery, without replacing any old graph. They remain local EXP/not human
quality or publication approval.

本机新版预测器已经加入 Veda Bundle 下拉框，不要放进 UNET 加载器。
推荐配方分别连接：H3 非裁剪底模 → Turbo v4 step600 LoRA → Veda Apply；
采样使用显式 **8 步**，`dual_clock_euler/native_flow`、video/audio shift
`12/3`。作者 Miowtion 的 8 步示例使用
`minimax_h3_turbo_v4_step600_ema.safetensors`；本机对应转换文件为
`minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`、强度 1。
本地转换 metadata 的 `sampler_steps=4` 是历史推断标签，不是作者 Veda
推荐步数，也不应借此改写旧四步工作流。Veda 的网格限制仍须单独满足。

This revision retains the predictor-v1 format and all twelve packed plans.
The unchanged public loader has passed actual menu/path execution and full
CPU loading of all 100 parameters: every bf16 value matched the official
FP8/per-head-scale dequantization. Its metadata says `step=100` and
`source_weights=leap0.8`; the official config describes 600 scratch-training
updates followed by 100 full-trajectory updates. Do not reject it merely
because the filename still says `600step`, or interpret training updates as
inference steps.

Eight new-revision native Chromium Queue cases completed the actual 8-step
recipe and all 400 attention calls, with strict complete 120-frame, 768×768,
24-fps, five-second H264/AAC outputs and native joint checkpoints. However,
the counterbalanced two-seed matrix **failed its repetition gate**: the first
seed's cold sparse run differed from its warm repeat in both latent values and
decoded picture/audio. Dense repeats and both second-seed mode repeats matched
exactly. The cold sparse prompt was slower (458.125 s versus its warm 216.078 s).
These are new-revision execution checks, not stable speed or human quality
approval; the failed cold case is retained.

A separate fixed-QKV, real-predictor single-layer diagnostic reproduced the
cold/warm difference with identical inputs and selected blocks, unchanged RNG
and precision flags. Default static-first shape generalization changed Flex
output; explicit dynamic compilation from the first call produced exact repeats
in that bounded diagnostic. The local adapter now requests `dynamic=True`;
this compilation-only correction was then tested in a **new** complete native
matrix, not by relabeling the earlier failed run. All eight cases completed the
original 8-step recipe, strict five-second AV and joint checkpoints in one
cache-disabled Core process. For both seeds, each mode's two latent/manifest
content and decoded RGB/PCM results were exactly equal, including the first
cold sparse run. The public sources, controllers and selected predictor stayed
unchanged throughout, and the owned service closed after completion.

Performance remains conditional: the first sparse prompt took 565.484 s and its
warm repeat 243.203 s; the second seed's sparse prompts took 238.343/237.547 s,
versus its dense prompts 321.797/322.766 s. Including the cold run, the first
seed did **not** meet the paired speed gate. The bounded warm advantage is not
a first-use or general speed qualification, and no human quality was accepted.
Seven new-revision full-length plans subsequently passed native eight-step
execution, complete AV decoding and their own source-bound checks: all four T37
plans plus 16:9, 1:1 and 4:3 at T72. The user requested representative validation
instead of the full twelve-case matrix; the remaining runs were stopped, with
the interrupted eighth case retained as incomplete. All twelve packed plans
remain selectable, but this is not a claim that every grid was executed or that
quality and speed were accepted. The Bundle menu lists only supported
`.safetensors` files, retaining existing valid choices and excluding HF cache
metadata; the folder category and cache files are unchanged.
The six EAV order controls, separate Relay and split/cold-HIGH results are
described below. The correction does not change the author's
predictor, masks, teacher weights, sigma schedule, audio wiring or legacy dense
path. Windows Flex prerequisites and default report-only behavior remain unchanged.
Historical results below used the earlier predictor SHA and do not automatically
qualify this update.

### New-revision composition checks (local)

The updated predictor also completed the original full eight-step recipe with
external Prompt Relay in an actual native Chromium Queue. Its complete
120-frame, 768×768, 24-fps H264/AAC output strictly decoded. Relay's partial-query
attention path produced 34,000 **dense delegated** calls and zero sparse calls;
this proves composition execution, not simultaneous Relay/sparse acceleration.

A separate 42-node native 4+4 graph completed report-only heuristic LOW at
384×384, the unchanged learned 3D x2 upscaler, and trained Veda HIGH at the exact
T37 768×768 plan. It observed 200 LOW dense and 200 HIGH sparse calls, external
report-only EAV, two authenticated portable stage saves and strict complete
five-second AV. A new 26-node fresh-Core native HIGH-only graph then restored
only this same-release verified LOW. No LOW sampling occurred; HIGH request/state
and complete decoded RGB/PCM exactly matched the full graph, with the original
LOW artifact and selected predictor unchanged.

These are bounded partial-sigma/composition mechanics, not the teacher-eight-step
quality recipe, first-use speed, every grid/material, human acceptance or a
release. The initial private EAV-order control wired its guider around Veda and
failed with zero observed Veda calls; that failed run is retained. A new corrected
six-control native matrix passed the original eight callbacks and complete
120-frame/768×768/24-fps H264/AAC plus joint checkpoints in every case. It used
Dense and Sparse controls, then both EAV modes in each node order. Veda→EAV
recorded 400 sparse calls and 400 EAV measurements; report-only matched its
Sparse control's full latent and decoded RGB/PCM exactly. Apply changed both
the saved video tensor and decoded picture, with maximum gain 1.031129 below
the unchanged 1.5 bound. EAV→Veda preserved the earlier owner: zero Veda calls
and an explicit bypass, with 400 EAV measurements. Report-only exactly matched
the Dense control; apply changed saved video and decoded picture with maximum
gain 1.035768 below 1.5. This is owner-preservation evidence, not simultaneous
reverse-order sparse acceleration. Public sources, nine controller/template
files, selected bundle and all five original teacher weight files retained
their complete SHA identities, and the owned Core closed. Cold/warm timing,
human picture/audio quality, other materials and all grids are separate gates.
Old-predictor effect receipts do not qualify this revision.

### Latest separate-stage qualification (local, not published)

One explicit native 4+4 graph completed LOW at 384×384 with the independent
heuristic in report-only mode, the unchanged learned 3D x2 upscaler, and HIGH
at the exact official T37 768×768 trained Veda plan. It recorded 200 LOW dense
and 200 HIGH sparse calls, externally configured report-only EAV and two
completed portable stage saves. Both API and actual native Chromium Queue
full runs and fresh-Core HIGH-only resumes passed strict complete five-second
H264/AAC decode. Cold runs loaded the completed LOW without executing it;
HIGH request/state and decoded picture/audio matched full exactly, and the
original LOW artifact was unchanged. The expanded 26-module historical batch
passed 566 CPU tests without CUDA initialization. Stage identity now binds
recognized actual Veda selector/predictor configuration without mutating MODEL;
unknown executable producers remain nonportable, preserved and unverified.

This is one **partial-sigma 4+4 mechanical case**, not the predictor's teacher
8-NFE quality recipe, stable acceleration, every grid/task/effect order or
human acceptance. Earlier slower first-run and visually changed heuristic
results remain separate failures to qualify speed/quality. T72/T102 proofs
remain single-NFE execution only, not complete long media. No publication
or Registry installation is implied.

Use the official [MiniMax H3 T2VA Veda 8NFE 600Step Preview bundle](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) in `ComfyUI/models/veda_scorers`. The earlier media-tested file is `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`, SHA-256 `ff4ba71e245a581c725b40e1af1d4f872cf8b8562407851594a8d2f4c7121f64`. The loader checks the Veda format, 50 layers, 56 heads, 128-dimensional heads, and file identity. It never downloads a model by itself.

Only the 12 exact grids inside that bundle can be selected: video latent time `37`, `72`, or `102` with spatial token grid `24×24` (1:1), `24×32` (4:3), `24×42` (16:9), or `42×24` (9:16). No nearest-plan substitution or arbitrary resizing occurs. The predictor was published for T2VA, 8 denoising evaluations, and the 600-step Turbo recipe; another task, LoRA, step count, or reference/keyframe path has no inherited quality claim. The local mechanical media test used non-pruned FL2VA INT8 ConvRot, a converted Turbo v4 step600 LoRA, native audio, `dual_clock_euler/native_flow`, shifts `12/3`, and the exact `1x1_t37` plan.

## Nodes and safety

1. `H3 Veda · Bundle (T2VA EXP)` selects the predictor from `models/veda_scorers`.
2. `H3 Veda · Apply Flex Sparse (T2VA EXP)` clones the input MODEL. Its default `report_only` observes a supported plan while delegating to the original dense attention. `apply_exp` uses the predictor and Flex tile mask. `fused_tile_io_exp` is a separate, default-off Triton gather/pool/scatter experiment, not the author's FA4 sparse attention kernel. An existing unverified attention or DiT patch owner is preserved unchanged: Veda is explicitly bypassed and the audit says so. Actual unsupported task/layout, missing kernel on a selected sparse path, or changed bundle still fails; no dense delegation is labeled sparse speedup.
3. `H3 Veda · Audit After Sample (T2VA EXP)` must follow the actual sampler and exposes calls, selected grid, mode, dense-delegated calls, and failure in ComfyUI history. A masked H3 call retains its mask and original dense backend; it is counted as delegated, never as a sparse speedup. It does not certify video quality or speedup.

On this Windows Torch 2.13/CUDA 13/Triton stack, an isolated DynamicVRAM run crashed inside native Triton. Therefore `apply_exp` refuses Windows DynamicVRAM before replacing the MODEL. The currently verified isolated command uses `--disable-dynamic-vram` with matching `triton-windows`; do not change a running user service merely to try the experiment. This safety gate may change only after a separate successful regression. `report_only` remains available with the original attention path. The released bundle and model may require substantial system RAM/VRAM; the 16 GiB card reached near-full VRAM even at 768×768.

On a Windows Python process whose default text encoding is CP936/GBK, the first Torch Inductor compile can fail while reading an upstream UTF-8 source template. Selected Flex `apply_exp` now checks this before cloning MODEL and gives a specific error. Set `PYTHONUTF8=1` **before starting ComfyUI** and restart the process; setting it inside an already running node is ineffective. This is an environment prerequisite, not a sparse speedup or a change to the original attention path.

The local CPU regression runner has an explicit `--tmp-path-mode isolated` option for independent, short per-test directories. It records their exact paths and retains inputs/failures; test assertions, production codec isolation, strict decoding and the default pytest mode are unchanged. On the local Windows environment, the eleven-module Veda/effects/media/harness batch passed all 223 tests in this mode, with CUDA uninitialized and 4,691 source files stable. The original default-temporary-directory child-start failures remain separate evidence, and their exact OS cause is unresolved. This regression result is not a Veda speed, video-quality, or default-environment repair claim.

Ordering matters for external effects. Applying Veda **after** an existing unverified EAV/Prompt Relay or DiT attention owner preserves that owner and reports `bypassed_unverified_owner_preserved`, with no sparse coverage. Applying the existing EAV node **after** Veda delegates its attention calculation to Veda and can then measure or scale the output. A separate real-canvas T2VA experiment below establishes this one ordering only. Prompt Relay splits its queries and supplies an additive bias; when such a partial-query/full-KV call reaches either Veda route, the original attention receives that call unchanged and the audit records `partial_query_preserved` with zero sparse coverage. CPU numerical tests cover bias preservation, and an isolated real-weight one-NFE Relay-before-sampler run recorded 4,250/4,250 dense-delegated calls with zero Veda sparse calls. **Neither is simultaneous Relay+Veda acceleration or a complete-media qualification.** A masked H3 call likewise preserves its mask via dense delegation. LOW/HIGH two-pass, cold resume, non-T2VA inputs, and full-media qualification on other grids remain separate work. The released bundle has only one spatial grid per aspect ratio and temporal length, so it cannot simply accelerate both resolutions of an ordinary same-aspect LOW-resolution → HIGH-resolution upscale: only a stage whose exact grid exists can use it. Existing workflows and their defaults are unchanged.

## Current evidence and limits

### Explicit stage content identity

The native-stage identity adapter recognizes the actual Core selector wrapping either native Veda route. It projects only an inspection clone, never unpatches the live MODEL, and binds captured settings, original native delegate, bundle file/metadata, live predictor parameter bytes, exact tile plans/head assignments, checked derived tile caches, source implementation and Torch/CUDA versions. Telemetry and valid cache population do not change pre/post operator identity; changed predictor bytes change it, while a corrupted plan/cache/file fails. Foreign selector/runtime/hook/kernel owners remain execution-local/nonportable. This descriptor is not a completed sample, quality or speed certificate. The 18-module CPU/effects/media/native-stage/old-seam batch passed 405 tests with CUDA uninitialized and 4,693 source files stable. Trained partial-sigma LOW/HIGH and fresh-process cold-HIGH media remain separate gates.

The first trained LOW stage exposed an integration error: its pre-sampling descriptor included Veda but the post-sampling effects projection omitted it, so the stage correctly refused a mismatched key set rather than producing certified completion. Both checks now use the same full projection. Two additional real tiny CPU Core-stage tests verify unchanged operator content can complete and changed content still rejects; the corrected nine-module stage/effects/old-seam batch passed 280 tests. The later 26-module regression passed 566 tests, with CUDA uninitialized and all 4,693 source files stable, including stage recovery, effects, media and five old seam modules. These counts describe separate batches, not an additive test total. The original failed run is retained; this CPU correction is not itself a cold-media qualification or a weakening of artifact integrity.

### One partial-sigma split and fresh-process HIGH recovery (mechanical gates)

A real 42-node native-stage graph ran independent 384×384 LOW heuristic `report_only` for four steps, the unchanged learned 3D ×2 upscaler, then the official exact-grid 768×768 trained Veda HIGH for four partial-sigma steps. External Stage EAV was `report_only` on both stages. LOW recorded 200 dense calls; HIGH 200 sparse calls and zero dense delegates. Both completed stage artifacts had portable, SHA-bound receipts. A fresh process ran a 26-node HIGH-only graph loading that LOW artifact without LOW execution. HIGH request and saved-state SHA matched the full graph exactly, and both strict-complete five-second H.264/AAC files had identical decoded RGB24 and PCM-f32 hashes. Original LOW and runtime sources remained unchanged; owned servers stopped.

The same full and HIGH-only configurations subsequently ran through actual native Chromium canvas Queue buttons in separate owned Core processes. Their accepted POSTs exactly matched the canvas and API control apart from non-executable UI metadata. Full Queue-to-history took 426.703 s; cold HIGH-only took 278.109 s. Both had the same stage request/state identities and complete decoded picture/audio as their API controls; cold executed Stage Load and HIGH, not LOW. All four outputs had MP4 SHA-256 `ff28816ec56e5e572ccb9aae4b64deadf61a5dfda513f5dc4ef8b24c52c42815`. Sources and original LOW were unchanged, and owned processes/ports closed. A connection-reset callback after completed cold execution is retained in its service log; it is not a failed sampler or hidden retry. These gates establish this particular partial-sigma connected stage/recovery configuration, not teacher-eight-step quality, stable acceleration, simultaneous sparse Relay, every grid/task or all two-pass routes. Human quality review remains pending.

### Startup-reserve diagnostic, not general acceleration

Two uninstrumented independent-process complete-AV controls used the same saved native prompt, seed, eight-NFE recipe, history polling and a 2.5 GiB isolated startup VRAM reserve. Trained sparse took 300.672 s and dense 315.125 s, each with 400 calls and strict 120-frame/24-fps/five-second 768×768 H.264/AAC. Each mode's MP4 and decoded RGB24/PCM-f32 SHA matched that mode's earlier 1 GiB-reserve output. The small cross-process difference, with different cache/process state, is not repeatable speed qualification or proof that reserve alone caused the change; earlier slower-sparse evidence remains valid. A connected native Chromium two-Queue sparse canvas at 2.5 GiB reserve completed both eight-NFE runs with 400 sparse calls and strict five-second AV each, taking 373.453 and 228.547 s. Its second seed changed, and no matched dense native canvas has yet qualified these timings. No human quality approval is inferred.

An isolated native Chromium canvas opened, serialized, and clicked Queue on a private 768×768, 124-generation-frame, 8-NFE T2VA graph. The history recorded 400/400 sparse layer calls; a separate Output Trim produced 120 frames at 24 fps (5.000 seconds), and the built-in Safe AV Save produced H.264 video plus AAC audio. Both streams passed strict complete FFmpeg decoding. A second native canvas used the same graph and seed except `report_only` versus `apply_exp` and the output filename. It also produced strictly decoded five-second H.264/AAC media with 400/400 original-dense calls.

The local 16 GiB RTX 4060 Ti **did not show an end-to-end speedup** in this matched first-run comparison: dense sampling took 10:31 (whole prompt 11:52); default Flex Veda took 12:44 (whole prompt 14:08). An earlier same-process eight-step A=dense → B=Veda diagnostic without media took 536.8 s versus 1184.9 s, also slower for Veda. Single-step warm A/B/A/B measurements sometimes favored Veda; the later full eight-step repeat-run measurements below separately establish a warm-run performance hint, not a first-run result. The optional fused tile I/O passed an isolated full-grid 56-head numerical comparison and a real-weight one-step execution; its same-process warm single-step result was 66.1 s versus dense 76.7 s, approximately the same Veda time as the default tile I/O. A subsequent native Chromium eight-step fused canvas passed 400/400 sparse calls and strict full decode of 120-frame 768×768 H.264/AAC at 24 fps / five seconds, but sampling took **16:16** and the whole prompt **17:38**. Its sparse attention wall time alone was 515.0 seconds. It was slower than both the matched dense canvas and the default tile-I/O Veda canvas, so fused tile I/O remains default-off and does not qualify as a first-run acceleration on this machine.

A same-process, one-NFE four-phase diagnostic compared 64 MiB and 16 MiB tile-copy budgets without media. After each setting's warm phase, 16 MiB took 70.1 s for the request / 13.3 s in the Veda attention path, while 64 MiB took 66.8 s / 10.4 s. Reducing the budget did not improve this local hot single-step result, so the 64 MiB default remains unchanged. Neither setting is an eight-step quality or speed qualification.

A later eight-NFE, real-weight diagnostic found a substantial **repeat-run effect**, but did not establish a general speedup. In one isolated process, two consecutive Flex-sparse runs took 618.5 then 191.1 s, each with 400 actual sparse calls; measured Veda attention time fell from 267.2 to 83.6 s. In a separate isolated process, two consecutive original-dense runs took 546.3 then 433.5 s, each with 400 original calls. The second Flex run was faster than the second dense run under these separate-process, same-route-repeat conditions, while the first Flex run remained slower. A separate A→B→A sequence suffered severe step-time drift on the return to dense; its A2 was deliberately interrupted and retained as a failed diagnostic, not used as a completed baseline. All services stopped and source remained stable. These timings have no VAE/media export, are not repeated across seeds or sessions, and cannot override the slower first-run native AV canvases or certify speedup for ordinary use. Predictor/model offload, JIT cache, process state and output quality remain unresolved.

A subsequent full-output API control used the exact same saved native T2VA prompt (apart from the selected Veda mode and output prefix), seed, 8-NFE recipe, 768×768 geometry, and history-polling protocol without a persistent WebSocket. Both runs had 400 actual attention calls, successful Core history, 120-frame/24-fps/exact-five-second H.264/AAC, and strict complete decoding. Original dense took **628.187 s** for the whole prompt; trained Flex sparse took **1266.219 s**, with 608.430 s reported inside its sparse attention route. Thus even after removing the WebSocket distinction, this local first full-output sparse run was about **2.02× slower**, not qualified as an acceleration. The sparse MP4 SHA-256 was `D02F17B85B66FADC85F810660C3D0051F62672FABB5E49078B7E5B3E573BE0B9`; the dense control was `8BE2822D726677FC36917F8E522D34551B4727EE6FEA5ECEFAA1BBF291CE6EED`. Two other connected full-output dense controls took 930.875 and 914.922 s for byte-identical dense media, while native Queue followed by closing Chromium took 660.265 s. Connection state correlates with those timings but is **not** a proven cause. The 16 GiB card used almost all VRAM, with substantial model offload; this is not yet a single-variable bottleneck diagnosis. The first sparse attempt without UTF-8 startup failed at Torch template decoding before sampling and remains a separate failure receipt, not a quality or timing result.

These are configuration-specific observations, not statistical quality, broad compatibility, or human video/audio approval. The sparse and dense contact sheets show the same rain-roof subject but visible luminance/detail differences; neither was fully viewed/listened to for acceptance. Prompt Relay, LOW/HIGH, cold resume, human review, and publish decision remain open. Experimental weights, private reports, and generated media are not included in a release.

### Other exact plans: one-step execution only

The exact `1x1_t72` and `1x1_t102` plans each completed one real H3 denoising evaluation at 768×768, with 243 and 345 native generation frames respectively. Each recorded 50 sparse layer calls, zero dense delegates, unchanged source snapshot and owned-process/port cleanup. Whole requests took 240.547 and 379.875 s, including model/text loading and compilation state. These extend only mechanical one-step execution to those two longer grids; no VAE decode, full-length media, eight-NFE speed or human quality is certified by them. Other longer-grid aspect ratios remain independently unqualified.

Separate isolated Core runs used the same real H3 weights and author predictor with 124 generation frames but omitted VAE decoding and media export. The 4:3 1024×768 plan matched `[37,24,32]`, 16:9 1344×768 matched `[37,24,42]`, and 9:16 768×1344 matched `[37,42,24]`. Each completed one NFE with 50/50 sparse layer calls, no dense delegation, exact plan audit, stable source, and a stopped isolated service. Their whole one-step requests took approximately 280, 445, and 310 seconds respectively. These are feasibility checks, **not** eight-step, five-second AV, quality, speedup, or T72/T102 qualifications. The 16 GiB GPU was near full; larger/longer plans should not be inferred safe from these runs.

### Veda-before-EAV composition evidence (not a speed or quality claim)

Two private 18-node/28-link native Chromium canvases put the official Veda Apply before the existing external EAV node, with both audits after the real sampler. Both used the same 768×768, `1x1_t37`, eight-NFE, five-second T2VA recipe and recorded 400/400 Veda sparse calls, eight EAV model forwards, 400 EAV temporal-CFI measurements, and strict full H.264/AAC decoding of 120 video frames plus audio. With EAV `report_only`, the MP4 SHA-256 was byte-identical to the prior Veda-only canvas; observed gain ranged 1.0–1.03356, and sampling/whole-prompt times were 18:58/20:21. With EAV `apply_exp`, its audit reported actual application, gain up to 1.03279, and the resulting MP4, decoded picture, and decoded audio all differed from the no-effect control. Decoded picture PSNR was 26.88 dB against that control, which measures a difference, **not** an improvement. The `apply_exp` sampling/whole-prompt times were 8:25/9:48, but these separate processes had dramatically different cold/warm behavior; neither run establishes a composition speedup. The video/audio have not received full human quality review. Reverse node order, Relay, other grids/tasks and two-pass/cold paths are not certified by this result.

## Independent model-free TripPool 64 experiment

The separate `H3 Veda · Heuristic TripPool 64 Apply (EXP)` and `H3 Veda · Heuristic Audit After Sample (EXP)` nodes implement an independent, model-free route inspired by [BSAI-ComfyUI-VedaSparse](https://github.com/xm6018924/BSAI-ComfyUI-VedaSparse). They do **not** load the official Miowtion trained predictor, the BSAI reserved distilled scorer, or an FA4 kernel. The default is `report_only` and delegates to the original dense attention. `apply_exp` explicitly uses 64-token, head-aware native H3 video tiles, Avg/Max/Min tile descriptors, a per-head top-k mask, always-retained conditioning rows and columns, and PyTorch FlexAttention. The runtime clones MODEL and its audit says `trained_predictor_used=false`. Existing unverified attention/DiT owners are preserved with an explicit bypass, not counted as sparse calls. The same Windows DynamicVRAM safety gate applies to selected Flex execution.

Unlike the trained T2VA predictor, this route does not require a scorer file or one of the official fixed spatial plans. It still requires the native 50-layer, 56-head, 128-dimensional H3 DiT, a valid final target video segment, a supported Flex kernel, and a live layout matching the actual packed rows. Masked calls and calls below `min_tokens` delegate to dense and are counted as such. Non-T2VA/reference inputs and other grids have geometry/unit coverage but **not** real-media or quality qualification. Three optional trailing controls reproduce the BSAI-style sampling interval and conditioning sink without changing existing graphs: `start_percent=0`, `end_percent=1`, and `sink_conditioning=exact_kv_and_rows` preserve the prior behavior. A non-default interval is converted through the selected MODEL's `model_sampling.percent_to_sigma`; outside it, calls explicitly delegate dense and the audit counts them. Missing or nonuniform live sigmas fail explicitly. `sink_conditioning=off` removes global conditioning **keys/values only from video query rows**; nonvideo query rows still attend all valid keys. This off mode can change prompt/audio/reference fidelity and has only mask/unit tests, not real-media or quality approval. No automatic fallback from a requested trained scorer to heuristic is provided.

In one isolated native Chromium 768×768 T2VA, 8-NFE, 120-frame/24-fps/five-second canvas, the 5% heuristic setting recorded 400/400 sparse calls, full strict H.264/AAC decoding, sampling 8:57 and whole-prompt 10:16. A second native canvas at 10% also recorded 400/400 sparse calls and strict five-second AV decoding, but took 10:52 sampling and 12:10 whole-prompt. The matched dense canvas was 10:31 sampling and 11:52 whole-prompt. Thus 5% showed only a single-run timing hint, while 10% was slower; neither establishes repeatable acceleration. Contact sheets at both heuristic settings are visibly darker and differently composed than dense. This is **not** a quality pass, a same-process statistical speed comparison, or a claim that the heuristic matches the official trained Veda route. Keep it opt-in and experimental pending matched repeats, more tasks/grids, and human video/audio review.

The optional interval and sink controls subsequently passed three isolated real-weight, one-NFE checks: full-window `sink_conditioning=off` made 50/50 sparse calls; window `0.5–1` made 0/50 sparse calls and delegated all 50 as `outside_sigma_window`; window `0–0.5` made 50/50 sparse calls. A private native Chromium Queue using the latter interval, 5% keep and the default exact-conditioning sink completed all eight NFE with **250 sparse and 150 window-delegated dense calls**. Its 768×768, 120-frame, 24-fps, exactly five-second H.264/AAC output passed strict full video/audio decoding. Sampling took about 16:41 and Queue-to-history 1083.516 s; a separate browser-closed dense control took 660.265 s. Process-state variance prevents a causal speed estimate, but this is plainly **not a speed qualification**. The new video's 0–4 s contact sheet is visibly darker and framed differently from dense, so neither visual equivalence nor full human audio/video acceptance is claimed. The sink-off mode has only the one-NFE real-weight check, not complete media qualification. These tests use private workflows and do not alter released examples.
