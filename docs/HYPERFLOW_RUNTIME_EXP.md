# HyperFlow independent runtime — EXP acceptance (2026-09-22)

## Continuous split execution reports

The combined split now records actual Core sampler callbacks separately for
LOW and HIGH, including local indices, absolute intervals and each callback's
local total. The existing `stage_nfe` and `total_nfe` fields remain compatible
but describe trained intervals; they are not counters of extra DiT forwards
inside CFG or user wrappers. `nfe_evidence` makes that distinction explicit.
Callback recording does not change sampling, noise, AV clocks or the exact
`x_sigma` handoff. The owned qualification controller also consumes existing
HyperFlow and Stage `report_json` outputs without changing producer inputs.

Actual tiny native 50+2 CPU tests cover 1+7, 4+4 and 7+1, exact AV equality
with and without an external callback, and all twelve real report schemas.
Earlier frozen probe runs are unchanged; a missing callback report in those
runs cannot be retroactively treated as observed execution.

This is an opt-in MiniMax-H3 pathway, not a generic acceleration LoRA and not
the existing Director `standard_4plus4_v1`. The old single and 4+4 recipes
remain the default for projects without HyperFlow settings. Earlier development
probes used an owned Core on port 8860; the 2026-09-22 formal-install probes
used separate owned Core ports 8873–8875. The formal plugin files are now
installed, but the user's running Core was not restarted or changed.

## Source and artifact identity

- User-provided mirror revision:
  `DeepBeepMeep/MiniMax-H3@7b61c8edb895aaf25b248f064e9726ffdcc7ec46`.
  Author implementation reviewed at
  `Video-Rebirth/hyperflow@1dd2f342aba5ab51da02b62885939655e8e268da`
  (code Apache-2.0). The adapter itself reports the **MiniMax H3 Community
  License Agreement**; its terms, not the code license, govern the weights.
- Original, non-pruned file:
  `F:/AI-T8-video-onekey/ComfyUI/models/hyperflow/loras/minimax_h3_hyperflow_8step_v1.0.safetensors`;
  2,795,328,008 bytes; locally calculated SHA-256
  `9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447`.
  Do not publish this file with the repository.
- Real numeric scan passed: all 632 original tensors finite and consumed,
  210 backbone/base-time patches plus 2 independent endpoint-time projections,
  including 52 QKV fusions; metadata rank/alpha 256, gate 0.25, video/audio
  shifts 12/3, nine raw sigma points. The loader requires 50 DiT plus 2
  refiner blocks and rejects a pruned model or missing endpoint branch rather
  than silently falling back to a single-time approximation.

## User-facing routes and contracts

| Route | Nodes / handoff | NFE | Status |
| --- | --- | ---: | --- |
| Single8 | dedicated Loader → typed Plan → AV Sampler → normal `BasicGuider` / `SamplerCustomAdvanced` | 8 | GPU executable, quality not certified |
| Continuous4+4 | two independently patched full-H3 Loader models → `MiniMaxH3HyperFlowSplitT8Advanced`; exact captured model-space `x_sigma`, no new noise or resize | 4+4 | Exact real-model latent parity to single8 for tested T2VA case |
| Upscale8+4 | LOW full8 clean latent → existing learned 3D H3 latent upscaler → HIGH typed absolute 4:8 Tail Plan / new-noise Sampler | 8+4 | Separate low-to-high experiment, quality not certified |
| PartialUpscale4+4 | LOW absolute 0:4 predicted clean `denoised_output` → learned 3D upscaler → HIGH absolute 4:8 new-noise Tail Plan | 4+4 | Formal Core generated 0.4 MP landscape and 0.5 MP portrait; quality not certified |

The HyperFlow weight selector is `hyperflow_file`, separate from low/high
content-LoRA arrays. Each stage may chain multiple compatibility-loader
nodes before its own HyperFlow Loader; both stages share the same base H3 and
the same original HyperFlow file. The Director compiler emits
`hyperflow8_single_v1`, `hyperflow8_continuous_split_exp_v1`,
`hyperflow8plus4_new_noise_upscale_exp_v1`, or
`hyperflow4plus4_partial_x0_upscale_exp_v1` only when explicitly selected.
The old `standard_4plus4_v1` remains a distinct graph and scheduler.
Director inspects each selected content-LoRA safetensors header and rejects a
two-time HyperFlow artifact in a content slot, including a copy with another
filename. The dedicated Loader also rejects a directly preceding generic
LoRA whose attached metadata declares `hyperflow=true`. If a content LoRA
patches the base time embedder, the Loader reports that the original
HyperFlow endpoint branch is unchanged; that stack is not quality-qualified.

The continuous split is **not** an image-upscale two-pass. Its absolute
boundary is video sigma ≈ 0.923077 and audio sigma = 0.75, not the old
Director 4+4 schedule. The 8+4 route is twelve evaluations; partial-upscale
4+4 is eight. Both deliberately re-noise AV latents at HIGH after learned
resize. Neither is one continuous trajectory or a proven quality boost. For
partial4+4, LOW's nonterminal `SamplerCustomAdvanced` output socket 1 is the
predicted clean x0 for the upscaler; socket 0 is not used as the clean picture.
The Director receipt records the expected trained grid, shifts, grid-contract
digest, absolute intervals and NFE; frozen HR1 batches separately hash the
actual selected model bytes.
The early 0.4 MP probe predates the additional compilation-receipt fields;
the later 0.5 MP and final-source 8+4 `director-report.json` files contain
them. The digest identifies the **expected v1 training-grid contract**, not
the exact loaded-file hash or a measured forward counter; runtime validates
the file metadata and the typed sampler asserts its step count. When wiring
the public nodes manually, explicitly connect LOW sampler output socket 1
to the upscaler; the HIGH plan check alone cannot detect a wrong LOW socket.

## Local execution evidence

All GPU runs used the full
`minimax_h3_fl2va_int8_convrot.safetensors`, Qwen3-VL NVFP4/AWQ encoder,
video INT8 VAE, audio FP32 VAE, the original HyperFlow file, fixed seed
123456789, and owned isolated Core. Every successful route completed native
AV decode and MP4/AAC delivery; this proves execution and file validity,
not subjective visual or audio quality.
In the table, the first six historical probe directories are relative to
`G:/CodexHome/.codex/worktrees/t8-h3-final-20260920/`; rows labeled
**Formal** are relative to this installed plugin directory. Neither location
is a published repository artifact.

| Probe | Prompt ID | Evidence directory | Result |
| --- | --- | --- | --- |
| Single8 256², 22 frames | `972beed0-86dc-4c5c-9335-7b5a4ded911c` | `artifacts/development/hyperflow-gpu-5e534aa836` | 8 forwards, H.264 256²/22 frames + AAC 32 kHz, Core 51.00 s |
| Continuous4+4, two loader owners | `c1f44454-9734-4227-896f-05fb5a10696b` | `artifacts/development/hyperflow-gpu-85dd7eff10` | 4+4 forwards, same AV delivery, Core 49.52 s |
| Upscale8+4 256²→512² | `6893a5fc-d879-4525-adfa-d05913bbf46e` | `artifacts/development/hyperflow-gpu-63accc4f67` | full8 + learned 3D resize + tail4, H.264 512²/22 frames + AAC, Core 61.65 s |
| Single8 vs continuous4+4, two distinct owners | `d1198543-6eb1-4530-99ea-95cd49a3ba1b` | `artifacts/development/hyperflow-gpu-925a32c808` | final video and audio latents each had max/mean/RMS absolute difference **0** |
| Director compiler actual 8+4 graph | `ded9940a-944f-412d-b37e-ce47c4f3ae7a` | `artifacts/development/hyperflow-gpu-a4b288cd07` | compiled project graph produced 448²/24-frame H.264 + AAC, Core 66.65 s |
| Director 8+4, two content-LoRA slots per stage | `ba7714bb-32cd-488e-890a-c289c6c6783c` | `artifacts/development/hyperflow-gpu-49deb21eba` | four compatibility-loader nodes in stage chains, MP4 output, Core 73.34 s |
| Formal Director 8+4, 448² | `c4f457a7-c486-49a1-83e0-9c135e96fcf6` | `artifacts/development/hyperflow-gpu-9467d3abc3` | 24 frames with full H.264/AAC decode, owned Core stopped |
| Final-source Director 8+4 regression, 448² | `cad8e9de-5130-4c23-8398-9152947d9883` | `artifacts/development/hyperflow-gpu-78c22bdbb9` | 24-frame H.264/AAC full decode; decoded RGB and PCM hashes exactly match the prior formal 8+4 baseline |
| Formal Director partial4+4, 0.4 MP 16:9 | `37d48ddb-1c49-44e8-816c-37235272a5a1` | `artifacts/development/hyperflow-gpu-3cebf8b894` | 832×480/24 frames, two 4-NFE stages, full H.264/AAC decode; owned Core stopped |
| Formal Director partial4+4, 0.5 MP 9:16 | `09e9375d-25e2-40eb-874b-2d7b6d3d47a6` | `artifacts/development/hyperflow-gpu-85b41f6d77` | actual 544×960/24 frames (0.522 MP, 32-grid), full H.264/AAC decode; owned Core stopped |
| Formal Director 8+4, two **different** content LoRAs | `5c52792c-f20e-4e1b-8f04-cb720db96c3b` | `artifacts/development/hyperflow-gpu-a2a82743be` | Five View + H3-World each in LOW/HIGH chains, 448²/24-frame H.264/AAC fully decoded; paired no-content output has different decoded RGB and PCM hashes |
| Formal real-model 1+7 parity | `2d0f1a01-bcdd-46d8-ace7-d11b4d03fdb4` | `artifacts/development/hyperflow-gpu-9918ab55a0` | final video/audio latent max_abs = 0 versus single8; two independent Loader owners |
| Formal real-model 7+1 parity | `c43112a9-c843-4667-86a7-135e1e7766c8` | `artifacts/development/hyperflow-gpu-91aa9e8889` | final video/audio latent max_abs = 0 versus single8; two independent Loader owners |

The first four-slot structural probe used the same Five View file twice per
stage at small strengths; it did not establish diversity. A later owned-Core
probe loaded a second installed H3-World adapter in both stages and completed
the full AV graph. Compared with the no-content same-seed 448² Director probe,
decoded RGB and PCM hashes differed. This is combined-patch execution evidence,
**not** isolated causal effect or quality of each individual adapter. The Core
dynamic-loader log reports 210 unique patch targets; overlapping content and
HyperFlow targets mean that count alone cannot prove every content patch's
individual contribution. Stage-isolated effect measurements remain unverified.

The first attempted two-loader split with the default comfy-aimdo compiler
failed inside native `malloc_graph_pop`/`malloc_graph_abort` (Windows access
violation / `aimdo memory compile error`):
`artifacts/development/hyperflow-gpu-1972ee24ce`. The identical split and
subsequent two-loader routes succeeded in an isolated Core started with
`--disable-comfy-compiler`; see each `server-command.json` and stderr. Core
`main.py` logs show comfy-aimdo initialized and DynamicVRAM staging, so the
observed tested flag combination was `aimdo_enabled=True` and
`args.disable_comfy_compiler=True` for success, vs True/False on failure
(derived from command, Core source and log, not a live introspection endpoint).
Director rejects dual-branch HyperFlow under the known unsafe flag pair; it
does not globally alter Core settings or silently change recipes.

An earlier continuation implementation round-tripped the partial nonterminal
latent through Core's generic clean-latent restart. That produced real
single8/split final differences (video max_abs 0.45708, audio 0.02118;
`artifacts/development/hyperflow-gpu-2192d5223a`). It was corrected by
capturing the LOW sampler's exact model-space `x_sigma` and injecting it into
the HIGH typed sampler. After correction, both same-owner and distinct-owner
GPU comparisons gave exact zero difference; the failed receipt is retained
so the reason for the direct-state handoff is auditable.

## Gates still open

导演台任务／声音、D3 与素材叠加的逐项源码门禁审核及待验收矩阵见 [HyperFlow 导演台组合审核](HYPERFLOW_DIRECTOR_COMBINATION_AUDIT.md)。该审核只补 CPU 拒绝回归，不把未跑的组合升格为 GPU 或人审通过。

- Human review of the entire video and audio (not just MP4 integrity or the
  8+4 first frame) is outstanding. Do not present this experiment as a
  picture-quality or speed improvement over Turbo/standard 4+4.
- Ref2VA, FL2VA, Drive Audio, Hybrid, soft/binary masks on full GPU, longer
  clips, and stage-isolated effects of different content-LoRA pairs have not
  completed the same runtime/quality matrix. The 0.4 MP landscape and 0.5 MP portrait runs above
  cover only partial4+4 T2VA/native, not every route/aspect combination.
  Native soft-mask row mapping has CPU tiny coverage only.
- The dedicated original weights are required. No automatic download,
  pruned fallback, generic-LoRA substitution or automatic Turbo stacking is
  permitted. If Core's allocator/compiler behavior changes, rerun the
  two-loader GPU matrix before removing the Director safety gate.
- The legacy native4+4 dual-model long-video node now explicitly rejects a
  HyperFlow-attached MODEL before hashing/cache/sampling. This prevents a
  user from silently sending the two-time owner through native-flow sigmas;
  it does **not** implement or qualify a HyperFlow long-video route.

Final formal scoped Python regression completed 435 passed / 3 optional skipped;
the 102-case Director JS state suite passed. `tests/test_hyperflow_advanced.py`
and `tests/test_director_generation.py` cover converter errors, the fixed AV grid,
typed-plan ownership, 1+7/4+4/7+1 toy x_sigma continuation, native 50+2
tiny-model mixed masks, A→B→A owner patch lifecycle, generic-HyperFlow
misrouting and content-time patch warnings, and Director legacy separation.
Ruff passed for the scoped implementation and probe script.

## 2026-09-22 formal follow-up: modes, dimensions and long-film boundary

Dedicated manual-node **single8** GPU also completed two additional conditioned modes using the full non-pruned H3 base: distinct first/last FL2VA at `artifacts/development/hyperflow-gpu-8e8c1f1fb4/` (22 frames, 256², full H.264/AAC decode; movie SHA `2be1052183bc8019f97549ab7224269aaff9a72ad98ec72c5c9c4941acb9f351`) and Ref2VA at `artifacts/development/hyperflow-gpu-f501c545cb/` (same format; SHA `d61b7723d28d2fbb1a9bb92eea1d2fa669449ba06f2b0cda096bc7a1a094aa55`). These prove the **manual dedicated nodes** can execute those fixed inputs; the Director HyperFlow compiler still deliberately gates to T2VA/native, and image/reference fidelity and audio quality have not been human-approved.

The formal Director **8+4**, in addition to partial4+4 above, completed the same exact 32-grid output sizes: 832×480/24 frames at `artifacts/development/hyperflow-gpu-1a0943109d/` (SHA `ac11cee189542b770d92ec99d7d071ad321e371e7cff6ca361cf7b39b755aa14`) and 544×960/24 frames at `artifacts/development/hyperflow-gpu-4e716f106f/` (SHA `4584fe146f44dde4d847b3c95e508c897f802b7053b6f1dda61c34f5a5dff0d3`). Both audits fully decode video/audio; neither proves an improvement over single8 or old 4+4.

P7 is a **separate** opt-in node, not a silent extension of the Director recipe or the legacy long-video node. Current-source T2VA/native 8-second two-segment GPU evidence, full AV audit, completed-chain cross-Core copy recovery and interrupted-chain HIGH-only resumption are in `docs/HYPERFLOW_LONG_VIDEO_EXP.md`. The new route's 896×448 example has no active content LoRA, so do not borrow the short Director two-content-LoRA result as a P7 multi-LoRA quality claim. The prior formal 435/3 Python and 102 JS numbers above remain their original test scope; later expanded regression and HR1 hardening have separate counts in the top ROADMAP. Human full-film/voice/seam review remains open.
