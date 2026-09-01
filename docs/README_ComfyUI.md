# MiniMax-H3 Turbo 4-step LoRA — ComfyUI conversion

> 2026-09-02 v1.64.0 MV Vocal Lock V3 official Ref2V correction: the current recommended workflow uses the
> official `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` at strength 1.0 with four
> Euler/simple steps, shifts 12/3, and 1024x768 output. A same-image, same-audio, same-failing-seed
> comparison removed the persistent double-face ghosting. This supersedes the earlier attribution to
> seed choice or unavoidable base-model reprojection: the failed r1-r3 route had combined a generic
> LarryVrh EMA Turbo LoRA with a non-official eight-step/shift-6:3 Ref2VA schedule.
>
> A real 32-second/five-scene r4 run then completed 5/5 scenes and 768/768 frames at 24fps. The complete
> song remained outside H3 and was muxed once after assembly. Strict video-only, audio-only, and combined
> decode passed; default multithreaded video decode repeated 20 times with zero anomalies after the final
> all-intra baseline packaging correction. Agent review of all scenes at 2fps found no duplicate face,
> background face, persistent halo, or obvious subject-edge smear. Official SyncNet measured isolated-
> vocal offsets `0/-1/0/-1/0` frames at 25fps, while a 400ms delayed-video control measured nine frames.
> The 32-second mechanical and proxy-visual gates pass. The user then completed review, reported that
> the 32-second result had no problem and was perfect, and explicitly removed the approximately
> 90-second requirement. Final human acceptance is bound to master SHA-256
> `e833277844e6980fdeacf9bdfd5c61ffe48aefdb3e1eba6869c363777b7dd75f`. Manifest `accepted` still
> means mechanically saved and contract-bound for future material; this specific master additionally
> has explicit human approval.
> No product node calls `/prompt` or any remote API.

> 2026-09-01 fully local MV / lip-scene route: three append-only Advanced EXP nodes analyze a
> complete local song on CPU, compile deterministic Ref2VA prompts, render scenes strictly serially
> through the connected local H3 `MODEL`, resume accepted scenes, and mux the complete original song
> once after video assembly. The implementation does not submit `/prompt` or call remote H3, LLM,
> TTS, music, or video APIs. This is audio-conditioned H3 performance orchestration rather than a
> phoneme-level lip solver; final lip motion, identity, acting and image quality require full-speed
> human review. Use the dated workflow under `examples/workflows/24-mv-lipsync`.

> 2026-08-30 FlashVSR v1.1 post-processing: three append-only Advanced EXP nodes load the
> official model folder, compile an explicit quality/memory plan, and restore decoded H3 frames
> at 2x or 4x while returning the exact original AUDIO object. `quality_locked` keeps the public
> LCSA `2.0/3.0/11` budget. `balanced_dynamic_exp` changes only eligible interior low-motion
> chunks and always protects the first, last and high-motion chunks. `memory_safe` keeps the fixed
> budget and uses same-seed feathered tiles plus staged offload. The LCSA mask is dispatched to a
> separately installed `spas_sage_attn` block-sparse Sage2 kernel; absence or incompatibility is an
> actionable error, never a silent dense fallback. Models are checked by required structure and
> loadability, not filename hash, byte size or pixel area. Official FlashVSR primarily targets 4x;
> the bundled 2x workflows remain conservative experiments and cannot recover missing identity,
> lip sync or source detail.

> 2026-08-27 v1.52.2 canvas policy: `1920x1088` is a warning/reference area, not a hard
> execution cap. Conditioning, Source AV, Long Video, Multi-Keyframe, Still Image, SPEED,
> Prompt Relay resource estimation and Environment Audit allow larger 32-aligned canvases and
> report user-owned VRAM/runtime/OOM risk. Existing `allow_above_reference_area` inputs remain only
> for old-workflow schema compatibility and no longer gate execution.

> 2026-08-29 PDD integration: the existing Advanced EXP node now prefers ComfyUI's official native
> PDD FinalLayer when its runtime semantics are present, and keeps the reviewed dynamic fallback for
> older cores. The converted files still require the dedicated node because they contain 258 backbone
> adapters plus four custom absolute 32-head banks. The node converts those banks to the official
> first-head-plus-offset padded-diff layout without using a core-version, model-hash or file-size gate.
> FL2VA and Ref2VA both completed serial native 736x416x22 real renders on official ComfyUI e7051b0,
> with exact 8 NFE/block 0-7 selection and strict finite H.264/AAC decode. Minimum free VRAM was
> 482/633MiB, so no universal 16GiB claim is made; earlier full-length fallback results remain valid.

> 2026-08-30 H3 Super low-Sigma route: use the independent identity-preserve workflow when
> the official LTX Stage-2 full denoise changes faces too much. Its exact schedule is
> `0.5 -> 0.412 -> 0.350 -> 0` (three Euler updates), with Dense Attention as the default.
> The official parity workflow is unchanged, and H3 audio continues to bypass LTX Stage 2.

> Frontend workflows are organized by purpose under `examples/workflows/`, including the new
> `19-pdd-acceleration` category. Each category contains an independent `README.md` with
> purpose, validated outcomes, usage guidance and explicit limitations. The same hierarchy is
> mirrored into the installed `MiniMax H3 T8` user-workflow menu; dated JSON filenames and graph
> contents are preserved.

> 2026-08-26 SLA quality correction: the append-only Turbo/SLA Profile Router now defaults to the
> corrected ordinary Alpha8 Turbo LoRA at eight NFE and 12/3 shifts. A serial 736x416x124 rerun using
> the difficult close-person to aerial final-frame transition strictly decoded, but full human review
> rejected the result because it entered a persistent forced scene/scale transition after about one
> second. Its runtime report contains eight ordinary Turbo forwards and zero SLA calls, so that clip is
> evidence of incompatible FL2VA anchors, not an SLA-kernel failure or success. The recommended workflow
> now repeats a same-scale anchor by default and tells users to replace it only with a compatible final
> frame. Its minimum observed free VRAM was only 418MiB, below this project's 512MiB gate. The SLA
> exact profile remains four model evaluations, 6/3 shifts and 85-percent dynamic sparsity: the official
> LightX2V `infer_steps=5` value denotes five sigma grid points, not five model evaluations. Released
> evidence covers the BF16 checkpoint family and LightX2V's FP8 recipe, not the local INT8 ConvRot base;
> the new quality-oriented exact profile therefore refuses INT8 instead of silently calling it upstream
> parity. Legacy SLA nodes remain loadable for diagnostics. The user-supplied file named `124f` in the
> latest report actually probes as 704x416, 22 frames and 0.9167 seconds, so its filename cannot support
> a conclusion about failure after one second.

> The current local 1.45.0 candidate appends a fail-closed Prompt Semantic Contract Audit as node
> 190 after the complete prior 189-node prefix, then appends a read-only NFE Run Contract compiler
> as node 191 without moving any prior ID. It checks only user-authored required/forbidden
> phrase groups plus exact dialogue and media-tag preservation. Empty anchors ABSTAIN; a mechanical
> pass still returns the original prompt until the user explicitly accepts the reviewed candidate.
> The captured `turns` to `stands still` provider regression is rejected. The current source has
> 133 workflows and 191 unique runtime nodes; the first 190 IDs exactly match the preceding package.
> The prior `final27` ZIP is a historical snapshot from before the Prompt Budget official/local
> boundary correction. `final29` is now historical because the Provider Router gained a
> fail-closed diagnostic for Ollama models that return only `message.thinking` without final
> `message.content`. The refreshed local-only `final30` package contains 297 entries, 132
> workflows, six Quick Start subgraphs and 190 nodes. That package is now historical: the current
> source adds the run-contract compiler and corrected NFE example wiring while preserving all 190
> earlier positions. Its exact filename, size and
> SHA remain in excluded local verification evidence. The complete source passes 1,140 CPU tests;
> this lexical audit and thinking-only refusal are not a universal semantic
> equivalence or prompt-quality claim.

> One further append-only Advanced EXP node now adds exact step-boundary checkpoint/resume for the
> project's `dual_clock_euler` sampler with the `native_flow` schedule. It is node 188 and defaults
> to disabled/no-write. Opt-in writes atomically preserve the completed packed joint-AV Euler state,
> original processed noise and latent, optional packed mask, and full sigma schedule in no-pickle
> safetensors. Resume validates the exact operator-declared model/run contracts, seed, runtime patch
> structure, AV layout, shifts, audio-velocity protocol and tensor digests before exposing only the
> remaining sigmas. A true two-process CPU split run matches the uninterrupted control bit-for-bit.
> This support deliberately excludes multistep-history samplers, ancestral/SDE RNG recovery,
> third-party sampler state and interruption inside a model forward; real H3 restart media equivalence
> remains unclaimed.
> The clean final package audit contains 292 entries, 131 frontend workflows, six Quick Start
> subgraphs and 188 unique runtime nodes. Isolated extraction preserved the exact 187-node prefix
> from the prior RAVEN package and appended only the NFE resume setup.

> The current local append-only RAVEN integration adds nodes 185-187 without changing the first
> 184 IDs or stable sampling. It does not reimplement RAVEN. The Profile node emits one exact
> parameter set to both the audit and the separately installed external sampler; the Guarded Loader
> checks plugin/model/CUDA/BF16 and the reviewed memory envelope before delegating; the Request
> Audit calls the external runtime's own T2VA conditioning, empty-latent and causal-MODEL contracts.
> The current 16GB GPU / 128GB host is deliberately outside the default gate, so no local real
> RAVEN generation, quality result or OOM-safety claim exists.

> The six Quick Start subgraph files use dated ASCII filenames while keeping their bilingual graph
> titles and NOTE text. This avoids a reproduced Registry-pack omission of Unicode paths on a
> Windows/GBK console; it does not change any subgraph graph, node schema, default, or saved stable
> workflow.

> Project v1.45.0 appends twenty-one opt-in planning/report/concat/provider/checkpoint nodes after all 163 v1.44 IDs: Audio
> Integrity Audit, Audio Perceptual Drift Audit, Speaker Routing Audit, and Prompt Budget + Role Compiler. They never edit audio,
> reassign a speaker, or truncate a prompt; the prompt compiler now preserves leading/trailing whitespace too. Risk findings return `ABSTAIN`; exact tokenizer counts
> are labelled exact only when the connected CLIP exposes countable token IDs. Four dated
> importable workflows include NOTE guidance. Real H3 threshold/listening calibration remains open,
> so these nodes do not claim to repair model leakage, pops, tail wrap, or voice swapping.
> A real CPU-only Qwen3-VL 8B/Boogu tokenizer run then compiled mixed Chinese/English text,
> `<Picture 1>`, `<Video 1>`, `<Audio 1>` and one explicit role binding without truncation. It
> returned `PASS`, 140 exact tokens versus a 153-token planning estimate, no findings and no
> warnings. Exact 7000-character PASS, 7001-character non-mutating ABSTAIN and a three-subject
> media contract also pass. The default 7000 ceiling matches the current official MiniMax H3 CLI
> submission contract. The validated ComfyUI MiniMax tokenizer has no 7000-character guard, so the
> official CLI submission rule is not described as a local open-weight tokenizer hard limit or architectural
> quality guarantee. Raising the local audit ceiling reports an official-compatibility warning and
> never raises the current official CLI limit. The connected count covers compiled text only; H3 Conditioning
> later inserts visual and timestamp tokens.
> The appended Prompt Provider Router keeps local passthrough as its no-network default and can
> explicitly call OpenAI-compatible servers (OpenAI, LM Studio or llama.cpp) or Ollama using the
> same pinned three-field H3 rewrite contract. Every request requires confirmation; non-loopback
> endpoints additionally require explicit remote permission, HTTPS and an API key read only from
> an environment variable. I2VA/L2VA/FL2VA upload only bounded JPEG keyframes, never raw audio.
> Ollama defaults `keep_alive=0`; OpenAI-compatible servers have no standardized unload call.
> CPU-only Ollama 0.32.15 probes with `deepseek-r1:1.5b` and `deepseek-r1:8b` both reached the real
> native chat endpoint. The 1.5B model exhausted 768 tokens, omitted two fields and damaged the
> Chinese `<d>` block; the 8B model returned all three fields but removed the dialogue. The default
> strict validator rejected both, and Ollama's loaded-model list was empty after each
> `keep_alive=0` response. A real slow-loopback test also changed response reading to one socket
> read per cancellation checkpoint and normalizes header/first-byte stalls to a bounded timeout.
> These are fail-closed and lifecycle results, not a provider-quality pass.
> The provider path now replaces each source `<d>...</d>` block with a deterministic opaque token
> before serialization. It restores a literal only when that exact token appears once inside
> `integrated_multimodal_description`; missing, edited, duplicated or misplaced tokens fail closed.
> A further real 8B run confirmed that the raw Chinese dialogue was not uploaded, but that model
> omitted the token and was therefore rejected with zero restoration. This improves privacy and
> prevents corruption; it does not manufacture compliant output from an unsuitable model.
> An appended `contract_repair_attempts` control can make zero to two text-only correction
> requests after deterministic validation fails. It defaults to zero, never resends reference
> images, never restores dialogue before validation, and never relaxes the final contract. With
> Ollama `keep_alive=0`, each repair can reload the model; enable it only when that cost is intended.
> A subsequent isolated CPU-only `deepseek-r1:8b` request passed the strict three-field and exact
> Chinese-dialogue contract in 81.964 seconds without needing repair; `ollama ps` was empty after
> the response and the isolated listener was stopped. It changed the requested turning action into
> standing still, so this is a positive structural contract result, not a semantic-quality pass.
> The reference-relative drift audit uses aligned 500ms windows and 24 gain-normalized log-power
> bands, ignores reference windows below -50dBFS, and requires three persistent windows. Against
> the previously human-reviewed Motion Recovery files, pass-1 returned PASS, the user-rejected
> pure pass-2 returned ABSTAIN over 1.4-3.6s, and the accepted 80% pass-1 blend returned PASS. This
> one-case calibration is a review cue only; it cannot prove distance, reverb, identity or cause.
> The same local batch now adds nine Creator Workspace/runtime nodes over the existing Studio Timeline: a
> non-destructive shot override, run-window/sidecar compiler, explicit shot/variant selector and
> pixel-preserving CPU A/B preview, plus an immutable run receipt and deterministic resume planner.
> They do not queue jobs, write/delete files, load models, replace
> Conditioning or alter sampler math. One real-data API session compiled a three-shot 22/22/39
> timeline, emitted three deterministic shot-B variants and a five-frame hold map, selected variant
> 2, and compared two actual 39-frame H3 candidates without changing source pixels. The labelled
> MP4 passed strict decode and peak VRAM did not rise above the 1,156.5MiB service baseline. Human
> usability and human A/B judgement remain open. Retention candidate classification is now
> implemented as a non-destructive plan; any authorized filesystem executor remains open. One separate low-load H3
> active-sampler cancellation/release probe has passed; resumed media is still open.
> One isolated CPU API graph recorded `completed` then `accepted` for the first shot and selected
> the second shot for rendering. An identical requeue reported all dependent nodes cached, while
> changing only the base seed invalidated them. This validates ordinary graph-cache identity, not
> H3-internal resume or automatic cache detection; receipt outcomes and cache observations remain
> explicit external facts.
> An additional opt-in pair binds the Creator workspace hash to the existing Long Video background
> manager and selects shots/seeds from accepted_count/retry_count. It reuses the established
> targeted queue deletion, running-prompt interrupt, history retry, process lease and accepted
> manifest; it does not add a second queue. `review_only` is the default, cross-workspace chain reuse
> fails closed, and a complete Long Video Candidate Save/Auto Accept terminal is mandatory. The
> adapter passed CPU/schema/frontend tests and one low-load live H3 cancellation/release run.
> A separate isolated CPU PromptServer run did make the bound prompt enter `running` and then used
> the real cancel route. It signalled interruption to the exact prompt ID, history emitted
> `execution_interrupted`, accepted/retry stayed zero and both queues emptied. This validates the
> CPU runtime cancellation plumbing. A later 256x256x22 H3 run cancelled the real sampler at
> progress 1/4, recorded `unload_all_models`, emptied both queues and returned observed device use
> to baseline+90MiB. Resumed media and automatic-accept quality remain open.
> The run-receipt workflow now ends with an additional non-destructive Retention Plan node. It
> validates the immutable ledger, applies each shot's retention policy and emits separate keep and
> proposed-delete manifests. Completed-but-unreviewed candidates are always kept; ambiguous accepted
> runs, keep/delete path overlap, or delete candidates without an explicit `path` fail closed. The
> path-review switch defaults off. Even when enabled, the node only reports
> `READY_FOR_EXTERNAL_EXECUTOR`: it never resolves, opens, moves or deletes a path, and this project
> does not include a destructive executor.
> A separate native ComfyUI workflow now loads two synchronized videos, preserves the existing
> pixel-safe visual comparison, saves a deliberately silent side-by-side review video, exposes
> independent A/B audio players and routes the same-content tracks through the reference-relative
> drift audit. The real 39-frame pair produced a 520x288/24fps silent comparison that passed strict
> video decode three of three times; both audio previews were emitted and the audit returned PASS
> with waveform correlation 0.9842, spectral-drift p90 0.0735 and level-delta p90 0.2308dB.
> This closes the mechanical audio-review path only; no human winner or perceptual equivalence is inferred.
> The reviewer later rejected both short replacement pages as `unsure`: about one second without a
> clear human face was not judgeable. The final replacement therefore uses SHA-locked `10A.jpg`,
> 1088x544x124 I2VA, eight NFE, shifts 12/3, one close Mandarin utterance and the same seed. The
> 4B/8B/native-32B arms each produce 5.167-second media. Creator reuses one 124-frame latent twice and
> compares one-decode native concat against separate-decode composition as exact 243-frame,
> 324,000-sample, 10.125-second arms. The final Creator and 4B/8B pairs were both marked assessable
> and tied in every reviewed visual/audio dimension with no blocking failure. A later controlled
> 4B/native-32B pair was also an assessable all-dimension tie; the reviewer noted that the outputs
> were not identical but felt similar on this simple task. Native 32B was run only after the unchanged
> 14,500MiB start gate passed and retained 643MiB minimum sampled free VRAM.
> A no-model objective pass found zero black/white/frozen-transition frames in all four arms.
> Creator one-decode has a lower numeric join-frame discontinuity than separate decode, but both
> tracks fall from dialogue tail to near-silence around 5.17 seconds. ClipProj 4B/8B video remains
> relatively similar while their audio differs materially in correlation and level. These signals
> guide where to inspect/listen; they are deliberately not converted into an automatic winner.
> A final convenience page combines the Creator and ClipProj long pairs while preserving independent
> keyed A/B randomization and assessability. It exports one JSON after both groups are reviewed;
> the private mapping remains outside the public HTML. These single-reviewer ties close only the fixed
> simple portrait gates; they do not establish cross-material equivalence, automatic Creator acceptance,
> ClipProj noninferiority or a universal 16GiB safety tier. Native 32B remains the default.
> Two pass-through compatibility audits then inspect separately installed ClipProj and Sol-Attn
> versions, dimensions, hardware and patch ownership without importing a kernel or loading a model.
> One native latent timeline node mechanically joins complete H3 AV latents by removing each later
> segment's repeated two latent steps/five video frames and deriving audio trims from the cumulative
> 24fps/40Hz clocks. It defaults the combined latent to CPU but cannot evict ComfyUI's cached inputs.
> One repeated low-load real H3 probe joined two 256x256x22 Turbo8 T2VA latents and decoded the
> result once. It produced exact 39-frame video and 52,000-sample lossless audio after dropping
> nine 40Hz audio steps/7,200 samples from the second segment. All three MP4 outputs repeated
> byte-for-byte and passed three strict video/audio/combined decodes. Against separately decoded
> composition, boundary video MAD fell from 0.11164 to 0.04269 and the single-sample audio jump
> from 0.02231 to 0.00104, while the adjacent 100ms level gap remained 8.32dB. The independently
> seeded segments still change state, and only 341.11/144.18MiB whole-device headroom remained;
> this is not a seamless, quality-superior, VRAM-saving or 16GiB-safe claim.
> A final append-only Native Latent Resume Manifest node computes an exact chunked SHA-256 over
> complete H3 video/audio latent samples, optional nested AV masks and supported non-volatile
> metadata. Empty expected JSON creates a baseline; supplying the old manifest defaults to an
> error on any content, checkpoint-ID, shape, dtype or mask mismatch. One- and eight-MiB chunking
> produced the same digest in CPU tests, and byte/mask/metadata mutations were detected. The node
> writes no files and cannot recover diffusion-internal NFE state, so this closes only checkpoint
> identity—not storage persistence, perceptual continuity, cache release or crash recovery itself.
> Two final append-only Native Latent Checkpoint Save/Load nodes persist an already-formed complete
> H3 nested AV latent, optional nested masks and supported metadata as no-pickle safetensors under a
> bounded ComfyUI output store. Save defaults to disabled, uses a unique filename, verifies the
> temporary payload before atomic placement, and returns the whole-file SHA plus the exact-content
> manifest. Load always verifies the embedded content manifest and can additionally require the
> independently retained manifest and file SHA. A completed Save in process 25280, process exit,
> then strict Load in process 8884 returned `MATCH_EXTERNAL`; video/audio tensors, dtypes, masks and
> metadata matched exactly. This is completed-latent process replacement, not recovery of an
> interrupted diffusion iteration, Transformer/sampler state, ComfyUI queue, CUDA allocator, or
> perceptual continuation.
> One further append-only Native Latent Continuation Concat node consumes the exact Long Video
> Planner and Conditioning reports that produced a sampled continuation. It verifies chain,
> segment, render geometry, timeline start/end, active motion context and the native motion-
> keyframe count, then removes the full proven 5/22/39-frame context on both the video and
> cumulative audio clocks. A 124-frame timeline plus a 124-frame continuation with 22-frame
> context produces 226 physical frames, video latent T=67 and audio latent T=377; a subsequent
> 39-frame-context continuation produces 311 frames. The safe default requires prior video and
> audio context. All segments remain in latent space for one final AV decode, and any final hidden
> tail is trimmed only after that decode. These CPU contracts do not prove that an arbitrary
> third-party sampler consumed the conditioning, perceptual continuity, lower VRAM or interrupted-
> NFE recovery.
> The real ClipProj 0.1.13 and Sol-Attn 0.6.2 source trees were then installed at fixed commits.
> Header probes identified the existing 8B encoder as Qwen3-VL/Boogu and authenticated the
> 41,990,896-byte v3.1 projection as 4096-to-5120. Real SM89/BF16 hardware plus a synthetic complete
> 50-block owner passed the Sol audit. Two complete 736x416x124 T2VA templates now show the actual
> external-node order. A separate low-load service then ran one fixed-seed 256x256x22, four-NFE
> T2VA per route. ClipProj 8B, native 32B and active Sol all passed strict video/audio/container
> decode. The 8B cold route used about 1201MiB less whole-device peak than the native 32B control in
> this one short probe, but was about 2.7 seconds slower. Sol first entered its kernel at 547 tokens only
> after the mechanical probe temporarily lowered `min_tokens` to 256. A previous
> `dense_percent=0.2` four-step run stayed entirely dense and was byte-identical to the native
> control, so the four-step workflow now defaults that field to zero while production
> `min_tokens` remains 4096. A later same-seed 1152x640x22 T2VA pair kept that production threshold
> and logged `Sol active (5139 tokens)` on 46/50 blocks in strict mode. Sol/dense completed in
> 49.281/41.828 seconds and peaked at 16,004.9/16,008.7MiB; because Sol ran first and dense second,
> the timing remains order/cache-confounded, and neither route cleared the 512MiB headroom gate.
> Both passed separate video/audio/container decode 3/3. Video SSIM was about 0.558 and PCM
> correlation about 0.719, proving a material route change rather than quality or audio parity.
> Human blind review, repeated-memory and general 16GiB gates remain open.
> ClipProj 8B also completed one separate I2VA visual-path probe: the first frame entered both the
> VAE keyframe and Qwen3-VL vision path with `has_reference_images=true`. At 256x256x22, four NFE,
> shifts 12/3 and seed 2608228201 it finished in 36.125 seconds, produced exact 22-frame H264 plus
> 32kHz stereo AAC, and passed strict video/audio/container decode three of three times. Whole-device
> peak was 15,207.4MiB with about 1,172.6MiB headroom. The Chinese/proper-name/short-dialogue prompt
> was encoded. A same-seed native-32B control later completed in 33.718 seconds at a 15,591.9MiB
> peak versus 36.125 seconds and 15,207.4MiB for 8B. Full-video SSIM was about 0.9293 and PCM
> correlation about 0.8984, but both routes showed an unwanted pseudo-Chinese subtitle-like overlay
> and the short utterance was not human-listened or ASR-accepted. This closes mechanical control,
> not parity or a general 16GiB tier.
> A second visual-path probe connected distinct first and last images for FL2VA, kept
> `has_reference_images=true`, and used `<Picture 1>`/`<Picture 2>` for the two anchors. The fixed
> 256x256x22, four-NFE, 12/3-shift run completed in 23.953 seconds and again passed strict
> video/audio/container decode three of three times. Whole-device peak was 15,358.0MiB with about
> 1,022.0MiB headroom. First/last anchor SSIM was about 0.8354/0.5095 after documented transforms;
> these are path-presence diagnostics, not quality scores. Long interpolation, listening, 0.7MP
> and general 16GiB gates remain open.
> A same-seed native-32B FL2VA control then took 31.078 seconds and peaked at 15,670.6MiB versus
> 23.953 seconds and 15,358.0MiB for 8B. Full-video SSIM was about 0.7090; first/last anchor SSIM
> was 0.8354/0.5095 for 8B and 0.8462/0.5302 for 32B. PCM correlation was only about 0.5608 with a
> substantial RMS difference, so listening remains mandatory and no audio non-inferiority claim is made.
> Ref2VA then completed one same-seed Stock20 mechanical comparison without a Turbo LoRA. The 8B
> and native-32B routes both emitted exact 256x256x22 H264/AAC and passed strict decode 3/3. The 8B
> route took 22.907 seconds versus 27.812 seconds, but peaked at 16,318.2MiB versus 16,018.2MiB and
> left only about 61.8MiB headroom. Three-frame SFace means were 0.319/0.302 respectively; neither
> path consistently cleared the project's usual 0.36 suggestion threshold. This closes execution
> and same-seed mechanical control only, not human quality parity, VRAM savings or 16GiB safety.

> Project v1.44.0 appends `MiniMaxH3LightX2VSLAKJSageComposerT8Advanced` after the strict SLA
> loader and runtime audit. It accepts only the complete KJNodes MiniMax H3 50-block Sage forward
> patch and gives each call one owner: SLA apply uses block-sparse Sage2, while dense control and
> non-SLA calls keep KJ Sage. The original SLA node, all first 160 IDs, schemas/defaults and stable
> sampling remain unchanged. A second dated FL2VA workflow documents the exact connection order.
> 966 CPU tests, Ruff, compileall, JSON parsing, the installed KJ source contract and Registry
> validation passed; no new GPU quality, speed, audio or general 16GiB claim is made.

> Project v1.43.0 keeps nodes 1-155 and their schemas/defaults unchanged, then appends five opt-in
> nodes for an external BlockSwap contract, LanPaint AV preparation/compositing, and a local 8B
> prompt rewriter with explicit unload. Six native Quick Start blueprints reuse existing nodes rather
> than changing their contracts. This release also permits face-detector aliases under
> `ComfyUI/models` to resolve through symlinks/junctions, adds an SM120 high-token Sage guard, and
> preserves compatibility with both reviewed MiniMax H3 row-mask/tokenizer core variants. The full
> CPU suite passed 951 tests; Ruff, compileall, 170 non-artifact JSON files, version consistency,
> `git diff --check`, and `comfy node validate` passed. No new GPU stress, BlockSwap/LanPaint quality,
> or universal 16GiB-safety claim is made.

> Project v1.42.0 appends the seventh Motion Recovery node as an automatic native-lazy gate and
> changes only the new analyzer's default to `auto_conservative_exp`. A real calm 736x416x124
> Stock20 clip automatically abstained, requested no pass-2 inputs, and produced byte-identical MP4
> plus identical decoded video/PCM hashes. Separate real I2VA, FL2VA, and Ref2VA 20+10-NFE runs
> completed as 124-frame/24fps/32kHz-stereo media and passed strict decoding. The I2VA run also
> produced `pass1_original`, `pass2_recovered_exp`, and `blend_exp` dialogue tracks; ASR preserved
> the target sentence in all three. Full human listening accepted the exact pass-1 default, rejected
> pure pass-2 recovery because its middle window suddenly sounded distant before returning, and
> accepted `blend_exp` only for this one clip at `pass1_mix=0.8`. Pure pass-2 therefore remains
> diagnostic-only, while blend remains opt-in rather than a new default. Observed whole-device
> headroom fell below 512MiB in all three generation
> routes, so no general 16GiB-safety or perceptual-quality claim is made.

> Project v1.40.0 appends EAV + BlockCache, EAV + STG, and EAV + Long Video as
> nodes 146-148 without changing the first 145 node IDs or stable sampling. Only low-load
> deterministic contracts, registration, importable workflows, and project/user SHA parity were
> checked; pressure, repeated-memory, quality, listening, speed, and general 16GiB claims remain
> explicitly unvalidated.
>
> Project v1.39.2 appends an isolated Enhance-A-Video + Prompt Relay composer as node 145. A
> standalone EAV model and standalone Relay model cannot be stacked because both own the same H3
> diffusion wrapper and optimized-attention entry. The composer authenticates the existing Relay
> binding, executes Relay routing first, then applies FETA gain only to target-video output rows in
> the same attention call. It adds no model forward and `disabled` preserves the exact Relay MODEL.
> This release includes one Stock20 T2VA frontend template and deterministic contract/runtime-audit
> handoff tests; real 0.7MP quality, listening, repeated-memory and general 16GiB gates remain open.

> Project v1.39.1 republishes the five learned-latent two-pass frontend templates as one coherent
> 4+4 workflow set: standard I2VA, native speech, `lock_source`, `remix_source=0.20`, and
> `reference_only`. Every graph uses four low-resolution plus four high-resolution joint-AV model
> calls, the high-resolution Conditioning follows the learned upscaler's aligned width/height, and
> the upscaler reports rather than prohibits outputs above the former 2MP reference area. This is a
> workflow/packaging release; it does not change the 144-node runtime registry or stable sampler.

> Project v1.39.0 appends an isolated Enhance-A-Video + Strict Sage composer without changing the
> preceding 143 node IDs, old schemas, or stable sampling. The composer owns the H3 FETA route and
> `sageattention.sageattn` HND backend, refuses external full-block attention patches, and never
> silently falls back to PyTorch attention. One real 1152x640x124 Stock20 T2VA probe completed
> 20x50 FETA measurements and 20x50 successful Sage calls with zero failures/fallbacks; the H.264/AAC
> output passed three strict video/audio/combined decodes. This single mechanical result does not
> establish better image quality, audio non-inferiority, acceleration, lower VRAM use, or universal
> 16GiB safety.

> Project v1.38.2 completes one real 1152x640x124 Stock20 Ref2VA and one task-Hybrid
> disabled/apply EAV pair. Both apply paths completed 20x50 runtime audits and all accepted media
> passed deterministic strict video/audio decoding. Ref2VA apply left about 417MiB whole-device
> headroom, below the 512MiB project floor, so visual superiority, audio non-inferiority and general
> 16GiB safety remain explicitly unproven. This patch also pins the offline strict decoder to one
> thread; no runtime node, schema, workflow or stable sampler changed.

> Project v1.38.1 adds deterministic Ref2VA/task-Hybrid reference-composer probe builders and a
> hash-traceable anonymous A/B review packager. These are validation tools only: they do not change
> the 143-node runtime contract, stable sampling, or existing workflows, and they do not turn the
> still-incomplete real 0.7MP reference-task matrix into a quality, audio, or 16GiB-safety claim.

> Project v1.38.0 appends an isolated Stock20 Enhance-A-Video Reference Composer for native
> Ref2VA and task-Hybrid packed layouts. The original EAV node keeps its previous schema and
> fail-closed reference behavior. The new route validates the exact native reference segment order
> and sizes, computes CFI from target-video Q/K only, and directly scales only target-video output
> rows. Two importable workflows and deterministic routing tests are included; real 0.7MP reference
> A/B, audio non-inferiority, visual superiority and general 16GiB safety remain unproven.

> Project v1.37.1 updates learned latent two-pass generation to a default 4+4 schedule, so the
> low- and high-resolution stages execute eight joint AV Transformer calls in total. The standard
> Mandarin speech probe completed at 1472x832x124 with strict video/audio decode and intelligible
> speech. Learned upscale no longer imposes a 2MP execution ban. Since v1.52.2, larger outputs no
> longer require a Conditioning opt-in and remain subject to user-owned VRAM and runtime risk. Saved
> workflows that explicitly use three or five refine calls remain compatible.

> Project v1.37.0 appends two isolated Enhance-A-Video / FETA Advanced nodes and five importable
> 0.7MP workflows without changing the preceding 140 node IDs or stable sampler mathematics. The
> clean-room H3 adapter computes the paper's temporal CFI from target-video Q/K, delegates actual
> attention to the existing backend, and directly scales only target-video output rows. Controlled
> Stock20 T2VA/I2VA/FL2VA/L2VA and strict corrected-Alpha8 Turbo8 T2VA pairs passed runtime audits
> and strict media decoding. They do not establish a stable visual-quality advantage, audio
> non-inferiority, or universal 16GiB safety; unsupported wrapper combinations remain fail-closed.

> Project v1.36.2 repairs the learned two-pass custom sampler at a partial schedule start. Comfy's
> generic KSAMPLER initially applies the video sigma to the whole packed AV latent; at pass-2 video
> sigma `0.9035`, H3 shift 12/3 requires audio sigma about `0.701`. Before the first model call the
> custom dual-clock sampler now reconstructs only the audio slice on that audio clock from the actual
> KSAMPLER noise/latent terms. The native I2VA example again lets pass 2 complete joint AV and connects
> its output directly to AV Decode. Same prompt/seed/model native Euler and repaired custom outputs
> were both judged normal by the user; their decoded PCM correlation is about `0.9491`.

> The v1.36.1 `first_pass + 0.0` Audio Audit path remains available only as an explicit lock tool.
> Although it proved latent preservation, listening showed that it froze an unfinished pass-1 audio
> estimate and sounded wrong. It is therefore removed from the native default graph. This correction
> was followed by a controlled speech run using one image, one 5.152s licensed LibriSpeech source,
> seed `2608215001`, and the same 4+3 learned two-pass schedule. `lock_source`,
> `remix_source=0.20`, `reference_only`, and native speech all produced valid 1472x832/24fps/32kHz
> stereo files, and the user completed full audio/lip review with no issues. This closes the four-mode
> gate only for that material and reviewer; it is not a universal lip-sync, voice-quality, or 16GB claim.

> Project v1.36.0 adds an append-only MiniMax H3 Prompt Relay suite: authenticated event plans,
> model-free timeline preview and resource estimates, H3 packed-attention routing, Studio Packet and
> Long Video bridges, and eleven generation templates covering T2VA/I2VA/FL2VA/L2VA/Ref2VA/Hybrid,
> reference video with its matching soundtrack, standalone reference audio, joint AV and corrected
> Alpha8 Turbo8 ordering. Eight controlled baseline/Relay pairs produced sixteen strict-decode-clean
> 736x416x124 AV files. One reviewer found the reference-video pair approximately equal and the
> standalone-reference-audio Relay clip somewhat better in prompt adherence; this remains a
> single-material result, not a general quality, audio, speed or 16GB-safety claim. Stable sampler
> mathematics and existing node schemas remain unchanged.

> The current unreleased prompt-tag compatibility fix preserves strict failure for ambiguous or
> explicitly disconnected reference tags, while accepting three mechanically unambiguous legacy
> cases: zero-based ordinals, stale ordinals when exactly one same-type medium is connected, and
> plain numbered media prose when no reference medium of that type exists. Every automatic mapping
> or prose fallback is surfaced in the node report. No node schema, input order, conditioning
> layout, or sampler math changes.

> Project v1.35.1 keeps the learned-latent nodes isolated and fixes the shipped I2VA graph against
> upstream workflow commit `64fc9d4`. The only user size control is now the learned upscaler's
> `scale_by=2.0`; its aligned width/height outputs directly drive high-resolution Conditioning.
> The graph uses native shift 12/3, `simple8` low calls 0-3, raw refine sigmas
> `0.9035,0.6316,0.3158,0`, and the SHA-verified `comfyui_alpha8` LightX2V LoRA through
> `LoraLoaderBypassModelOnly`. The superseded plain conversion is explicitly rejected because it
> applies about 16x excessive update. One 736x416 to 1472x832 I2VA run completed all seven joint AV
> forwards and produced a strict-decode-clean 124-frame, 24fps, 32kHz-stereo H.265 output. This is
> one-case mechanical and visual-collapse evidence only: broader perceptual superiority and
> universal 16GB safety remain unproven. Stable interpolation and sampler math are unchanged.

> Project v1.33.1 records the completed SPEED blind review and formal 100-clip calibration before
> any further promotion. T2VA,
> FL2VA and Ref2VA all preferred the full-resolution baseline for overall quality, motion and audio;
> Ref2VA reference adherence also preferred baseline. The reviewer explicitly identified FL2VA
> SPEED A and Ref2VA SPEED B as visibly broken. The exact manual 14-coarse/6-full schedules are
> therefore rejected as stable defaults even though their exact-profile speedups remain real.
> Five append-only Advanced nodes now add an aspect-safe 24fps/17n+5 calibration window, accumulate
> cross-batch spatial spectra after independent natural-video windows pass through the target H3 video
> VAE, save/load one atomic sufficient-statistics file, finalize only provenance- and diversity-reviewed
> datasets with at least 100 actual unique clips, and bind a profile to full-file model/VAE SHA-256 fingerprints.
> Load follows complete file content rather than stale ComfyUI cache. No calibrated quality fix is claimed yet.
> A read-only source-curation tool additionally checks metadata, strict decode, exact file/decoded-window
> duplicates and heuristic temporal near-duplicates. Its first scan of 188 MiniMaxH3 outputs left only
> 3 provisional candidates, 55 manual-review files and 130 rejected files; it therefore prevents the
> existing A/B and derived outputs from being counted as a 100-clip calibration dataset.
> A broader five-root H3 scan found 1,275 files but still only 3 provisional candidates. A separately
> curated 3-image x 3-seed I2VA Stock20 probe accumulated 9/9 clips and fitted A=29.5579671,
> beta=2.3662343, R-squared=0.9971147; it remains `research_probe_only` because 9 is below 100.
> A strict scan of 358 local input videos retained 19 provisional candidates. All 19 were encoded as a
> local-video-corpus proxy and fitted A=26.5381849, beta=2.1534428, R-squared=0.9955738, but this is
> still not a formally reviewed 100-natural-video corpus encoded by the exact H3 VAE and therefore
> cannot drive a new quality A/B.
> Dataset profiles are additionally bound to the exact H3 video latent C/T/H/W grid; another
> resolution or 17n+5 duration fails closed instead of reusing the fit. Embedded ComfyUI metadata
> curation hashes provenance/content identifiers without emitting raw prompts, workflows or paths.
> The restored formal corpus now contains 100 reviewed windows from pinned VChitect-T2V-Dataverse
> sources encoded through the exact H3 video VAE. It fitted A=29.9641867, beta=2.3183721 and
> R-squared=0.9951512 and passed the dataset-profile contract. One controlled 736x416x124 Stock20
> T2VA comparison nevertheless failed the product gates: full resolution took 243.203s at
> 12504.6MiB peak, while calibrated SPEED took 248.688s at 16175.8MiB peak and retained only about
> 203.7MiB headroom. Its original H.264 also failed strict decoding on one frame; a non-overwriting
> re-encode exists only for optional blind inspection. The current implementation therefore remains
> EXP and cannot claim acceleration, memory safety, quality, or audio non-inferiority.
> The dated T2VA SPEED frontend workflow now loads the formal 100-clip dataset, recomputes exact
> model/VAE fingerprints and runs `delta_optimal`, with three visible notes preserving the failed
> speed, memory, decode and blind-review verdict. The other SPEED task workflows are labeled as
> historical mechanical examples because no route-specific formal profile exists. All ten project
> copies match their installed ComfyUI user-menu copies by SHA-256.
> The v1.33.1 source gate passes 718 project tests, changed-scope Ruff, compileall, 125-node
> append-only registration and stable sampler SHA protection against ComfyUI
> `187eda8ef5e588c6a5765cad53e482765edae052`.

> Project v1.32.1 adds controlled same-input, same-seed, same-20-NFE full-resolution comparisons
> for SPEED T2VA, FL2VA remix and Ref2VA image. Exact-profile end-to-end speedups were 2.179x,
> 2.209x and 2.299x; peak VRAM changed by -77.0MiB, +33.9MiB and -162.5MiB, so SPEED is a
> compute-time optimization rather than a memory-safety feature. The decoder gate now uses
> FFmpeg `-xerror -err_detect explode`; it rejected one older corrupt T2VA SPEED stream, which
> was regenerated and passed 3/3. The final six files pass mechanical A/V checks and are packaged
> for anonymous full-video/audio review. Quality, audio and reference non-inferiority remain unproven.
> The release gate passed 664 tests, full Ruff, compileall, 126 non-artifact JSON parses, stable
> sampler SHA protection and project/user SHA parity for all 70 frontend workflows.

> Project v1.32.0 completes representative real-GPU mechanical runs for SPEED T2VA, I2VA,
> FL2VA, L2VA, Ref2VA, Hybrid and an explicit Turbo8 scope. Every 1024x576x124 output passed
> three strict decodes with finite 32kHz stereo audio, but the worst 16GB headroom was only about
> 122MiB and no same-input full-resolution/perceptual baseline has established speed or quality.
> Six dated frontend workflows include multiple parameter and review notes. The 655-test, full-Ruff,
> compileall, 70-workflow JSON/live-schema and stable-sampler-hash release gates passed. SPEED remains EXP.

> Project v1.31.1 adds a modality-stable AV noise node and targeted H3 clone-family release for
> the isolated SPEED experiment. A real 1056x608, 124-frame Stock20 run completed and passed three
> strict decodes, but its minimum 16GB headroom was about 376MiB, below the 512MiB release gate;
> SPEED therefore remains EXP and makes no universal memory, quality, or audio claim.

> Project v1.31.0 appends one isolated H3 Detail Mixer Advanced sampler without changing the
> existing five detail nodes, stable schemas, defaults or sampler math. Tail subdivision,
> model-time bias, STG and joint-AV RF Restart are separate opt-in toggles and all default off.
> The report separates integrator NFE from STG weak-branch and total planned joint-AV Transformer
> forwards. Temporal Detail remains a decoded-IMAGE post-process and the example bypasses its
> audio from AV Decode. No universal quality, audio or 16GB safety claim is made.
> The final source gate passed 616 project tests, changed-scope Ruff, compileall, strict parsing and
> project/user hash parity for all 64 frontend workflows, plus whitelist-only ComfyUI startup.
> A 256x256x22 native FL2VA INT8 GPU probe with two base steps plus one tail step, Bias and STG
> completed in 39.38 seconds and saved all 22 frames. It is mechanical integration evidence only.

> Project v1.30.1 is a frontend-workflow compatibility hotfix. It does not change a stable node
> schema, default or sampler. The converter and repair tool now serialize the complete live-schema
> input order and connected slots. Forty project workflows plus forty installed user copies were
> repaired; 610 tests and a strict 123-workflow JSON/schema scan passed.

> Project v1.30.0 appends four isolated H3 SPEED Advanced nodes. They implement the official
> spatial progressive DCT expansion, kappa rescaling and sigma alignment as a whole-chain H3
> sampler that rebuilds stage-specific AV conditioning. WAN spectrum constants are never reused.
> Strict mode is T2VA/native-audio/exactly-20-step Stock20 only; multimodal stage rebuilding is explicit EXP.
> Code, theory and CPU/static validation are complete, but no real H3 GPU generation has been run,
> so speed, quality, audio non-inferiority and 16GB safety claims remain false.
> The final source gate passed 606 project tests, Ruff, compileall, and all 63 frontend workflow JSON parses.

> Project v1.29.0 also includes five isolated MiniMax H3 detail experiments: final-interval AV
> subdivision, smooth shared-Transformer model-time bias, true joint-AV rectified-flow restart,
> H3 skip-block spatio-temporal guidance and decoded-frame motion-gated luma detail enhancement.
> The older default-off Dynamic Guidance and extra-tail-NFE examples remain available. These are
> Advanced examples. They are generation-time experiments, not face restoration. A 1.0-to-1.0
> guidance curve preserves the BasicGuider route; every inserted tail point is one full joint A/V
> DiT forward. All five red-Hanfu 1152x640x124 candidates completed and each new file passed three
> strict A/V decodes; the reused upstream eight-step baseline retains its previously accepted bad
> frame and therefore fails strict decode. Human quality/audio review remains pending. See the main
> README and the five `H3_Hanfu_*_Advanced_EXP.json` workflows.

The converter lives in the project-local `tools/` directory. Model weights are
kept outside this code repository and installed through ComfyUI's standard
model directories. Conversion adds the required `diffusion_model.` prefix; it
does not merge, transpose, rescale, or otherwise modify tensor values.

## Requirements

- ComfyUI **0.30.0 or newer** with native MiniMax-H3 support.
- A **non-pruned** MiniMax-H3 diffusion model:
  - `minimax_h3_fl2va_bf16.safetensors`, or
  - `minimax_h3_fl2va_int8_convrot.safetensors`.
- For R2V, use the corresponding non-pruned `ref2va_bf16` or
  `ref2va_int8_convrot` model.

Do not use a `*_pruned_*` diffusion model for a complete application of this
LoRA. Pruned checkpoints replace each AdaLN input with an 8-dimensional curve
basis, while this LoRA was trained against the original 2688-dimensional
AdaLN input. The other 208 adapter modules match, but 51 AdaLN adapters do not.
The bypass loader can therefore fail at runtime on a pruned model.

One exact-checkpoint-specific Experimental exception was completed on 2026-08-10 for
`10Eros_Max_h3_fl2va_bf16_test4_pruned.safetensors` with SHA-256
`f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062`. Its dedicated
`curveproj1025` LoRA converts all 51 AdaLN adapters to the target's 8-dimensional curve basis and
adds the required bias deltas. This does not make the original 518-tensor LoRA generally compatible
with pruned models, and the converted file must not be used on a different checkpoint merely because
its filename also contains `pruned`.

## Version 1.27.0 native SAM3.1 multi-person Face Refine

Version 1.27.0 appends six isolated Advanced nodes after the preceding 101 IDs. Existing Face Refine,
Conditioning, sampler, workflow and `sampling.py` contracts do not change. The route uses ComfyUI's current
native `SAM31Tracker` to detect and propagate two or three person masks per shot. A behavioral capability
probe requires `track_video_with_detection`; old SAM3 or unknown wrappers fail closed. Track colors and
indices are shot-local only and reset after cuts.

Authorized single-person reference images are represented as in-memory profiles. Pinned OpenCV Zoo YuNet
localizes faces and pinned Apache-2.0 SFace emits CPU identity suggestions. Suggestions are one-to-one per
shot and must pass score and margin gates; `{"0:0":"Character_A"}` style JSON remains authoritative. No
profile is persisted by these nodes, and a face embedding is not treated as identity proof.

Each repair job covers one character, one shot and one legal `17n+5` window. The default reviewed profile is
73 frames (about 3.04 seconds at 24fps), MANUAL512, crop factor 2.5, 21/51 trajectory smoothing,
relative-to-clip 0.8/0.35 denoise and the
existing face-only 24/24 stitch. Character branches execute sequentially and feed a source-bound composite
state. Acceptance defaults to false, overlapping masks reject, and original audio is locked and re-muxed
after all accepted candidates. The default SAM release policy calls only selective model-and-clone unload,
GC and `soft_empty_cache`; it never calls global model unload.

The job now also exposes an optional `target_face_px` scale mode for manual canvases. Legacy workflows omit
the new optional fields and retain crop-factor behavior. The regenerated two/three-person examples explicitly
target about 300px face height in a 512 crop and use Turbo 8 steps; their reports include the effective crop
factor, achieved face-height range and source-boundary-limited frame count. This guarantees crop-space scale,
not native detail: an already enlarged or defocused source cannot recover information merely by resizing.
The single-person recommended workflow now also uses the Turbo 8-step review profile while retaining its
human-selected MANUAL512/crop-2.5 route. Every single/two/three-person frontend workflow includes an embedded
Markdown note with the intended use and exact review parameters.

Face Refine is a structural re-generation tool, not a sharpening or super-resolution stage. It can repair
collapsed eyes, mouths, noses or face geometry when the source face is otherwise clear. If the source is
defocused, low-resolution or merely enlarged from a tiny raster, H3 usually preserves that soft visual style;
the crop target only ensures enough H3 canvas area and cannot recreate absent source texture.

Install `sam3.1_multiplex_fp16.safetensors` under `models/checkpoints` and
`face_recognition_sface_2021dec.onnx` under `models/face_detection`. The locally verified SHA-256 values are
`9BA99C92703C2E8B4F47DE2D34A539BB8E18923049E238B780D70DBE6368EB03` and
`0BA9FBFA01B5270C96627C4EF784DA859931E02F04419C829E83484087C34E79` respectively. Weights are not
distributed by this plugin.

A real 240x416x22 two-person chain completed native tracking, automatic Alice/Bob SFace assignment, two
sequential MANUAL512 H3 branches and final composition. The strict-decode output is 22 frames at 24fps with
SHA-256 `C74000515CFED4DB8A7D6E1DCD428F4AF379D3CEA89A432C3AE5EEC806F818E2`. A second 240x416x22 source
contained four people; the default `person with a visible face` prompt selected the three repairable visible
faces and omitted the back-facing person. Three reviewed manual assignments, SFace-guarded repair plans and
sequential H3 branches completed in 95.78 seconds under prompt
`0c37c0b3-e910-405f-9b3f-0a159c048b9e`. The final 22-frame H.264/AAC SHA-256 is
`C3CCB956397AC7497E8241DAB97D057ABAFFC20C625945662DE2608917B4DC42`; source and output decoded-audio
SHA-256 are both `3645A04B3F853F324732FFB9779EE1C95B01F6E5F68C6A07968ECBEDAAD552C1`.

That run exposed a real compositor-contract bug before it passed: Parity Stitch treats only `alpha == 0` as
mask exterior, while the original multi-person compositor incorrectly thresholded `alpha <= 1e-6` as exterior
and rejected the second person's legitimate feather tail. The compositor now uses `alpha > 0`, validates finite
0..1 mask values, and has a sub-micro-alpha regression. It still rejects actual outside-mask changes and mask
overlap. Sampled free VRAM reached approximately 489MiB in the two-person run, 450MiB in an earlier single
branch and about 375MiB after the complete three-person cold run. These are operational passes below the
512MiB project gate; neither universal 16GiB safety nor perceptual restoration is claimed.

A longer 608x448x73 follow-up (3.042 seconds) used the stricter example prompt
`front-facing person with a visible face`. Unlike the broader prompt on this four-person shot, it retained the
intended left armored man, center woman and right yellow-headscarf man while omitting the back-facing person.
All three roughly 50-100px source faces were planned on MANUAL512 canvases and all three H3 branches completed
sequentially. The third 24px feather region overlapped 50,621 already accepted pixels, so the default `reject`
policy stopped; after review, `keep_old_exp` preserved the earlier pixels and applied only the third person's
non-overlapping region. The strict-decode-clean 608x448, 73-frame output SHA-256 is
`AB26FC42A0FD9EFA5DA32877100554F1487165DEF2498BCC0495DD7638F656BB`; source and result decoded PCM MD5
are both `4c7905d4a36f6f9c456b7e074b52707e`. Five labeled zoomed time samples showed no sampled catastrophic
identity swap or face collapse, but the visible change was modest and no blinded perceptual promotion is claimed.

A later clear-source two-person acceptance run used a native 1920x1408, 69-frame, 24fps side-profile clip.
SAM3.1 tracked all 69 frames; the first legal 56-frame H3 window used two clear single-person references,
MANUAL512, a 300px target face height, relative-to-clip 0.8/0.35 and sequential Turbo 8-step branches, with the
remaining 13 frames retained as an untreated control tail. After watching the complete result, the user confirmed
that the damaged facial structure was repaired and clarified that the earlier blurred-source failure was a task
boundary rather than proof that Face Refine could not work. This is one user acceptance on a clear-source fixture,
not a universal restoration, identity, deblur or memory guarantee.

Since 1.27.2, the current example may request a 73-frame H3 window from that same 69-frame source. Repair Job
repeats the final source frame for four model-context positions, while Composite discards those positions and
returns the exact 69-frame source timeline with its untouched audio. This bounded behavior applies only when the
shortfall is at most 16 frames. Reference profiles now default to `dominant_face_auto`, which filters clearly
smaller/lower-confidence YuNet false positives but still rejects two similarly plausible real faces.

## Version 1.26.0 audited author-parity correction

Version 1.26.0 adds no node ID and changes no stable sampler. It corrects only the newly appended
MANUAL512 REL Parity path after a source-level, same-frame comparison with
`Carasibana/ComfyUI-H3-FaceRefine@79a97ce5`. Ultralytics now receives BGR ndarray input; the paste mask
uses the author's centred smoothed-face rectangle even when the crop is frame-edge clamped; per-frame
denoise derives face size from `crop height / crop_factor`; colour matching runs after warp in source
coordinates; and the real 89-frame batch is passed directly to the video VAE. A latent shape mismatch
still fails closed—there is no pixel-tail duplication, latent trim, pad or hidden resize.

On the fixed fixture, the post-fix crop mean absolute error against the fixed upstream node is
`0.00000677`, maximum denoise-curve error is `0.00001423`, and complete synthetic stitch mean absolute
error is `0.00000117`. The live package chain then completed under prompt
`1ed411fa-9b91-45c4-801d-7f45b3597fe5` in 112.48 seconds. Output SHA-256 is
`0DD8C79F95B01647F3BF345B6503C83A5860BE99BA66D8D72114BD274E9A0884`; strict decode reports 89 frames,
320x320 at 24fps and 32kHz stereo audio. Decoded audio MD5 remains identical to source and author target
(`26d40526bd022d7237ba183bd8777966`). Full-video SSIM to the selected author target improved from
0.955273 to 0.967059. After watching the complete author-target/T8-v2 side-by-side, the user judged both
results equally good, so the fixed fixture/seed/model-stack subjective non-inferiority gate passes. The observed
15,823/16,380MiB whole-device peak leaves about 557MiB only for this run; cross-input quality and universal
16GiB safety are not inferred.

## Version 1.25.0 human-selected MANUAL512 REL Face Refine baseline

Version 1.25.0 keeps the preceding 100 node IDs, schemas and defaults unchanged, then appends
`MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced`. The recommended Parity example now explicitly
uses `manual_512`, crop factor `2.5`, `relative_to_clip`, video-mask strengths `0.8/0.35`, 21/51 trajectory
smoothing and the existing face-only 24px dilation/24px source-feather stitch. The original soundtrack
continues through the locked zero audio mask and final source mux.

The appended node is a fail-closed mechanical guard, not a quality oracle. It binds the Plan, latent injection,
denoise and stitch reports by SHA, verifies the selected settings and requires at least 200 crop-space face pixels. It
passes the candidate tensor through unchanged and does not route through the 1.24.0 source-similarity proxy
gate. On the fixed 89-frame local fixture, the 512 crop produced approximately 205-312px faces from
105-195px source faces and 1.60-1.95x crop magnification. The strict-decode output SHA-256 is
`19EA5844643B962F6FD197E34705861916D69F7EA70F3E00A2DF022D6A017399`; the user selected it as the best
of the six complete comparison videos. This one-fixture human choice does not establish universal identity,
restoration quality or 16GiB safety; sampled headroom reached only 161MiB.

The recommended workflow pins the actually reviewed stack: Ref2VA pruned INT8, Qwen3-VL NVFP4, the official
video/audio VAEs, the alpha8 T8-converted FL2V Turbo LoRA at 0.75, two identity references, locked source audio,
face YOLO, `er_sde + simple + 4 steps + denoise 0.45 + seed 42`. Comfy decoded the source as 89 frames. The
Parity path now explicitly duplicates the final crop once for the legal 90-frame H3 internal grid, records that
policy in the latent report, and discards exactly that one internal tail before stitching. Final media stays at the
original 89 frames; arbitrary source-frame trimming or hidden temporal fitting remains forbidden. Existing Plan
defaults still require a native H3 grid; only the reviewed example opts into the guarded one-frame exception.

The same API was executed end to end through this package's own six Parity nodes under prompt ID
`57741215-c23b-4a9b-87b7-7288ce175ff1` in 107.41 seconds. Strict decode produced 89 frames at 320x320/24fps
plus 32kHz stereo audio; SHA-256 is `B91BBE09C2AF4266EDD2975760A13749A0DB819054BE6C8118E144F0D4AF3097`.
Decoded audio MD5 is bit-identical to both source and the selected candidate (`26d40526bd022d7237ba183bd8777966`).
Video SSIM against the selected candidate is 0.955273. This closes real mechanical execution, timeline and
soundtrack preservation, not subjective equal-or-better quality; the full side-by-side remains the authority.

## Version 1.24.0 Face Refine parity plus source-fallback quality gate

Version 1.23.0 appended four Advanced node IDs after the unchanged 95-node prefix. They reproduce the
auditable mechanics of `Carasibana/ComfyUI-H3-FaceRefine@79a97ce5`: separate reflected-Gaussian
21-frame center and 51-frame size smoothing, crop factor 3, an automatically selected square canvas
capped at 768, a face-size-dependent video denoise mask (0.8/0.35 at 30/120px, smoothed over 9 frames),
an exactly zero audio mask, and face-only 24px dilation/24px source feather stitching. The example uses
`er_sde`, `simple`, four steps, base denoise 0.45, LightX2V FL2V Turbo at 0.75 and seed 42, then muxes
the untouched source soundtrack. Version 1.24.0 preserves those 99 IDs and appends one
`MiniMaxH3FaceRefineQualityGateT8Advanced` node. It refuses non-Parity candidates whose pixels changed
outside the stitch mask, then accepts only continuous runs passing conservative source-relative structure,
face-delta, measured-sharpness and residual-jitter thresholds. Rejected frames return to the exact source
tensor. A passing proxy still does not prove identity or restoration quality. See `THIRD_PARTY_NOTICES.md`.

Real local probing has three T8 candidates. The exact 0.8/0.35 strength route and an all-50 Hybrid
weight-only variant both produced obvious facial distortion and are rejected. A single-variable
0.45/0.15 route was visibly less destructive and improved face-region SSIM from 0.46179 to 0.76044,
but its median face-Laplacian ratio was only 0.50535; it remains a manual-review EXP candidate rather
than a quality preset. The fixed upstream four-node code was also run with the same local model stack:
0.8/0.35 measured face SSIM/motion correlation 0.49077/0.42102; 0.45/0.15 measured
0.77776/0.70423 but retained only a 0.56297 median sharpness ratio, so it was also rejected. The exact
face/person YOLO weights are installed and hash-checked; face YOLO participated in that run while person
fallback was disabled. The source still lacks the author's GGUF, embedded prompt, full references and
isolated vocals, so this is fixed-code local execution rather than the author's original environment.

The default high-strength T8 candidate was then passed through the new quality gate. Its accepted-mask
video contained zero non-black frames: all 124 generated face frames were rejected and the source tensor
was returned. A separately encoded source-only Comfy pass versus the gated encode measured full/face SSIM
0.999885/0.996810 and face MAE 0.000471; remaining differences are from separate H.264 encodes. This proves
the tested regression can be contained, not that any face was restored.
The complete local gate passes 548 tests, full Ruff, compileall, 104 non-artifact project JSON
documents, live object-info loading of the new schema and the unchanged stable sampling hash.

## Version 1.22.1 Face Refine extended validation

Version 1.22.1 changes no node ID, input, output, default, model weight or stable sampling path. It adds
offline exact-plan/host-memory probing, randomized local source-versus-candidate blind review plus strict
reveal analysis, fixed-threshold YuNet evaluation on a user-provided local WIDER FACE validation copy,
controlled tracker crossing and candidate proxy summarization. None downloads models or datasets at
runtime, and WIDER images are not distributed with this plugin.

On ComfyUI `0.33.0@7fe8a61385`, an aspect-safe 736x416x124 Face Refine chain completed three cold and
three warm full-H3 runs. Minimum whole-device headroom was 717.6/922.1MiB and post-run private-memory
spread was 3.2/79.1MiB. One 736x416x362 cold run also completed with 1176.2MiB total-minus-used headroom
and 41311.6MiB peak process private memory. This is exact-stack execution evidence, not a universal 16GiB
safety tier.

The pinned YuNet 2023mar model was evaluated at IoU 0.5 on 3226 WIDER FACE validation images and 39123
valid faces. The node default 0.35 threshold produced 0.6223 precision and 0.6499 recall; 0.60 produced
0.8610 precision and 0.5694 recall, while under-16px recall fell from 0.4360 to 0.3225. The default is not
changed from aggregate F1 alone because this feature targets far faces. A real 124-frame group clip and a
362-frame hard negative also proved that detection/tracking is not identity-safe. One reviewer completed all
six randomized pairs before reveal: source won overall and identity 6-0, while motion was tied in all six.
Candidate mean scores were 1/5 for identity, expression/mouth, temporal stability, seam and naturalness versus
5/5 for source; every note identified candidate facial distortion and repeated jitter. This rejects the current
six candidates for the fixed source/settings, but the preregistered five-reviewer panel remains incomplete and
the repeated source can reveal the control over time. A deterministic five-scenario crossing matrix also showed
that three frames of target occlusion switched the geometry-only tracker from A to B for the final 31 frames.
Six real candidate proxies also provided no automatic non-inferiority signal: face SSIM mean was 0.4791-0.5392,
candidate/source face Laplacian median ratio was 0.5493-0.6835 and face-motion-difference correlation mean was
0.3581-0.4071. These metrics do not measure identity or replace blind review, so automatic acceptance,
perceptual/identity promotion, automatic acceptance and general memory safety remain denied. See
`FACE_REFINE_ADVANCED_VALIDATION.md`.

## Version 1.18 Advanced Studio and diagnostic routes

Version 1.18 appends Advanced/Experimental routes without changing the existing H3 conditioning or
stable dual-clock sampler. Start with `MiniMaxH3EnvironmentAuditT8Advanced`; it is read-only and a
pass means only that no known blocker was found. Qwen prefix caching and MLP activation chunking
default to report-only. Context IR uses local validation by default, and an external visual provider
requires explicit upload confirmation while never receiving raw audio.
The audit reports cumulative process I/O/page-fault and pinned-memory/GPU-health state; use
`tools/validate_h3_vram.py run` for before/after workload deltas and the conservative
`fits/fits_with_thrashing/unsafe/unknown` classification. A single audit snapshot never proves
that the current workflow caused the observed cumulative reads.

In the current 1.45 development checkpoint, selecting `sage_attention` also makes this audit retain
the exact SageAttention package/core import error, every core symbol required by the current KJNodes
MiniMax H3 patch, the wheel's reported CUDA architectures and the active-GPU match. It performs no
attention call. This separates incompatible wheels and architecture-probe failures without silently
falling back. The Audio Lock Source and Quick Audio Drive examples likewise distinguish preserved
`mux_audio` from deterministic lip synchronization: H3 can condition visual performance on an audio
reference, but it does not enforce phoneme-exact mouth motion.

The stable default sampler has also been run against the H3-era legacy ComfyUI
`0.30.0@563b98eef` and current `cbbc9dab1` using the same plugin, model files, workflow and seed.
Both runs produced 22/22 byte-identical PNG frames; their 32kHz stereo audio correlation was
0.999688 with 36.12dB SNR and a one-int16-LSB maximum difference. This is evidence for the stable
`dual_clock_euler` legacy velocity branch only, not a blanket claim for every Advanced route or
every historical ComfyUI release. Separately, the current `1.18.2@c7f5080` plugin imported into
that exact legacy snapshot without traceback: all 86 plugin node IDs and all 24 appended Advanced
schemas appeared in `/object_info`. A real Trajectory Advanced follow-up then completed full, 2+2
split and save/load/resume at 256x256x22 in 3/3 GPU prompts; the final checkpoint pair was byte-exact.
Minimum headroom was only 94.153MiB, below the 512MiB project gate. This closes only that short
Advanced trajectory route. Four model-free/no-write API graphs also passed: Environment Audit, all
seven Studio/Prompt/Repair-planning nodes, both local Context IR nodes and both Reel plan/no-write
nodes. Qwen Prefix Cache report-only, `memory_lru_exp` wrapper installation and Stats passed. A
follow-up then used a native 48-frame reference video and completed OFF, cache prime and real HIT plus
full A/V generation. Stats recorded one miss and one hit; HIT was 21.80% faster than OFF, but output
was non-exact (mean video SSIM 0.950934, audio correlation 0.953028) and minimum headroom was only
334.508MiB. This is one short-route compatibility result, not a lossless or 16GiB-safe claim.
AV Decode Safety decoded 22 frames
and audio; Activation Chunk correctly refused the old unknown H3 source contract before sampling.
The four Repair execution nodes then consumed an already persisted real-chain plan with acceptance
off; Bind, Stage, Accept and Compose completed, the source manifest plus all 27 accepted assets stayed
unchanged, and only a new base-rollback validation render was written. An isolated two-segment fixture
then exercised `accept_repair=true` and repair-overlay composition on
the same old core. Replacement index 1 and unselected index 0 were preserved, the base manifest and
accepted asset hashes remained unchanged, and the output contained 44 video frames. Its AAC stream
duration was 58,688 samples versus 58,667 logical samples (+21), while decoding returned codec padding;
this is mechanical transaction evidence, not sample-exact delivery or real-H3 repair quality.
Scheduled Audio Injection also
completed both its default `report_only` and actual `scheduled_injection` 256x256x22 one-step routes;
each emitted 22 frames plus audio, but minimum whole-device headroom was only 15.685/97.335MiB and the
apply pass does not prove speech suppression. All 24 newly appended
Advanced node IDs therefore have route-specific execution or explicit fail-closed evidence on this
exact old core. The scope remains bounded: Qwen covers one short reference-video route, Activation
refused application, Repair acceptance used an isolated fixture, and Scheduled Audio retains a
negative quality result. Other modes, arbitrary legacy releases and general 16GiB safety are not
implied.

All file mutations are opt-in: repair acceptance, Reel Delivery composition and trajectory
checkpoint saving default false. Scheduled drive-audio injection defaults to bypass because its
first real A/B did not stop ASR-detected extra tail speech. AV Decode Safety defaults to preflight
only, and its current-headroom report is not a future VAE peak prediction. Current H3 regular decode
also uses internal 256-pixel tiles on larger canvases, while explicit tile controls are ignored by
the H3 first-stage alias; missing global tile coordinates are therefore high-risk in either mode.
A validation-only direct full-canvas spatial-coordinate substitution was then tested on three
736x416 source reconstructions. The 256x256 one-tile control was bit-exact, but all three tiled
cases lost SSIM/PSNR and worsened seam ratios with visible grid/ghost artifacts, so that direct
remedy was rejected and was not merged.
The supplied workflows in
`examples/workflows` retain these safe defaults.

The bounded verification record is in `VERIFICATION_REPORT.md`. A 736x416x124 controlled A/B rejects
activation chunking as a memory optimization for the current fused TensorWise INT8 path. The final
Trajectory v2 contract uses Load.resume_noise for direct internal-x-sigma transport, not DisableNoise.
Its 736x416x124 and 256x256x362 full-versus-2+2 final AV latents were bit-exact; the 124-frame three-cold/
three-warm matrix completed 18/18 prompts and 6/6 paired comparisons. The 362-frame full run left only
520.51MiB, and paired warm split+resume was not faster than full, so this still does not establish a
universal 16GiB safety or throughput benefit. AV Decode likewise has no tiled-equivalence claim.
On current ComfyUI `v0.32.0-16@ddbaa8752`, a separate six-prompt recheck completed full, split and
resume at both 124 and 362 frames; each full/resume checkpoint pair had an identical SHA-256. Full-run
headroom was 749.019/548.502MiB respectively. The 362 result remains a close 256x256-only pass, not a
higher-resolution or general 16GiB safety tier.
The Qwen prefix cache now has three fresh-process cold pairs and three same-process warm pairs:
every hit arm was faster (paired mean 11.97%/11.01%), but outputs remained non-bit-exact, one warm
audio pair dropped to 0.2323 correlation, and minimum headroom was only 75.63/168.08MiB. It stays
report-only EXP and is not a lossless, VRAM-saving, or 16GiB-safe feature.
ComfyUI later moved to `v0.32.0-15@86aedfd9`, adding merged projection, fixed-KV, prefetch and
in-place residual paths to Llama/Qwen. The cache now hashes the directly invoked TransformerBlock
contract as well as Llama and Attention, and executes prefix/suffix inference explicitly without
autograd. A current-core CPU causal-equivalence probe passed, followed by a real same-process 32B
NVFP4 OFF/HIT pair: the 108.283MiB entry recorded one hit after one miss, elapsed time changed from
13.297s to 9.375s, video SSIM mean/minimum was 0.951217/0.924603 and finite 32kHz stereo audio
correlation was 0.956522. OFF/HIT headroom was only 116.998/337.583MiB, so this proves current-core
compatibility, not losslessness or 16GiB safety.
ComfyUI then advanced to `v0.32.0-16@ddbaa8752`, moving MiniMax projection-format detection without
changing the Qwen forward path used here. The exact-source contract, tiny-Llama equivalence probe,
447-test project regression and a native 48-frame video-reference full A/V pair all passed again.
That pair recorded a real 110.744MiB hit; OFF/HIT elapsed time was 25.266/15.578s, video SSIM
mean/minimum was 0.950934/0.944633 and audio correlation was 0.953029. Whole-device headroom was only
344.340/338.833MiB, so the 512MiB safety gate still failed and the result remains non-exact.
The exact short multi-reference and video-reference mechanics are now also exercised. Two image
references produced real cold and warm hits from a 60.70MiB entry and reduced elapsed time by
6.04%/6.87%, but video SSIM mean/minimum were 0.91869/0.91130, audio correlation was 0.95956 and
minimum headroom was 311.85MiB. A native 48-frame, 2-second, 24fps video-reference full A/V pair hit
a 110.74MiB entry and reduced elapsed time by 13.81% and peak device use by 166.31MiB, while leaving
only 145.15/311.46MiB OFF/HIT headroom; its video SSIM mean/minimum were 0.95093/0.94463 and audio
correlation was 0.95303. Both paths are non-exact one-step probes, not perceptual, fixed-speedup or
16GiB-safety evidence.
A further same-process warm matrix covered three two-image material combinations and two seeds each.
All 6/6 pairs produced real hits and every HIT arm was faster, with a mean elapsed-time change of
-11.09%. Video SSIM averaged 0.9314 across pairs with a 0.8531 minimum frame; audio correlation
averaged 0.9771. Post-pair process private memory had no 256MiB upward staircase, but whole-device
headroom fell to 111.93MiB. The one-step contact sheet is unsuitable for perceptual acceptance, so
human non-inferiority at a useful generation profile remains open.
A Stock20 follow-up used one seed from each of the same three material combinations and
conditioning-only primes. All 3/3 full HIT arms were real and faster, averaging -5.00% elapsed time,
but full diffusion amplified the numerical difference: video SSIM averaged 0.8227 across pairs
(pair range 0.6790-0.9073, minimum frame 0.6052), while audio correlation averaged 0.7188 and ranged
from 0.2603 to 0.9894. Minimum headroom was 190.68MiB. These automated quality and safety results do
not pass promotion; the cache remains report-only pending blind review and independent hardware.

Version 1.18.1 adds no node or schema. It hardens Reel Delivery around external termination:
the mixed PCM stage is validated in a temporary file before atomic replacement, one OS advisory
lock serializes a project root, and the next run removes only matching orphan temporary files before
reusing hash-verified phases. A Windows/NTFS 30-minute soak completed 50 independently addressed
clip paths, dialogue/music/ambience/SFX lanes, 43,200 frames and 86,400,000 48kHz samples. Recovery
also completed after killing the audio FFmpeg child, final-mux FFmpeg child and parent Python process.
The 50 clip paths were hardlinks to one small fixture, so that soak alone did not prove codec diversity.
A separate Windows/NTFS composition then mixed synthetic H.264/AAC, HEVC/MP3 and VP9/Opus 128x96
24fps sources plus WAV/FLAC/Opus/AAC lanes. It produced exactly 132 frames, a 264,000-sample plan and
5.500-second output stream; source hashes stayed unchanged, the repeat reused both phases and the
output hash stayed stable. Version 1.18.2 fixes a separate MP4 timing defect without changing any node
schema: FFmpeg's default 1000-unit movie timescale quantized a 58-frame/48kHz plan from 116,000 logical
AAC samples to 115,968. The mux now uses `LCM(24, sample_rate)` and validates the final container header
before atomic replacement. Isolated 32/44.1/48kHz cases all reached zero logical-sample error. Three
real 256x256 H3 clips then produced exact 58-frame/116,000-sample output, and three distinct real
736x416x124 H3 clips produced exact 348-frame/696,000-sample, 14.5-second output with 12-frame
transitions. Source hashes, phase reuse and repeated output hashes remained stable. AAC decoder tail
padding is still expected and is not a lossless-PCM claim. Local real-H3 and 736x416 delivery mechanics
are closed. A derived-1080p follow-up placed the same three real-H3 clips into 1920x1088 canvases.
PyAV/libx264 auto-threading first produced one native crash and then 3/3 streams with decodable H.264
reference/CABAC errors. Reel now scopes x264 to one thread at two megapixels or above and invalidates
old high-resolution phases lacking the exact policy marker; lower resolutions keep auto-threading.
The production high-resolution path now applies the versioned `ffmpeg_single_thread_xerror_v2`
contract to both the encoded phase and final mux temporary file before their atomic replacements.
Three independent projects passed both strict decodes and their phase/final SHA-256 values were
identical across runs. Each result was exact 348 frames,
696,000 logical samples and 14.5 seconds; the transition-tail estimate was 71.72MiB. This proves derived
1920x1088 file delivery mechanics, not native H3 1080p generation quality. Two local FFmpeg 7.1
Windows builds showed nondeterministic failures when auto-thread decoding the same high-resolution
H.264 bytes, while single-thread FFmpeg and PyAV/libavcodec 62 repeated decoding passed. The claim is
therefore scoped to the explicit single-thread validation contract; other players and untested
FFmpeg builds remain open. A separate Ubuntu 24.04.4 WSL2 run used a Linux kernel, ext4 `/tmp`,
Linux FFmpeg 7.0.2 and PyAV 18.1.0 against the same production Reel module. Two H.264/AAC clips plus
one FLAC lane produced exact 66 frames and 132,000 logical 48kHz samples. A repeat reused the video
and audio phases and preserved the output SHA-256; source hashes stayed unchanged, no orphan
temporary file remained, a real POSIX lock contender timed out, and the lock was reacquired after
the holder exited. This closes one low-resolution Linux/POSIX mechanical route, not bare-metal
Linux, macOS, high-resolution Linux, arbitrary FFmpeg builds or cross-GPU behavior.

Selective Repair additionally survived six hard-kill boundaries on an isolated 14-segment,
60-second accepted chain without changing the base manifest or 27 accepted assets. A real H3
segment-7 replacement composed to exact 1,440-frame/1,920,000-sample outputs, but the outgoing
boundary regressed because segment 8 still depended on the original segment 7: adjacent-frame SSIM
fell from 0.932967 to 0.803963 and the repaired outgoing audio level gap reached 19.396dB. Accept and compose
now remove only exact-destination `.*.tmp` files while holding their OS locks. A rebuilt copy of the
same real 14-segment chain repeated all six process kills: expected exit codes, durable markers,
base/27-asset immutability and retries passed; the half-copy and mid-audio orphans were both reported
before retry and absent afterward. This closes crash-clean behavior only for those Windows/NTFS
killpoints. Cascading dependent-segment regeneration, blind quality review and cross-platform
filesystems remain open.

## Install and connect

1. Copy either converted `*_comfyui.safetensors` file to
   `ComfyUI/models/loras/`.
2. Update ComfyUI and restart it.
3. Add **Load LoRA (Bypass, Model Only) (for debugging)** after **Load Diffusion
   Model**. Connect its model output wherever the diffusion model was connected.
4. Start with LoRA strength `1.0`. The upstream discussion reports that values
   around `1.5–2.2` can look stronger, but that is an empirical preference, not
   part of the trained LoRA math.
5. Use **MiniMax H3 Dual-Clock Sampler (T8)** from
   `custom_nodes/minimax-h3-audio-T8`, with `steps=4`, video shift `12`, and
   audio shift `3`. Keep `sampler=dual_clock_euler` and
   `scheduler=native_flow` for the original verified path. Its `model` output goes to the guider; its `sampler` and
   `sigmas` outputs go to `SamplerCustomAdvanced`. Connect the same H3 AV latent
   to both the dual-clock node and `SamplerCustomAdvanced.latent_image`.
6. To test more audio integration steps without changing the stable workflow,
   use the separate **MiniMax H3 Multi-Rate Sampler (EXP/T8)**. Start with
   `video_steps=4`, `audio_steps=8`; then compare 4/10 with the same seed.
   `audio_steps` is the number of full joint H3 DiT calls, so 4/10 costs about
   2.5 times as much as stable 4/4. The Turbo LoRA is still trained for four
   steps, so extra audio microsteps are experimental and not guaranteed to win.

Do not combine the dual-clock node with `MiniMax H3 Sigma Shift`,
`KSamplerSelect`, or an external scheduler node. The node replaces all three.
Version 1.7.0 keeps the 1.3.3 internal sampler and scheduler dropdowns while preserving
the original defaults. Alternative ComfyUI samplers use native `ModelSamplingAV`
and are exposed only when the installed ComfyUI has FLOW_AV support; alternative
schedulers change the sigma grid and are not a quality guarantee for a four-step
Turbo LoRA. Old workflow/API JSON may omit both new fields and retains the
original behavior.
The same no-extra-scheduler rule applies to the EXP node.

The bypass loader is recommended because it computes the author's intended
runtime expression `base(x) + B(A(x))`. A regular LoRA loader may round small
updates when it materializes them into BF16 weights, and it cannot faithfully
patch quantized weights in the same way.

## Ready-to-import workflows

Version 1.17.0 retains all 61 Version 1.16.0 node IDs and appends one isolated
`MiniMaxH3HybridCompatibilityAuditT8Advanced` node. Put it after every MODEL-changing node and
sampler setup, then route its passthrough MODEL to `BasicGuider`. Connecting final H3 Conditioning
also verifies Long Video/MultiKeyframe pairing and actual reference modalities.

The default `report_only` mode never blocks and returns the exact same MODEL object. The optional
`block_hard_conflicts` mode rejects invalid Hybrid offset-set structure, Hybrid/LoRA order or AdaLN
overlap, incomplete Block Cache/Sage contracts, Long Video/MultiKeyframe conflicts, mismatched
Conditioning, and configured current-VRAM/host-commit gate failures. It recognizes stock, stable
dual-clock/native AV and EXP multi-rate sampling without changing sampler mathematics.

When a T8 VRAM policy is connected to the Hybrid Loader, a small policy-application provenance
attachment now follows MODEL clones. `require_applied_vram_policy=true` distinguishes a real fixed/
auto reserve from missing or report-only policy. Current 512 MiB VRAM and 16 GiB host-commit gates
are not peak predictions. Passing the audit is mechanical compatibility only; quality, de-waxing,
reference identity and universal 16 GiB safety remain unproven, and `memory_safe_claim=false`.
See `docs/HYBRID_COMPATIBILITY_AUDIT.md` and import
`examples/workflows/09-hybrid-model/2026-08-09_H3_Hybrid_Compatibility_Audit_Stock20_EXP.json` or
`hybrid_compatibility_audit_api.json`.

A follow-up exact-profile matrix on 2026-08-13 used 736x416, 124 frames,
Stock20, the 27.69 MiB Hybrid artifact, KJ H3 Sage, default-threshold T8 H3
Block Cache, the 4 GiB policy, and strict audit. Three fresh-process cold runs
and three same-process warm runs all succeeded and each cached 6/20 forwards.
Worst whole-device headroom was 766.38 MiB, maximum positive warm-baseline
movement was 82.94 MiB, and all six 124-frame PNG sequences plus FLAC files
were byte-identical. This passes only the exact local mechanical/repeatability
gate; cache-off quality, multiple materials/seeds, other GPUs, and universal
memory safety remain unproven.

A same-stack Cache OFF control then completed three cold and three warm runs,
with three interleaved warm OFF/ON pairs in one process. Mean end-to-end time
fell from 169.93 s OFF to 129.19 s ON (23.98%); sampler time fell from 146.24 s
to 105.31 s (27.99%), with 6/20 cache hits each time. The treatment was not
bit-exact: mean video SSIM was 0.8432 (minimum frame 0.7577), uint8 MAE was
10.37, and audio correlation/SNR were 0.9207/7.99 dB. One OFF cold run left
only 239.40 MiB, below the 512 MiB gate. This proves a repeatable performance
benefit only for this exact local profile. The later six-pair human screen is
single-reviewer evidence only, and `memory_safe_claim=false` remains unchanged.

The follow-up was extended to three visual material classes with two seeds each:
portrait, high-frequency mechanical dragon, and a rooftop superhero scene. Five
warm controlled pairs saved 22.05-28.47% end-to-end (24.35% mean) and
27.94-33.05% sampler time (29.07% mean), with 6-7/20 hits. Across all six
quality pairs, video SSIM ranged from 0.5192 to 0.9373 (0.7020 mean; minimum
single frame 0.4774), while audio correlation ranged from 0.8792 to 0.9806
(0.9329 mean). No pair was bit-exact. One human reviewer then scored the
randomized package as six video ties and six audio ties. The reviewer saw every
B side as slightly lighter, but B mapped to Cache OFF in the first three pairs
and Cache ON in the last three. Decoded signal statistics also found B brighter
in only 1/6 pairs and slightly less saturated in 4/6, so the observation is not
attributable to threshold 0.12. This is a single-reviewer smoke screen, not
statistical perceptual non-inferiority or a universal default recommendation.

A follow-up calibrated 0.08 and 0.10 on the two most divergent pairs, then
extended the more conservative 0.08 setting to the complete three-material,
two-seed matrix. Threshold 0.10 showed a non-monotonic superhero-audio
regression and is not recommendation evidence. At 0.08, five valid warm pairs
saved 9.05-14.38% end-to-end (12.35% mean) and 12.20-18.39% sampler time
(15.10% mean), with 3-4/20 hits. Six-pair video SSIM averaged 0.8598 (pair
range 0.6013-0.9840; minimum frame 0.5294), and audio correlation averaged
0.9635 (range 0.8883-0.9927). Both proxies improved over 0.12 in all six pairs,
but difficult cases remain materially different. The randomized OFF-versus-0.08
package was then scored by one human reviewer before reveal. Video favored 0.08
once with five ties and no Cache-OFF win; the reviewer saw a slight difference
in real-person material and no discernible difference in animated material.
Audio produced five ties and one low-confidence Cache-OFF preference in the
effectively silent sixth pair. This passes only a single-reviewer subjective
smoke screen. No node or legacy workflow default changed.

An additional anonymous model-side review sampled six timestamps per side plus
each pair's maximum-difference frame. It found no black frame or sampled-frame
structural collapse and recorded six visual ties. Objective checks found no
clipping in 12 audio tracks, but no human listening was performed and both
superhero-seed-2 tracks were effectively silent. This low-confidence screen
did not by itself establish motion/audio non-inferiority. The subsequent human
screen above still does not prove statistical non-inferiority or losslessness.

Version 1.16.0 retains all 60 Version 1.15.1 node IDs and appends one isolated
Hybrid Artifact Maintenance Advanced output node. Its API and frontend examples
default to side-effect-free inspection. Mutating actions require explicit confirmation and a
positive operation epoch; verified files are moved to a recoverable same-volume quarantine, not
permanently deleted. Exact path derivation, atomic fsynced journals, per-file SHA-256, stale-owner
checks, tampered-journal refusal, and a real worker-process kill/recovery test guard the feature.
It never scans source checkpoints, unloads a MODEL, or releases VRAM. See
`docs/HYBRID_ARTIFACT_MAINTENANCE.md`.

Version 1.15.1 retains all 60 Version 1.15.0 node IDs and updates the opt-in
`examples/workflows/09-hybrid-model/2026-08-09_H3_Hybrid_Model_VBAR_Headroom_Stock20_EXP.json` / `hybrid_model_vbar_headroom_api.json` pair. It
connects a reportable 4.0 GiB total-reserve policy directly to the Hybrid Loader, guaranteeing that
ComfyUI reserve and AIMDO simple headroom are set before the stock diffusion-model load. The policy
uses a direct AIMDO setter, does not reinitialize devices or alter startup `--vram-headroom`, and
does not globally unload models in this fixed-policy example.

On the exact RTX 4060 Ti 16 GiB, 736x416, 124-frame Hybrid Stock20 validation graph, the 4.0 GiB
setting passed three cold and three warm runs with at least 1028.117 MiB and 1401.415 MiB headroom,
respectively. Decoded same-seed video and PCM matched the no-policy baseline bit-for-bit. This does
not generalize to other resolutions, frame counts, GPUs, concurrent CUDA users, or host-commit
conditions; `memory_safe` and `never_oom` remain false.

Version 1.14.0 added the opt-in
`examples/workflows/09-hybrid-model/2026-08-09_H3_Hybrid_Model_Advanced_Stock20_EXP.json` workflow and
`tests/fixtures/api/hybrid_model_advanced_api.json`. Audio-only and mixed-reference variants are provided as
`examples/workflows/09-hybrid-model/2026-08-09_H3_Hybrid_Model_Audio_Reference_Stock20_EXP.json`,
`examples/workflows/09-hybrid-model/2026-08-09_H3_Hybrid_Model_Mixed_Reference_Stock20_EXP.json`,
`hybrid_model_audio_reference_api.json`, and `hybrid_model_mixed_reference_api.json`.
The graph records the selected FL2VA/Ref2VA file and curve hashes for diagnostics and artifact
integrity, but does not use reference-model fingerprints as an allowlist. It builds or reuses a
27.69 MiB curve-rebased target-slice artifact under
`ComfyUI/models/h3_hybrid_artifacts`, and then applies it to a MODEL loaded through ComfyUI's stock
diffusion loader. It does not create a second full fused checkpoint. Keep the order Hybrid Loader →
optional LoRA. `auto_match_reference_modalities_exp` reads the connected Conditioning and selects the
smallest video/audio modality-row recipe for actual extra references; this is not a best-quality selector.
The resumable sequential matrix tool writes blind-review media and `matrix_summary.json/csv`, with optional
local-only ASR, face and speaker signals. Fifteen real Stock20 follow-up runs completed, but the minimum
whole-device headroom was only 41.34 MiB; no recipe is yet a proven quality winner, reference-only route,
de-wax fix, or universal 16 GB safe tier. See
`docs/HYBRID_MODEL_ADVANCED_VALIDATION.md` for the exact hashes and current pilot limits.

Three base sampler-comparison frontend workflows are installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/01-basic-generation/`: stable 4/4, experimental 4/8,
and experimental 4/10. They share the same seed, prompt, EMA LoRA, loaders, and
MP4 settings for direct comparison. Drag a JSON file into the ComfyUI canvas or
open it from the Workflows menu.

Three stable 4/4 source-audio workflows are installed under
`ComfyUI/user/default/workflows/MiniMax H3 T8/02-audio-control/`:
`2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json`,
`2026-08-06_H3_Audio_Remix_Source_Stable_4V4A.json`, and
`2026-08-06_H3_Audio_Reference_Only_Stable_4V4A.json`. Each uses a 5-second Audio Window,
736x416 canvas, 124-frame legal H3 context, exact synchronized Output Trim, and
the explicit dual-clock Euler/native-flow defaults. Lock mode routes the clean
Conditioning `mux_audio` to the final MP4; remix and reference-only route decoded
model audio instead. Upload or select a source file in `Load Audio` before queuing.

Version 1.12.0 also installs two opt-in dialogue-safe audio workflows under
`ComfyUI/user/default/workflows/MiniMax H3 T8/05-speech-dialogue/` without changing any old
workflow: `2026-08-10_H3_Dialogue_Safe_Master_EXP.json` accepts already independent speech/music/ambience/SFX
stems and keeps the background running after verified speech ends;
`2026-08-09_H3_Dialogue_Timed_Background_Bed_Lock_EXP.json` is a two-pass H3 graph that locks an independent
dialogue-free background bed after an explicit 40Hz latent boundary. The first path is sample-exact
stem assembly. The second path is not source separation or a sample-exact cut: the real H3 Audio
VAE encoded a standard 124-frame window to 206 steps against the 207-step AV clock, so the example
explicitly selects `fit_reported`, and its decoder showed roughly 0.3 seconds of temporal influence
after the latent boundary. Both workflows use placeholders that must be replaced before queuing.

The project also includes the isolated experimental long-video workflow
`examples/workflows/04-long-video/2026-08-09_H3_Long_Video_22F_EXP.json` and API graph
`tests/fixtures/api/long_video_segment_api.json`. They plan and execute one bounded segment
at a time, store only a checksummed AV latent tail, and use a cloned-MODEL object
patch rather than a process-global MiniMax H3 monkey patch. Intermediate segments
must preserve the sampled tail; only a Planner-marked final segment may trim it
and automatically disables the next context checkpoint. The examples use core
`CreateVideo -> SaveVideo`; this avoids VideoHelperSuite's `apad + -shortest`
ending the AAC stream roughly 79-90ms before the video in the tested standalone
segments. A real four-step, three-segment run produced exact A/V stream durations.
A controlled single-case comparison found a single last frame materially worse
than 22-frame context, while video-VAE re-encoding did not beat the direct sampler
latent route and took longer. Audio boundary click risk, long-chain degradation,
and VRAM safety tiers are still unresolved, so the feature remains experimental.

Version 1.5.0 adds a safer accepted-state route without removing that P1 example:

- `tests/fixtures/api/long_video_candidate_accept_api.json` loads only an accepted parent,
  saves a non-mutating candidate, previews it with acceptance off by default, and
  promotes it through a locked, checksummed, atomic manifest only after review;
- `examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Accepted_22F_EXP.json` provides the same
  review-first route as a drag-and-drop frontend workflow;
- replacing segment N is explicit and invalidates every accepted segment after N,
  because those outputs were conditioned on the old parent chain;
- `tests/fixtures/api/long_video_compose_api.json` verifies the contiguous final manifest and
  composes it while holding only one decoded video frame and one segment PCM buffer;
- audio sample boundaries are absolute over the 24fps timeline. The optional cosine
  bridge preserves the exact sample count and reduces an instantaneous value step,
  but it is not proof of perceptually seamless, phase-continuous, or lossless audio.

The Long Video Conditioning node now exposes a default-off identity-reference experiment through
three inputs appended after the old schema. `first_frame_reuse=segment0_only` is the unchanged
default. `persistent_identity_reference` adds non-timeline image references only on continuation
segments while segment 0 remains controlled by the exact `first_frame`. The optional
`persistent_identity_image` accepts a continuation-only face or upper-body crop. The compatible
`single_reference` strategy prefers that crop and falls back to the full first frame;
`scene_plus_identity` sends the full scene and crop as two separate image references and fails
closed when the crop is missing. Persistent and user references together remain capped at nine;
use `task_type=auto` or `Hybrid`.

Old API JSON and widget prefixes remain valid because every new input is appended with a default.
No model or full video history is added, and stable sampling is untouched. A continuation does gain
one or two reference blocks and VAE encodes, so sequence length, runtime, and VRAM can rise and the
references can compete with motion context. The original full-first-frame strategy failed its
32-second identity-depth gate: source cosine fell from 0.613 in continuation 1 to 0.134 in
continuation 7.

A motion-rich three-seed crop-only probe improved legacy by pooled mean/median
+0.08272/+0.09845 on 54/59 paired detected frames, but one seed regressed against the full-scene
strategy. The subsequent scene-plus-crop strategy completed all six two-segment cold chains and
12/12 prompts with byte-identical segment 0 inside each comparison. It improved legacy by
+0.11278/+0.11628 on 56/59 paired frames and the full-scene strategy by +0.08454/+0.09455 on
52/60; every seed had a positive median delta and contact sheets showed active football/arm motion.

One independent scene-plus-crop 32-second/eight-segment chain then completed 8/8 prompts without
OOM, retry, or cache reuse and produced exact 768-frame/32.000-second video and audio. Continuation
identity medians were 0.699/0.639/0.644/0.609/0.737/0.601/0.574, for a last/first retention ratio
of 0.821. It improved legacy by pooled +0.42945/+0.46670 mean/median and the full-scene strategy by
+0.33771/+0.35802. Minimum free VRAM was 3906.07 MiB and post-15-second occupancy returned to
1231.63 MiB in this fixed local profile only. The predeclared motion gate still failed: continuation
2 and 5 flow-P90 ratios were 0.546/0.538 versus legacy, and continuation 5 temporal MAD was 0.647.
Two-fps strips show continuing ball, arm, and pose changes rather than a freeze, but the
action-amplitude/trajectory warning is real. The feature therefore remains EXP and default-off;
unseen-seed/multi-source intermediate validation and a motion-regression remedy are required before
any three-seed 60-second matrix or identity-lock/action-safe/memory-safe claim.

The accepted-state implementation has synthetic MP4 tests and one real 124/102/102-frame
H3 three-segment compose A/B. The 5ms bridge reduced the two post-AAC boundary jumps from
about 0.04226/0.03509 to 0.00434/0.00704 while preserving the 328-frame timeline. This is
an amplitude-discontinuity metric, not a listening test; the later 14-segment run below adds
one long-chain case, while multi-material validation is still required before any seamless,
arbitrary-length, or no-OOM claim.

Version 1.6.0 adds a human-reviewed total-duration resume route:

- `examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json` is the recommended
  frontend graph and is installed under `MiniMax H3 T8/04-long-video/`;
- `tests/fixtures/api/long_video_auto_resume_api.json` is the equivalent API graph;
- one `MiniMaxH3LongVideoOrchestratorT8` input defines the total duration, a fixed
  legal H3 render window, overlap context, global/per-segment prompts, a seed policy,
  steps, video/audio shifts, sampler, and scheduler;
- the same sampling outputs drive `MiniMaxH3DualClockSamplerT8` and the candidate's
  machine-generated `sampling_summary`, so changing a real sampler parameter cannot silently
  leave the accepted-state identity at an old manually entered value;
- total duration is quantized once at 24fps. With the 124-frame window and 22-frame
  context, 60 seconds is exactly `124 + 12*102 + 92 = 1440` output frames while every
  internal sampling window remains 124 frames;
- the accepted manifest length selects the next segment. Conflicting accepted timeline
  settings are rejected, and a complete final manifest blocks downstream sampling;
- the workflow deliberately remains review-first: queue a candidate with acceptance off,
  preview it, accept it in a separate queue, reset acceptance to false, then queue the next
  segment. It is not a background auto-queue, pause/cancel, or automatic model-unload system.

Version 1.7.0 adds a separate, explicitly enabled background route without changing the
review-first workflow:

- `2026-08-09_H3_Long_Video_Background_22F_EXP.json` and `long_video_background_api.json` connect
  `Background Start` before expensive work and `Auto Accept & Continue` as the only terminal;
- `2026-08-09_H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json` is the ready-to-import
  two-image variant: the full scene drives exact segment 0, while a same-subject face or
  upper-body crop joins the full scene as two continuation-only identity references;
- `review_only` is the safe default. `auto_accept_and_continue` accepts every successful
  candidate without human review and validates/queues exactly one next prompt at a time;
- node buttons and REST routes expose status, pause-after-current, resume, and targeted cancel;
- retry reuses the exact API prompt. It never silently lowers resolution, frame count, context,
  sampler settings, steps, or seed, and stops after the configured additional attempts;
- the selected policy is requested after every durable acceptance, including continue, pause,
  and final. `clear_execution_cache` sets `free_memory=true` with `unload_models=false`;
  `unload_all_models` is a stronger global ComfyUI unload, not an H3-only release; `keep_loaded`
  requests neither. A release failure preserves the accepted manifest and stops without regeneration;
- `background_job.json` persists state and a prompt hash, not the prompt body. Error persistence
  excludes `current_inputs/current_outputs`, media tensors, and prompts. After a server restart,
  status reconciles a stale active prompt to `detached` and reports whether to queue the workflow
  once or compose an already complete manifest. Queue once to reattach the in-memory prompt
  snapshot to an incomplete accepted manifest;
- automatic final composition is optional. A post-accept composition error stops the job and
  leaves the complete manifest for the standalone Compose Accepted node instead of retrying from
  an already advanced manifest;
- prompt snapshots remove ComfyUI's runtime `is_changed` cache fingerprints before requeueing.
  Without this sanitization, `keep_loaded` can incorrectly cache the entire next segment instead
  of rerunning the orchestrator, sampler, save, and terminal nodes.

Live model-free checks completed two-segment auto queue/composition, pause-after-current then
resume, targeted cancellation with no accepted manifest, and one exact-prompt retry followed by
bounded failure. A real H3 executor probe used FL2VA INT8, Standard Turbo EMA LoRA, NVFP4 CLIP,
both H3 VAEs, 256x256, a 124-frame window, 22-frame context, one step, DynamicVRAM headroom 2GiB,
and `unload_all_models`. Two distinct prompt IDs succeeded; manifest revision 2 contains
`124+20=144` frames, and the final H.264/AAC video and audio streams are both exactly 6.000s.

A repeated release-policy matrix used three paired seeds at 736x416, four steps, a 124-frame
window, 22-frame context, and two segments. Each policy ran in three fresh-process cold trials,
then one unmeasured same-process primer plus three measured warm trials. All 21 chains succeeded
without retry/OOM; the 18 measured chains were bit-identical per seed across segment videos,
accepted AV-tail tensor payloads, and final H.264/AAC files.

| Policy | Cold runtime mean | Warm runtime mean | Cold/warm device-peak mean | Cold/warm post-15s mean |
|---|---:|---:|---:|---:|
| `keep_loaded` | 170.89s | 153.10s | 13,449.95 / 13,467.30MiB | 8,083.22 / 7,987.22MiB |
| `clear_execution_cache` | 188.28s | 185.08s | 13,434.52 / 13,408.94MiB | 1,229.63 / 1,229.63MiB |
| `unload_all_models` | 189.08s | 197.13s | 13,421.52 / 13,384.03MiB | 1,229.63 / 1,229.63MiB |

Every paired peak difference was below the project's 128MiB material-difference threshold.
`keep_loaded` gained 17.39s cold and 31.97s warm versus the default, but retained about
6.85/6.76GiB more device memory. Global unload gained no repeatable peak advantage and was
12.05s slower than the default in warm trials. `clear_execution_cache` therefore remains the
balanced default. The result remains specific to this local model/GPU/profile.
This is a mechanical background/reload result, not a quality benchmark or a general four-step,
high-resolution, cross-GPU `memory_safe` claim.

A real hard-kill probe terminated ComfyUI after segment 0 was durably accepted at manifest
revision 1 and the next prompt was active. On restart, status reported `detached`, one accepted
segment, and `queue_workflow_once`. Requeueing the same workflow resumed at segment 1 under a new
job linked to the previous job, without changing or rewriting segment 0's candidate, video, or AV
tail tensors. Revision 2 completed with 144 frames and exact 6.000s A/V/container durations;
whole-device recovery peak was 13,537.02MiB in the native-v2 repeat. Two independent real H3 chains then interleaved on
the same ComfyUI queue as `A0 -> B0 -> A1 -> B1`; both completed under isolated prompt/job IDs,
parents, manifests, output roots, and final files, with a 13,511.44MiB device peak and no OOM.

The manifest commit lock is now an OS-owned `manifest.lock.v2`, automatically released on process
death. Same-host Windows/NTFS tests passed one-winner same-slot acceptance, four processes times
25 protected updates with all 100 retained, and forced owner termination followed by acquisition
within two seconds. A chain-wide background lease rejects a second ComfyUI process before it can
generate; after killing the first owner, a third process reattached with the old `previous_job_id`.
Live legacy locks are respected, dead legacy residue remains rollback-compatible, unknown schemas
fail closed without backup rollback, same-schema additive fields survive a later write, and corrupt
auxiliary background state is quarantined before manifest-led recovery.

Accepted manifests and background states now use separate schema-2 format markers. A valid
schema-1 file is normalized to schema 2 in memory without changing the raw file on read. The next
protected manifest write atomically upgrades the primary while preserving the raw schema-1
backup; background reattachment writes schema 2 and records the previous schema. Read-only
validation against an existing real H3 schema-1 chain preserved both original hashes, and the new
hard-kill plus dual-chain real probe wrote native schema 2 before and after recovery. Unknown
future schemas still fail closed.

Eight deterministic acceptance fault-injection cases now cover missing/corrupt accepted-asset
repair, candidate-id and archived-path collision refusal, context-copy failure, failure after the
backup write but before primary replacement, and missing-primary recovery. A valid backup is now
authoritative when the primary is absent even if the caller permits a new chain; an unknown or
corrupt backup cannot be replaced by an empty manifest. Retrying the same reviewed candidate after
a copy/commit failure completes without losing the prior revision. These tests cover named program
boundaries, not arbitrary CPU instructions or storage power loss.

The two highest-risk points also passed real Windows/NTFS process termination. A worker held the
actual `manifest.lock.v2` and was killed after either (a) the accepted MP4 was fully copied but the
context and manifest were not, or (b) revision 1 was written to the backup but revision 2 had not
replaced the primary. Three independent rounds per pair produced 6/6 successful recoveries: the
OS lock released automatically, the old/no manifest remained authoritative, and the same
candidate recommitted in under two seconds with valid hashes and revisions. No worker remained.
This used small test media rather than a live H3 CUDA generation and does not emulate machine power
loss or a network filesystem.

This validates the local on-disk schema-1 to schema-2 contract. It does not validate a complete
upgrade/downgrade matrix across separately released plugin/ComfyUI builds,
network/shared-filesystem locking, simultaneous CUDA execution, or multi-GPU parallelism.

A representative four-step background chain was then completed on the local RTX 4060 Ti 16GB
with FL2VA INT8, Standard Turbo LoRA, 736x416, a 124-frame window, 22-frame AV context,
DynamicVRAM headroom 2GiB, and global `unload_all_models` between segments. All 14 distinct
prompts succeeded once without retry or OOM. Manifest revision 14 is exactly
`124 + 12*102 + 92 = 1440` frames and 1,920,000 audio samples; the automatic H.264/AAC video,
audio, and container streams are all exactly 60.000 seconds. Half-second polling observed a
12,823MiB whole-device peak and 3,556MiB minimum free margin. The running maximum was essentially
flat after segment 3, with only a further 39MiB increase at segment 9 and no later staircase.
All 13 visual contact pairs avoided an obvious hard cut in this walking-shot sample. Audio
adjacent-window level change still reached 13.75dB; the 5ms bridge reduced median single-sample
jump about 96.2% but cannot repair level, semantics, rhythm, or lip sync. This is one local
prompt/seed profile, not a cross-GPU or universal memory-safe claim.

A follow-up real 256x256 one-step/two-segment probe verified the corrected final-release timing.
At completion, whole-device use was about 8,124MiB and state recorded
`last_release_policy=unload_all_models`; within 15 seconds it automatically fell to about 1,230MiB,
a 6,894MiB drop without a manual `/free` call. This validates the final release request, not
universal side-effect-free reload behavior for every third-party model.

All 146 current project tests, Ruff, ComfyUI whitelist import, and live `/object_info`/route probes pass
for this checkpoint. The live instance registered 25 T8 nodes. The earlier 1.6.0 checkpoint exposed the
auto-resume workflow through `/userdata`. A real DynamicVRAM probe using non-pruned FL2VA
INT8, Standard Turbo LoRA, NVFP4 H3 CLIP, both H3 VAEs, 736x416, a 124-frame internal
window, one step, and a one-second request also completed candidate generation, acceptance,
complete-chain blocking, and accepted-file composition. Candidate and final outputs both
contain 24 frames at 24fps with exactly 1.0-second video, audio, and container streams.
This is execution-path evidence, not a four-step/multi-segment quality or VRAM safety result.
The post-probe single-source sampling hardening was also rerun against the real model: changing
only Orchestrator `steps` to 1 drove the sampler and produced candidate metadata
`1-step dual_clock_euler/native_flow shift12/3`; acceptance, complete-chain blocking, and
composition succeeded again.

A later real four-step auto-resume probe requested six seconds and generated two accepted
segments: 124 frames followed by a 20-frame final segment conditioned on the accepted 22-frame
AV tail. The final manifest covers exactly 144 frames, and both unbridged and 5ms-bridge outputs
contain 24fps/144 frames with 6.000-second video, audio, and container streams. The post-AAC
single-sample boundary jump fell by about 80.2%, but the adjacent audio windows still differed
by about 33.3dB in level. The video contact sheet has no obvious identity or composition cut,
yet boundary MAD and SSIM discontinuity were the largest among 16 nearby intra-segment
transitions. Device peaks were about 15,461.4 and 16,181.5MiB; the latter leaves only about
198MiB, below the 512MiB safety gate. This is a successful bounded two-segment execution, not
a seamless-audio, long-chain, or 16GB-safe result.

A subsequent uninterrupted four-step DynamicVRAM run completed the full 60-second plan in
14 accepted segments: `124 + 12*102 + 92 = 1440` frames. Both the unbridged and 5ms-bridge
assemblies report 24fps/1440 frames and exactly 60.000-second video, audio, and container
streams. No explicit `/free` request was issued between segments. The 14 measured device peaks
ranged from about 15,480.0 to 16,228.2MiB; the descriptive warm-peak slope was about
+28.0MiB/segment and the baselines did not form a monotonic staircase. This single run therefore
does not show a cumulative VRAM leak, but five segments left less than 512MiB and the worst left
only about 151.3MiB. It is not a validated 16GB safety tier, and 0.25-second polling may miss
shorter spikes.

Across all 13 video seams, median/max MAD were about 0.01618/0.01906 and median/min SSIM were
about 0.96374/0.92868. Contact sheets do not show a hard subject/background cut at the worst
seams, but the 14-segment timeline shows gradual appearance and exposure drift; these metrics do
not prove identity preservation. Audio degradation is material: the median adjacent half-second
level change was about -9.51dB, the largest absolute change was about 40.83dB, and the final
segment's above-8kHz energy ratio was about 36.30dB below the first segment. The bridge reduced
the median post-AAC single-sample boundary jump by about 97.23%, but cannot restore level,
timbre, speech semantics, or lip sync. This proves the bounded/resumable 60-second execution
path, not seamless or lossless long-form quality. The later fixed-prompt three-base-seed cold
gate closes only mechanical/memory repeatability; different prompts/materials, same-seed
whole-chain warm repeats, dialogue/lip-sync, fast motion, rhythmic music, blind listening,
ASR/speaker checks, and cross-configuration VRAM profiles remain open.

After accepting the final segment, the first uncached full re-queue can terminate with the
expected ComfyUI `ExecutionBlocked` status (empty traceback, execution stops at the
Orchestrator). A cached re-queue may instead report success while running only the review node.
Both paths leave the candidate count at 14 and perform no extra sampling; the former is a safe
completion signal rather than a generation failure.

The 5-frame versus 22-frame comparison now includes repeated 0.3M and 0.6M matrices rather than
only single probes. Each resolution used two accepted-segment-0 baselines whose MP4 and video/audio
tail tensors were bit-identical. Three paired seeds were run in alternating order with a fresh
isolated ComfyUI process for every cold trial, then again after a same-process primer for the warm
matrix. Every matching context+seed cold/warm candidate was bit-identical, and VRAM was polled at
0.10-second intervals.

At 736x416, the cold 5/22-frame absolute device-peak means were 15,279.5/15,224.0MiB, while paired
`22-5` differences were +96.6, -78.3, and -184.9MiB. In contrast, the sampler PyTorch-pool means
were repeatably 3,189.9/3,495.3MiB: 5 frames saved about 305MiB locally. Cold runtime means were
86.53/93.08 seconds and warm means were 69.27/78.01 seconds. The warm process reached only 97.6MiB
free, with five of six measured trials below the 512MiB gate. Across these three seeds, the 22-frame
route had better mean video MAD/SSIM and audio level/NCC evidence.

At 1056x608, cold 5/22-frame absolute peak means were 15,739.0/15,724.2MiB, but paired differences
swung from -752.0 to +696.7MiB. Sampler-pool means remained stable at 5,753.4/6,381.2MiB, so 5 frames
saved about 628MiB of local pool and reduced cold/warm runtime means from 230.38/218.40 seconds to
200.29/187.89 seconds. All six warm trials failed the 512MiB margin gate; the minimum margin was
33.6MiB. Five frames had lower mean MAD/higher SSIM here, which may also reflect suppressed motion;
22 frames had much better mean audio level/NCC, and one of its three contacts showed a clear
front-to-profile boundary change.

The 39-frame treatment then reused the 736x416 baselines and the same three seeds for three fresh
cold starts and three post-primer warm runs. All six runs succeeded, matching cold/warm outputs
were bit-identical, and the 5/22/39 segment-0 MP4 plus AV tail tensors were identical. The
39-frame sampler-pool mean was about 3,799MiB, a repeatable 303-304MiB increase over 22 frames;
cold/warm runtime means were 101.65/87.38 seconds. All three warm trials failed the 512MiB gate,
with only 77.35MiB free at worst. Manual inspection found one acceptably continuous boundary,
one visible pose/framing jump, and one severe identity/shot discontinuity.

Five frames therefore remains `fast_context_5_experimental`: its compute and sampler-pool savings
are repeatable, but an absolute device-peak advantage is not. Twenty-two frames remains the current
balanced default candidate. Thirty-nine frames is now `context_39_high_risk_experimental`, not a
quality or safety tier. The 1056x608/39-frame treatment was denied by the predefined gate rather
than forced: its 22-frame warm control already had all six trials below 512MiB and only 33.6MiB
free at worst. This is not evidence that 39 frames must OOM; it is evidence that the unchanged
configuration cannot establish a safety tier and carries material OOM risk. No configuration
receives a `memory_safe` label before a hardware/model/resolution/plugin-specific gate passes.

A later controlled memory-policy matrix fixed DynamicVRAM headroom at 2.0GiB. Stock and Sage
each completed three cold and three warm trials at 736x416 and 1056x608 with every trial above
512MiB and matching strategy+seed cold/warm outputs bit-identical. Default Block Cache hit 0 of
4 forwards and cannot skip the first full forward. Sage was faster but produced a higher
whole-device peak than Stock at equal headroom and material shot/pose/trajectory divergence in
two of three 1056x608 seeds, so it remains a high-risk approximate speed experiment.

Stock+headroom-2.0 then completed a second uninterrupted 60-second/14-segment 736x416 chain with
2739.41MiB minimum free margin. Both assemblies are exact 24fps/1440-frame, 60.000-second AV
streams. Relative to the previous same-prompt/same-seed Stock headroom-0.5 chain, all 14 segment
MP4 hashes and all 13 continuation AV tensor payloads were identical; median peak fell by about
2635MiB and total generation time increased about 1.63%. This is a validated local conservative
profile for the exact RTX 4060 Ti 16GiB/model/resolution/window/context/plugin contract, not a
general `memory_safe` tier or never-OOM promise.

The same conservative profile was then repeated as three independent ComfyUI cold starts with
base seeds `2608082000`, `2608083101`, and `2608083202`, while prompt, model, canvas, render window,
context, and sampling remained fixed. All 42/42 segments completed once without OOM, retry, or
candidate reuse. Manifest/parent/revision continuity, candidate and accepted video/context
SHA-256 values, 1440 frames, 1,920,000 samples, completion blocking, and six exact 60.000-second
assemblies were independently verified. Per-chain maximum peaks were 13,640.09, 13,414.01, and
13,426.72MiB; the worst free margin was 2739.41MiB and no segment fell below 512MiB. This closes
the fixed local profile's cross-base-seed cold-start mechanical/memory gate, not same-seed
whole-chain warm repeats, cross-prompt/material coverage, other GPUs, or desktop-load profiles.

The long-term quality gate failed. All three 14-segment middle-frame timelines accumulate facial
age and identity drift, most severely for seed `2608083101`. Across the three chains, maximum
adjacent half-second audio level gaps were 23.59-48.06dB, descriptive NCC medians were only
0.127-0.206, and the final segment's above-8kHz energy ratio was 9.66-36.30dB below the first.
The 5ms bridge reduced median post-AAC single-sample jumps by 94.93%-97.33%, but cannot repair
level, timbre, semantics, or recursive dulling. The local report is
`artifacts/long-video-generation-check/stock-headroom2-60s-multiseed/analysis/REPORT.md`.

On the same ComfyUI commit, a core-only `EmptyMiniMaxH3LatentAV -> VAEDecodeAudio`
graph independently reproduces a CUDA-input/CPU-filter mismatch under `--novram` at the
MiniMax H3 audio VAE upsampler. The T8 AV Decode node is therefore not the source of that
failure. The DynamicVRAM route succeeds; `--novram` H3 audio decode is not advertised as
compatible until ComfyUI or a separately validated local workaround resolves the buffer move.

## Reproduce the conversion

```powershell
$sourceDir = '<path-to-source-loras>'
$outputDir = '<path-to-converted-loras>'
python .\tools\convert_minimax_h3_lora_for_comfyui.py `
  "$sourceDir\minimax_h3_turbo_4步加速.safetensors" `
  "$sourceDir\minimax_h3_turbo_4步加速ema.safetensors" `
  --output-dir $outputDir
```

The converter is strict: it checks the MiniMax-H3 metadata, all 259 expected
adapter modules, all 518 tensor names/shapes/dtypes, and bitwise tensor equality
after saving. It writes through a temporary file and never changes the sources.

For the exact 10Eros curve-pruned checkpoint, use the separate no-overwrite tool rather than the
generic prefix converter:

```powershell
$sourceLora = '<path-to-converted-LightX2V-LoRA>'
$targetModel = '<path-to-exact-pruned-model>'
$timeReference = '<path-to-full-FL2VA-time-reference>'
$outputLora = '<new-output-path>'
$coreAblation = '<optional-new-core208-output-path>'
python .\tools\convert_minimax_h3_turbo_for_pruned_curve.py `
  --lora $sourceLora `
  --pruned-model $targetModel `
  --time-embedder-reference $timeReference `
  --output $outputLora `
  --core208-output $coreAblation `
  --expected-lora-sha256 35946f9f2957c2766e28b627c88169535249dd07a3040ce3c2c8c99951fdbc7b `
  --expected-pruned-model-sha256 f82cc3f723b080e7ae94a7c98f95aa989e387618d0bdc940133dfbd9f432c062 `
  --expected-time-reference-sha256 7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5 `
  --expected-table-sha256 ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e
```

The main output has 259 A/B adapters plus 51 FP32 `.diff_b` tensors (569 tensors total). The tool
keeps the 208 directly compatible adapters and every AdaLN B bit-identical, fits the 51 A tensors
over the 1025-point time curve with an affine intercept, refuses existing outputs, validates readback,
and re-hashes all three inputs after publication. The generated SHA-256 is
`6c2f38d45dfa3fc282a48de3171b6946a5e6d46e13f832c43b93734f6d12edf5`. Use the bypass loader at
strength 1.0 first. The current evidence is one 256x256/124-frame four-step AV smoke plus static
validation, not a high-resolution or multi-seed quality release. Generated model files and local
sidecars remain outside Git; see `VERIFICATION_REPORT.md` for the durable public summary.

## Variant choice

- Standard: usually sharper on fast motion.
- EMA: time-averaged; usually smoother.

These descriptions come from the upstream model card. Compare them with the
same prompt and seed.

## Verified mapping

| Source module | ComfyUI target | Count | Full/non-pruned | Pruned |
|---|---|---:|---:|---:|
| `blocks.*.{attn,mlp}` | `diffusion_model.blocks.*.{attn,mlp}` | 200 | 200 | 200 |
| `blocks.*.adaln_proj.linear` | prefixed same path | 50 | 50 | 0 |
| `token_refiner.blocks.*.{attn,mlp}` | prefixed same path | 8 | 8 | 8 |
| `final_layer.adaln_proj.linear` | prefixed same path | 1 | 1 | 0 |
| **Total** |  | **259** | **259** | **208** |

The files target the generic ComfyUI LoRA convention recognized by
`comfy.lora.model_lora_keys_unet()` and preserve the upstream scale semantics:
no `alpha` tensor means scale `1.0`, matching `W_eff = W + B @ A`.

## Sources checked

- [Original MiniMax-H3 Turbo LoRA repository](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- [Upstream ComfyUI conversion discussion #1](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/1)
- [Upstream sampler/loading discussion #6](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/discussions/6)
- [Official ComfyUI MiniMax-H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI LoRA key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py)
- [ComfyUI bypass loader](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_lora_debug.py)

## Skin Finish file-output safety

The opt-in Skin Finish P1 file nodes are source-selected by default. When a candidate is explicitly
accepted, all three routes copy approved source-audio packet payloads, encode only the SDR video stream
with single-thread libx264, and run single-thread FFmpeg `-xerror -err_detect explode` validation
before atomic publication. A missing FFmpeg executable or any decoder diagnostic fails closed and
leaves the source untouched.

All three file-output routes share `sdr_8bit_rec709_compatible_v1`. The check does not rely only on
ComfyUI's reported bit depth: it reads PyAV pixel-component bit counts, pixel-format names and FFmpeg's
integer color-primary, transfer and matrix enums. It accepts unmarked or conventional BT.709/legacy
SDR-compatible 8-bit sources and rejects higher-bit-depth formats, PQ, HLG, linear/Log transfer,
BT.2020/P3 primaries and BT.2020/ICTCP-style matrices before encoding. Approved source primaries,
transfer, matrix and range metadata are copied to the H.264 output and recorded in the report. This is
an explicit rejection/preservation contract; it is not HDR tone mapping or wide-gamut support.

`MiniMaxH3SkinFinishVideoStreamT8Advanced` accepts an untrimmed file-backed `VIDEO` directly. Its
first pass retains only pinned-YuNet face metadata and small source digests; the second pass processes
at most the configured bounded frame chunk and immediately encodes it. It does not materialize a
complete ComfyUI `IMAGE` batch. This reduces memory owned by this post-process only and is not a
claim about the H3 generation peak, arbitrary video lengths, codecs, HDR or universal 16GiB safety.

`MiniMaxH3SkinFinishQualityVideoStreamT8Advanced` preserves that released node's schema and default
behavior, but installs a private bounded chunk processor only inside the new append-only node. Pass 1
still retains YuNet metadata only. Pass 2 lazily loads the pinned CPU ParseNet, builds semantic skin
masks, applies the non-generative Skin Finish, source-detail Frequency Split and Texture Guard, then
runs Safety Audit with exactly one prior source/candidate/mask frame at each chunk boundary. The model
is released in `finally`; no full IMAGE candidate or semantic-mask batch is retained. With
`accept_candidate=false` it performs no analysis, does not load ParseNet and writes no file. A
960x544x5, two-CPU-thread real probe used three chunks with a two-frame peak, found semantic skin on
5/5 frames, had zero frequency/texture/audit rejection, strictly decoded H.264, preserved audio packet
payloads and decoded PCM exactly, and observed about 1.88GiB peak process working set versus about
10.37GiB in the earlier complete 124-frame IMAGE-chain diagnostic.

One unique 736x416x768, 24fps, 32-second H3 final file was then processed exactly once with two CPU
threads and two-frame chunks. The run completed all 384 chunks in 1579.530986 seconds with about
1.974GiB observed peak process working set. ParseNet was semantic-ready on 690 frames; 78 frames were
kept as exact source because no reliable semantic candidate survived the source-bound gates. Frequency
Split and Texture Guard each rejected 26 affected frames to source, while Safety Audit rejected zero
frames or chunks. The largest cross-chunk treatment jump was 0.00052124. The 768-frame H.264 candidate
strictly decoded, source/candidate audio packet payloads and decoded PCM were exact, and ParseNet was
released without persistent cache. The report is
`artifacts/skin-finish-quality-stream-long-32s-20260825/validation_report.json`, SHA-256
`31B3033E507CBDCC87933EC75CD61037EC1047306B46F70804D931F4A8B2D2F8`. This closes one 32-second
bounded resource and media-preservation gate, not visual preference, arbitrary duration, repeated-run
or universal memory certification. Anonymous human review ID `9f33c46592ab` resolved to
`ABSTAIN_SOURCE_INSUFFICIENT`; the source did not visibly expose enough oily-skin defect for an
aesthetic decision.

A separate dated Oil Control Stream workflow uses the same node without changing its schema or
defaults. It pins `oil_control`, amount 0.35, texture retention 0.90, shine control 0.35, two-frame
chunks and CRF 16 for footage that visibly contains forehead, nose-bridge or cheek shine. On the
requested v1.0 eight-step LoRA speaking close-up, one 960x544x124 run processed all 124 frames with no
source fallback or Frequency Split, Texture Guard or Safety Audit rejection. It strictly decoded,
preserved source audio packet payloads and decoded PCM exactly, and reached about 1.998GiB peak process
working set. Review ID `d4eb04003a44` resolved to `ABSTAIN_UNSURE`: eight criteria were ties, two were
abstentions, neither side had a hard failure, and the reviewer wrote that they seemed about the same.
This closes the pending review without establishing a visible benefit. The workflow remains a targeted
starting point, not permission to force visible treatment onto footage with no oily-skin defect.

`MiniMaxH3SkinFinishSpecularFrequencyT8Advanced` is an append-only experimental alternative to the
ordinary Frequency Split, not a replacement. It runs the unchanged split first and then restores only
the darker highlight treatment that split lost where bright semantic skin, positive local source
detail and the input candidate agree. The RGB correction is a convex interpolation between the
frequency result and the input Skin Finish candidate, so it cannot invent a stronger subtraction than
that candidate. A suppression value of zero returns the ordinary split candidate exactly; source mask
exterior, alpha or auxiliary channels and AUDIO remain unchanged, and source stays selected by default.

A single six-frame CPU calibration used the same balanced raw Skin Finish parameters for the ordinary,
0.35 and 0.65 specular routes. All routes passed Texture Guard and Safety Audit. Final mean luma change
over the brightest skin decile was -0.00006744, -0.00014774 and -0.00020280 respectively, while the
source-relative texture proxy was 0.99780405, 0.98907673 and 0.98415595. The labelled contact sheet
still looked subtle, so no full-video run, default workflow or efficacy claim was added. A subsequent
candidate-bounded 3% run used maximum raw oil-control parameters: ordinary, 0.65 and 1.0 retained only
24.5%, 29.7% and 32.5% of raw average treatment while preserving texture proxies of 0.99784, 0.99344
and 0.99124. All six frames passed both guards, but the visual difference remained weak.

For diagnosis only, the pinned CineStyle `e7d5fac` file was separately downloaded and dynamically
executed on the same six frames and exact T8 semantic mask; none of its code was copied or vendored.
Its defaults were more visible: raw mean change was 0.01130743 with texture proxy 0.66553, and the T8-
guarded result retained 0.00675145 with texture 0.88940. However, the brightest skin decile became
brighter by 0.00382215 and the raw upstream output was not bit-exact outside the supplied mask. This
shows a stronger smoothing/brightening trade-off, not a scientifically better oil-control answer.
The T8 node remains display-referred SDR candidate-intent restoration, not physical reflectance
separation, deblur or pore reconstruction.

`MiniMaxH3SkinFinishSurfaceT8Advanced` is the subsequent clean-room append-only candidate. It uses an
independent scalar-luma guided-filter base and bounded RGB surface correction on each frame; it does
not copy CineStyle's Matchbox passes, constants, weights or dependencies. The first version treated
only compact positive detail and was measurably safe but still visually weak. Wide oily highlights can
live in the guided base and therefore have little positive high-frequency residual. The corrected node
adds a bounded photographic luminance shoulder over that guided base. This is display-referred tone
finishing, not the skin-reflectance, illumination and geometry model required for physical facial-
specular separation.

Defaults remain conservative: amount 0.65, surface smoothing 0.70, texture keep 0.85, compact-highlight
compression 0.65, broad-highlight compression 0.45 from luma 0.68 to 0.94, blemish balance 0.35, a 2%
short-side radius capped at 32 pixels, two-frame CPU chunks and `accept_candidate=false`. Mask
exterior, alpha or auxiliary channels and AUDIO remain exact source contracts; mask, texture, change
and clipping failures reject that frame to source.

The corrected low-load calibration compared only current Quality Stream and one Surface candidate on
the same pinned six frames. Both passed Texture Guard and Safety Audit with exact mask exterior. Final
masked mean change increased from 0.00013440 to 0.00840873 while the texture proxy remained 0.99715394;
the brightest-skin-decile luma change increased from -0.00002611 to -0.01996538. A subsequent unique
960x544x124 file-stream validation completed 62 two-frame chunks with 124/124 semantic faces, two
Surface and two Texture Guard source fallbacks, zero Safety Audit failures and maximum temporal effect
jump 0.00381594. Strict video decode passed; all source audio packet payloads and decoded PCM were
exact. The CPU-two-thread run took 944.394768 seconds and peaked at about 1996.301MiB process working
set without loading H3 or SAM. Candidate SHA-256 is
`0DD7F64AA7B1E16C893C27B20165A79B25A45294EA5029EF468AF8C8EAF7D0E7`. Anonymous review
`b3aad4e0d57b` is complete and hash-bound. The source won overall, skin naturalness, shine/highlight,
tone evenness and halo/edges; the other five criteria were ties, the candidate won none and neither
side had a hard failure. The Surface candidate therefore remains disconnected from workflows and has
no perceptible-benefit claim. This result rejects promotion of this parameter set; it does not prove
that all possible guided-surface methods are ineffective.

Surface v2 addresses the rejected dimensions without merely increasing the same shoulder. Broad
highlight energy is now the positive difference from a larger, mask-weighted local skin-illumination
estimate; uniformly bright skin therefore receives no broad correction. A two-pixel inside-only gate
fades treatment to zero at hard semantic-mask boundaries, and probability-valued ParseNet masks use
their positive support for geometry while retaining their original confidence for blending. The same
box average is evaluated as mathematically equivalent horizontal and vertical passes, reducing the
six-frame Surface stage from about 57.18 seconds to 4.35 seconds on two CPU threads.

The sole v5 static candidate (`0.90 / 0.25 / 0.96 / 0.90 / 0.90 / 0.10 / 2.5%`) passed all six frames.
After Texture Guard its masked mean change was 0.00387333, brightest-skin-decile luma change was
-0.00787700, texture proxy was 0.99280846 and the two-pixel-boundary/interior change ratio was
0.67757654. One 960x544x124 stream then completed 62 two-frame chunks with 124/124 semantic faces,
zero Surface or Texture Guard fallback, zero Safety Audit failure, maximum internal temporal jump
0.00173517, exact 163-packet audio payload and exact decoded PCM. Runtime was 741.01245 seconds and
peak working set was about 1997.340MiB without H3 or SAM. A mapping-blind public A/B audit reported
maximum ROI temporal jump 0.00052624 and p99 difference edge 0.01638918. Anonymous review
`8e89bff3bc95` completed with all ten criteria tied, no candidate or source wins and no hard failures.
This removes the clear subjective regression seen in v1 but does not establish a perceptible benefit.
The node therefore remains experimental and disconnected, with no default or workflow promotion.
Further work should use a materially different surface model rather than stronger tuning of the same
display-referred shoulder.

Accepted Quality Stream runs now perform a host-memory preflight before constructing the processor or
loading ParseNet. Where host available memory is measurable, less than 2,048MiB returns the exact
source VIDEO with `ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN`; no parser is loaded and no file is
written. The floor is evidence-derived from the reviewed run's approximately 1,163.129MiB process
working-set increase and leaves about 884.871MiB additional availability. It is deliberately fixed and
not user-lowerable. `accept_candidate=false` skips even the measurement. Platforms where physical
availability cannot be measured proceed only with the bounded route and emit an explicit report
warning, preserving portability without pretending the RAM gate was checked.

### Skin Finish full-IMAGE RAM preflight

The Basic and Advanced P0 IMAGE routes must retain a complete candidate, two complete float32 mask
outputs and a complete float16 RGB difference image for their public output contract. Before any of
those outputs, face-plan masks or processing chunks are allocated, the node now derives an
incremental CPU-memory estimate from the actual frame count, height, width, channel count, input
dtype, configured chunk size, proxy geometry and mask source. The estimate sums the retained outputs,
mask preparation, bounded full-resolution scratch and proxy scratch, multiplies that component total
by 1.5 and adds a fixed 512MiB headroom. Neither factor is exposed as a user-lowerable widget.

On Windows the gate compares this same required floor against both available physical RAM and
available commit. If either measurable value is below the estimate, execution returns the exact
source with `ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_CANDIDATE_ALLOCATED` before mask preparation or
candidate processing. The zero mask and difference audit outputs are broadcast zero views in this
rejected path, so the fail-closed response does not recreate the very full-batch allocation that was
blocked. If the platform exposes neither measurement, the report says
`ALLOW_MEASUREMENT_UNAVAILABLE_BOUNDED_CPU_ROUTE`; this preserves the existing bounded CPU route but
does not claim that the RAM floor passed.

This is an incremental post-process estimate after the input IMAGE already exists. It is not a total
ComfyUI graph estimator, does not reserve memory atomically against other processes, makes no GPU
memory claim and cannot establish universal 16GiB or arbitrary-workflow safety. File-backed long
video should continue to use the bounded Quality Stream route rather than materializing a complete
IMAGE batch.

### Skin Finish Texture Guard (Advanced EXP)

Place the append-only Texture Guard after an existing Skin Finish source/candidate/mask triplet.
It protects source deep shadows and near-clipped highlights with a smooth exposure gate, then rejects
each complete frame to source if the candidate adds too many clipped pixels or falls below a
source-relative high-pass RMS floor. The default remains source-selected and AUDIO is passed through
as the same object. High-pass energy can include noise, so this node is a mechanical anti-overprocessing
guard rather than an automatic beauty, pore, sharpness, identity or natural-skin score. It currently
assumes ordinary 0..1 SDR display values and makes no HDR, Log, wide-gamut or linear-light claim.

### Skin Finish Semantic Mask (Advanced EXP)

`MiniMaxH3SkinFinishSemanticMaskT8Advanced` is an append-only, optional CPU parser that consumes the
exact source `IMAGE` batch and a source-bound Face Refine Plan. It outputs a semantic skin `MASK`, a
sampled audit preview and a JSON report; it does not modify frames or audio. Connect its mask to
`MiniMaxH3SkinFinishAdvancedT8` with `mask_source=external_exact`, then optionally place Texture Guard
after the candidate. Both acceptance switches remain false in the example workflow.

The only accepted checkpoint is
`ComfyUI/models/facedetection/parsing_parsenet.pth`, exactly 85,331,193 bytes with SHA-256
`3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2`. Runtime download and arbitrary
model paths are absent; PyTorch must support `torch.load(..., weights_only=True)`. Missing FaceXLib,
missing or mismatched weights, stale source geometry/pixels, malformed plans and inference failures
produce an empty mask and `ABSTAIN`/`REJECT`, never a full-screen fallback. The model runs on CPU,
has no persistent cache, and is released in `finally` without unloading any ComfyUI/H3 model.

The pinned checkpoint uses the ParseNet/CelebAMask-HQ order: class 1 is skin, 2 nose, 4/5 eyes,
6/7 brows, 10-12 mouth/lips, 13 hair, 17 neck and 18 cloth. The default selects only skin; nose,
glasses, eyes, brows, mouth/lips, hair, hats, earrings, necklaces and cloth are protected, while neck
and ears are not selected. Do not substitute the differently ordered FaceXLib BiSeNet example list.

The current Face Refine Plan retains an upright face box but not five-point landmarks, so ParseNet
uses an expanded square crop rather than affine alignment. This route describes one selected face
track only. It does not solve SAM3.1 multi-person identity assignment, profile/occlusion safety,
deblur, face reconstruction, natural pores or aesthetic quality. The dated workflow is
`examples/workflows/17-skin-finish/2026-08-24_H3_Skin_Finish_Semantic_Mask_Advanced_EXP.json`.

### Skin Finish Multi-Person Semantic Mask (Advanced EXP)

`MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced` is a separate append-only route for a
source-bound `H3_T8_SAM31_MULTIFACE_TRACK_PLAN`. It does not rerun SAM3.1. For each frame and
shot-local person mask, it chooses one unique pinned-YuNet detection, sorts the viewpoint-dependent
eye and mouth pairs by image x-coordinate, and estimates an OpenCV LMEDS similarity transform from
the five landmarks to the standard FFHQ 512 template. The aligned crop is parsed by the same pinned
CPU ParseNet; skin and protected-feature masks are inverse-warped and intersected with that exact
person track. A single YuNet detection cannot be reused by another track.

An optional `H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT` may add Character labels to the report across
shot resets. Those labels are manual/SFace suggestions, not identity proof, and do not alter masks,
skin parameters or acceptance. Missing landmarks are never propagated from another frame. Invalid
source/plan/assignment hashes, non-ready plans, ambiguous faces, excessive alignment residual,
abnormal skin area, parser failures or insufficient ready-frame coverage return an empty mask and
`ABSTAIN`. The default workflow keeps Skin Finish, Texture Guard and Video Finalize acceptance off.

SAM3.1 is expected to offload after producing the plan. The node then finishes and releases YuNet
before loading ParseNet, runs ParseNet on CPU without a persistent cache, and releases it in
`finally`; it does not call global `unload_all_models()`. The dated workflow is
`examples/workflows/17-skin-finish/2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json`.
One 960x704 six-frame/two-person low-load run completed 12/12 real YuNet five-point and pinned
ParseNet masks, but used deterministic source-bound left/right person regions rather than loading
SAM3.1 again. It therefore proves the parser/alignment/intersection mechanics, not live SAM quality,
automatic scene-cut detection, full-video continuity, identity truth or aesthetic improvement.

### Skin Finish Per-Person routing diagnostics (Advanced EXP)

`MiniMaxH3SkinFinishPerPersonT8Advanced` keeps the existing precedence of exact `shot:track`, reviewed
Character, optional default profile and exact source. Its report now includes, for every resolved
route, a display-referred SDR Rec.709 luma proxy, mean and maximum RGB treatment magnitude, and the
fractions of treated pixels touching low or high clipping. These values are observational review aids;
they never select a candidate or claim skin-tone fairness, beauty, identity or naturalness.

Deterministic CPU fixtures cover two tracks moving through and past each other, exact-source fallback
where their masks overlap, and a two-shot case where track numbers and screen sides swap but the
hash-bound reviewed Character mapping remains stable. A separate dark/light fixture confirms that both
routes receive independent diagnostics, remain finite, avoid new clipping in that fixture and preserve
every pixel outside the owned semantic masks. This closes routing arithmetic and report coverage only.
Real occlusion/re-identification, cross-shot identity truth, different-skin-tone fairness and full-video
human preference still require representative review.

### Skin Finish Dichromatic Specular (Advanced EXP)

`MiniMaxH3SkinFinishDichromaticT8Advanced` is an isolated, append-only research candidate and is not
connected to any workflow. In linear sRGB it applies a neutral-illuminant dichromatic approximation:
the pixel must have a positive achromatic specular estimate, locally diluted chroma and a consistent
direction relative to a masked diffuse-colour estimate before any correction is allowed. Uniform
same-chromaticity bright skin is intentionally unchanged, and near-neutral diffuse colours receive
low confidence because the separation is ill-conditioned. The node is frame-independent, fades only
inside the semantic-mask edge, keeps the exterior and auxiliary channels exact, passes AUDIO as the
same object, enforces bounded-change/texture/clipping gates and selects source by default.

One fixed six-frame calibration passed 6/6. One 960x544x124 bounded file stream then used 62 two-frame
CPU chunks, accepted 124/124 semantic faces, returned six frames to source through the stage and
Texture Guard contracts, reported zero Safety Audit failures, strictly decoded all frames and kept
all 163 AAC packet payloads plus decoded PCM exact. Runtime was 732.857519 seconds on two CPU threads,
with about 1964.262MiB peak working set and no H3/SAM load. The mapping-blind public A/B temporal audit
for review `b2e13261f44e` reported maximum face-ROI effect jump 0.00057450 and maximum p99 difference
edge 0.01626730, with exact PCM and no gross temporal-delta warning. The valid anonymous review then
revealed B as source: source won seven criteria, three tied, candidate won zero, and neither side hard-
failed. After the mapping had been revealed, the reviewer watched again and corrected the qualitative
description to “基本一样”. That correction is post-reveal and therefore does not overwrite the blind
JSON or count as another blind vote. The conservative conclusion is simply that perceptible benefit
was not established. This parameterized route stays disconnected, with no recommended workflow/default
or quality claim. It also does not provide physical BRDF recovery, deblur, pore generation or identity
repair.

## Classic-paper Advanced/EXP nodes (2026-08-28)

This batch adds six opt-in routes without changing any previously released node schema or workflow:

- **RAFT Motion Audit / Mask Propagation** reads a local torchvision-compatible RAFT Small or Large
  checkpoint from `models/optical_flow`. The audit reports motion, cuts and forward/backward
  consistency. Propagation transports a reviewed mask between explicit keyframes; it does not assign
  identities, sharpen frames or repair faces.
- **Trajectory Fun Control** converts normalized bounding-box keyframes into an interpolated path,
  preview and Fun Control conditioning video. It is inspired by creator-facing trajectory control,
  but does not claim to reproduce TrailBlazer's U-Net attention edits.
- **RealBasicVSR Restore** reads an MMagic-compatible checkpoint from `models/upscale_models`, processes
  overlapping temporal windows serially and either returns native size or x4 output. Audio is passed
  through unchanged. A real 32-frame H3 clip showed that strength `0.65` raised mean Laplacian variance
  from 273.10 to 1059.52 but visibly produced over-sharpening and bright edge halos. The default was
  therefore reduced to `0.30`; the same clip reached 531.80 with roughly half the mean pixel change
  (0.00804 instead of 0.01750) and materially less ringing. This is one fixed low-load review, not a
  universal optimum. It is a post-process, so it cannot recover missing identity or fix lip sync.
- **FreeNoise Long Video** supplies a deterministic shared video-noise pool to either in-node long-video
  loop. It leaves audio noise native. Because H3 still renders independent continuation windows, this
  is a noise-rescheduling adaptation rather than a full reproduction of FreeNoise's one-latent sliding
  temporal attention.
- **Dual-Clock AYS Schedule Contract** defaults to the existing native-flow schedule. Manual mode accepts
  only a complete, strictly descending `1 -> 0` base-sigma list and maps it through separate video/audio
  shifts. Schedules published for SD, SDXL or SVD are not labelled H3-optimal.
- **CADS Visual Reference Annealing** applies the paper's condition-noise interpolation and optional
  moment rescaling to visual reference/keyframe latents only. It does not anneal audio references or the
  target audio stream. H3 quality is not calibrated, so use a fixed-seed A/B and review identity,
  endpoint, action and composition adherence before accepting a candidate.

The dated ComfyUI workflows are under `03-image-video-edit`, `04-long-video` and `07-motion-detail`.
FreeInit and PAG are deliberately not exposed: the current H3 joint audio/video flow and packed
attention layout do not yet provide a validated joint re-noising or isolated perturbed-attention
contract. A same-name approximation would have undefined audio behavior or conflicting attention
ownership.

## 1.60.0 compatibility and FastH3 VSA additions (2026-08-30)

- H3 Audio VAE encoding now disables the legacy aligned-length tail crop only for the H3 audio VAE,
  preserving the final non-aligned latent step. Recent cores and non-H3 VAEs remain unchanged.
- FastH3 Preview v1 can now use its 50 learned compression gates with the tile-64, 90% sparse
  Comfy Kitchen VSA API. Plain T2VA is the only supported packed layout; missing capabilities and other
  layouts fall back to dense four-step inference with an explicit report.
- H3 Fun Control resolves the recent official `MODEL_PATCH` contract first and the former ControlNet
  conditioning contract second. Selection is structural and never gated by a filename, hash or byte size.
- `MiniMaxH3LongVideoSamplingPlanT8Advanced` appends an optional tail-subdivision or independent
  low-sigma second pass to both in-node long-video runners. Disabled/disconnected preserves the old route;
  Prompt Relay remains global, EAV remains owned by the main pass, and preview-cache state is shared.

The dated frontend examples are
`10-speed/2026-08-30_H3_FastH3_VSA_T2VA_4Step_0p4MP_Advanced_EXP.json` and
`04-long-video/2026-08-30_H3_In_Node_Long_Video_Prompt_Relay_EAV_Manual_Second_Pass_Advanced_EXP.json`.
These changes make no universal quality, speed, memory or 16 GiB claim.

## Community compatibility additions (2026-08-29)

These six append-only Advanced/EXP nodes leave existing node IDs, widgets and workflows unchanged:

- `MiniMaxH3LoRACompatibilityLoaderT8Advanced` adds the direct MiniMax H3 LoRA module aliases from
  ComfyUI PR #15662, allowing DiffSynth-Studio/ModelScope-style adapters to use ComfyUI's native patch
  loader. Filename and size are informational only; there is no hash, filename or size execution gate.
- `MiniMaxH3TimedImageReferenceT8Advanced` and `MiniMaxH3TimedVideoReferenceT8Advanced` add tagged,
  time-indexed Qwen semantic references without consuming native `minimax_refs` slots. They are semantic
  prompt guidance, not identity or pixel control. Timed Video consumes a decoded CFR `IMAGE` batch and
  explicit source FPS; it does not claim native VFR-container support.
- `MiniMaxH3ChunkedTwoPassPlanT8Advanced` and `MiniMaxH3ChunkedTwoPassUpscaleT8Advanced` divide the
  learned-latent second pass into serial temporal chunks. The default `full_frame_safe` route preserves
  H3's global spatial context inside each chunk, crossfades temporal overlaps, and returns the exact input
  audio latent. `independent_tiles_exp` is retained only for research: a real render showed persistent
  tile-specific texture divergence, so it is not a recommended quality route. No project pixel-area
  ceiling is added; full-frame memory and runtime remain user-owned.
- `MiniMaxH3FastH34StepSetupT8Advanced` configures the published FastH3 Preview v1 **T2VA-only** route at
  four Euler NFE, CFG 1 and video/audio shifts 12/3. Its VSA profile uses the official VSA/Data-Free adapter's
  50 learned compression gates through a compatible Comfy Kitchen `sol_attn`; missing capability falls back
  to dense and is reported. The old `t2va_fl2va` value remains load-compatible but unsupported layouts are
  never presented as validated VSA.

The existing `MiniMaxH3LongVideoSeamDriftT8Advanced` remains the recommended conservative tone route.
Unlike a full-frame `frame_shift/gain_bias/lut` correction, it only proposes bounded RGB gain/offset in
same-shot transition frames, fades the correction out, abstains on cuts/flash/black/HDR-like inputs and
does not touch audio. Use report-only first and accept only after reviewing the seam.
