# Standard H3 → LTX video latent candidate

## Later separate-refiner validation (local, not published)

An explicit new native H3 teacher-eight-step **73-frame** joint AV source was
generated, saved and authenticated; no 124-frame source was silently cropped or
padded. Ordinary and Identity each completed six actual native Chromium
full/fresh refiner pairs: no effect, EAV report/apply, Relay, and combined EAV
report/apply. Each fresh run restores only the verified H3 input and performs
the original three LTX sampler callbacks again. There is no H3 regeneration,
additional upscaler, RGB detour, duration edit or LTX audio conversion.

All 24 outputs strictly decode 73 frames at 832×480/24fps in H264/AAC. Each
full/fresh candidate tensor and complete decoded RGB/PCM matches exactly; every
decoded PCM equals the actual H3 source. The original five selected model files,
pinned adapter/source and immutable inputs stayed unchanged. EAV apply reached
all 48 video blocks for three steps with non-unit gains below the unchanged
1.5 bound; Relay reached all 48 blocks with 288 backend delegate calls.

Effect telemetry still says `observed_not_certified`. A separate matching actual
sampler-return/callback proof is **live-only**, not authorization to restore
completed LTX sampling across processes. These new checks do not qualify the
different RGB route, arbitrary materials/backends, human picture/audio quality
or publication. Historical RGB encoder failures and old accepted clips remain
unchanged. The earlier released interface and evidence below retain their own
scope, rather than inheriting acceptance of these new candidates.

Scoped completed EXP source, rebased on released v1.84.0; registered as the343rd
node after all342 existing IDs, including Sol and the two Semantic Bridge nodes.
A conversion/save template is in
`examples/workflows/35-h3-ltx-latent`; it requires local input and asset setup.
The fresh public-route refinement below is complete. The rebased candidate's
624 selected CPU tests and actual Core343 registration passed; the342-ID prefix
and all272 existing graph/Sol contents match the release with explicit LF
comparison. Formal-core delivery, actual native CPU and CUDA/BF16 latent
save/load checks and the three-clip picture/audio/lipsync human review completed.
Event/seam ratings remain NA; no universal quality or16GB guarantee follows.
See [the latest source/acceptance scope](H16_SOURCE_UPDATE_20260917.md). These test
counts are scoped, not whole-repository or end-to-end quality claims.
Do not label the existing accepted Prepared LTX clip as acceptance of this new entry.

Later progress notes below retain historical failed/pending attempts; this
header and the linked current summary supersede their status, not their evidence.

## Boundary

`MiniMaxH3LTXLatentAdapterEXPT8` takes standard Comfy LATENT, either plain H3 video
or joint H3 video/audio. It returns only LTX video LATENT, the original unmodified
H3 object, actual output frames, FPS and a report. Use the existing learned H3
upscaler upstream when needed; this node never adds a second upscaler or RGB detour.
Neither H3 audio nor masks/reference conditioning are reinterpreted as LTX inputs.

Source frames must be explicitly specified on the H3 17n+5 grid at24fps, with an
exact explicit reference prefix if present. Current temporal policies:

| Input | Policy | Output | Audio consequence |
| --- | --- | --- | --- |
| 73 | exact | 73 /24fps | No duration change |
| 124 | exact | Explicit rejection | Choose a policy, not silent crop |
| 124 | pad_to_ltx_grid | 129 /24fps | Separately extend/handle audio tail |
| 124 | crop_to_ltx_grid | 121 /24fps | Explicitly trim/reconcile audio |

The node does not mux audio or use `-shortest`. Spatial canvas stays unchanged and
must already align to32 pixels. It has no artificial source-frame ceiling.

## External assets and ownership

- Sana source144085566a866f9784f3798d4c8d1603f3adbccf, directory
  `models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/h3_ltx_adapter`.
- Efficient-Large-Model/H3-to-LTX-Latent-Adapter model revision
  1792c42689a0f22de880eaf57a187c6a373a636d, with original config.json/model.safetensors.
- Weight SHA256170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac.

No external code or weights are redistributed. Source files are authenticated
with LF-normalized hashes and executed from those checked bytes in a unique owned
package, without altering sys.path or importing unverified siblings/pycache.
Config/weights are checked before and after execution. Model loading preserves
the host CPU RNG. CPU/F32 is the conservative initial default; CUDA/BF16 is explicit,
not a silent fallback. Each call releases its own model and temporary modules.
Cancellation is checked between module forwards; no global model unload/cache purge.

## Evidence from the new interface

Historical GPU evidence paths below are under `artifacts/h16` in the isolated
`bridge-h16-meridian-20260917` checkout. The current rebased candidate is
`meridian-h16-resume-20260917`; its evidence is under `artifacts/resume`.

- `standard-latent-contract-cpu-v1`:58 selected tests, including overlapping
  package/workflow tests. Not a full repository test count.
- `standard-latent-actual-cpu-v1`:actual194,759,504-parameter adapter on the saved
  real73-frame H3 latent; wrapper output bit-identical to direct upstream CPU/F32.
- `standard-node-actual-cpu-v1`:actual node class with pinned owned loader,
  bit-identical to that reference; source AV and CPU RNG unchanged, modules cleaned.
- `standard-node-actual-cuda-bf16-v1`:same real input through CUDA/BF16 node;
  finite `[1,128,10,15,26]` output. Adapter peak allocated459,738,624 bytes,
  reserved528,482,304; after owned model cleanup998,400 allocated bytes (output).
  These are adapter-only allocator readings, not whole LTX pipeline requirements.
- `standard-adapter-native-ltx-decode-v1`:both outputs decoded with actual Comfy
  LTX VAE to73×480×832 RGB, encoded H264 with all97 original AAC packet payloads
  retained. Both strict video/audio checks passed. Decoded CPU/F32 vs CUDA/BF16
  MAE0.001014, PSNR55.76dB; this is numeric proximity, not human acceptance.
  VAE peak allocated5,459,514,368 bytes is separate from the adapter-only number.

One inspected frame is nonblack and contains the expected two-person indoor scene;
that does not qualify motion, likeness or refinement. No new diffusion was run.
- `registration342-cpu-v1`:106 selected CPU tests passed,1703 sources unchanged,
  CUDA not initialized. Not a full repository regression.
- `standard-node-native-core-v1`:actual native LoadLatent → registered adapter →
  native SaveLatent completed on CPU, bit-identical to direct node output.
  Native open/save/export, full-page reload and second API export were identical.
  The first visual layout overlapped outputs; layout-only v2 fixed that and was
  opened/saved/exported again. Both saved graph and actual API audit passed.
  All browser actions used an isolated CPU service; no UI injection or generation.
  Owned service exited and was cleaned. This does not qualify fresh refinement.

Next: compare through the existing refinement
route on this unaccepted short draft, then include the bound result in collective
human review. Do not regenerate or overwrite the old accepted LTX sample.

## Fresh refinement input preparation

`standard-node-LTX-audio-input-v1` completed with the actual native LTX audio VAE,
exit0 and no owned children. It consumed the previously generated CUDA/BF16
standard-node video latent without rerunning H3, the adapter or an upscaler.
The matching source AAC decoded to98304 stereo samples at32kHz; native audio
encoding returned77 latent frames, explicitly conformed to76 for the73-frame
LTX timeline. The final refiner delivery will retain the original AAC, not this
conditioning waveform or the newly refined audio. No audio loudness correction.

`standard-node-LTX-public-refinement-v1/preparation.json` binds4709 assets and
the exact Core/LTX/Sana source revisions for the existing public Prepared nodes.
The upstream fixed generic Stage2 prompt intentionally applies to new drafts;
its separately recovered, weights-only-loadable text cache was fully hashed.
No new Gemma inference is necessary. The original failed text-encoding job is
not relabelled successful; only its verified metadata recovery is reused.

Next explicit command after the current GPU owner exits:
`python -X utf8 -B artifacts/h16/run_h16_ltx_prepared.py --run`.
This calls the real public bundle and generation node classes, runs a fresh
three-update joint refiner then separate VAE decode with stage ownership/cache
checks. Preparation alone is not a successful refined clip or human acceptance.

## Fresh public-route refinement completed

The command above completed successfully in this development checkout. Do not
rerun it as though preparation were still pending. The actual public bundle and
generation nodes produced a new73-frame832×480H264 clip, SHA
`4a6142ccef579fe6c8254daaad2132ad8146b6f2d5223c35cd80357d889c11b7`.
There were three joint Euler forwards/144 block calls,22.66seconds including
weight transfers; the separately owned tiled VAE decode took4.19seconds.
These are stage measurements, not full-job latency or a speed comparison.
All97 original AAC packets and timestamps were retained, with no reencoding;
full AV decoding passed. Both generation and decode exited0 and their Windows
Job Objects had zero remaining owned processes after cleanup.

The source was the new standard-node adapter result; no H3 generation, adapter,
extra upscaler or text encoder was rerun. The refiner uses inverse-quantized
existing INT8 weights with BF16 LoRA fusion, not INT8 activation execution and
not an original BF16 checkpoint. The historical output filename contains
`learned2x`, but this test adds no second upscaler. First/middle/last frame
inspection shows nonblack two-person indoor images; motion/likeness/quality
remain for collective human review. This does not change the accepted old clip.
