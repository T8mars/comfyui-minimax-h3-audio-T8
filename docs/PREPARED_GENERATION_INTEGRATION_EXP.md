# Prepared Tao / LTX generation integration — GitHub EXP

## Local split Prepared LTX and external effects (not published)

The separate Bundle → Generate Refined Latent → Decode Frozen Latent nodes
have now completed one actual native Chromium generation queue, a fresh-Core
Load Frozen Refined Latent → Decode queue, and a third-Core cache queue.
The unchanged native three-update worker performed three joint forwards and
144 blocks; the fresh decoder did not generate again. The 73-frame, 832×480,
24-fps H264/AAC clip passed complete strict decode; its original AAC packets
were retained and decoded audio matched the input. Both cache stages then
returned without launching either worker. This is one mechanical input case,
not human quality, arbitrary input or full split-project acceptance.

The append-only `Prepared LTX · External EAV / Relay` node is a **new local
candidate**. It now has its own later native combination qualification below;
it does not inherit that earlier baseline. Connect it between
Bundle and each Generate / Load / Decode input, with a fresh chain ID. Connect
the separate Stage EAV Config for video FETA. Optional Relay uses the separate
`Prepared LTX · External Relay Timeline` node: it reads the actual SHA-bound
CPU prepared video/metadata, without text/model/sampler/VAE work. Connect its
LTX 8n+1 plan plus a local `t8_prepared_ltx_relay_caches_v1` JSON manifest
containing the plan hash and one `{event_index, prompt, path, sha256}` cache per
event. Paths are absolute; each prompt must be that event's actual local text.
The caches must contain independently encoded native Gemma/AV-connector
contexts accepted by the pinned prepared cache provider. Ordinary Core CLIP
conditioning and the fixed global cache cannot be relabelled event caches.
This binding node does not implicitly encode text, load models or sample.

An additional local EXP `Prepared LTX · Encode Relay Event Caches` node now
provides that **separate explicit encoding step**. Connect the original Bundle
and the Prepared Relay Timeline to it, select the native INT8 Gemma checkpoint
from the normal `models/text_encoders` menu, and connect its manifest-path
output to External EAV / Relay. Keep the original Bundle connected to the
encoder; connect the effects-bound Bundle to Generate / Load / Decode only.
The original global prompt cache stays unchanged. The timeline's compiled
prompt is a display/planning output, not a substitute for encoded features.

Encoding uses the matching pinned Gemma/AV-connector cache provider, not an
arbitrary Core CLIP. It runs in an owned serial worker with CPU/GPU weight
leases, requiring 64 GiB free host RAM at startup, 8 GiB host and 2 GiB GPU
runtime reserve. It does not run diffusion or a VAE. Use an absolute shared
serial-lease path and a fresh `cache_id` when text, checkpoints, implementation
or import environment changes. A matching committed cache is loaded without
another encoder worker; bad hashes are errors, not requests to encode again.
The global-only plan has an explicit zero-encoder/zero-GPU bypass. The normal
full/fresh-cold sampling stages remain separate from this cache stage.

Relay `apply_exp` appends those actual post-connector video context segments
and applies bounded event bias over video text cross-attention; audio context
is untouched. Disabled/report-only keeps the original video context and
native output. EAV observes native post-norm/RoPE temporal Q/K, preserves the
original masks/projection/gates, and applies a bounded gain only when explicitly
selected. Unknown forward producers are retained and reported uncovered;
genuine bad file/hash/shape/kernel/gain errors remain errors. No sigma, noise
generator order, step count, audio delivery or old workflow changes.

The new configuration and event-cache contents participate in the bundle and
engine identity. Old successful files remain historical evidence; they are not
adopted into new-source chains or manually assigned new fingerprints. The
focused 188 CPU tests and real pinned upstream masked/gated CUDA tiny A/B/A
verify contracts/delegation only. A separate trained-weight EAV A/B/A performed
nine joint forwards: the two report-only outputs exactly matched each other and
the original global-cache video/audio baseline; explicit EAV changed the video.
Two actual local-event caches also completed independent native INT8 Gemma and
AV-connector encoding with serial CPU/GPU weight leases, without sampling or
changing the global cache. These are separate numeric/cache qualifications,
not by themselves new-node complete media or human quality. A four-phase
real-weight global → Relay → Relay+EAV → global control
also completed twelve joint forwards, with 144 Relay applications in each
selected phase, 96 EAV applications in the combined phase, exact global-control
video/audio repeat, and restoration of instance forwards/attention delegates.
The eleven focused modules subsequently passed 202 CPU tests. A separate,
new-source native Chromium Queue then actually executed the public Plan →
Encode → Effects Bind → Generate combination with both effects `apply_exp`:
two independent Gemma/connector event encodings, three joint forwards,
144 Relay and 96 EAV applications. The saved video/joint-audio tensors exactly
matched the independently controlled combined numeric case. A fresh Core
loaded the frozen generation and decoded only, while Encode hit its own cache.
A third Core queue hit all three caches without running any worker or changing
the original generation record. Both decoded outputs were the same strict
complete 73-frame 832×480 H264/AAC clip, with original decoded audio unchanged.
This is one indoor two-person/teacup input, not full human viewing, arbitrary
materials, all backends or general quality acceptance.
Six additional editable EAV / Relay / combined full and decode-only EXP graphs
are built by `tools/build_formal_prepared_ltx_effect_workflows.py`; old graphs
are not modified. Blank local events are an explicit global-only bypass.
All six graphs subsequently passed actual native imports, visible independent
EAV/local-Relay edits, Save As, browser reload/reopen, named-edge and widget
audit in a separate CPU Core with queue/history empty. Edited QA graphs were
kept private and never queued; this is UI serialization, not new inference.

A separate encoder-only native Queue was actually interrupted through the
owned Core API during native Gemma event encoding. Core reported cancellation,
no cache was committed and the assigned Job ended with zero owned processes.
A fresh Core reused the same incomplete cache namespace and encoded only the
two events; another fresh Core read the committed cache without a worker.
The cancelled evidence stayed unchanged, output features matched the earlier
real encoder exactly, and no diffusion or VAE ran in any of these queues.
This encoder-only cancellation qualification remains separate. A new chain
then queued five real native canvases in five fresh owned Core processes,
using the explicitly external completed encoder manifest: combined generation
was interrupted after an actual joint forward, generation-only retry completed
the original three forwards, decode was interrupted during actual tiled VAE
work, decode-only retry completed, and the last queue launched no worker.
Neither cancelled stage committed a completion record; failed stage contents
stayed unchanged through retries and every owned Job ended with zero active
processes. The generation record survived decode cancellation unchanged.
The retried clip strictly decoded all 73 frames at 832×480 with original audio;
its decoded picture/audio matched the earlier one-input combined clip.
This is cancellation/recovery mechanics for that indoor teacup input, not
arbitrary materials, providers, long durations or human quality acceptance.
The expanded sixteen modules passed
273 CPU tests with CUDA uninitialized and a stable source snapshot.
The original missing-digest worker failure remains failed; its later
source-bound repeat passed completion and owned-process cleanup.

## 2026-09-17 completed H16 source additions (latest status)

The specific standard-LATENT/public LTX refinement and Tao two-request public
generation/decode routes below completed formal delivery and bound human review.
Both actual Tao cancellation and injected fault were followed by successful
ordinary H3 eight-step generation in the same Core with owned-process cleanup.
See [the scoped source update](H16_SOURCE_UPDATE_20260917.md).
This supersedes the pending statements immediately below and in historical
sections; older failures, cache-adoption evidence and unassessed routes remain
limited to their original scope. No arbitrary-input or universal16GB guarantee.

The new standard-LATENT H3→LTX draft completed the actual public refinement and
decode nodes; see [the bound short result](H16_STANDARD_LATENT_ADAPTER_EXP.md).
It retains original AAC and remains human-pending. Older statements below about
no fresh wrapped inference describe their historical snapshot, not this new run.

A separate `tao_stream` bundle kind is now wired through the same two existing
Prepared nodes. The original `tao5s` and `ltx_refine` schemas remain unchanged.
Actual switched-prompt two-request generation and repaired public-route decoding
have completed; see [the bound lifecycle/media evidence](H16_TAOMATE_STREAM_LIFECYCLE_EXP.md).
Actual cancellation/fault recovery and human voice/seam quality remain separate
pending gates. Neither public media success nor the624 selected rebased CPU
tests qualify every input, model combination or end-to-end GPU budget.

Use the existing bundle loader → generation/decode node connection. Generate a
new local bundle with `tools/prepare_generation_bundle.py --kind tao_stream
--generation-request GENERATION.json --decode-request DECODE.json --output BUNDLE.json`.
The CLI performs inventories/hashes only, not model downloads, teacher synthesis
or GPU work. These files are explicit prepared manifests, not Comfy workflow JSON.

The generation manifest has `source`, `source_revision`, `base`, `adapter`,
`teacher`, `download_receipt`, and an ordered `stream_requests` list. Each item
contains only `request_index` (contiguous from0), `text_features`, `milestones`,
and `audio_seed`. Every file must refer to its matching real prepared request;
changing prompt text in JSON does not create new embeddings or Base10 states.
The decode manifest has `core`, the same `source`, `video_vae`, and `audio_vae`.
The tool sets route schemas and the exact request count. Source/model/teacher
inventories and all nested text/milestone identities are bound into the bundle.

The public `noise_seed` binds video seed0; subsequent requests use seed+index
modulo2^64. Teacher audio seeds remain the per-request prepared values. The
worker validates raw5120 BF16 text, tags, teacher metadata, finite matching
states3/6/9 and global continuation counts before allocating model weights.
One model/runtime owns the entire request sequence. Native transport prefixes
are verified against previous clean tails and stripped once, without blending
or repeating request-local audio. All requests must succeed before a joined
generation stage is promoted. Failure/cancellation does not resume half a KV
stream; retry restarts generation, whereas a completed generation with failed
decode resumes only decode. Different input/seed/code requires a new chain ID.

Each native request publishes5seconds; two requests are10seconds. This is the
pinned native recipe, not an arbitrary-duration/streaming preview promise.
The first request owns124 native frames, subsequent requests119; globally
rounded audio counts are207 then198/198/199…, not207 per segment. Final VAE
decode is joint, and endpoint-preserving video/audio delivery mapping is applied
once over the whole timeline to produce24fps H264/AAC MP4. No volume boost,
normalization, repeated PCM, or extra RGB float archive is produced by default.

The new backend/contract files participate in engine identities. Existing cached
chains from older code therefore cannot be silently relabelled current results.
Keep the historical accepted media; do not regenerate them merely for a new
fingerprint. New public-route media, real cancellation/next ordinary task checks,
integrated regression, formal-core delivery and collective review remain gates.

## Current qualification (supersedes progress notes below)

The 2026-09-13 collective human review has now been received. The specific
H3-to-LTX refined clip was accepted for picture, motion and audio/lipsync; do not
regenerate or request acceptance again for that same clip. The separate Tao
new-dialogue five-second recovered clip is now accepted for picture, motion and
audio/lipsync too. Its original generation job still failed the finalization
memory guard and lacks a success report; independent saved-latent recovery and
media acceptance do not convert that failed job into reliable generation.
Do not adopt that failed job using the original success-only checkpoint tool.
Older Dance picture/identity and both depth results were rejected. These outcomes do
not qualify fresh packaged GPU workers. The GitHub delivery remains EXP and keeps
that limitation explicit rather than treating accepted recovered media as a new run.

Before the finalization change below, both real Core PromptExecutor two-node graphs completed explicit original-pilot
checkpoint adoption: `prepared-cached-executor-tao5s-cpu-v3-20260913/report.json`
and `prepared-cached-executor-ltx_refine-cpu-v3-20260913/report.json` under local
artifacts. Both returned generation/decode cache hits, exact original movie SHA,
no generation/decode attempts and CUDA initialized=False. The event transport
was in-memory, not an HTTP/browser test. The packaged workers have not received
a new full GPU run; these results must not be described as fresh inference.

`tools/migrate_prepared_checkpoint.py` now binds original requests, terminal
reports, exact seeds, model/input/text identities, actual source-root revisions
and unchanged native math helpers. LTX additionally requires the CPU media
recovery receipt and its good MP4; the corrupt first MP4 is not accepted. The
v2 source-bound migration qualified eight Tao helpers and five LTX helpers.
The migration is deliberately limited to the evidenced original pilot settings.
Python and dependency versions/locations now participate in the fingerprint.

The combined focused suite passed109 tests, including20 synthetic migration
counterexamples. After that snapshot, cleanup diagnostics gained a fallback for
missing Python3.11 `add_note`;45 process/runtime tests passed, including two new
cases. This was run on Python3.12, not a full Python3.10 environment qualification.
No sampling math changed. This cleanup-source change made the older v2 checkpoint
fingerprint stale. The v3 explicit migration and both real Core traversals above
were completed after that change; never edit a fingerprint by hand or rerun a
passed GPU sample to disguise a migration.

Complete input inventories and native CPU preflight now passed for both routes:
Tao29 assets/three directories/two source pins, LTX4709 assets/two runtime
directories/three source pins. These inventories do not generate new teachers,
text features or H3-to-LTX conversions. Both templates were also imported into an
isolated native Comfy browser and exported using Export (API); all execution
values and edges matched the Core-qualified graphs. No prompt was queued in that
browser check. Actual local archive extraction/import passed330 nodes and243
workflow JSONs, with exactly three Prepared CLIs, their standalone help and
malformed-bundle rejection checked from the extracted directory. CUDA remained
uninitialized. This qualifies archive wiring and CPU contracts only, not fresh GPU inference or human acceptance.
The original package result predates human-review documentation updates.
Documentation-only repacks must retain the same package gates before delivery.
Pinned sources, tested dependencies and distribution limits are documented in
[Prepared sources and environment](PREPARED_SOURCES_AND_ENVIRONMENT.md).

## Status and intended delivery

### Tao finalization lifecycle correction (CPU evidence, not a new GPU pass)

The actual worker entry, exercised with tiny CPU model/runtime doubles, retained
both model and pipeline into postflight input hashing. Its regression failed
before the change (one failure, three passing fault paths). Generation now runs
inside a separate owned function; the function returns and cyclic garbage is
collected before postflight hashing. No model or latent object is returned.
The five new tests include weak-reference checks with a deliberately cyclic
pipeline, generation/cleanup/postflight failures, exact saved fixture tensors,
stage memory telemetry, and AST equality of the entire original generation body
after excluding progress messages. The combined Prepared suite passed116 tests.

Progress events now include process RSS/private memory when available and system
available memory at save, receipt, release and postflight phases. These readings
are diagnostics, not exact peaks or proof of the historical RAM failure cause.
The old failed job/report and its separately accepted recovered media are unchanged;
no resource threshold, sampling algorithm or teacher content was changed.

This worker edit invalidates older engine fingerprints, including the v3 bundles.
Explicit v4 migrations re-check the same eight Tao/five LTX math helpers and original
successful pilot evidence; they are not adoption of the failed new-dialogue job.
Fresh wrapped full-model GPU execution remains unqualified. Do not rerun an accepted
sample simply to replace honest cache-adoption evidence with a fresh-inference claim.

This is an unfinished reusable generation/decode integration, not a playback-only
node and not a newly qualified GPU route. Two candidate nodes are now registered
in the GitHub EXP tree (330 total); they are deployed as experimental interfaces,
not promoted to a fully qualified Registry route. Do not describe them as stable
or fresh end-to-end GPU-qualified before the remaining gates below are complete.
The already generated Tao, LTX and RGB Dance clips remain the review evidence;
do not regenerate those passed cases simply to exercise a new wrapper.

The intended workflows consume explicit prepared inputs, create new video
latents when the generation cache is absent, then decode in a separate owned
process after the generation process has exited. Prepared inputs are real
prerequisites, not an arbitrary prompt/latent coercion:

- Tao: matching Base10 teacher and text features, one native five-second request,
  existing official base and Tao adapter. Video seed can differ; the audio seed
  must remain compatible with the prepared teacher. No teacher rerun is implied.
- LTX: already normalized/converted LTX AV latents and matching post-connector
  text cache. Native three Euler updates, LoRA0.8, single-generator video-then-
  audio noise order. Native CFR24/8n+1 geometry and a bounded input token envelope;
  only the saved73-frame2048×1024 case has actual GPU evidence. Original AAC is
  retained for delivery; generated LTX audio is not the delivered track.

## Implemented controller boundaries

`prepared_generation_runtime.py` owns per-chain OS locks and a separate serial
GPU lease, hash-checked stage records, attempt-specific output directories and
atomic state promotion. A verified empty/failed chain is bound to the original
seed/settings before launching workers. A different fingerprint or disabled
resume requires a new chain ID; no overwrite or silent regeneration occurs.
Decode-only cache states, unknown stages, files escaping the chain directory,
changed file contents and changed packaged backend identity fail closed.

The controller identifies checkpoint adoption explicitly in reports; it does
not call adopted outputs freshly generated. All checkpoint sources are checked
before copying any, and copied bytes are checked again. The real saved-sample
migration/provenance tool is still required; a hand-authored checkpoint alone
is not proof that the expected model settings produced the saved clip.

`prepared_process.py` reuses the existing Windows Job Object with kill-on-close
and stdin startup gate. No worker task code executes before ownership is
assigned. This includes descendants spawned between telemetry polls or just
before the worker exits. A successful worker that leaves descendants is a
failure, not a completed stage. Cleanup tries remaining operations even after
one fails; an original cancel/failure retains its identity and receives cleanup
diagnostics. Cleanup uncertainty prevents success and the next stage.

## Evidence

`artifacts/prepared-orchestration-cpu-v2-20260913/tests.xml`:43 tests passed,
CUDA_INITIALIZED=False. This supersedes the overlapping17 process and22 cache
early runs; do not add their counts to43.

- 21 process/controller cases: actual small Windows CPU child processes, gated
  ownership, normal exit, hang/timeout, active cancellation, orphan descendants,
  worker report/digest checks, resource-guard failure, plus injected cleanup and
  receipt failures and malformed deadlines.
- 22 cache/state cases: explicitly synthetic artifacts and fake resource/model
  calls, serial stage order, complete/partial resume, failed-first binding,
  setting/backend changes, corrupted states/files, checkpoint adoption and
  invalid chain names. These are not model inference or media-quality evidence.

Ruff passed for the controller/contract/process and these fixtures/tests. All
test processes exited. No GPU worker, sample regeneration, deployment, commit
or push occurred during this integration step.

## Copied backend provenance and remaining gates

`prepared_backend` contains exact copies of the local G-drive prototype helpers
that produced the existing samples. Source originals are retained. Hash checks
show unchanged math helpers including Tao local runtime/session/transport,
bounded retention, host cache, leased hook, prepared pipeline, weight offload,
Tao media; and LTX dequantized LoRA, LoRA contract, text offload, fusion gate and
prepared input boundary. `resource_guard.py` is the existing controller helper.

The separately copied four worker wrappers and LTX media wrapper were edited
for packaged imports, explicit prepared seeds/geometry and frame count. Their
new full GPU execution has not been run. Parent process/cache tests do not
qualify these altered wrappers. Upstream source and licensing identities must
remain explicit in the final bundle/package inventory.

Delivery gates now completed, within the stated scope:

1. Strict manifest/source/runtime identities and native CPU preflight for both
   original inputs. Main host dependency identity is versions/locations, not
   adversarial whole-environment attestation; tracked Git scope is explicit.
2. Explicit preparation and evidence-bound checkpoint migration, including the
   LTX CPU media recovery instead of the damaged initial MP4.
3. Two lazy nodes, two templates, real Core execution/checkpoint traversal and
   native browser Export(API) value/edge parity. These are cache-adoption runs,
   not fresh wrapped GPU inference or arbitrary input qualification.
4. Local archive membership/hash, actual extracted schema/workflow imports and
   three standalone Prepared tool checks. No external weights/runtime or private
   handoff files are distributed; source notices retain licensing limitations.

Remaining: resolve generation finalization reliability without rerunning the
accepted Tao clip, and resolve or explicitly disposition rejected Dance/depth.
The published accepted-picture continuation is now transplanted with CPU
regression coverage, retaining its previous scoped eight-second human evidence,
not claiming a fresh GPU run in this research tree. Both depth routes remain rejected experimental
examples, not recommended solutions. No publication is authorized for this new
development. Never rerun passed pilots just to turn cache-adoption evidence
into a claim of fresh inference.

## Current input/node evidence

`prepared-contract-cpu-v1-20260913/tests.xml`:80 tests passed, including the
previous43 plus37 new manifest/real-file/source tests. CUDA initialized=False.
`prepared-nodes-cpu-v1-20260913/tests.xml`:8 additional schema/lazy-handoff/template
tests passed. These two result sets cover different tests; handoff tests use
explicit synthetic media and a mocked runner, not model inference.

`prepared-workflows-core-v1-20260913/report.json`:actual Core registered330
nodes; both two-node/one-edge templates passed API validation and serializer
audit. Nothing was queued; not native browser roundtrip evidence. Nodes return
original saved-file previews, avoiding a second SaveVideo audio encode.

`tools/prepare_generation_bundle.py` creates new identity manifests without
overwriting prior ones. `artifacts/prepared-input-bundles-v1-20260913/ltx_refine.json`
contains4709 actual file identities,2 isolated runtime directory inventories and
3 source pins. No teacher/model execution or checkpoint adoption occurs here.

`prepared-native-inputs-ltx-cpu-v1-20260913/report.json`:actual native LTX AV
input loading and prompt-cache matching on CPU; video[1,128,10,32,64],
audio[1,8,76,16], text contexts[1,1024,4096] and[1,1024,2048], CUDA=False.
This reuses existing prepared inputs, not text encoding or model generation.

Tao shared `prepared_backend/taomate_input_preflight.py` now verifies the actual
teacher prompt/seed and all three saved3/6/9milestones before the generation
worker queries CUDA. An actual CPU call with the saved request passed:44tokens,
hidden[44,5120], clean audio[2,32,207], zero new teacher forwards, CUDA=False.
The full Tao bundle hashing operation completed separately (29 assets, three
directories, two source pins), followed by its native CPU preflight and v3 Core
checkpoint traversal. No hashing or model process remains active for those runs.
