# Prepared Relay encoder: scoped source admission

The v9 sampler-call baseline retains all 103 v8 sites unchanged and adds
exactly two reviewed calls in `prepared_ltx_relay_cache.run_cache_encoding`:

| Call | Actual operation | Fixed boundary |
| --- | --- | --- |
| `reader.sample()` | Read NVML GPU and host resource state | `NvmlResourceReader`, no diffusion |
| `run_worker(..., SPEC, ...)` | Owned native event-cache encoding Job | Only `ltx_relay_cache_worker.py`, native Gemma encode / AV connector, no sampler or VAE |

The AST scanner and fail-closed handling of new/changed sites are unchanged.
These are explicit text-only interfaces, not approval of new diffusion paths
or a generic exemption for functions named `sample` or `run_worker`.

The controller validates source/environment/checkpoint content and exact plan,
owns serial leases, enforces resource headroom and Windows Job cleanup, and
only commits complete SHA-bound per-event caches. Tests retain failure/retry,
late cancellation, corrupt state/manifest/assets and global-only no-worker
boundaries. Real native Chromium Queue independently exercised the public
encoder then generated with Relay/EAV, fresh decoded without regeneration and
third-process read all caches without launching a worker. Strict complete
73-frame 832×480 H264/AAC and original audio equality qualified one input.

No private models, source pins, process reports or media are included here.
Real encoder-only native Queue cancellation during Gemma, assigned-Job cleanup,
fresh-Core encoding-only retry and third-Core cache read also subsequently
passed, without promoting partial caches or altering failed evidence.
Human quality, all materials/backends and combined generation/decode
cancellation remain separate gates; the historical v8 red report is not rewritten.
