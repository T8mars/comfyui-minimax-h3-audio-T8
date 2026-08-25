# MiniMax H3 Audio Refine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three append-only MiniMax H3 Audio Refine Advanced EXP nodes that audit, plan, and assemble an exact uncached partial-tail dual-clock audio resample without changing any existing node contract.

**Architecture:** Keep policy, hashing, decisions, sigma derivation, and assembly in one focused domain module, and keep ComfyUI V3 schemas in a separate node wrapper module. Reuse the existing strong conditioning contract, runtime telemetry, joint AV validation, and stable dual-clock sampler; Setup returns a deterministic NOISE and BasicGuider-equivalent GUIDER so the supported route cannot silently use CFG greater than one.

**Tech Stack:** Python 3.10+, PyTorch, current ComfyUI `comfy_api.latest.io`, `comfy.nested_tensor`, `comfy.samplers`, pytest, Ruff.

---

## File map

- Create `audio_refine_advanced.py`: pure contracts, decisions, hashing, mask classification, resource checks, partial-tail math, deterministic/bypass noise, BasicGuider assembly, and setup orchestration.
- Create `nodes_audio_refine_advanced.py`: three ComfyUI V3 node schemas and thin execution adapters.
- Create `tests/test_audio_refine_advanced.py`: all red/green mechanical behavior, including real `SamplerCustomAdvanced` no-op integration with lightweight fakes.
- Modify `nodes.py`: import and append the three classes after the current position 210 list.
- Modify `features.json`: append only the three node IDs and add a scoped `audio_refine_advanced` feature record after implementation is mechanically verified.
- Modify `tests/test_preflight_and_registration.py`: change only the expected count and append-only tail assertion.
- Keep `sampling.py`, existing node schemas, existing workflows, `SKILL.md`, and `roadmap.md` unchanged in the mechanical implementation commit.

Runtime positions 211, 212, and 213 are reserved respectively for Audit, Plan, and Dual-Clock Setup; the existing positions 0..210 remain byte-for-byte ordered.

### Task 1: Contract primitives and deterministic hashing

**Files:**
- Create: `tests/test_audio_refine_advanced.py`
- Create: `audio_refine_advanced.py`

- [ ] **Step 1: Write the failing contract tests**

Add tests that import these exact public symbols and therefore fail while the module is absent:

```python
from h3_audio_t8_pkg.audio_refine_advanced import (
    AUDIO_REFINE_AUDIT_SCHEMA,
    AUDIO_REFINE_PLAN_SCHEMA,
    AUDIO_REFINE_AUDIT_TYPE,
    AUDIO_REFINE_PLAN_TYPE,
    canonical_json,
    classify_audio_refine_latent,
)

def test_audio_refine_contract_constants_are_stable():
    assert AUDIO_REFINE_AUDIT_TYPE == "H3_T8_AUDIO_REFINE_AUDIT"
    assert AUDIO_REFINE_PLAN_TYPE == "H3_T8_AUDIO_REFINE_PLAN"
    assert AUDIO_REFINE_AUDIT_SCHEMA == "t8.minimax_h3.audio_refine.audit.v1"
    assert AUDIO_REFINE_PLAN_SCHEMA == "t8.minimax_h3.audio_refine.plan.v1"

def test_joint_av_manifest_is_content_bound_and_deterministic():
    first = classify_audio_refine_latent(_latent())
    second = classify_audio_refine_latent(_latent())
    assert canonical_json(first["manifest"]) == canonical_json(second["manifest"])
    assert first["manifest_sha256"] == second["manifest_sha256"]
    changed = classify_audio_refine_latent(_latent(audio_offset=1.0))
    assert changed["manifest_sha256"] != first["manifest_sha256"]
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
F:\AI-T8-video-onekey\python\python.exe -m pytest -q tests\test_audio_refine_advanced.py
```

Expected: collection fails with `ModuleNotFoundError: h3_audio_t8_pkg.audio_refine_advanced`.

- [ ] **Step 3: Implement the minimum contract module**

Create constants, canonical JSON with `allow_nan=False`, SHA-256 helpers, chunked tensor hashing, and `classify_audio_refine_latent()`. The classifier must call `nested_av_parts()`, require channels 24/32 and stereo 2, reject sparse/quantized/meta/non-finite tensors, hash video/audio contents, and return no tensor references:

```python
AUDIO_REFINE_AUDIT_TYPE = "H3_T8_AUDIO_REFINE_AUDIT"
AUDIO_REFINE_PLAN_TYPE = "H3_T8_AUDIO_REFINE_PLAN"
AUDIO_REFINE_AUDIT_SCHEMA = "t8.minimax_h3.audio_refine.audit.v1"
AUDIO_REFINE_PLAN_SCHEMA = "t8.minimax_h3.audio_refine.plan.v1"

def canonical_json(value, *, indent=None):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":") if indent is None else None,
                      indent=indent, allow_nan=False)
```

- [ ] **Step 4: Run GREEN**

Run the Task 1 test command. Expected: contract tests pass; later not-yet-written behaviors are absent rather than skipped.

### Task 2: Audit decisions, locks, patch stack, and resource gates

**Files:**
- Modify: `tests/test_audio_refine_advanced.py`
- Modify: `audio_refine_advanced.py`

- [ ] **Step 1: Write failing Audit tests**

Add direct tests for `audit_audio_refine()` using injected `runtime_snapshot_fn` and lightweight MODEL fakes. Cover:

```python
def test_audit_allows_native_full_audio_with_clean_stack():
    audit, decision, report = _audit()
    assert decision == "ALLOW"
    assert audit["schema"] == AUDIO_REFINE_AUDIT_SCHEMA
    assert audit["decision"] == "ALLOW"
    assert json.loads(report)["decision"] == "ALLOW"

@pytest.mark.parametrize("mode,expected", [
    ("lock_source", "ABSTAIN_AUDIO_LOCKED"),
    ("remix_source", "ABSTAIN_REMIX_SOURCE_NOT_VALIDATED"),
])
def test_audit_abstains_for_protected_source_modes(mode, expected):
    audit, decision, _ = _audit(conditioning_report=f"task=I2VA\naudio_mode={mode}")
    assert decision == "ABSTAIN"
    assert expected in audit["reason_codes"]

def test_audit_abstains_when_headroom_is_unknown_or_below_floor():
    assert _audit(runtime={"gpu": {}, "host": {}})[1] == "ABSTAIN"
    assert _audit(runtime=_runtime(free_mib=511))[1] == "ABSTAIN"
    assert _audit(runtime=_runtime(commit_gib=15.99))[1] == "ABSTAIN"

def test_audit_abstains_for_transformer_patch_replacement():
    model = _model(patches_replace={"attention": {("double_block", 0): object()}})
    audit, decision, _ = _audit(model=model)
    assert decision == "ABSTAIN"
    assert "ABSTAIN_PATCH_STACK_UNVALIDATED" in audit["reason_codes"]
```

Also cover all-zero audio mask, fractional mask, legacy video-only mask, optional protected audio, malformed media map/report, wrong channel/rank, non-finite latent, non-H3 model, and conditioning content change.

- [ ] **Step 2: Run RED**

Run only the new Audit tests with `pytest -q tests/test_audio_refine_advanced.py -k audit`. Expected: imports succeed but `audit_audio_refine` is missing.

- [ ] **Step 3: Implement Audit minimally**

Implement the exact public signature `audit_audio_refine(*, model, positive, av_latent, conditioned_prompt, media_map_json, conditioning_report, protected_audio=None, minimum_free_vram_mib=512, minimum_commit_headroom_gib=16.0, hash_chunk_megabytes=8, runtime_snapshot_fn=runtime_snapshot)` and return `(audit_descriptor, final_decision, canonical_json(report, indent=2))` after the checks below.

Requirements:

- Use `compile_nfe_run_contract()` for the prompt/media/report/positive strong contract.
- Parse exactly one `audio_mode=` line.
- Use `split_noise_masks()` and classify audio mask as absent/full/locked/fractional/invalid.
- Inspect MODEL class, base model, model sampling, ModelPatcher weight patch structure, attachment keys, transformer wrappers, and `patches_replace` without hashing weights.
- Add reason objects with stable `code`, `severity`, and `message`.
- Resolve final decision with `REJECT > ABSTAIN > ALLOW`.
- Enforce non-lowerable 512 MiB and 16 GiB floors.
- Store only scalar manifests, hashes, shapes, model object ID, gates, and runtime evidence.

- [ ] **Step 4: Run GREEN**

Run `pytest -q tests/test_audio_refine_advanced.py -k audit`. Expected: all Audit tests pass with no CUDA initialization.

### Task 3: Exact KSampler-equivalent partial-tail Plan

**Files:**
- Modify: `tests/test_audio_refine_advanced.py`
- Modify: `audio_refine_advanced.py`

- [ ] **Step 1: Write failing sigma and decision tests**

```python
def test_default_partial_tail_matches_reference():
    plan, decision, _ = plan_audio_refine(_allow_audit(), 4, 0.5, 42)
    assert decision == "ALLOW"
    assert plan["full_steps"] == 8
    assert plan["actual_refine_nfe"] == 4
    assert plan["base_sigmas"] == pytest.approx([.5, .375, .25, .125, 0])
    assert plan["video_sigmas"] == pytest.approx([
        12/13, 4.5/5.125, .8, 1.5/2.375, 0,
    ])
    assert plan["audio_sigmas"] == pytest.approx([.75, 9/14, .5, .3, 0])

@pytest.mark.parametrize("steps,denoise", [(1,.35), (4,.35), (6,.5), (8,1.0)])
def test_partial_tail_uses_current_ksampler_integer_rule(steps, denoise):
    plan, _, _ = plan_audio_refine(_allow_audit(), steps, denoise, 1)
    assert plan["full_steps"] == int(steps / denoise)
    assert len(plan["video_sigmas"]) == steps + 1

def test_plan_never_upgrades_audit_decision():
    plan, decision, _ = plan_audio_refine(_abstain_audit(), 4, .5, 1)
    assert decision == "ABSTAIN"
    assert plan["decision"] == "ABSTAIN"
```

Add invalid range, non-finite value, unsupported model strategy, and tampered Audit descriptor tests.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_audio_refine_advanced.py -k 'plan or partial_tail'`. Expected: `plan_audio_refine` is missing.

- [ ] **Step 3: Implement Plan minimally**

Implement `plan_audio_refine(audit, refine_steps, audio_denoise, refine_seed, model_strategy="connected_model_explicit")`. Use exactly:

```python
full_steps = int(refine_steps / audio_denoise)
base_sigmas = [(refine_steps - k) / full_steps for k in range(refine_steps + 1)]
video_sigmas = [shift_sigma(value, 12.0) for value in base_sigmas]
audio_sigmas = [shift_sigma(value, 3.0) for value in base_sigmas]
```

Sign the canonical payload, record requested/effective denoise, fixed shifts/sampler/scheduler/CFG/masks, and preserve Audit reason codes without upgrading its decision.

- [ ] **Step 4: Run GREEN**

Run the Task 3 test command. Expected: all Plan and sigma tests pass.

### Task 4: Setup assembly, deterministic noise, BasicGuider, and bypass

**Files:**
- Modify: `tests/test_audio_refine_advanced.py`
- Modify: `audio_refine_advanced.py`

- [ ] **Step 1: Write failing Setup tests**

Use monkeypatch only at the Comfy boundary; test real domain behavior:

```python
def test_setup_allow_uses_full_schedule_then_exact_tail(monkeypatch):
    called = {}
    def fake_setup(model, latent, steps, shift_video, shift_audio, sampler_name, scheduler):
        called.update(steps=steps, shifts=(shift_video, shift_audio))
        return _patched_model(), _sampler(), native_flow_sigmas(steps, shift_video)
    result = setup_audio_refine(
        plan=_plan(), model=_model(), positive=_positive(), av_latent=_latent(),
        setup_sampling_fn=fake_setup, runtime_snapshot_fn=lambda: _runtime(),
    )
    assert called == {"steps": 8, "shifts": (12.0, 3.0)}
    assert result.sigmas.tolist() == pytest.approx(_plan()["video_sigmas"])
```

Add tests that prove:

- Setup does not invoke MODEL forward.
- input samples are the same objects before downstream sampling.
- video mask is float32 all zero and audio mask float32 all one.
- deterministic noise is identical for one seed and differs for another.
- Guider has CFG 1 and only positive conditioning.
- MODEL/conditioning/latent contract mismatch rejects before clone/noise/mask creation.
- a newly insufficient runtime gate lowers ALLOW to ABSTAIN.
- ABSTAIN returns empty SIGMAS and bypass noise whose `generate_noise()` returns the original samples reference.
- current real `SamplerCustomAdvanced.execute()` with empty SIGMAS performs zero `prepare_sampling` and zero model forward, and returns equal samples.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_audio_refine_advanced.py -k setup`. Expected: Setup symbols are missing.

- [ ] **Step 3: Implement Setup minimally**

Implement:

```python
class AudioRefineRandomNoise:
    def __init__(self, seed): self.seed = int(seed)
    def generate_noise(self, input_latent):
        return comfy.sample.prepare_noise(input_latent["samples"], self.seed,
                                          input_latent.get("batch_index"))

class AudioRefineBypassNoise:
    seed = 0
    def generate_noise(self, input_latent):
        return input_latent["samples"]

class AudioRefineBasicGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, positive):
        super().__init__(model)
        self.inner_set_conds({"positive": positive})
```

`setup_audio_refine()` must verify descriptors and bindings before allocations, rerun resource gates, call `setup_sampling_fn(model, av_latent, plan["full_steps"], 12.0, 3.0, "dual_clock_euler", "native_flow")`, slice `full_sigmas[-(plan["actual_refine_nfe"] + 1):]`, build exact 0/1 nested masks, and return a typed dataclass/tuple with the seven node outputs. ABSTAIN returns original MODEL/LATENT, bypass noise, BasicGuider, a never-called KSampler object, and `torch.empty(0, dtype=torch.float32)`.

- [ ] **Step 4: Run GREEN**

Run all `tests/test_audio_refine_advanced.py`. Expected: all domain and Setup tests pass serially without H3/CUDA.

### Task 5: ComfyUI V3 schemas

**Files:**
- Modify: `tests/test_audio_refine_advanced.py`
- Create: `nodes_audio_refine_advanced.py`

- [ ] **Step 1: Write failing schema tests**

Assert exact IDs, categories, experimental flags, input order, output order, defaults, non-lowerable resource minima, and `fingerprint_inputs()` NaN for Audit and Setup runtime telemetry:

```python
assert [schema.node_id for schema in schemas] == [
    "MiniMaxH3AudioRefineAuditT8Advanced",
    "MiniMaxH3AudioRefinePlanT8Advanced",
    "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
]
assert [item.id for item in schemas[2].outputs] == [
    "model", "noise", "guider", "sampler", "sigmas", "latent", "report_json",
]
```

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_audio_refine_advanced.py -k schema`. Expected: node module missing.

- [ ] **Step 3: Implement thin wrappers**

Define `AudioRefineAuditIO = io.Custom(AUDIO_REFINE_AUDIT_TYPE)` and `AudioRefinePlanIO = io.Custom(AUDIO_REFINE_PLAN_TYPE)`; define three `io.ComfyNode` classes whose `execute()` methods only call the domain functions and return `io.NodeOutput`. The Audit optional AUDIO input is `protected_audio`; Plan has one model strategy option; Setup accepts plan/model/positive/av_latent and returns seven outputs.

- [ ] **Step 4: Run GREEN**

Run all new tests. Expected: schema and execute-adapter tests pass.

### Task 6: Append-only registration and metadata

**Files:**
- Modify: `nodes.py`
- Modify: `features.json`
- Modify: `tests/test_preflight_and_registration.py`
- Modify: `tests/test_audio_refine_advanced.py`

- [ ] **Step 1: Write failing registration tests**

Snapshot the first 211 IDs from `features.json`, then assert runtime IDs preserve that exact prefix and append:

```python
assert ids[:211] == old_ids
assert ids[211:] == [
    "MiniMaxH3AudioRefineAuditT8Advanced",
    "MiniMaxH3AudioRefinePlanT8Advanced",
    "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
]
assert len(ids) == 214
```

Update existing count assertions only after the new test fails at 211.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_audio_refine_advanced.py -k registration tests/test_preflight_and_registration.py::test_all_nodes_register_with_unique_ids_and_valid_schemas`. Expected: missing tail and count mismatch.

- [ ] **Step 3: Append registration**

Import `AUDIO_REFINE_ADVANCED_NODE_CLASSES` in `nodes.py` and append `*AUDIO_REFINE_ADVANCED_NODE_CLASSES` after `*SKIN_FINISH_DICHROMATIC_NODE_CLASSES`. Append the three IDs to `features.json["nodes"]`; add a feature record that states exact uncached scope, fixed CFG/shifts/masks, resource gates, no quality guarantee, no Frozen Cache, and no old workflow change.

- [ ] **Step 4: Run GREEN**

Run the Task 6 command. Expected: runtime and metadata have 214 unique IDs and prefix 0..210 is unchanged.

### Task 7: Mechanical regression and implementation commit

**Files:**
- Verify all files above.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONPATH='F:\AI-T8-video-onekey\ComfyUI'
F:\AI-T8-video-onekey\python\python.exe -m pytest -q \
  tests\test_audio_refine_advanced.py \
  tests\test_core.py tests\test_sampling.py tests\test_conditioning.py \
  tests\test_nfe_run_contract_advanced.py tests\test_preflight_and_registration.py
```

Expected: zero failures.

- [ ] **Step 2: Run all artifact-independent project tests**

Generate the test list from `tests/test_*.py`, excluding only tests whose collection imports ignored `artifacts/` source media. Run serially with `--maxfail=1`; record every exclusion and its collection error rather than claiming an unqualified full-suite pass.

- [ ] **Step 3: Run static checks**

```powershell
F:\AI-T8-video-onekey\python\python.exe -m ruff check audio_refine_advanced.py nodes_audio_refine_advanced.py tests\test_audio_refine_advanced.py nodes.py
F:\AI-T8-video-onekey\python\python.exe -m py_compile audio_refine_advanced.py nodes_audio_refine_advanced.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Verify old workflows and old registry prefix**

Hash every tracked `examples/workflows/**/*.json` against branch base and assert no path or byte hash changed. Import the extension and compare runtime node IDs 0..210 with base `features.json`.

- [ ] **Step 5: Commit mechanical implementation**

Stage only the design, plan, new Audio Refine modules/tests, `nodes.py`, `features.json`, and the registration test. Commit:

```powershell
git commit -m "feat: add exact H3 audio refine setup"
```

### Task 8: One low-load real H3 smoke

**Files:**
- Create: `tools/run_audio_refine_smoke.py`
- Create local ignored evidence under `artifacts/audio-refine-smoke-20260826/`

- [ ] **Step 1: Write a dry-run test before any runner code**

The test must prove default dry-run starts no server, queues no prompt, rejects an active user service, caps execution at one prompt, checks at least 512 MiB current VRAM and 16 GiB commit headroom, and fixes Turbo4 + Refine4 + denoise0.5 + shift12/3 + CFG1.

- [ ] **Step 2: Run RED, implement the minimum bounded runner, then run GREEN**

Use a private localhost port and private user/temp/output directories. Do not call the user's 8188 service, do not run concurrently, do not retry model generation, and do not run a matrix.

- [ ] **Step 3: Execute one confirmed smoke only if preflight passes**

The output must prove: 4 refinement forwards, finite joint latent, exact video latent equality before/after refine, decodable 24fps video, finite 32kHz stereo audio, and no persistent plugin-owned cache. If the resource gate abstains, record the ABSTAIN evidence and do not force execution.

- [ ] **Step 4: Commit the smoke runner only after its dry-run tests pass**

Do not commit generated media or machine-specific paths. Quality A/B, user blind listening, Quality Gate, workflows, README, and Frozen Cache remain separate phases after this mechanical smoke.

### Task 9: Post-smoke Quality Gate and user workflow

The single mechanical run completed both 4-step passes and strict media decode, but proved that ComfyUI's zero video mask alone does not guarantee byte-identical returned video latent. Add one append-only Quality Gate at position214 using TDD. It defaults to the original, rejects non-finite/shape/rate/channel/duration failures, treats signal metrics as review cues, and after explicit human acceptance splices only candidate audio latent into the exact original video latent.

Add a dated frontend workflow under`examples/workflows/18-audio-refine`, six or more canvas NOTE nodes, a per-directory README, root README/index entries, and an identical ComfyUI user workflow copy. Preserve old positions0..210 and every pre-existing workflow byte. Run one fixed0.6–0.7MP quality pair only after mechanical and resource gates pass; no matrix, concurrency, stress or cross-GPU run. Final completion remains blocked only on the user's blind listening judgment; Frozen Cache stays deferred.
