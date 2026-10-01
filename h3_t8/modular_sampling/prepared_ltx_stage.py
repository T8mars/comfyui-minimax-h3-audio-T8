"""Opt-in prepared LTX refinement and decode as independently owned stages.

The legacy combined controller and its cache namespace are intentionally untouched.
This controller reuses its fixed workers, resource guard and request identities, but
never starts the other stage implicitly.
"""

import json
from pathlib import Path
import re
import uuid

from ..prepared_backend.backend_files import sha
from ..prepared_backend.resource_guard import (
    GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease,
)
from ..prepared_generation_contract import (
    decode_request, engine_sources, fingerprint, generation_request, job_directory,
    validate_bundle,
)
from ..prepared_generation_runtime import (
    ROUTES, run_worker, validate_stages, verify_assets, verify_record, write_json,
)
from ..prepared_identity import absolute_path

STATE_SCHEMA = "t8_modular_prepared_ltx_v1"
RECEIPT_SCHEMA = "t8_modular_prepared_ltx_generation_v1"
NAMESPACE = "MiniMaxH3-ModularPreparedLTX"


def _sources():
    sources = engine_sources()
    this_file = Path(__file__).resolve()
    sources[str(this_file)] = sha(this_file)
    return sources


def _root(output_directory, chain_id):
    legacy = job_directory(output_directory, chain_id)
    output = legacy.parent.parent
    root = output / NAMESPACE / chain_id
    if root.parent.is_symlink() or root.is_symlink() or not root.resolve().is_relative_to(output):
        raise ValueError("Modular LTX job leaves the selected output directory")
    return root


def _context(bundle, output_directory, chain_id, noise_seed):
    validate_bundle(bundle)
    if bundle["kind"] != "ltx_refine":
        raise ValueError("This split stage accepts only prepared LTX refinement bundles")
    sources = _sources()
    identity = fingerprint(bundle, noise_seed, sources)
    return _root(output_directory, chain_id), sources, identity


def _read_state(root, identity, *, required, resume_existing=True):
    state_path = root / "state.json"
    if not state_path.exists():
        if required:
            raise ValueError("No committed modular LTX generation stage exists")
        return {"schema": STATE_SCHEMA, "kind": "ltx_refine", "fingerprint": identity,
                "stages": {}, "human_review": "pending"}
    state = json.loads(state_path.read_text(encoding="utf8"))
    if (not isinstance(state, dict) or set(state) != {"schema", "kind", "fingerprint", "stages", "human_review"}
            or state["schema"] != STATE_SCHEMA or state["kind"] != "ltx_refine"
            or state["human_review"] != "pending" or not isinstance(state["stages"], dict)):
        raise ValueError("Invalid modular LTX state")
    if not resume_existing or state["fingerprint"] != identity:
        raise ValueError("Existing modular LTX chain differs or resume is disabled; use a new chain_id")
    validate_stages(state["stages"])
    for record in state["stages"].values():
        verify_record(record, root)
    return state


def _receipt(root, state, noise_seed):
    record = state["stages"]["generation"]
    path = verify_record(record, root)
    return {"schema": RECEIPT_SCHEMA, "fingerprint": state["fingerprint"],
            "chain_id": root.name, "noise_seed": noise_seed,
            "state_path": str((root / "state.json").resolve()),
            "latent_path": str(path), "sha256": record["sha256"]}


def _verify_receipt(receipt, root, state, noise_seed):
    expected = _receipt(root, state, noise_seed)
    if not isinstance(receipt, dict) or receipt != expected:
        raise ValueError("Generation receipt differs from the committed LTX state")


def _run_one(stage_name, bundle, root, noise_seed, sources, identity, lease_path, interrupt, state):
    lease = Path(absolute_path(str(lease_path)))
    lease.parent.mkdir(parents=True, exist_ok=True)
    with SerialProbeLease(lease), NvmlResourceReader() as reader:
        interrupt()
        guard = ResourceGuard(GuardPolicy(
            startup_free_ram_bytes=(64 if stage_name == "generation" else 16) * 1024**3,
            minimum_free_gpu_bytes=2 * 1024**3,
            minimum_free_ram_bytes=8 * 1024**3,
        ))
        sample = reader.sample()
        if reason := guard.observe(sample, startup=True):
            raise RuntimeError("Prepared LTX stage startup guard: " + reason)
        if stage_name == "generation":
            request = generation_request(bundle, noise_seed, sample["gpu_uuid"], sources)
            spec = ROUTES["ltx_refine"][0]
        else:
            latent = verify_record(state["stages"]["generation"], root)
            request = decode_request(bundle, latent, sample["gpu_uuid"], sources)
            spec = ROUTES["ltx_refine"][1]
        record = run_worker(root / f"{stage_name}-{uuid.uuid4().hex}", spec,
                            request, reader, guard, interrupt)
        interrupt()
        if _sources() != sources or fingerprint(bundle, noise_seed, sources) != identity:
            raise RuntimeError("Prepared LTX implementation or settings changed while running")
        return record


def run_generation(bundle, *, output_directory, chain_id, noise_seed, resume_existing,
                   lease_path, interrupt):
    if type(resume_existing) is not bool:
        raise ValueError("resume_existing must be a boolean")
    root, sources, identity = _context(bundle, output_directory, chain_id, noise_seed)
    root.mkdir(parents=True, exist_ok=True)
    with SerialProbeLease(root / "job.lock"):
        state = _read_state(root, identity, required=False, resume_existing=resume_existing)
        verify_assets(bundle, interrupt)
        write_json(root / "state.json", state)
        hit = "generation" in state["stages"]
        if not hit:
            record = _run_one("generation", bundle, root, noise_seed, sources, identity,
                              lease_path, interrupt, state)
            state["stages"]["generation"] = record
            write_json(root / "state.json", state)
        receipt = _receipt(root, state, noise_seed)
        return receipt, {"status": "refined_latent_ready_pending_decode",
            "latent_path": receipt["latent_path"], "sha256": receipt["sha256"],
            "state_path": receipt["state_path"], "generation_ran": not hit,
            "decode_ran": False, "human_review": "pending"}


def load_generation(bundle, *, output_directory, chain_id, noise_seed,
                    expected_sha256, interrupt):
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("Explicit generation SHA256 is required")
    root, _, identity = _context(bundle, output_directory, chain_id, noise_seed)
    if not root.is_dir():
        raise ValueError("No committed modular LTX generation stage exists")
    with SerialProbeLease(root / "job.lock"):
        state = _read_state(root, identity, required=True)
        verify_assets(bundle, interrupt)
        if "generation" not in state["stages"]:
            raise ValueError("No committed LTX generation artifact to load")
        receipt = _receipt(root, state, noise_seed)
        if receipt["sha256"] != expected_sha256:
            raise ValueError("Explicit LTX generation SHA256 differs from committed artifact")
        return receipt, {"status": "refined_latent_loaded_no_generation",
            "latent_path": receipt["latent_path"], "sha256": receipt["sha256"],
            "state_path": receipt["state_path"], "generation_ran": False,
            "decode_ran": False, "human_review": "pending"}


def run_decode(bundle, receipt, *, output_directory, lease_path, interrupt):
    if (not isinstance(receipt, dict) or set(receipt) != {"schema", "fingerprint", "chain_id",
            "noise_seed", "state_path", "latent_path", "sha256"}
            or receipt.get("schema") != RECEIPT_SCHEMA):
        raise ValueError("Expected an explicit modular LTX generation receipt")
    root, sources, identity = _context(bundle, output_directory, receipt["chain_id"], receipt["noise_seed"])
    if not root.is_dir():
        raise ValueError("No committed modular LTX generation stage exists")
    with SerialProbeLease(root / "job.lock"):
        state = _read_state(root, identity, required=True)
        verify_assets(bundle, interrupt)
        if "generation" not in state["stages"]:
            raise ValueError("Decode cannot start without a committed generation stage")
        _verify_receipt(receipt, root, state, receipt["noise_seed"])
        hit = "decode" in state["stages"]
        if not hit:
            record = _run_one("decode", bundle, root, receipt["noise_seed"], sources,
                              identity, lease_path, interrupt, state)
            state["stages"]["decode"] = record
            write_json(root / "state.json", state)
        movie = verify_record(state["stages"]["decode"], root)
        report = {"status": "prepared_ltx_video_ready_pending_review",
            "fingerprint": identity, "movie": str(movie), "sha256": sha(movie),
            "state_path": str((root / "state.json").resolve()),
            "generation_ran": False, "decode_ran": not hit,
            "human_review": "pending"}
        write_json(root / "last_decode.json", report)
        return movie, report
