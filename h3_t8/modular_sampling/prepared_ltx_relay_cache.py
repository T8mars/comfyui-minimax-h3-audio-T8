"""Explicit, independently cached Prepared LTX event encoding; never sampling."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from ..prepared_backend.backend_files import sha
from ..prepared_backend.ltx_effects_contract import RELAY_SCHEMA, validate_plan, validate_relay
from ..prepared_backend.resource_guard import (
    file_identity, GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease,
)
from ..prepared_generation_contract import engine_sources, environment_identity, job_directory, validate_bundle
from ..prepared_generation_runtime import run_worker, verify_assets, verify_record, write_json
from ..prepared_identity import absolute_path, linked

SCHEMA = "t8_prepared_ltx_relay_encode_state_v1"
SUCCESS = "native_INT8_event_caches_prepared_not_sampling"
SPEC = ("ltx_relay_cache_worker.py", SUCCESS, "encoded-contexts.safetensors", 3600)
NAMESPACE = "MiniMaxH3-PreparedLTXRelayCaches"


def _sources():
    return {**engine_sources(), str(Path(__file__).resolve()): sha(Path(__file__))}


def _root(output_directory, cache_id):
    output = job_directory(output_directory, cache_id).parent.parent
    root = output / NAMESPACE / cache_id
    if (not root.resolve().is_relative_to(output) or any(path.exists() and linked(path)
            for path in (root.parent, root))):
        raise ValueError("Prepared Relay cache leaves its owned output directory")
    return root


def _verify_manifest(path, digest, plan, bundle, root):
    path = Path(absolute_path(str(path)))
    if (linked(path) or not path.resolve(strict=True).is_relative_to(root.resolve())
            or not 0 < path.stat().st_size <= 1024**2 or sha(path) != digest):
        raise ValueError("Prepared Relay manifest content/path changed")
    manifest = json.loads(path.read_text(encoding="utf8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "plan_hash", "caches"}
            or manifest["schema"] != RELAY_SCHEMA or manifest["plan_hash"] != plan["plan_hash"]):
        raise ValueError("Prepared Relay cache manifest differs from its encoded plan")
    validate_relay({"mode": "report_only", "max_workspace_mib": 32, "plan": plan,
                   "caches": manifest["caches"]}, bundle["generation"]["geometry"], bundle["generation"]["prompt"])
    for cache in manifest["caches"]:
        current = Path(absolute_path(cache["path"]))
        if (linked(current) or not current.resolve(strict=True).is_relative_to(path.parent.resolve())
                or sha(current) != cache["sha256"]):
            raise ValueError("Prepared Relay event cache content/path changed")
    return str(path.resolve())


def run_cache_encoding(bundle, plan, *, gemma_path, output_directory, cache_id,
                       resume_existing, lease_path, interrupt):
    validate_bundle(bundle)
    if bundle["kind"] != "ltx_refine" or type(resume_existing) is not bool:
        raise ValueError("Prepared Relay encoding needs an LTX bundle and boolean resume flag")
    bundle, plan = deepcopy(bundle), deepcopy(plan)
    validate_plan(plan, bundle["generation"]["geometry"], bundle["generation"]["prompt"])
    interrupt()
    gemma = file_identity(absolute_path(gemma_path)) if plan["events"] else None
    if gemma is not None and linked(Path(gemma_path)):
        raise ValueError("Prepared Relay text checkpoint cannot be a link")
    sources = _sources()
    provenance = {"bundle": bundle, "plan": plan, "gemma": gemma,
                  "sources": sources, "environment": environment_identity()}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    root = _root(output_directory, cache_id)
    root.mkdir(parents=True, exist_ok=True)
    with SerialProbeLease(root / "job.lock"):
        state_path = root / "state.json"
        state = {"schema": SCHEMA, "fingerprint": identity, "record": None, "manifest_sha256": None}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf8"))
            if (not isinstance(state, dict) or set(state) != {"schema", "fingerprint", "record", "manifest_sha256"}
                    or state["schema"] != SCHEMA):
                raise ValueError("Invalid Prepared Relay cache state")
            if not resume_existing or state["fingerprint"] != identity:
                raise ValueError("Prepared Relay cache settings/source differ; use a new cache_id")
        verify_assets(bundle, interrupt)
        write_json(state_path, state)
        hit = state["manifest_sha256"] is not None
        if hit:
            if plan["events"]:
                primary = verify_record(state["record"], root)
                if state["record"]["report"].get("status") != SUCCESS:
                    raise ValueError("Prepared Relay cache worker did not complete")
                manifest = primary.parent / "relay-caches.json"
            else:
                if state["record"] is not None:
                    raise ValueError("Global-only Relay cannot have an encoder worker record")
                manifest = root / "global-only-relay-caches.json"
        elif not plan["events"]:
            manifest = root / "global-only-relay-caches.json"
            write_json(manifest, {"schema": RELAY_SCHEMA, "plan_hash": plan["plan_hash"], "caches": []})
            state["manifest_sha256"] = sha(manifest)
        else:
            lease = Path(absolute_path(str(lease_path)))
            lease.parent.mkdir(parents=True, exist_ok=True)
            with SerialProbeLease(lease), NvmlResourceReader() as reader:
                interrupt()
                guard = ResourceGuard(GuardPolicy(startup_free_ram_bytes=64 * 1024**3,
                    minimum_free_gpu_bytes=2 * 1024**3, minimum_free_ram_bytes=8 * 1024**3))
                sample = reader.sample()
                if reason := guard.observe(sample, startup=True):
                    raise RuntimeError("Prepared Relay text encoding startup guard: " + reason)
                request = {"schema": "t8_prepared_ltx_native_relay_encode_v1",
                    "isolated_paths": bundle["generation"]["isolated_paths"],
                    "base": bundle["generation"]["base"], "gemma": gemma["path"],
                    "geometry": bundle["generation"]["geometry"], "prompt": bundle["generation"]["prompt"],
                    "gpu_uuid": sample["gpu_uuid"], "plan": plan,
                    "identities": {**{asset["path"]: asset["sha256"] for asset in bundle["assets"]},
                                   **sources, gemma["path"]: gemma["sha256"]}}
                record = run_worker(root / ("encode-" + uuid.uuid4().hex), SPEC, request, reader, guard, interrupt)
                interrupt()
                manifest = Path(record["path"]).parent / "relay-caches.json"
                state["record"] = record
                state["manifest_sha256"] = record["report"]["manifest_sha256"]
        path = _verify_manifest(manifest, state["manifest_sha256"], plan, bundle, root)
        interrupt()
        verify_assets(bundle, interrupt)
        if _sources() != sources or (gemma is not None and file_identity(gemma["path"]) != gemma):
            raise RuntimeError("Prepared Relay encoder implementation/checkpoint changed while running")
        write_json(state_path, state)
        return path, {"status": "event_caches_ready_not_sampled" if plan["events"] else "global_only_bypass",
            "plan_hash": plan["plan_hash"], "manifest_sha256": state["manifest_sha256"],
            "event_count": len(plan["events"]), "encoder_ran": bool(plan["events"]) and not hit,
            "cache_hit": hit, "sampling_ran": False, "original_bundle_unchanged": True,
            "quality_accepted": False, "state_path": str(state_path.resolve())}
