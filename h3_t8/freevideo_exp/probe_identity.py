"""Bind the two small probes to actual compute source, environment and GPU."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .runtime import canonical, digest, write_json_new


def identity_for(config, hardware):
    rows = [row for row in config["files"] if row["kind"] == "source" and (
        Path(row["path"]).is_relative_to(Path(config["source_root"]))
        or Path(row["path"]).is_relative_to(Path(config["vdn_root"]))
        or Path(row["path"]) == Path(config["python"]))]
    environment = {}
    for name in ("torch", "triton"):
        distributions = importlib.metadata.packages_distributions().get(name, [name])
        if len(distributions) != 1:
            raise ValueError("Ambiguous external distribution: " + name)
        distribution = importlib.metadata.distribution(distributions[0])
        record = distribution.read_text("RECORD")
        if record is None:
            raise ValueError("External runtime distribution has no identity RECORD: " + name)
        environment[name] = dict(distribution=distributions[0], version=distribution.version,
                                record_sha256=hashlib.sha256(record.encode()).hexdigest())
    if not hardware.gpu_uuid or not hardware.driver_version:
        raise ValueError("Cannot bind FreeVideo probes without the actual GPU UUID and driver")
    return dict(schema="t8-freevideo-kernel-identity-v1", python=str(Path(sys.executable).resolve()),
                compute_source_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
                environment=environment, gpu=hardware.gpu_name, gpu_uuid=hardware.gpu_uuid,
                capability=list(hardware.capability), driver=hardware.driver_version,
                torch=hardware.torch_version, cuda=hardware.cuda_version, system=hardware.system)


def validate_probes(config, probes, current):
    path = Path(config["home"]) / "kernel-probe-binding.json"
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("identity") != current:
        raise ValueError("FreeVideo GPU/runtime/kernel source differs from the probed identity")
    for name in ("linear", "cudnn"):
        row = probes.get(name, {})
        log = Path(config["home"]) / ("probe-" + name + ".log")
        if (binding.get("logs", {}).get(name) != digest(log) or row.get("log_sha256") != digest(log)
                or row.get("status") != "complete" or row.get("finite") is not True
                or any(row.get(key) != current[key] for key in ("torch", "cuda", "gpu", "system", "capability"))):
            raise ValueError("FreeVideo probe result/log identity mismatch: " + name)


def bind(config, identity, probes, *, origin):
    # Create-only. Binding old logs is explicitly labelled, never a new kernel run.
    write_json_new(Path(config["home"]) / "kernel-probe-binding.json",
                   dict(identity=identity, logs={name:row["log_sha256"] for name,row in probes.items()}, origin=origin))
    validate_probes(config, probes, identity)
