"""Explicitly reuse unchanged-compute probe evidence, never claim new execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def identify(config, config_path):
    code = "import sys,json;sys.path[:0]=sys.argv[1:3];from freevideo_engine.paths import add_vdn;add_vdn();from freevideo_engine.hardware import _detect_local;from freevideo_exp.probe_identity import identity_for;print(json.dumps(identity_for(json.load(open(sys.argv[3],encoding='utf8')),_detect_local())))"
    env = dict(os.environ, FREEVIDEO_VDN_ROOT=config["vdn_root"])
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    output = subprocess.check_output([config["python"], "-I", "-B", "-X", "utf8", "-c", code,
        str(ROOT / "h3_t8"), config["source_root"], str(config_path)], env=env,
        text=True, encoding="utf8", timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return json.loads(output.strip().splitlines()[-1])


def main():
    from h3_t8.freevideo_quality.runtime import config_for, load_model
    from h3_t8.freevideo_exp.runtime import digest, write_json_new
    from h3_t8.freevideo_exp.probe_identity import bind, validate_probes
    from h3_t8.freevideo_exp.masked_probe import validate as validate_masked
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_config", type=Path)
    parser.add_argument("--original-config", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.runtime_config)
    config = config_for(model, full=True)
    original = json.loads(args.original_config.read_text(encoding="utf8"))
    home, old_home = Path(config["home"]), Path(original["home"])
    if home.resolve() == old_home.resolve():
        raise ValueError("Evidence reuse needs a separate destination home")
    for name in ("kernel-probes.json", "kernel-probe-binding.json", "probe-linear.log",
                 "probe-cudnn.log", "masked-probe-v1", "probe-reuse-origin.json"):
        if (home / name).exists():
            raise FileExistsError("Never overwrite/relabel existing evidence: " + str(home / name))
    current = identify(config, args.runtime_config)
    if identify(original, args.original_config) != current:
        raise ValueError("Original and current actual compute identities differ")
    probes = json.loads((old_home / "kernel-probes.json").read_text(encoding="utf8"))
    validate_probes(original, probes, current)
    validate_masked(original, current)
    if set(probes) != {"linear", "cudnn"}:
        raise ValueError("Expected exactly the two original bounded kernel probes")
    for name in probes:
        row = probes[name]
        parsed = [json.loads(line) for line in (old_home / ("probe-" + name + ".log")).read_text(
            encoding="utf8").splitlines() if line.startswith('{"backend":')]
        if not parsed or dict(parsed[-1], log_sha256=row["log_sha256"]) != row:
            raise ValueError("Original complete log/result disagree: " + name)
        limit = .01 if name == "linear" else .006
        if row.get("relative_rmse_limit") != limit or not 0 <= row.get("relative_rmse", 1) <= limit:
            raise ValueError("Original numerical gate failed: " + name)
    masked = json.loads((old_home / "masked-probe-v1/result.json").read_text(encoding="utf8"))
    parsed_masked = [json.loads(line) for line in (old_home / "masked-probe-v1/log.txt").read_text(
        encoding="utf8").splitlines() if line.startswith('{"status":')]
    if not parsed_masked or parsed_masked[-1] != masked:
        raise ValueError("Original masked log/result disagree")
    files = ("kernel-probes.json", "probe-linear.log", "probe-cudnn.log",
             "masked-probe-v1/result.json", "masked-probe-v1/log.txt")
    origin = dict(schema="t8-freevideo-probe-evidence-reuse-v1", new_GPU_probe_executions=0,
        original_config=str(args.original_config.resolve()), original_config_sha256=digest(args.original_config),
        current_config=str(args.runtime_config.resolve()), current_config_sha256=model.config_sha256,
        original_home=str(old_home.resolve()), identity=current,
        original_binding_sha256=digest(old_home / "kernel-probe-binding.json"),
        files={name: dict(original_path=str((old_home / name).resolve()), sha256=digest(old_home / name)) for name in files},
        scope="Original v0.2.3 tiny executions retained; UI adapter epoch changed, compute source/environment/GPU unchanged")
    if identify(config, args.runtime_config) != current:
        raise ValueError("Compute identity changed while inspecting evidence")
    # All source bytes have been verified. Create-only copies preserve their original epoch.
    (home / "masked-probe-v1").mkdir(parents=True, exist_ok=False)
    for name in files:
        with (home / name).open("xb") as stream:
            stream.write((old_home / name).read_bytes())
        if digest(home / name) != origin["files"][name]["sha256"]:
            raise ValueError("Copied evidence SHA mismatch: " + name)
    write_json_new(home / "probe-reuse-origin.json", origin)
    bind(config, current, probes, origin=origin)
    validate_masked(config, current)
    print(json.dumps(dict(status="pass", new_GPU_probe_executions=0,
        original_home=str(old_home), current_home=str(home), identity=current), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
