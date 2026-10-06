"""Two new-source tiny probes, optionally one masked tiny, never old log rebinding."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3_t8.freevideo_quality.runtime import config_for, load_model  # noqa: E402 - explicit CLI project bootstrap
from h3_t8.freevideo_exp.runtime import digest, write_json_new  # noqa: E402
from h3_t8.freevideo_exp.probe_identity import bind  # noqa: E402
from h3_t8.freevideo_exp.supervision import owned_process  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_config", type=Path)
    parser.add_argument("--masked", action="store_true")
    args = parser.parse_args()
    config = config_for(load_model(args.runtime_config), full=True)
    home = Path(config["home"])
    output = home / "kernel-probes.json"
    if output.exists():
        raise FileExistsError("Probe result exists: inspect it; do not overwrite or relabel old logs")
    env = dict(os.environ, FREEVIDEO_HOME=str(home), FREEVIDEO_VDN_ROOT=config["vdn_root"],
        TRITON_CACHE_DIR=str(home / "triton"), TORCHINDUCTOR_CACHE_DIR=str(home / "inductor"),
        TORCHINDUCTOR_COMPILE_THREADS="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "FREEVIDEO_RUNTIME_LOCK_FD", "FREEVIDEO_RUNTIME_LOCK_HANDLE"):
        env.pop(key, None)

    def identify():
        code = "import sys,json;sys.path[:0]=sys.argv[1:3];from freevideo_engine.paths import add_vdn;add_vdn();from freevideo_engine.hardware import _detect_local;from freevideo_exp.probe_identity import identity_for;print(json.dumps(identity_for(json.load(open(sys.argv[3],encoding='utf8')),_detect_local())))"
        result = subprocess.check_output([config["python"], "-I", "-B", "-X", "utf8", "-c", code,
            str(ROOT / "h3_t8"), config["source_root"], str(args.runtime_config)], env=env,
            text=True, encoding="utf8", timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return json.loads(result.strip().splitlines()[-1])

    identity, results = identify(), {}
    for backend in ("linear", "cudnn"):
        run = home / ("tiny-" + backend)
        run.mkdir(exist_ok=False)
        command = [config["python"], "-I", "-B", "-X", "utf8", str(ROOT / "h3_t8/freevideo_quality/probe_worker.py"),
                   str(run), config["source_root"], backend]
        log = home / ("probe-" + backend + ".log")
        with log.open("x", encoding="utf8") as stream, owned_process(command, env, run, stream) as child:
            if child.wait(timeout=300):
                raise RuntimeError("Probe failed; original log retained: " + str(log))
        rows = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines() if line.startswith('{"backend":')]
        if not rows or rows[-1].get("status") != "complete":
            raise ValueError("Missing complete tiny result: " + backend)
        results[backend] = dict(rows[-1], log_sha256=digest(log))
        print(json.dumps(results[backend]), flush=True)
    if identify() != identity:
        raise ValueError("Compute source/environment changed across probes")
    config_for(load_model(args.runtime_config), full=True)
    write_json_new(output, results)
    bind(config, identity, results, origin="New v0.2.3 source, two new owned tiny executions; no old log reuse")
    if args.masked:
        run = home / "masked-probe-v1"
        run.mkdir(exist_ok=False)
        code = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('freevideo_exp.masked_probe',run_name='__main__')"
        command = [config["python"], "-I", "-B", "-X", "utf8", "-c", code, str(ROOT / "h3_t8"), str(args.runtime_config), str(run)]
        with (run / "log.txt").open("x", encoding="utf8") as stream, owned_process(command, env, run, stream) as child:
            if child.wait(timeout=120):
                raise RuntimeError("New-source masked tiny failed; log retained")
        print((run / "result.json").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
