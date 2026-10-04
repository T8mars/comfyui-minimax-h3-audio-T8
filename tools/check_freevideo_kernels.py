"""Run only the two selected tiny kernels in external Python, preserving evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from freevideo_exp.runtime import config_for, digest, load_model, write_json_new
from freevideo_exp.probe_identity import bind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_config")
    parser.add_argument("--bind-existing", action="store_true", help="Bind this session's existing logs without rerunning kernels; never overwrite")
    args = parser.parse_args()
    config = config_for(load_model(args.runtime_config), full=True)
    home = Path(config["home"])
    result_path = home / "kernel-probes.json"
    if result_path.exists() and not args.bind_existing:
        raise FileExistsError("Probe result already exists; inspect it rather than rerunning kernels")
    results = {}
    env = dict(os.environ)
    env.update(FREEVIDEO_HOME=str(home), FREEVIDEO_VDN_ROOT=config["vdn_root"],
               TRITON_CACHE_DIR=str(home / "triton"), TORCHINDUCTOR_CACHE_DIR=str(home / "inductor"), TORCHINDUCTOR_COMPILE_THREADS="1")
    def identify():
        code = "import json,sys;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from freevideo_engine.paths import add_vdn;add_vdn();from freevideo_engine.hardware import _detect_local;from freevideo_exp.probe_identity import identity_for;c=json.load(open(sys.argv[3],encoding='utf8'));print(json.dumps(identity_for(c,_detect_local())))"
        output = subprocess.check_output([config["python"], "-I", "-B", "-X", "utf8", "-c", code,
                                          str(Path(__file__).parents[1] / "h3_t8"), config["source_root"], args.runtime_config],
                                         env=env, text=True, encoding="utf-8", timeout=60,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return json.loads(output.strip().splitlines()[-1])
    before = identify()
    if args.bind_existing:
        results = json.loads(result_path.read_text(encoding="utf-8"))
        earliest = min((home / ("probe-" + name + ".log")).stat().st_mtime_ns for name in ("linear", "cudnn"))
        # Existing evidence belongs to this session. Refuse if compute source was edited after those probes.
        for row in config["files"]:
            path = Path(row["path"])
            if row["kind"] == "source" and (path.is_relative_to(Path(config["source_root"])) or path.is_relative_to(Path(config["vdn_root"]))):
                if path.stat().st_mtime_ns > earliest:
                    raise ValueError("Cannot retrospectively bind edited compute source: " + str(path))
        bind(config, before, results, origin="Current-session existing logs, environment identity captured after probes; no new kernel executions")
        print(json.dumps(dict(status="existing_logs_bound", identity=before)))
        return
    for backend in ("linear", "cudnn"):
        code = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));from freevideo_engine.paths import add_vdn;add_vdn();sys.argv=['probe',sys.argv.pop(1)];runpy.run_module('freevideo_engine.probe',run_name='__main__')"
        command = [config["python"], "-I", "-B", "-X", "utf8", "-c", code, config["source_root"], backend]
        # Diagnostic subprocess output is kept even on failure, not a claimed pass.
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", timeout=300,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        log = home / ("probe-" + backend + ".log")
        with log.open("x", encoding="utf-8") as stream:
            stream.write(result.stdout)
        parsed = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"backend":')]
        if not parsed:
            raise RuntimeError("Probe returned no result; see " + str(log))
        results[backend] = dict(parsed[-1], log_sha256=digest(log))
        print(json.dumps(results[backend]), flush=True)
        if result.returncode or parsed[-1].get("status") != "complete":
            write_json_new(home / "kernel-probes-failed.json", results)
            raise RuntimeError("Kernel compatibility failed; no backend fallback: " + backend)
    write_json_new(result_path, results)
    if identify() != before:
        raise ValueError("FreeVideo probe environment changed during compatibility checks")
    bind(config, before, results, origin="Identity captured before and after the two tiny kernel probes")


if __name__ == "__main__":
    main()
