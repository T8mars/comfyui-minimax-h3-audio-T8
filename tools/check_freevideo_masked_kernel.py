"""Run exactly one additive-bias cuDNN tiny in an owned isolated child."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from freevideo_exp.runtime import config_for, load_model
from freevideo_exp.supervision import owned_process


def main():
    config = config_for(load_model(sys.argv[1]), full=True)
    root = Path(config["home"]) / "masked-probe-v1"
    root.mkdir(exist_ok=False)
    env = dict(os.environ, FREEVIDEO_HOME=config["home"], FREEVIDEO_VDN_ROOT=config["vdn_root"],
        TRITON_CACHE_DIR=str(Path(config["home"]) / "triton"), TORCHINDUCTOR_CACHE_DIR=str(Path(config["home"]) / "inductor"))
    code = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('freevideo_exp.masked_probe',run_name='__main__')"
    command = [config["python"], "-I", "-B", "-X", "utf8", "-c", code,
        str(Path(__file__).resolve().parents[1] / "h3_t8"), str(Path(sys.argv[1]).resolve()), str(root)]
    with (root / "log.txt").open("x", encoding="utf8") as log:
        with owned_process(command, env, root, log) as process:
            if process.wait(timeout=120):
                raise RuntimeError("Masked probe failed; preserved log: " + str(root))
    print(json.dumps(json.loads((root / "result.json").read_text(encoding="utf8"))))


if __name__ == "__main__":
    main()
