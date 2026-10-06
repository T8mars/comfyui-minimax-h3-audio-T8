"""Tiny probe launch gate: no engine/Torch import before owned Job assignment."""
from pathlib import Path
import runpy
import sys
import time

root, source, backend = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
started = time.monotonic()
while not (root / "launch-gate").is_file():
    if time.monotonic() - started > 30:
        raise RuntimeError("Tiny probe was not assigned to its supervisor")
    time.sleep(.05)
sys.path.insert(0, source)
from freevideo_engine.paths import add_vdn
add_vdn()
sys.argv = ["probe", backend]
runpy.run_module("freevideo_engine.probe", run_name="__main__")
