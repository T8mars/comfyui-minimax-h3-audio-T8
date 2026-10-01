from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_pkg"
sys.path.insert(0, str(PACKAGE_ROOT / "tests"))
sys.path.insert(0, str(PACKAGE_ROOT / "h3_t8"))

if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def restore_cpu_thread_count_after_each_test():
    """A test's explicit CPU setting must not leak into the next test.

    Do not force one numerical configuration: keep the actual incoming count,
    including a runner's deliberate setting, and restore it even on failure.
    """
    import torch
    previous = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(previous)
