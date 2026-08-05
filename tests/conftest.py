from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_pkg"
sys.path.insert(0, str(PACKAGE_ROOT / "tests"))

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
