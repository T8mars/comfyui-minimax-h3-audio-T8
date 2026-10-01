"""Run the actual frontend state modules with memory-only platform substitutes."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_director_state_sequences():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the frontend state regression suite")
    suite = Path(__file__).with_suffix(".mjs").with_name("director_state_regression.mjs")
    result = subprocess.run(
        [node, "--test", str(suite), str(suite.with_name("director_creation_regression.mjs")), str(suite.with_name("director_audit20_regression.mjs")), str(suite.with_name("director_postaccept_regression.mjs"))], capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
