"""Real Windows owned workers must not mutate an imported frozen source tree."""
import os
from pathlib import Path

import pytest

from h3_audio_t8_pkg.dlss_fi_backend.process import IsolatedTaskError, run_isolated


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows Job, never a mocked process")
@pytest.mark.parametrize("fail", [False, True])
def test_owned_worker_import_preserves_frozen_file_set_even_on_error(tmp_path, fail):
    helper = tmp_path / "frozen_sibling.py"
    helper.write_text("VALUE = 19\n", encoding="utf8")
    task = tmp_path / "worker.py"
    task.write_text("import sys\nimport frozen_sibling\n"
        "assert frozen_sibling.VALUE == 19\n"
        "print('dont_write_bytecode=' + str(sys.dont_write_bytecode), flush=True)\n"
        + ("raise RuntimeError('owned_import_failure')\n" if fail else ""), encoding="utf8")
    original = {path.name: path.read_bytes() for path in (helper, task)}
    if fail:
        with pytest.raises(IsolatedTaskError) as caught:
            run_isolated(task, [], timeout=15)
        receipt = caught.value.receipt
        assert receipt["status"] == "child_failed" and "owned_import_failure" in receipt["stderr_tail"]
    else:
        receipt = run_isolated(task, [], timeout=15)
        assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["job_assigned_before_task"] and receipt["active_after_cleanup"] == 0
    assert "dont_write_bytecode=True" in receipt["stdout_tail"]
    assert {path.name: path.read_bytes() for path in (helper, task)} == original
    assert not list(Path(tmp_path).rglob("*.pyc")), "Owned frozen tree gained bytecode during import"
