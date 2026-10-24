"""New source preparation keeps all old source/config defaults intact."""

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def tool():
    path = ROOT / "tools/prepare_h21_freevideo_source.py"
    spec = importlib.util.spec_from_file_location("h21_explicit_source_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_original_source_default_after_each_case():
    helper = tool().source
    original = helper.FREEVIDEO_REVISION
    try:
        yield
    finally:
        helper.FREEVIDEO_REVISION = original


def test_actual_old_preparer_default_survives_successful_new_preparation(monkeypatch):
    module = tool()
    old = module.source.FREEVIDEO_REVISION
    captured = []
    monkeypatch.setattr(sys, "argv", ["prepare_h21_freevideo_source.py", "--output", "new-only"])
    monkeypatch.setattr(module.source, "main", lambda: captured.append(module.source.FREEVIDEO_REVISION))
    module.main()
    assert captured == ["f2c25e6e4f82d5ad9ed966998423762503d98e5c"]
    assert module.source.FREEVIDEO_REVISION == old


def test_old_default_survives_real_helper_failure_without_retry(monkeypatch):
    module = tool()
    old, seen = module.source.FREEVIDEO_REVISION, []
    monkeypatch.setattr(sys, "argv", ["prepare_h21_freevideo_source.py"])

    def failing():
        seen.append(module.source.FREEVIDEO_REVISION)
        raise OSError("Retain a real Git/source read error")

    monkeypatch.setattr(module.source, "main", failing)
    with pytest.raises(OSError, match="Git/source read error"):
        module.main()
    assert seen == [module.REVISION] and module.source.FREEVIDEO_REVISION == old


def test_old_audio_flag_cannot_silently_select_another_source_family(monkeypatch):
    module = tool()
    old = module.source.FREEVIDEO_REVISION
    monkeypatch.setattr(sys, "argv", ["prepare_h21_freevideo_source.py", "--audio-reference-v3"])
    monkeypatch.setattr(module.source, "main", lambda: pytest.fail("No download for contradictory explicit source selection"))
    with pytest.raises(ValueError, match="not the old audio-v3"):
        module.main()
    assert module.source.FREEVIDEO_REVISION == old
