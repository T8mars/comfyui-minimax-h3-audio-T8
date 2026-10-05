"""Actual JS recipes through existing snapshots, candidates and persistence."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg.director_project import new_project, ProjectStore
from h3_audio_t8_pkg.director_radar import (
    apply_operation, candidate_status, compiled_local, render_binding,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe_values():
    url = (ROOT / "web/director/continuity_recipes.mjs").as_uri()
    result = subprocess.run([shutil.which("node"), "--input-type=module", "-e",
        f"import {{continuityRecipe}} from {json.dumps(url)};console.log(JSON.stringify(['continuity','offscreen'].map(continuityRecipe)));"],
        capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_actual_js_form_boundaries_and_snapshot_defaults():
    result = subprocess.run([shutil.which("node"), "--test", str(ROOT / "tests/director_continuity_recipes.mjs")],
        capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_html_module_selection_preserves_both_native_core_entries():
    html = (ROOT / "web/director/index.html").read_text(encoding="utf8")
    declaration = next(line for line in html.splitlines() if line.startswith("const directorModules = "))
    paths = ["/minimax_h3_t8/director/ui", "/api/minimax_h3_t8/director/ui", "/extensions/minimax-h3-audio-T8/director/index.html"]
    script = "const paths=" + json.dumps(paths) + ";console.log(JSON.stringify(paths.map(pathname=>{const location={pathname};" + declaration + "return directorModules;})));"
    result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
        capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["/extensions/minimax-h3-audio-T8/director/"] * 2 + ["./"]
    for filename in ("creation_library.mjs", "workbench.mjs", "modal_ui.mjs"):
        assert "import(directorModules + '" + filename + "')" in html
        assert (ROOT / "web/director" / filename).is_file()


@pytest.mark.parametrize("index", [0, 1])
def test_recipe_uses_existing_explicit_candidate_save_reopen_and_revert(index, tmp_path):
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "原作者稿：镜头里的B看向门口。"
    original = json.loads(json.dumps(project))
    rule = recipe_values()[index]
    rule["parameters"] = {"global_timeline": "全片0–5秒，音频32000Hz；本镜0–5秒。${do_not_execute}"}
    project = apply_operation(project, sid, "skill_create", rule)
    shot = project["doc"]["shots"][0]
    assert compiled_local(project, shot) == original["doc"]["shots"][0]["simplePrompt"]
    assert "promptCandidate" not in shot
    text = render_binding(shot["skillBindings"][0])
    assert "${do_not_execute}" in text  # single-pass literal only
    assert len(project["doc"].get("sharedSkills", [])) == 0
    project = apply_operation(project, sid, "candidate_create", {"facts": False, "intent": True})
    shot = project["doc"]["shots"][0]
    assert candidate_status(project, shot) == "pending"
    assert compiled_local(project, shot) == shot["simplePrompt"]
    project = apply_operation(project, sid, "candidate_accept", {"confirm": True, "text_sha256": shot["promptCandidate"]["text_sha256"]})
    shot = project["doc"]["shots"][0]
    assert compiled_local(project, shot).endswith(text)
    assert shot["simplePrompt"] == original["doc"]["shots"][0]["simplePrompt"]
    assert project["assets"] == original["assets"]
    assert shot["events"] == original["doc"]["shots"][0]["events"]
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    saved = store.save(project, 0)
    reopened = store.load(saved["id"])
    assert reopened["doc"]["shots"][0]["skillBindings"] == shot["skillBindings"]
    assert candidate_status(reopened, reopened["doc"]["shots"][0]) == "active"
    reverted = apply_operation(reopened, sid, "candidate_revert", {})
    assert compiled_local(reverted, reverted["doc"]["shots"][0]) == shot["simplePrompt"]
    # Draft parameter change cannot silently run an already adopted candidate.
    reopened["doc"]["shots"][0]["skillBindings"][0]["parameters"]["global_timeline"] = "全片5–10秒"
    assert candidate_status(reopened, reopened["doc"]["shots"][0]) == "stale"
    with pytest.raises(ValueError, match="依赖已改变"):
        compiled_local(reopened, reopened["doc"]["shots"][0])
