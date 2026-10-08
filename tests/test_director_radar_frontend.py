"""Actual exported frontend key/hash functions and Python/JS input parity."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg.director_radar import _inputs, apply_operation
from test_director_radar_history import cross_project
from test_director_radar import accept

ROOT = Path(__file__).resolve().parents[1]


def test_actual_frontend_key_and_lan_hash_regression():
    node = shutil.which("node")
    assert node, "This frontend qualification requires actual Node.js"
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests" / "director_radar_regression.mjs")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_readonly_diagnostics_text_module():
    node = shutil.which("node")
    assert node, "Diagnostics text qualification requires actual Node.js"
    result = subprocess.run([node, "--test", str(ROOT / "tests/director_diagnostics_text.mjs")],
        capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("facts,intent", [(False, True), (True, True), (False, False)])
def test_actual_python_js_candidate_consumption_parity(facts, intent):
    project, sid = cross_project()
    first = project["doc"]["shots"][0]
    shot = project["doc"]["shots"][1]
    shot["sourceEvidence"] = first["sourceEvidence"]
    project = accept(
        apply_operation(
            project, sid, "candidate_create", {"facts": facts, "intent": intent}
        ),
        sid,
    )
    shot = project["doc"]["shots"][1]
    expected = _inputs(project, shot, shot["promptCandidate"]["options"])
    url = (ROOT / "web" / "director" / "radar_state.mjs").as_uri()
    script = f"import {{radarCandidateInputState}} from {json.dumps(url)};let text='';for await(const chunk of process.stdin)text+=chunk;const p=JSON.parse(text);console.log(JSON.stringify(radarCandidateInputState(p.doc,p.doc.shots[1],p.assets).inputs));"
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        input=json.dumps(project),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
