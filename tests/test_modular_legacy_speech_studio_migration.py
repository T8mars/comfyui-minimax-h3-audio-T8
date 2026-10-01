"""Execute the old SpeechStudio positional migration against frozen workflows."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web/legacy_speech_studio_widgets.js"
OLD = (ROOT / "examples/workflows/05-speech-dialogue/"
       "2026-08-09_H3_Speech_Dialogue_Two_Speaker_Stock20_EXP.json")


def test_old_speech_studio_preserves_values_and_inserts_fixed_seed_control_only():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend migration test")
    graph = json.loads(OLD.read_text(encoding="utf-8"))
    speech = next(item for item in graph["nodes"] if item["type"] == "MiniMaxH3SpeechStudioT8")
    assert len(speech["widgets_values"]) == 21
    program = r"""
const fs = require('fs'), assert = require('assert');
let extension;
const app = {registerExtension(value) {extension = value;}};
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;\r?\n/, '').replace('export function ', 'function ');
const migrate = new Function('app', source + '; return migrateLegacySpeechStudioGraph;')(app);
const graph = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const prior = graph.nodes.find(node => node.type === 'MiniMaxH3SpeechStudioT8').widgets_values.slice();
const raw = fs.readFileSync(process.argv[2], 'utf8');
extension.beforeConfigureGraph(graph);
const actual = graph.nodes.find(node => node.type === 'MiniMaxH3SpeechStudioT8').widgets_values;
assert.deepEqual(actual, [...prior.slice(0, 2), 'fixed', ...prior.slice(2)]);
assert.equal(migrate(graph), 0);
assert.equal(fs.readFileSync(process.argv[2], 'utf8'), raw);
const malformed = {nodes:[{type:'MiniMaxH3SpeechStudioT8',
 widgets_values:[...prior.slice(0, 2), 'fixed', ...prior.slice(2)]}]};
assert.equal(migrate(malformed), 0);
const other = {nodes:[{type:'OtherNode', widgets_values:prior}]};
assert.equal(migrate(other), 0);
"""
    subprocess.run([node, "-e", program, str(SOURCE), str(OLD)],
                   check=True, capture_output=True, text=True, timeout=30)
