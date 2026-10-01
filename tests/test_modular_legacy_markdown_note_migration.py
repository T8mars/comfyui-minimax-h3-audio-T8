"""Old MarkdownNote scalar text survives in-memory native canvas import."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web/legacy_markdown_note_widgets.js"
OLD_SCALAR = (ROOT / "examples/workflows/04-long-video/"
              "2026-08-22_H3_Enhance_A_Video_Long_Video_Accepted_22F_Stock20_Advanced_EXP.json")
OLD_ARRAY = (ROOT / "examples/candidates/fasth3-v2-20260916/"
             "FastH3_V2_Dense_First_Frame_T8_Memory_EXP.json")


def test_scalar_notes_wrap_without_changing_old_files_or_existing_arrays():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend migration test")
    scalar = json.loads(OLD_SCALAR.read_text(encoding="utf-8"))
    array = json.loads(OLD_ARRAY.read_text(encoding="utf-8"))
    assert all(isinstance(item["widgets_values"], str)
               for item in scalar["nodes"] if item["type"] == "MarkdownNote")
    assert all(isinstance(item["widgets_values"], list)
               for item in array["nodes"] if item["type"] == "MarkdownNote")
    program = r"""
const fs = require('fs'), assert = require('assert');
let extension;
const app = {registerExtension(value) {extension = value;}};
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;\r?\n/, '').replace('export function ', 'function ');
const migrate = new Function('app', source + '; return migrateLegacyMarkdownNotes;')(app);
const scalar = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const array = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const original = fs.readFileSync(process.argv[2], 'utf8');
const oldTexts = scalar.nodes.filter(n => n.type === 'MarkdownNote').map(n => n.widgets_values);
assert.equal(migrate(scalar), oldTexts.length);
assert.deepEqual(scalar.nodes.filter(n => n.type === 'MarkdownNote').map(n => n.widgets_values),
 oldTexts.map(text => [text]));
assert.equal(migrate(scalar), 0);
assert.equal(migrate(array), 0);
assert.equal(fs.readFileSync(process.argv[2], 'utf8'), original);
assert.equal(migrate({nodes:[{type:'OtherNode',widgets_values:'keep'},
 {type:'MarkdownNote',widgets_values:['keep']},
 {type:'MarkdownNote',widgets_values:'named',widgets_values_named:{text:'named'}}]}), 0);
"""
    subprocess.run([node, "-e", program, str(SOURCE), str(OLD_SCALAR), str(OLD_ARRAY)],
                   check=True, capture_output=True, text=True, timeout=30)
