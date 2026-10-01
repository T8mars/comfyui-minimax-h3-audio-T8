"""Old CADS/NFE widget slots stay stable on native canvas import."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web/legacy_m0_widget_slots.js"
CADS = (ROOT / "examples/workflows/03-image-video-edit/"
        "2026-08-28_H3_CADS_Visual_Reference_Annealing_Advanced_EXP.json")
NFE = (ROOT / "examples/workflows/04-long-video/"
       "2026-08-23_H3_Dual_Clock_NFE_Checkpoint_Resume_Advanced_EXP.json")
FACE = (ROOT / "examples/workflows/06-face-refine/"
        "2026-09-05_H3_Face_Refine_Window_Studio_Serial_Advanced_EXP.json")
DETAIL = (ROOT / "examples/workflows/07-motion-detail/"
          "2026-08-18_H3_Hanfu_Detail_Mixer_Advanced_EXP.json")
RESTART = (ROOT / "examples/workflows/07-motion-detail/"
           "2026-08-18_H3_Hanfu_RF_Restart_Advanced_EXP.json")
MOTION = (ROOT / "examples/workflows/07-motion-detail/"
          "2026-08-22_H3_Motion_Recovery_Windowed_Stock20_Advanced_EXP.json")
TRAJECTORY = (ROOT / "examples/workflows/12-system-memory/"
              "2026-08-13_H3_Trajectory_Probe_Advanced_EXP.json")
TWO_PASS_DETAIL = (ROOT / "examples/workflows/13-latent-upscale/"
                   "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Lock_Source_Advanced_EXP.json")
FLASH = ROOT / "examples/workflows/23-flashvsr"


def test_old_cads_fixed_seed_and_connected_nfe_slot_migrate_in_memory_only():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend migration test")
    cads = json.loads(CADS.read_text(encoding="utf-8"))
    nfe = json.loads(NFE.read_text(encoding="utf-8"))
    assert next(item for item in cads["nodes"] if item["id"] == 14)["widgets_values"][-1] == 26082801
    assert next(item for item in nfe["nodes"] if item["id"] == 18)["widgets_values"] == [8]
    program = r"""
const fs = require('fs'), assert = require('assert');
let extension;
const app = {registerExtension(value) {extension = value;}};
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;\r?\n/, '').replace('export function ', 'function ');
const migrate = new Function('app', source + '; return migrateLegacyM0WidgetSlots;')(app);
const cads = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const nfe = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const face = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const detail = JSON.parse(fs.readFileSync(process.argv[5], 'utf8'));
const restart = JSON.parse(fs.readFileSync(process.argv[6], 'utf8'));
const motion = JSON.parse(fs.readFileSync(process.argv[7], 'utf8'));
const trajectory = JSON.parse(fs.readFileSync(process.argv[8], 'utf8'));
const twoPassDetail = JSON.parse(fs.readFileSync(process.argv[9], 'utf8'));
const flashFiles = fs.readdirSync(process.argv[10]).filter(name => name.endsWith('.json'));
assert.equal(flashFiles.length, 3);
for (const name of flashFiles) {
    const file = require('path').join(process.argv[10], name);
    const raw = fs.readFileSync(file, 'utf8');
    const graph = JSON.parse(raw);
    const old = graph.nodes.find(n => n.type === 'MiniMaxH3FlashVSRRestoreT8Advanced');
    const original = [...old.widgets_values];
    assert.equal(migrate(graph), 1);
    assert.deepEqual(graph.nodes.find(n => n.id === old.id).widgets_values,
        [...original.slice(0, 2), 'fixed', ...original.slice(2)]);
    assert.equal(migrate(graph), 0);
    assert.equal(fs.readFileSync(file, 'utf8'), raw);
}
for (const values of [[2, 1, 'true', 'offload_after'], [3, 1, true, 'offload_after'],
    [2, -1, true, 'offload_after'], [2, 1, true, 'unknown'],
    [2, 1, 'randomize', true, 'keep_loaded']]) {
    const invalid = {nodes: [{type:'MiniMaxH3FlashVSRRestoreT8Advanced', widgets_values:values}]};
    assert.equal(migrate(invalid), 0);
}
assert.equal(migrate({nodes:[null]}), 0);
assert.equal(migrate({nodes:[{type:'MiniMaxH3FlashVSRRestoreT8Advanced',
    widgets_values:[2, 1, true, 'offload_after'], widgets_values_named:{seed:1}}]}), 0);
const rawCads = fs.readFileSync(process.argv[2], 'utf8');
const rawNfe = fs.readFileSync(process.argv[3], 'utf8');
assert.equal(migrate(cads), 1);
assert.deepEqual(cads.nodes.find(n => n.id === 14).widgets_values,
 [0.1,0.6,0.9,1,'paper_independent',26082801,'fixed']);
assert.equal(migrate(nfe), 1);
assert.deepEqual(nfe.nodes.find(n => n.id === 18).widgets_values, ['', '', '', 8]);
assert.equal(migrate(face), 1);
assert.deepEqual(face.nodes.find(n => n.id === 27).widgets_values, [0, 'edge_hold_exp']);
assert.equal(migrate(detail), 1);
assert.deepEqual(detail.nodes.find(n => n.id === 8).widgets_values.slice(-2),
 [2608183001, 'fixed']);
assert.equal(migrate(restart), 1);
assert.deepEqual(restart.nodes.find(n => n.id === 12).widgets_values.slice(-2),
 [2608183001, 'fixed']);
assert.equal(migrate(motion), 1);
assert.deepEqual(motion.nodes.find(n => n.id === 13).widgets_values,
 [209, 0, 'fixed', 12, 'hot_ranges_only']);
assert.equal(migrate(trajectory), 1);
assert.deepEqual(trajectory.nodes.find(n => n.id === 13).widgets_values,
 [2, 4096, 123456789, 'fixed']);
assert.equal(migrate(twoPassDetail), 1);
assert.deepEqual(twoPassDetail.nodes.find(n => n.id === 16).widgets_values.slice(-2),
 [2608215001, 'fixed']);
assert.equal(migrate(cads), 0);
assert.equal(migrate(nfe), 0);
assert.equal(fs.readFileSync(process.argv[2], 'utf8'), rawCads);
assert.equal(fs.readFileSync(process.argv[3], 'utf8'), rawNfe);
const bad = {nodes:[{type:'MiniMaxH3NFERunContractT8Advanced',
 widgets_values:[8],inputs:[{name:'conditioned_prompt',link:1}]}]};
assert.equal(migrate(bad), 0);
"""
    subprocess.run([node, "-e", program, str(SOURCE), str(CADS), str(NFE),
                    str(FACE), str(DETAIL), str(RESTART), str(MOTION), str(TRAJECTORY),
                    str(TWO_PASS_DETAIL), str(FLASH)],
                   check=True, capture_output=True, text=True, timeout=30)
