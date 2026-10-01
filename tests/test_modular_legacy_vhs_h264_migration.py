"""Execute the frontend import bridge against a frozen old workflow."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web/legacy_vhs_h264_widgets.js"
OLD = (ROOT / "examples/workflows/05-speech-dialogue/"
       "2026-08-09_H3_Dialogue_Timed_Background_Bed_Lock_EXP.json")
DETAIL = ROOT / "examples/workflows/07-motion-detail"
DETAIL_FILES = (
    "2026-08-18_H3_Hanfu_Model_Time_Bias_Advanced_EXP.json",
    "2026-08-18_H3_Hanfu_RF_Restart_Advanced_EXP.json",
    "2026-08-18_H3_Hanfu_STG_Advanced_EXP.json",
    "2026-08-18_H3_Hanfu_Tail_Detail_3Step_Advanced_EXP.json",
    "2026-08-18_H3_Hanfu_Temporal_Detail_Advanced_EXP.json",
)


def test_old_vhs_h264_values_are_mapped_in_memory_only_and_other_formats_unchanged():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend migration test")
    graph = json.loads(OLD.read_text(encoding="utf-8"))
    vhs = next(item for item in graph["nodes"] if item["type"] == "VHS_VideoCombine")
    assert vhs["widgets_values"] == [
        24, 0, "MiniMaxH3_T8_DialogueSafe/timed_background_bed_lock",
        "video/h264-mp4", "yuv420p", 19, True, False, False, True]
    program = r"""
const fs = require('fs'), assert = require('assert');
let extension;
const app = {registerExtension(value) {extension = value;}};
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;\r?\n/, '').replace('export function ', 'function ');
const migrate = new Function('app', source + '; return migrateLegacyVhsH264Graph;')(app);
const old = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const before = JSON.stringify(old);
assert.equal(extension.beforeConfigureGraph(old), undefined);
assert.equal(JSON.stringify(JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))), before);
const result = old.nodes.find(node => node.type === 'VHS_VideoCombine').widgets_values;
assert.deepEqual(result, {frame_rate:24, loop_count:0,
 filename_prefix:'MiniMaxH3_T8_DialogueSafe/timed_background_bed_lock',
 format:'video/h264-mp4', pix_fmt:'yuv420p', crf:19, save_metadata:true,
 trim_to_audio:false, pingpong:false, save_output:true});
assert.equal(migrate(old), 0);
const webm = {nodes:[{type:'VHS_VideoCombine', widgets_values:
 [24,0,'x','video/webm','yuv420p',20,true,false,false,true]}]};
assert.equal(migrate(webm), 0);
const malformed = {nodes:[{type:'VHS_VideoCombine', widgets_values:
 [24,0,'x','video/h264-mp4',false,19,true,false,false,true]}]};
assert.equal(migrate(malformed), 0);
const other = {nodes:[{type:'OtherNode', widgets_values:
 [24,0,'x','video/h264-mp4','yuv420p',19,true,false,false,true]}]};
assert.equal(migrate(other), 0);
"""
    subprocess.run([node, "-e", program, str(SOURCE), str(OLD)],
                   check=True, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("filename", DETAIL_FILES)
def test_old_api_order_h264_detail_graph_maps_to_named_controls(filename):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend migration test")
    graph_path = DETAIL / filename
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    vhs = next(item for item in graph["nodes"] if item["type"] == "VHS_VideoCombine")
    assert vhs["widgets_values"] == [
        19, vhs["widgets_values"][1], "video/h264-mp4", 24, 0, False,
        "yuv420p", True, True, False]
    program = r"""
const fs = require('fs'), assert = require('assert');
const app = {registerExtension() {}};
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;\r?\n/, '').replace('export function ', 'function ');
const migrate = new Function('app', source + '; return migrateLegacyVhsH264Graph;')(app);
const graph = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const raw = fs.readFileSync(process.argv[2], 'utf8');
const old = graph.nodes.find(node => node.type === 'VHS_VideoCombine');
const prefix = old.widgets_values[1];
assert.equal(migrate(graph), 1);
const migrated = graph.nodes.find(node => node.type === 'VHS_VideoCombine');
assert.deepEqual(migrated.widgets_values, {frame_rate:24, loop_count:0,
 filename_prefix:prefix, format:'video/h264-mp4', pix_fmt:'yuv420p', crf:19,
 save_metadata:true, trim_to_audio:false, pingpong:false, save_output:true});
assert.equal(migrate(graph), 0);
assert.equal(fs.readFileSync(process.argv[2], 'utf8'), raw);
const malformed = {nodes:[{type:'VHS_VideoCombine', widgets_values:
 [19,prefix,'video/h264-mp4',24,0,false,false,true,true,false]}]};
assert.equal(migrate(malformed), 0);
"""
    subprocess.run([node, "-e", program, str(SOURCE), str(graph_path)],
                   check=True, capture_output=True, text=True, timeout=30)
