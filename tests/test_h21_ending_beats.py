"""The new plain-text template uses the real existing candidate pipeline."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project, compile_project
from h3_audio_t8_pkg.director_radar import apply_operation, candidate_status, render_binding


ROOT = Path(__file__).resolve().parents[1]


def recipe():
    url = (ROOT / "web/director/continuity_recipes.mjs").as_uri()
    script = (
        "import {continuityRecipe} from " + json.dumps(url) + ";"
        "console.log(JSON.stringify(continuityRecipe('ending_beats')));"
    )
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("ending", ["A用手里的钥匙开箱，箱盖保持打开", "A没有钥匙，箱盖始终关闭"])
def test_explicit_counterfactual_ending_is_pending_then_adopted_without_new_engine(tmp_path, ending):
    project = new_project()
    sid = project["current"]
    project["doc"]["shots"][0]["simplePrompt"] = "原稿保留；不是自动结算或已观察结果。"
    before = deepcopy(project)
    selected = recipe()
    selected["parameters"] = {
        "global_timeline": "全片0–5秒，本镜局部0–5秒，秒",
        "start_state": "只有A和原有箱子；钥匙有无由本次用户条件明确填写",
        "visible_ending": ending,
        "allowed_changes": "仅执行明确结局中的动作",
        "local_beats": "0–2秒展示手；2–4秒完成指定动作；4–5秒清楚保持结局",
        "literal_dialogue": "A：原文${never_execute}，只说一次",
        "preserve": "同一人物与同一箱子，不凭空产生钥匙或替换道具",
    }
    result = apply_operation(project, sid, "skill_create", selected)
    result = apply_operation(result, sid, "candidate_create", {"facts": False, "intent": True})
    shot = result["doc"]["shots"][0]
    assert candidate_status(result, shot) == "pending"
    assert ending in shot["promptCandidate"]["text"]
    assert "${never_execute}" in render_binding(shot["skillBindings"][0])
    assert compile_project(result)["shots"][0]["prompt"] == compile_project(before)["shots"][0]["prompt"]
    assert result["assets"] == before["assets"] and result["doc"]["global"] == before["doc"]["global"]
    assert "sourceEvidence" not in shot and shot["events"] == before["doc"]["shots"][0]["events"]
    adopted = apply_operation(result, sid, "candidate_accept", {
        "confirm": True, "text_sha256": shot["promptCandidate"]["text_sha256"],
    })
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    saved = store.save(adopted, 0)
    reopened = store.load(saved["id"])
    assert ending in compile_project(reopened)["shots"][0]["prompt"]
    assert reopened["doc"]["shots"][0]["simplePrompt"] == before["doc"]["shots"][0]["simplePrompt"]
    reverted = apply_operation(reopened, sid, "candidate_revert", {})
    assert compile_project(reverted)["shots"][0]["prompt"] == compile_project(before)["shots"][0]["prompt"]


def test_two_local_music_beats_remain_literal_without_retiming_or_forced_relay():
    project = new_project()
    sid = project["current"]
    before = deepcopy(project)
    selected = recipe()
    selected["parameters"] = {
        "local_beats": "本段Audio1为歌曲13–18秒；本镜局部0–2秒举剑，2–5秒收势。下一段局部0点另写，不把全片13秒当本镜13秒。",
        "literal_dialogue": "无对白，保留用户所选原生声音模式",
    }
    result = apply_operation(project, sid, "skill_create", selected)
    shot = result["doc"]["shots"][0]
    text = render_binding(shot["skillBindings"][0])
    assert "歌曲13–18秒" in text and "不强制Prompt Relay" in text
    assert shot["rev"] == before["doc"]["shots"][0]["rev"] + 1
    for key, value in before["doc"]["shots"][0].items():
        if key not in ("skillBindings", "rev"):
            assert shot[key] == value, key
    assert result["doc"]["global"] == before["doc"]["global"]
    assert "promptCandidate" not in shot and "sourceEvidence" not in shot


def test_new_template_keeps_nonempty_form_and_returns_detached_schema():
    url = (ROOT / "web/director/continuity_recipes.mjs").as_uri()
    script = (
        "import assert from 'node:assert/strict';"
        "import {continuityRecipe,fillContinuityRecipeForm} from " + json.dumps(url) + ";"
        "const fields=Object.fromEntries(['name','text','scope','parameter_schema','parameters'].map(k=>[k,{value:k.includes('parameter')?'{}':''}]));"
        "fillContinuityRecipeForm(fields,'ending_beats');const before=structuredClone(fields);"
        "assert.throws(()=>fillContinuityRecipeForm(fields,'ending_beats'),/已有草稿/);assert.deepEqual(fields,before);"
        "const copy=continuityRecipe('ending_beats');copy.parameter_schema.visible_ending.default='changed';"
        "assert.notEqual(continuityRecipe('ending_beats').parameter_schema.visible_ending.default,'changed');"
    )
    result = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
