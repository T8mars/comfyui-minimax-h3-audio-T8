"""Fail-closed audit of native browser saves of frozen legacy workflows."""

from copy import deepcopy
import hashlib
import json

import pytest

from tools.audit_modular_legacy_browser_save import (
    LANPAINT_BUTTON,
    LANPAINT_FIELDS,
    TASK_LABELS_SHA256,
    audit_pair,
    fetch_core_schemas,
    normalize_known_ui_changes,
    schema_default_appends,
)


def _fixture(tmp_path):
    original = {"nodes": [{"id": 1, "type": "Old", "mode": 0,
                           "inputs": [], "outputs": [], "widgets_values": [4]}], "links": []}
    saved = deepcopy(original)
    saved["nodes"][0]["widgets_values"].append("default_exp")
    saved["nodes"][0]["widgets_values_named"] = {"steps": 4, "opt": "default_exp"}
    current = {"nodes": [{"id": "Old", "info": {
        "input_order": {"required": ["steps"], "optional": ["opt"]},
        "input": {"required": {"steps": ["INT", {"default": 4}]},
                  "optional": {"opt": ["COMBO", {"default": "default_exp",
                                                 "options": ["default_exp", "other"]}]}}}}]}
    root = tmp_path / "project"
    profile = tmp_path / "profile"
    root.mkdir()
    (profile / "user/default/workflows").mkdir(parents=True)
    source = root / "old.json"
    target = profile / "user/default/workflows/saved.json"
    source.write_text(json.dumps(original), encoding="utf8")
    target.write_text(json.dumps(saved), encoding="utf8")
    baseline = {"files": {"old.json": hashlib.sha256(source.read_bytes()).hexdigest()}}
    return root, profile, baseline, current, original, saved


def test_only_current_schema_optional_defaults_may_be_appended(tmp_path):
    _, _, _, current, original, saved = _fixture(tmp_path)
    nodes = {"Old": current["nodes"][0]}
    assert schema_default_appends(original, saved, nodes) == {1: ["default_exp"]}

    changed = deepcopy(saved)
    changed["nodes"][0]["widgets_values"][-1] = "other"
    changed["nodes"][0]["widgets_values_named"]["opt"] = "other"
    with pytest.raises(ValueError, match="default"):
        schema_default_appends(original, changed, nodes)
    moved = deepcopy(current)
    moved["nodes"][0]["info"]["input_order"] = {"required": ["steps", "opt"], "optional": []}
    moved["nodes"][0]["info"]["input"]["required"]["opt"] = moved["nodes"][0]["info"]["input"]["optional"].pop("opt")
    with pytest.raises(ValueError, match="non-optional"):
        schema_default_appends(original, saved, {"Old": moved["nodes"][0]})


def test_classic_combo_list_schema_still_requires_exact_optional_default(tmp_path):
    _, _, _, current, original, saved = _fixture(tmp_path)
    current["nodes"][0]["info"]["input"]["optional"]["opt"][0] = ["default_exp", "other"]
    registry = {"Old": current["nodes"][0]}
    assert schema_default_appends(original, saved, registry) == {1: ["default_exp"]}
    saved["nodes"][0]["widgets_values"][-1] = "other"
    saved["nodes"][0]["widgets_values_named"]["opt"] = "other"
    with pytest.raises(ValueError, match="default"):
        schema_default_appends(original, saved, registry)


@pytest.mark.parametrize("damage", [None, "value", "bool", "button", "name", "source", "schema"])
def test_lanpaint_only_known_frontend_button_can_be_normalized(damage, monkeypatch):
    from tools import audit_modular_legacy_browser_save as module
    # Unit fixture does not require an installed third-party plugin. The real
    # browser batch separately freezes and verifies its complete source files.
    expected_source_sha = module.LANPAINT_INFO_SHA256
    monkeypatch.setattr(module, "_sha", lambda path: expected_source_sha)
    kind = "LanPaint_SamplerCustomAdvanced"
    values = [5, 5.0, .2, "Image First", "MiniMax H3 AV local repair"]
    old = {"nodes": [{"id": 14, "type": kind, "widgets_values": values}]}
    new = deepcopy(old)
    node = new["nodes"][0]
    node["widgets_values"] = [*values, "lanpaint_star_button"]
    node["widgets_values_named"] = dict(zip([*LANPAINT_FIELDS, LANPAINT_BUTTON], node["widgets_values"]))
    specs = [["INT", {}], ["FLOAT", {}], ["FLOAT", {}], [["Image First"], {}], ["STRING", {}]]
    current = {kind: {"info": {"input_order": {"required": LANPAINT_FIELDS},
                               "input": {"required": dict(zip(LANPAINT_FIELDS, specs))}}}}
    if damage in ("value", "bool"):
        node["widgets_values"][0] = True if damage == "bool" else 6
        node["widgets_values_named"][LANPAINT_FIELDS[0]] = node["widgets_values"][0]
    elif damage == "button":
        node["widgets_values"][-1] = "changed"
    elif damage == "name":
        node["widgets_values_named"]["wrong"] = node["widgets_values_named"].pop(LANPAINT_BUTTON)
    elif damage == "source":
        monkeypatch.setattr(module, "LANPAINT_INFO_SHA256", "0" * 64)
    elif damage == "schema":
        current[kind]["info"]["input_order"]["required"] = LANPAINT_FIELDS[:-1]
    if damage:
        with pytest.raises(ValueError, match="LanPaint"):
            normalize_known_ui_changes(old, new, current, labels_sha256=TASK_LABELS_SHA256)
    else:
        normalized, changes = normalize_known_ui_changes(old, new, current, labels_sha256=TASK_LABELS_SHA256)
        assert normalized["nodes"][0]["widgets_values"] == values
        assert changes["14"]["kind"] == "external_lanpaint_info_button_only"
        assert len(new["nodes"][0]["widgets_values"]) == 6


def test_forced_string_input_is_not_a_browser_widget(tmp_path):
    _, _, _, current, original, saved = _fixture(tmp_path)
    info = current["nodes"][0]["info"]
    info["input_order"]["optional"].insert(0, "connected_text")
    info["input"]["optional"]["connected_text"] = [
        "STRING", {"forceInput": True, "multiline": False}]
    nodes = {"Old": current["nodes"][0]}
    assert schema_default_appends(original, saved, nodes) == {1: ["default_exp"]}

    changed = deepcopy(saved)
    changed["nodes"][0]["widgets_values_named"]["opt"] = "other"
    changed["nodes"][0]["widgets_values"][-1] = "other"
    with pytest.raises(ValueError, match="default"):
        schema_default_appends(original, changed, nodes)


def test_browser_integer_json_for_whole_float_default_is_not_a_change(tmp_path):
    _, _, _, current, original, saved = _fixture(tmp_path)
    info = current["nodes"][0]["info"]
    info["input"]["optional"]["opt"] = ["FLOAT", {"default": 2.0}]
    saved["nodes"][0]["widgets_values"][-1] = 2
    saved["nodes"][0]["widgets_values_named"]["opt"] = 2
    assert schema_default_appends(original, saved, {"Old": current["nodes"][0]}) == {1: [2]}
    saved["nodes"][0]["widgets_values"][-1] = True
    saved["nodes"][0]["widgets_values_named"]["opt"] = True
    with pytest.raises(ValueError, match="default"):
        schema_default_appends(original, saved, {"Old": current["nodes"][0]})


def test_chunked_plan_only_normalizes_current_disconnected_width_height_outputs():
    kind = "MiniMaxH3ChunkedTwoPassPlanT8Advanced"
    old_outputs = [{"name": "plan", "type": "PLAN", "links": [3]},
                   {"name": "report_json", "type": "STRING", "links": []}]
    old = {"nodes": [{"id": 14, "type": kind, "outputs": old_outputs}]}
    new = deepcopy(old)
    new_outputs = new["nodes"][0]["outputs"]
    new_outputs.extend([{"name": "width", "type": "INT", "links": None},
                        {"name": "height", "type": "INT", "links": None}])
    current = {kind: {"info": {"output_name": ["plan", "report_json", "width", "height"],
                               "output": ["PLAN", "STRING", "INT", "INT"]}}}
    normalized, changes = normalize_known_ui_changes(
        old, new, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"] == old_outputs
    assert changes["14"]["fields"] == ["width", "height"]
    new["nodes"][0]["outputs"][2]["links"] = [99]
    with pytest.raises(ValueError, match="unexpected or connected output"):
        normalize_known_ui_changes(old, new, current, labels_sha256=TASK_LABELS_SHA256)


def test_vhs_named_old_workflow_preserves_encoder_controls_and_only_adds_preview():
    old_values = {"frame_rate": 24, "loop_count": 0, "filename_prefix": "x",
                  "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
                  "save_metadata": True, "trim_to_audio": False,
                  "pingpong": False, "save_output": True}
    old = {"nodes": [{"id": 1, "type": "VHS_VideoCombine", "widgets_values": old_values}]}
    new = deepcopy(old)
    preview = {"hidden": False, "paused": False, "params": {}}
    new["nodes"][0]["widgets_values"] = {**old_values, "videopreview": preview}
    new["nodes"][0]["widgets_values_named"] = deepcopy(new["nodes"][0]["widgets_values"])
    current = {"VHS_VideoCombine": {"info": {"input": {"required": {
        "format": [["video/h264-mp4"], {"formats": {"video/h264-mp4": [
            ["pix_fmt", ["yuv420p"]], ["crf", "INT"],
            ["save_metadata", "BOOLEAN"], ["trim_to_audio", "BOOLEAN"]]}}]}}}}}
    normalized, changes = normalize_known_ui_changes(
        old, new, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == list(old_values.values())
    assert changes["1"] == {"kind": "vhs_current_schema_defaults", "fields": ["videopreview"]}
    changed = deepcopy(new)
    changed["nodes"][0]["widgets_values"]["crf"] = 28
    changed["nodes"][0]["widgets_values_named"]["crf"] = 28
    with pytest.raises(ValueError, match="encoder settings"):
        normalize_known_ui_changes(old, changed, current, labels_sha256=TASK_LABELS_SHA256)


def test_task_label_normalization_with_independent_optional_default_append():
    old = {"nodes": [{"id": 2, "type": "MiniMaxH3AudioConditioningT8",
                      "widgets_values": ["T2VA"]}]}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = ["T2VA — 文生音视频", True]
    new["nodes"][0]["widgets_values_named"] = {
        "task_type": "T2VA — 文生音视频", "optional_flag": True}
    normalized, changes = normalize_known_ui_changes(
        old, new, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == ["T2VA", True]
    assert changes["2"]["kind"] == "task_type_display_only"


def test_vhs_ten_position_h264_requires_exact_semantics_after_native_save():
    values = [24, 0, "old/output", "video/h264-mp4", "yuv420p", 19,
              True, False, False, True]
    old = {"nodes": [{"id": 4, "type": "VHS_VideoCombine", "widgets_values": values}]}
    fields = ["frame_rate", "loop_count", "filename_prefix", "format",
              "pix_fmt", "crf", "save_metadata", "trim_to_audio", "pingpong", "save_output"]
    new = deepcopy(old)
    current = {"VHS_VideoCombine": {"info": {"input": {"required": {
        "format": [["video/h264-mp4"], {"formats": {"video/h264-mp4": [
            ["pix_fmt", ["yuv420p"]], ["crf", "INT"],
            ["save_metadata", "BOOLEAN"], ["trim_to_audio", "BOOLEAN"]]}}]}}}}}
    mapped = dict(zip(fields, values, strict=True))
    mapped["videopreview"] = {"hidden": False, "paused": False, "params": {}}
    new["nodes"][0]["widgets_values"] = mapped
    new["nodes"][0]["widgets_values_named"] = deepcopy(mapped)
    normalized, changes = normalize_known_ui_changes(
        old, new, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == values
    assert changes["4"]["fields"] == ["videopreview"]
    corrupted = deepcopy(new)
    corrupted["nodes"][0]["widgets_values"]["save_output"] = 19
    corrupted["nodes"][0]["widgets_values_named"]["save_output"] = 19
    with pytest.raises(ValueError, match="controls were not preserved"):
        normalize_known_ui_changes(old, corrupted, current, labels_sha256=TASK_LABELS_SHA256)


def test_vhs_api_order_h264_requires_exact_semantics_after_native_save():
    values = [19, "old/output", "video/h264-mp4", 24, 0, False,
              "yuv420p", True, True, False]
    old = {"nodes": [{"id": 4, "type": "VHS_VideoCombine", "widgets_values": values}]}
    fields = ["frame_rate", "loop_count", "filename_prefix", "format",
              "pix_fmt", "crf", "save_metadata", "trim_to_audio", "pingpong", "save_output"]
    mapped = {"frame_rate": 24, "loop_count": 0, "filename_prefix": "old/output",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
              "save_metadata": True, "trim_to_audio": False,
              "pingpong": False, "save_output": True}
    current = {"VHS_VideoCombine": {"info": {"input": {"required": {
        "format": [["video/h264-mp4"], {"formats": {"video/h264-mp4": [
            ["pix_fmt", ["yuv420p"]], ["crf", "INT"],
            ["save_metadata", "BOOLEAN"], ["trim_to_audio", "BOOLEAN"]]}}]}}}}}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = {**mapped,
        "videopreview": {"hidden": False, "paused": False, "params": {}}}
    new["nodes"][0]["widgets_values_named"] = deepcopy(new["nodes"][0]["widgets_values"])
    normalized, changes = normalize_known_ui_changes(
        old, new, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == values
    assert list(normalized["nodes"][0]["widgets_values_named"]) != fields
    assert changes["4"]["fields"] == ["videopreview"]
    corrupted = deepcopy(new)
    corrupted["nodes"][0]["widgets_values"]["trim_to_audio"] = True
    corrupted["nodes"][0]["widgets_values_named"]["trim_to_audio"] = True
    with pytest.raises(ValueError, match="API-order H.264 controls"):
        normalize_known_ui_changes(old, corrupted, current, labels_sha256=TASK_LABELS_SHA256)


def test_markdown_scalar_migration_requires_exact_saved_text():
    text = "## Important old note\nKeep this text."
    old = {"nodes": [{"id": 21, "type": "MarkdownNote", "widgets_values": text}]}
    new = {"nodes": [{"id": 21, "type": "MarkdownNote",
                      "widgets_values": [text], "widgets_values_named": {"text": text}}]}
    normalized, changes = normalize_known_ui_changes(
        old, new, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == [text]
    assert changes["21"] == {"kind": "markdown_scalar_wrapped_in_memory"}
    new["nodes"][0]["widgets_values"] = ["#"]
    new["nodes"][0]["widgets_values_named"] = {"text": "#"}
    with pytest.raises(ValueError, match="scalar text changed"):
        normalize_known_ui_changes(old, new, {}, labels_sha256=TASK_LABELS_SHA256)


def test_readable_only_audio_report_is_empty_nonserializing_append():
    kind = "MiniMaxH3NodeSourceDiagnosticT8"
    old = {"nodes": [{"id": 1, "type": kind, "widgets_values": ["target"]}]}
    saved = {"nodes": [{"id": 1, "type": kind, "widgets_values": ["target", ""],
                        "widgets_values_named": {"node_id": "target", "t8_readable_report": ""}}]}
    current = {kind: {"info": {"input_order": {"required": ["node_id"]},
                               "input": {"required": {"node_id": ["STRING", {"default": "target"}]}}}}}
    assert schema_default_appends(old, saved, current) == {1: [""]}
    saved["nodes"][0]["widgets_values"][-1] = "unexpected"
    saved["nodes"][0]["widgets_values_named"]["t8_readable_report"] = "unexpected"
    with pytest.raises(ValueError, match="readable-only report changed"):
        schema_default_appends(old, saved, current)


@pytest.mark.parametrize(("kind", "panel"), [
    ("MiniMaxH3SkinFinishPreviewAuditT8Advanced", "t8_skin_finish_preview"),
    ("MiniMaxH3TopazVideoEXPT8", "topaz_parameter_reference"),
    ("MiniMaxH3TAEH3SamplingPreviewEXPT8", "t8_taeh3_preview"),
])
def test_empty_panel_appends_preserve_all_real_widget_values(kind, panel, monkeypatch):
    old = {"nodes": [{"id": 1, "type": kind, "widgets_values": [0.5, False]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"].append("")
    saved["nodes"][0]["widgets_values_named"] = {"amount": 0.5, "accept": False, panel: ""}
    info = {"input_order": {"required": ["image", "amount"], "optional": ["accept"]},
            "input": {"required": {"image": ["IMAGE", {}], "amount": ["FLOAT", {}]},
                      "optional": {"accept": ["BOOLEAN", {}]}}}
    current = {kind: {"info": info}}
    assert schema_default_appends(old, saved, current) == {1: [""]}
    for field, value in (("accept", True), ("accept", 0), (panel, "unexpected")):
        changed = deepcopy(saved)
        named = changed["nodes"][0]["widgets_values_named"]
        named[field] = value
        changed["nodes"][0]["widgets_values"] = list(named.values())
        with pytest.raises(ValueError, match="nonserializing panel changed"):
            schema_default_appends(old, changed, current)
    changed = deepcopy(saved)
    changed["nodes"][0]["widgets_values_named"]["wrong_panel"] = (
        changed["nodes"][0]["widgets_values_named"].pop(panel))
    with pytest.raises(ValueError, match="nonserializing panel changed"):
        schema_default_appends(old, changed, current)
    monkeypatch.setattr("tools.audit_modular_legacy_browser_save._sha", lambda _: "bad")
    with pytest.raises(ValueError, match="nonserializing panel changed"):
        schema_default_appends(old, saved, current)


def test_flashvsr_import_bridge_cannot_hide_changed_restore_controls():
    old = {"nodes": [{"id": 6, "type": "MiniMaxH3FlashVSRRestoreT8Advanced",
                       "widgets_values": [2, 26083001, True, "offload_after"]}]}
    saved = deepcopy(old)
    fields = ["scale", "seed", "control_after_generate", "color_fix", "release_policy"]
    values = [2, 26083001, "fixed", True, "offload_after"]
    saved["nodes"][0]["widgets_values"] = values
    saved["nodes"][0]["widgets_values_named"] = dict(zip(fields, values, strict=True))
    normalized, changes = normalize_known_ui_changes(
        old, saved, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == old["nodes"][0]["widgets_values"]
    assert changes["6"]["kind"] == "flashvsr_fixed_seed_control_inserted"
    for index, wrong in ((0, 4), (1, 1), (2, "randomize"), (3, False), (3, 1),
                         (4, "keep_loaded")):
        changed = deepcopy(saved)
        changed["nodes"][0]["widgets_values"][index] = wrong
        changed["nodes"][0]["widgets_values_named"][fields[index]] = wrong
        with pytest.raises(ValueError, match="FlashVSR seed control shifted"):
            normalize_known_ui_changes(old, changed, {}, labels_sha256=TASK_LABELS_SHA256)


def test_cads_seed_and_nfe_connected_slots_are_exact_import_bridges():
    cads = {"nodes": [{"id": 14, "type": "MiniMaxH3CADSVisualReferenceT8Advanced",
                       "widgets_values": [0.1, 0.6, 0.9, 1, "paper_independent", 26082801]}]}
    latest = deepcopy(cads)
    latest["nodes"][0]["widgets_values"].append("fixed")
    latest["nodes"][0]["widgets_values_named"] = {
        "noise_scale": 0.1, "tau1": 0.6, "tau2": 0.9, "rescale_mix": 1,
        "noise_mode": "paper_independent", "seed": 26082801,
        "control_after_generate": "fixed"}
    normalized, changes = normalize_known_ui_changes(
        cads, latest, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == cads["nodes"][0]["widgets_values"]
    assert changes["14"]["kind"] == "legacy_fixed_seed_control_inserted"
    latest["nodes"][0]["widgets_values"][-1] = "randomize"
    latest["nodes"][0]["widgets_values_named"]["control_after_generate"] = "randomize"
    with pytest.raises(ValueError, match="fixed seed changed"):
        normalize_known_ui_changes(cads, latest, {}, labels_sha256=TASK_LABELS_SHA256)

    linked = [{"name": name, "link": index} for index, name in enumerate(
        ("conditioned_prompt", "media_map_json", "conditioning_report"), 1)]
    nfe = {"nodes": [{"id": 18, "type": "MiniMaxH3NFERunContractT8Advanced",
                      "inputs": linked, "widgets_values": [8]}]}
    saved = deepcopy(nfe)
    saved["nodes"][0]["widgets_values"] = ["", "", "", 8]
    saved["nodes"][0]["widgets_values_named"] = {
        "conditioned_prompt": "", "media_map_json": "", "conditioning_report": "",
        "hash_chunk_megabytes": 8}
    normalized, changes = normalize_known_ui_changes(
        nfe, saved, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == [8]
    assert changes["18"]["kind"] == "nfe_connected_string_slots_inserted"
    saved["nodes"][0]["widgets_values"][-1] = 16
    saved["nodes"][0]["widgets_values_named"]["hash_chunk_megabytes"] = 16
    with pytest.raises(ValueError, match="NFE connected widget slots changed"):
        normalize_known_ui_changes(nfe, saved, {}, labels_sha256=TASK_LABELS_SHA256)


def test_disconnected_obsolete_savevideo_outputs_do_not_hide_link_changes():
    old = {"nodes": [{"id": 13, "type": "SaveVideo", "outputs": [
        {"name": "video_url", "type": "STRING"}, {"name": "video", "type": "VIDEO"}]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["outputs"][0] = {"name": "video", "type": "VIDEO", "links": None}
    current = {"SaveVideo": {"info": {"output_name": ["video"], "output": ["VIDEO"]}}}
    normalized, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"] == old["nodes"][0]["outputs"]
    assert changes["13"]["kind"] == "current_core_disconnected_savevideo_output_replacement"
    saved["nodes"][0]["outputs"][0]["links"] = [7]
    untouched, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert untouched["nodes"][0]["outputs"][0]["links"] == [7]
    assert changes == {}


def test_current_previewimage_can_only_append_its_unconnected_schema_output():
    old = {"nodes": [{"id": 31, "type": "PreviewImage", "outputs": []}]}
    saved = deepcopy(old)
    saved["nodes"][0]["outputs"] = [{"name": "images", "type": "IMAGE", "links": None}]
    current = {"PreviewImage": {"info": {"output": ["IMAGE"], "output_name": ["images"]}}}
    normalized, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"] == []
    assert changes["31"]["kind"] == "current_schema_disconnected_outputs"
    saved["nodes"][0]["outputs"][0]["links"] = [1]
    with pytest.raises(ValueError, match="unexpected or connected output"):
        normalize_known_ui_changes(old, saved, current, labels_sha256=TASK_LABELS_SHA256)


def test_motion_window_and_face_connected_widget_migrations_are_exact():
    motion_old = {"nodes": [{"id": 13, "type": "MiniMaxH3MotionSegmentPlanT8Advanced",
                             "widgets_values": [209, 0, 12, "hot_ranges_only"]}]}
    motion_saved = deepcopy(motion_old)
    motion_saved["nodes"][0]["widgets_values"] = [209, 0, "fixed", 12, "hot_ranges_only"]
    motion_saved["nodes"][0]["widgets_values_named"] = {
        "max_expanded_frames": 209, "window_index": 0,
        "control_after_generate": "fixed", "handle_frames": 12,
        "coverage": "hot_ranges_only"}
    normalized, changes = normalize_known_ui_changes(
        motion_old, motion_saved, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == motion_old["nodes"][0]["widgets_values"]
    assert changes["13"]["kind"] == "motion_fixed_window_index_inserted"
    motion_saved["nodes"][0]["widgets_values"][2] = "randomize"
    motion_saved["nodes"][0]["widgets_values_named"]["control_after_generate"] = "randomize"
    with pytest.raises(ValueError, match="shifted old fields"):
        normalize_known_ui_changes(motion_old, motion_saved, {}, labels_sha256=TASK_LABELS_SHA256)

    face_old = {"nodes": [{"id": 27, "type": "MiniMaxH3FaceRefineWindowExtractT8Advanced",
                           "inputs": [{"name": "window_index", "link": 56}],
                           "widgets_values": ["edge_hold_exp"]}]}
    face_saved = deepcopy(face_old)
    face_saved["nodes"][0]["widgets_values"] = [0, "edge_hold_exp"]
    face_saved["nodes"][0]["widgets_values_named"] = {
        "window_index": 0, "pad_policy": "edge_hold_exp"}
    normalized, changes = normalize_known_ui_changes(
        face_old, face_saved, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == ["edge_hold_exp"]
    assert changes["27"]["kind"] == "face_connected_window_index_inserted"
    face_saved["nodes"][0]["widgets_values"][-1] = "reject"
    face_saved["nodes"][0]["widgets_values_named"]["pad_policy"] = "reject"
    with pytest.raises(ValueError, match="shifted pad policy"):
        normalize_known_ui_changes(face_old, face_saved, {}, labels_sha256=TASK_LABELS_SHA256)


def test_creator_background_buttons_are_only_nonserializing_null_appends():
    kind = "MiniMaxH3CreatorBackgroundStartT8Advanced"
    old = {"nodes": [{"id": 4, "type": kind, "widgets_values": [
        "chain", "review_only", 2, 2.0, "clear_execution_cache"]}]}
    buttons = ["status / 状态", "pause / 当前段后暂停", "resume / 继续", "cancel / 取消"]
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"] += [None] * 4
    saved["nodes"][0]["widgets_values_named"] = {
        "chain_id": "chain", "execution_mode": "review_only", "max_retries": 2,
        "retry_delay_seconds": 2.0, "release_policy": "clear_execution_cache",
        **dict.fromkeys(buttons)}
    current = {kind: {"info": {"input_order": {
        "required": ["workspace", "chain_id", "execution_mode", "max_retries",
                     "retry_delay_seconds", "release_policy"]},
        "input": {"required": {"workspace": ["H3_T8_CREATOR_WORKSPACE", {}],
                               "chain_id": ["STRING", {}],
                               "execution_mode": ["COMBO", {}],
                               "max_retries": ["INT", {}],
                               "retry_delay_seconds": ["FLOAT", {}],
                               "release_policy": ["COMBO", {}]}}}}}
    assert schema_default_appends(old, saved, current) == {4: [None] * 4}
    saved["nodes"][0]["widgets_values"][-1] = "unexpected"
    saved["nodes"][0]["widgets_values_named"][buttons[-1]] = "unexpected"
    with pytest.raises(ValueError, match="nonserializing buttons changed"):
        schema_default_appends(old, saved, current)


def test_lora_output_label_only_preserves_link_identity():
    kind = "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
    old = {"nodes": [{"id": 2, "type": kind, "outputs": [
        {"name": "MODEL", "type": "MODEL", "links": [2]},
        {"name": "report_json", "type": "STRING", "links": []}]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["outputs"][0]["name"] = "model"
    current = {kind: {"info": {"output_name": ["model", "report_json"],
                               "output": ["MODEL", "STRING"]}}}
    normalized, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"][0]["name"] == "MODEL"
    assert changes["2"]["kind"] == "lora_model_output_label_only"
    saved["nodes"][0]["outputs"][0]["links"] = [99]
    untouched, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert untouched["nodes"][0]["outputs"][0]["links"] == [99]
    assert changes == {}


@pytest.mark.parametrize("kind,old_name", [
    ("PreviewAny", "output"), ("PrimitiveStringMultiline", "value")])
def test_core_string_output_label_only_preserves_link_identity(kind, old_name):
    old = {"nodes": [{"id": 1, "type": kind,
                      "outputs": [{"name": old_name, "type": "STRING", "links": [2]}]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["outputs"][0]["name"] = "STRING"
    current = {kind: {"info": {"output_name": ["STRING"], "output": ["STRING"]}}}
    normalized, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"][0]["name"] == old_name
    assert changes["1"]["kind"] == "core_string_output_label_only"
    saved["nodes"][0]["outputs"][0]["links"] = [3]
    untouched, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert untouched["nodes"][0]["outputs"][0]["links"] == [3]
    assert changes == {}


def test_studio_timeline_nonserializing_preview_is_only_inert_append():
    old = {"nodes": [{"id": 7, "type": "MiniMaxH3StudioTimelineT8Advanced",
                      "widgets_values": ["project"]}]}
    new = deepcopy(old)
    new["nodes"][0]["widgets_values"] = ["project", ""]
    new["nodes"][0]["widgets_values_named"] = {
        "project_id": "project", "t8_timeline_preview": ""}
    current = {"MiniMaxH3StudioTimelineT8Advanced": {"info": {}}}
    assert schema_default_appends(old, new, current) == {7: [""]}
    new["nodes"][0]["widgets_values"][-1] = "unexpected"
    new["nodes"][0]["widgets_values_named"]["t8_timeline_preview"] = "unexpected"
    with pytest.raises(ValueError, match="preview changed"):
        schema_default_appends(old, new, current)


def test_current_core_video_metadata_output_changes_are_only_disconnected():
    old_outputs = [{"name": name, "type": kind, "links": ([] if name != "images" else [3])}
                   for name, kind in (("images", "IMAGE"), ("audio", "AUDIO"),
                                      ("fps", "FLOAT"), ("bit_depth", "INT"))]
    new_outputs = deepcopy(old_outputs)
    new_outputs[3]["type"] = "COMBO"
    new_outputs.append({"name": "color_space", "type": "COMBO", "links": []})
    old = {"nodes": [{"id": 2, "type": "GetVideoComponents", "outputs": old_outputs}]}
    new = {"nodes": [{"id": 2, "type": "GetVideoComponents", "outputs": new_outputs}]}
    info = {"output_name": [item["name"] for item in new_outputs],
            "output": [item["type"] for item in new_outputs]}
    normalized, changes = normalize_known_ui_changes(
        old, new, {"GetVideoComponents": {"info": info}}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["outputs"] == old_outputs
    assert changes["2"]["fields"] == ["bit_depth", "color_space"]
    connected = deepcopy(new)
    connected["nodes"][0]["outputs"][3]["links"] = [99]
    unchanged, changes = normalize_known_ui_changes(
        old, connected, {"GetVideoComponents": {"info": info}}, labels_sha256=TASK_LABELS_SHA256)
    assert unchanged["nodes"][0]["outputs"][3]["links"] == [99]
    assert changes == {}


def test_speech_studio_old_seed_control_insert_is_verified_not_hidden_shift():
    old_values = [0, 123, 10, 32, 20, "res_multistep", "simple", 12, 3,
                  "none", "off", "", "English", 0.85, True, "off", "",
                  0.86, True, -1, "keep_loaded"]
    old = {"nodes": [{"id": 9, "type": "MiniMaxH3SpeechStudioT8",
                      "widgets_values": old_values}]}
    new = deepcopy(old)
    values = old_values[:2] + ["fixed"] + old_values[2:]
    names = ["segment_index", "seed", "control_after_generate"] + [f"field_{i}" for i in range(19)]
    new["nodes"][0]["widgets_values"] = values
    new["nodes"][0]["widgets_values_named"] = dict(zip(names, values, strict=True))
    normalized, changes = normalize_known_ui_changes(
        old, new, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == old_values
    assert changes["9"]["kind"] == "speech_studio_fixed_seed_control_inserted"
    changed = deepcopy(new)
    changed["nodes"][0]["widgets_values"][3] = 999
    with pytest.raises(ValueError, match="shifted"):
        normalize_known_ui_changes(old, changed, {}, labels_sha256=TASK_LABELS_SHA256)


def test_pair_binds_frozen_source_and_browser_saved_file(tmp_path):
    root, profile, baseline, current, _, _ = _fixture(tmp_path)
    result = audit_pair(root, profile, baseline, current, "old.json", "saved.json")
    assert result["semantic_audit"]["nodes"] == 1
    assert result["appended_optional_defaults"] == {"1": ["default_exp"]}
    with pytest.raises(ValueError, match="bare JSON filename"):
        audit_pair(root, profile, baseline, current, "old.json", "../saved.json")
    (root / "old.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="unchanged M0"):
        audit_pair(root, profile, baseline, current, "old.json", "saved.json")


def test_core_load_image_frontend_upload_default_is_narrow():
    old = {"nodes": [{"id": 6, "type": "LoadImage", "widgets_values": ["10A.jpg"]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"].append("image")
    saved["nodes"][0]["widgets_values_named"] = {"image": "10A.jpg", "upload": "image"}
    core = {"LoadImage": {"info": {"input": {"required": {
        "image": [[], {"image_upload": True}]}}}}}
    assert schema_default_appends(old, saved, core) == {6: ["image"]}
    saved["nodes"][0]["widgets_values"][-1] = "url"
    with pytest.raises(ValueError, match="frontend default"):
        schema_default_appends(old, saved, core)


def test_core_load_image_mask_upload_default_preserves_channel():
    old = {"nodes": [{"id": 26, "type": "LoadImageMask", "widgets_values": [
        "subject_mask.png", "red"]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"].append("image")
    saved["nodes"][0]["widgets_values_named"] = {
        "image": "subject_mask.png", "channel": "red", "upload": "image"}
    core = {"LoadImageMask": {"info": {"input": {"required": {
        "image": [[], {"image_upload": True}],
        "channel": [["alpha", "red", "green", "blue"]]}}}}}
    assert schema_default_appends(old, saved, core) == {26: ["image"]}
    saved["nodes"][0]["widgets_values"][-1] = "url"
    with pytest.raises(ValueError, match="frontend default"):
        schema_default_appends(old, saved, core)


def test_core_save_video_dynamic_defaults_preserve_existing_choice():
    encoding = ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto"}, {"key": "nvenc"}]}]
    nested_codec = ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "auto"}, {"key": "h264", "inputs": {"optional": {"encoding": encoding}}}]}]
    format_spec = ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "auto"}, {"key": "mp4", "inputs": {"required": {"codec": nested_codec}}}]}]
    top_codec = ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "auto"}, {"key": "h264"}]}]
    info = {"input": {"required": {"format": format_spec},
                      "optional": {"codec": top_codec}}}
    core = {"SaveVideo": {"info": info}}
    old = {"nodes": [{"id": 25, "type": "SaveVideo", "widgets_values": [
        "prefix", "mp4", "h264", "auto"]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"].append("auto")
    saved["nodes"][0]["widgets_values_named"] = {
        "filename_prefix": "prefix", "format": "mp4", "format.codec": "h264",
        "format.codec.encoding": "auto", "codec": "auto"}
    assert schema_default_appends(old, saved, core) == {25: ["auto"]}
    saved["nodes"][0]["widgets_values"][-1] = "h264"
    with pytest.raises(ValueError, match="non-default"):
        schema_default_appends(old, saved, core)


def test_core_schema_source_must_be_explicit_loopback():
    for url in ("https://127.0.0.1:8223", "http://localhost:8223",
                "http://127.0.0.1:8223/other", "http://127.0.0.1"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            fetch_core_schemas(url)


@pytest.mark.parametrize(("kind", "fields", "buttons"), [
    ("MiniMaxH3LongVideoBackgroundStartT8",
     ["chain_id", "execution_mode", "max_retries", "retry_delay_seconds", "release_policy"],
     ["status / 状态", "pause / 当前段后暂停", "resume / 继续", "cancel / 取消"]),
    ("MiniMaxH3DirectorProjectT8", ["project_json", "shot_id"], ["打开曜石导演台"]),
    ("MiniMaxH3MeridianCameraEXPT8", ["frames", "preset", "strength", "camera_plan"],
     ["打开大号运镜／时间编辑器", "重置规范路径，使用节点预设", "二维预设规划（不运行GPU）"]),
])
def test_nonserializing_action_buttons_are_inert_and_exact(kind, fields, buttons):
    old_values = list(range(len(fields)))
    old = {"nodes": [{"id": 1, "type": kind, "widgets_values": old_values}]}
    saved = deepcopy(old)
    values = old_values + [None] * len(buttons)
    saved["nodes"][0]["widgets_values"] = values
    saved["nodes"][0]["widgets_values_named"] = dict(zip(fields + buttons, values, strict=True))
    info = {"input_order": {"required": fields, "optional": []},
            "input": {"required": {field: ["STRING", {"default": ""}] for field in fields}}}
    current = {kind: {"info": info}}
    assert schema_default_appends(old, saved, current) == {1: [None] * len(buttons)}
    changed = deepcopy(saved)
    changed["nodes"][0]["widgets_values"][-1] = "unexpected"
    with pytest.raises(ValueError, match="nonserializing buttons"):
        schema_default_appends(old, changed, current)
    changed = deepcopy(saved)
    changed["nodes"][0]["widgets_values_named"][buttons[-1] + "x"] = (
        changed["nodes"][0]["widgets_values_named"].pop(buttons[-1]))
    with pytest.raises(ValueError, match="nonserializing buttons"):
        schema_default_appends(old, changed, current)


def test_vhs_current_format_defaults_are_checked_before_normalization():
    base = ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"]
    values = [24, 0, "prefix", "video/h265-mp4", False, True]
    old = {"nodes": [{"id": 21, "type": "VHS_VideoCombine", "widgets_values": values}]}
    current = {"VHS_VideoCombine": {"info": {
        "input_order": {"required": ["images", *base]},
        "input": {"required": {"format": [["video/h265-mp4"], {"formats": {
            "video/h265-mp4": [["pix_fmt", ["yuv420p10le", "yuv420p"]],
                                ["crf", "INT", {"default": 22}]]}}]}}}}}
    actual = dict(zip(base, values, strict=True))
    actual.update(pix_fmt="yuv420p10le", crf=22,
                  videopreview={"hidden": False, "paused": False, "params": {}})
    saved = {"nodes": [{"id": 21, "type": "VHS_VideoCombine",
                        "widgets_values": actual, "widgets_values_named": deepcopy(actual)}]}
    normalized, changes = normalize_known_ui_changes(
        old, saved, current, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == values
    assert changes["21"]["fields"] == ["pix_fmt", "crf", "videopreview"]
    saved["nodes"][0]["widgets_values"]["crf"] = 18
    saved["nodes"][0]["widgets_values_named"]["crf"] = 18
    with pytest.raises(ValueError, match="non-default"):
        normalize_known_ui_changes(old, saved, current, labels_sha256=TASK_LABELS_SHA256)


def test_task_type_localized_label_requires_known_js_source():
    old = {"nodes": [{"id": 7, "type": "MiniMaxH3AudioConditioningT8",
                      "widgets_values": ["prompt", "FL2VA"]}]}
    saved = deepcopy(old)
    saved["nodes"][0]["widgets_values"][1] = "FL2VA — 首尾帧生音视频"
    saved["nodes"][0]["widgets_values_named"] = {
        "prompt": "prompt", "task_type": "FL2VA — 首尾帧生音视频"}
    normalized, changes = normalize_known_ui_changes(
        old, saved, {}, labels_sha256=TASK_LABELS_SHA256)
    assert normalized["nodes"][0]["widgets_values"] == ["prompt", "FL2VA"]
    assert changes["7"]["kind"] == "task_type_display_only"
    with pytest.raises(ValueError, match="display label changed"):
        normalize_known_ui_changes(old, saved, {}, labels_sha256="bad")
