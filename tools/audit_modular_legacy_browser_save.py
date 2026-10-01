"""Audit isolated native-browser saves of SHA-frozen legacy frontend graphs.

The browser must perform the open/save/reopen action separately. This tool binds
the resulting files to the original M0 corpus and allows only appended optional
widgets whose values equal the current live-schema defaults.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from tools.audit_modular_browser_roundtrip import audit_roundtrip
from tools.audit_modular_sampling_compat import SCHEMA, write_new

ROOT = Path(__file__).resolve().parents[1]
WIDGET_TYPES = frozenset({"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"})
CORE_UI_TYPES = ("LoadImage", "LoadImageMask", "LoadAudio", "LoadVideo",
                 "CreateVideo", "SaveVideo", "GetVideoComponents",
                 "VHS_VideoCombine", "PreviewAny", "PreviewImage")
TASK_LABELS_SHA256 = "c250b2bb39f40b2d107a2105e5cf61cc4d4c6755555df90b5d4e363d4c9723f9"
TIMELINE_UI_SHA256 = "e36410f57e0f36ae07c245889b99a6a16cf3b1e64f077441fcd50fb97d19193e"
SPEECH_MIGRATION_SHA256 = "046b6411958e39c9888cde8139606ed55083f3d10f0eb85d310bd862df186bd9"
MARKDOWN_MIGRATION_SHA256 = "77c849843926d9481da1576e178dc8ea66a1aa9e29d69456849b844ab01e3ce3"
VHS_MIGRATION_SHA256 = "d015a6aa993b1c541af68961eb06eb292d22eddcc10333cfb8e276e2b12b3ef9"
READABLE_AUDIO_SHA256 = "20e8099690cc9f69a517daffa7bd664535ac27ebea205598eae351a281d8b15f"
M0_WIDGET_SLOTS_SHA256 = "b20ee9b771a218e4e17cfc44598dd01ae1ac5729fe3844c0595e3305b2db6d14"
LANPAINT_INFO_SHA256 = "c6a60f29ca4f29b864063e05013ab256e8e7b6911fd600f2352ddc161a8e6740"
LANPAINT_FIELDS = ["LanPaint_NumSteps", "LanPaint_Lambda", "LanPaint_StepSize",
                   "LanPaint_PromptMode", "LanPaint_Info"]
LANPAINT_BUTTON = "More Info, Bug Report, Star on GitHub ⭐"
TAIL_SEED_COUNTS = {
    "MiniMaxH3CADSVisualReferenceT8Advanced": 6,
    "MiniMaxH3DetailMixerSamplerT8Advanced": 21,
    "MiniMaxH3RectifiedFlowRestartSamplerT8Advanced": 6,
    "MiniMaxH3TrajectoryProbeT8Advanced": 3,
    "MiniMaxH3TwoPassDetailMixerT8Advanced": 19,
}
READABLE_REPORT_TYPES = frozenset({"MiniMaxH3NodeSourceDiagnosticT8",
                                   "MiniMaxH3AudioSourceExplanationT8"})
NON_SERIALIZING_PANELS = {
    "MiniMaxH3SkinFinishPreviewAuditT8Advanced": (
        "web/skin_finish_preview.js",
        "7efcfca519fa29c40230c3a390fc668e7e169ee1473cdaff0296110c878c6b3c",
        "t8_skin_finish_preview",
    ),
    "MiniMaxH3TopazVideoEXPT8": (
        "web/topaz_parameter_guide.js",
        "7b0cb9ddbffb5674eb4a98853cca893f8bc8c72b5f9ff424a7721411a4cd61ff",
        "topaz_parameter_reference",
    ),
    "MiniMaxH3TAEH3SamplingPreviewEXPT8": (
        "web/taeh3_preview.js",
        "e1d97a2b01c4bf0e9ae184bca143d13bddcd42aff406927efcfd1d4339ef3ecc",
        "t8_taeh3_preview",
    ),
}
VHS_API_ORDER_FIELDS = ["crf", "filename_prefix", "format", "frame_rate",
                        "loop_count", "pingpong", "pix_fmt", "save_metadata",
                        "save_output", "trim_to_audio"]
NON_SERIALIZING_BUTTONS = {
    "MiniMaxH3MeridianCameraEXPT8": (
        "web/meridian_editor.js",
        "0e7622118cb00c2377795b7fa65c3937d5f0604e6bf58614e63a8416f470698f",
        ["打开大号运镜／时间编辑器", "重置规范路径，使用节点预设", "二维预设规划（不运行GPU）"],
    ),
    "MiniMaxH3LongVideoBackgroundStartT8": (
        "web/long_video_background.js",
        "3b3661eb9f4efa7df7e73478c898fc85a2bd150de4e30fd912c1e4cf285e59d0",
        ["status / 状态", "pause / 当前段后暂停", "resume / 继续", "cancel / 取消"],
    ),
    "MiniMaxH3DirectorProjectT8": (
        "web/director.js",
        "b377e9193073f99f07df29251cd866bd155db65cff115e39a14716e433c5cf78",
        ["打开曜石导演台"],
    ),
    "MiniMaxH3CreatorBackgroundStartT8Advanced": (
        "web/long_video_background.js",
        "3b3661eb9f4efa7df7e73478c898fc85a2bd150de4e30fd912c1e4cf285e59d0",
        ["status / 状态", "pause / 当前段后暂停", "resume / 继续", "cancel / 取消"],
    ),
}
TASK_LABELS = {
    "auto": "auto — 自动判断", "T2VA": "T2VA — 文生音视频",
    "I2VA": "I2VA — 图生音视频（首帧）", "FL2VA": "FL2VA — 首尾帧生音视频",
    "L2VA": "L2VA — 尾帧生音视频", "Ref2VA": "Ref2VA — 参考生音视频",
    "Hybrid": "Hybrid — 关键帧+参考混合生成",
}
KNOWN_DISCONNECTED_OUTPUT_APPENDS = {
    "MiniMaxH3ChunkedTwoPassPlanT8Advanced": [("width", "INT"), ("height", "INT")],
    "PreviewAny": [("STRING", "STRING")],
    "PreviewImage": [("images", "IMAGE")],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_json_default(expected, actual):
    # JavaScript JSON serialization writes a whole-valued FLOAT (2.0) as 2.
    # BOOLEAN must not pass as numeric 0/1, and all nonnumeric types stay exact.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def fetch_core_schemas(base_url: str) -> dict[str, dict]:
    """Read only the installed UI schemas needed for browser normalization."""
    parsed = urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or not parsed.port):
        raise ValueError("Core schema URL must be an explicit 127.0.0.1 HTTP port")
    result = {}
    for name in CORE_UI_TYPES:
        with urlopen(f"http://127.0.0.1:{parsed.port}/object_info/{name}", timeout=10) as response:
            payload = json.load(response)
        info = payload.get(name)
        if not isinstance(info, dict) or not isinstance(info.get("input"), dict):
            raise ValueError(f"Core did not return the {name} schema")
        result[name] = {"id": name, "info": info}
    return result


def _dynamic_option_keys(spec: list) -> list[str]:
    if not (isinstance(spec, list) and len(spec) > 1 and spec[0] == "COMFY_DYNAMICCOMBO_V3"):
        raise ValueError("Core dynamic widget schema changed")
    return [option["key"] for option in spec[1]["options"]]


def _core_ui_default_appends(old: dict, new: dict, info: dict) -> list:
    """Accept only known frontend-only/default fields, not arbitrary Core changes."""
    kind = new["type"]
    old_values = list(old.get("widgets_values") or [])
    new_values = list(new.get("widgets_values") or [])
    named = new.get("widgets_values_named") or {}
    fields = list(named)
    if len(fields) != len(new_values):
        raise ValueError(f"Core widget names changed: {kind}")
    if kind in ("LoadImage", "LoadImageMask"):
        image = info["input"]["required"]["image"]
        expected_fields = (["image", "upload"] if kind == "LoadImage"
                           else ["image", "channel", "upload"])
        if (not image[1].get("image_upload") or fields != expected_fields
                or len(old_values) != len(expected_fields) - 1
                or new_values[-1] != "image"):
            raise ValueError(f"{kind} upload widget is not the known frontend default")
        return ["image"]
    if kind in ("LoadAudio", "LoadVideo"):
        field, marker, default = (("audio", "audio_upload", "") if kind == "LoadAudio"
                                  else ("file", "video_upload", "image"))
        source = info["input"]["required"][field]
        if (not source[1].get(marker) or fields != [field, "upload"]
                or len(old_values) != 1 or new_values[-1] != default):
            raise ValueError(f"{kind} upload widget is not the known frontend default")
        return [default]
    if kind != "SaveVideo":
        raise ValueError(f"Unsupported Core widget normalization: {kind}")
    required = info["input"]["required"]
    optional = info["input"]["optional"]
    formats = _dynamic_option_keys(required["format"])
    codecs = _dynamic_option_keys(optional["codec"])
    if (not formats or formats[0] != "auto" or not codecs or codecs[0] != "auto"
            or not fields or fields[0] != "filename_prefix" or fields[-1] != "codec"
            or len(old_values) < 1 or fields[1:3] != ["format", "format.codec"]
            or any(field not in ("filename_prefix", "format", "format.codec",
                                 "format.codec.encoding", "codec") for field in fields)):
        raise ValueError("SaveVideo dynamic widget layout changed")
    if new_values[1] not in formats or new_values[2] not in codecs:
        raise ValueError("SaveVideo selected option is outside current Core schema")
    if "format.codec.encoding" in fields and fields != [
            "filename_prefix", "format", "format.codec", "format.codec.encoding", "codec"]:
        raise ValueError("SaveVideo encoding widget order changed")
    if "format.codec.encoding" not in fields and fields != [
            "filename_prefix", "format", "format.codec", "codec"]:
        raise ValueError("SaveVideo widget order changed")
    defaults = {"format": "auto", "format.codec": "auto", "codec": "auto"}
    if "format.codec.encoding" in fields:
        format_options = {entry["key"]: entry for entry in required["format"][1]["options"]}
        chosen_format = format_options.get(old_values[1])
        codec_spec = (chosen_format or {}).get("inputs", {}).get("required", {}).get("codec")
        codec_options = ({entry["key"]: entry for entry in codec_spec[1]["options"]}
                         if isinstance(codec_spec, list) and len(codec_spec) > 1 else {})
        chosen_codec = codec_options.get(old_values[2])
        encoding_spec = (chosen_codec or {}).get("inputs", {}).get("optional", {}).get("encoding")
        if (not isinstance(encoding_spec, list)
                or _dynamic_option_keys(encoding_spec)[0] != "auto"):
            raise ValueError("SaveVideo encoding has no known auto default")
        defaults["format.codec.encoding"] = "auto"
    if any(field not in defaults or new_values[index] != defaults[field]
           for index, field in enumerate(fields) if index >= len(old_values)):
        raise ValueError("SaveVideo appended a non-default dynamic widget")
    return new_values[len(old_values):]


def _vhs_default(spec: list):
    if len(spec) == 2 and isinstance(spec[1], list) and spec[1]:
        return spec[1][0]
    if len(spec) >= 3 and isinstance(spec[2], dict) and "default" in spec[2]:
        return spec[2]["default"]
    raise ValueError("VHS dynamic format has no auditable default")


def _normalize_vhs(old: dict, new: dict, info: dict) -> list[str]:
    old_values = old.get("widgets_values")
    actual = new.get("widgets_values")
    named = new.get("widgets_values_named")
    base = ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"]
    if (isinstance(old_values, list) and len(old_values) == 10
            and old_values[2] == "video/h264-mp4"):
        expected = dict(zip(VHS_API_ORDER_FIELDS, old_values, strict=True))
        fields = ["frame_rate", "loop_count", "filename_prefix", "format",
                  "pix_fmt", "crf", "save_metadata", "trim_to_audio",
                  "pingpong", "save_output"]
        formats = info["input"]["required"]["format"][1].get("formats", {})
        dynamic = formats.get("video/h264-mp4", [])
        if (_sha(ROOT / "web/legacy_vhs_h264_widgets.js") != VHS_MIGRATION_SHA256
                or [spec[0] for spec in dynamic] != fields[4:8]
                or not isinstance(actual, dict) or actual != named
                or list(actual) != [*fields, "videopreview"]
                or any(not _same_json_default(expected[field], actual[field])
                       for field in fields)
                or actual["videopreview"] != {"hidden": False, "paused": False, "params": {}}):
            raise ValueError("VHS API-order H.264 controls were not preserved")
        new["widgets_values"] = list(old_values)
        new["widgets_values_named"] = expected
        return ["videopreview"]
    if (isinstance(old_values, list) and len(old_values) == 10
            and old_values[3] == "video/h264-mp4"):
        fields = ["frame_rate", "loop_count", "filename_prefix", "format",
                  "pix_fmt", "crf", "save_metadata", "trim_to_audio",
                  "pingpong", "save_output"]
        formats = info["input"]["required"]["format"][1].get("formats", {})
        dynamic = formats.get("video/h264-mp4", [])
        if ([spec[0] for spec in dynamic] != fields[4:8]
                or not isinstance(actual, dict) or actual != named
                or list(actual) != [*fields, "videopreview"]
                or not all(_same_json_default(value, actual[name])
                           for name, value in zip(fields, old_values, strict=True))
                or actual["videopreview"] != {"hidden": False, "paused": False, "params": {}}):
            raise ValueError("VHS ten-position H.264 controls were not preserved")
        new["widgets_values"] = list(old_values)
        new["widgets_values_named"] = dict(zip(fields, old_values, strict=True))
        return ["videopreview"]
    if isinstance(old_values, dict):
        # Some older VHS workflows already saved a named object rather than
        # the six-value positional form. The current UI may append only its
        # inert preview state; do not normalize changed encoder controls.
        format_spec = info["input"]["required"]["format"]
        formats = format_spec[1].get("formats")
        selected = old_values.get("format")
        if (not isinstance(formats, dict) or selected not in format_spec[0]
                or selected not in formats or not isinstance(actual, dict)
                or actual != named or list(actual) != [*old_values, "videopreview"]
                or set(base) - set(old_values)):
            raise ValueError("VHS named-widget migration layout changed")
        dynamic = {spec[0] for spec in formats[selected]}
        if set(old_values) != set(base) | dynamic or any(
                not _same_json_default(value, actual[name])
                for name, value in old_values.items()):
            raise ValueError("VHS named encoder settings changed")
        if actual["videopreview"] != {"hidden": False, "paused": False, "params": {}}:
            raise ValueError("VHS preview state changed")
        new["widgets_values"] = list(old_values.values())
        new["widgets_values_named"] = dict(old_values)
        return ["videopreview"]
    if (not isinstance(old_values, list) or len(old_values) != len(base)
            or not isinstance(actual, dict) or actual != named
            or info["input_order"]["required"] != ["images", *base]):
        raise ValueError("VHS saved-widget migration layout changed")
    format_spec = info["input"]["required"]["format"]
    if (old_values[3] not in format_spec[0]
            or not isinstance(format_spec[1].get("formats"), dict)):
        raise ValueError("VHS original video format is not installed")
    options = format_spec[1]["formats"].get(old_values[3])
    if not isinstance(options, list):
        raise ValueError("VHS selected format has no current schema")
    dynamic = {spec[0]: _vhs_default(spec) for spec in options}
    preview = {"hidden": False, "paused": False, "params": {}}
    if set(actual) != set(base) | set(dynamic) | {"videopreview"}:
        raise ValueError("VHS saved unexpected or missing widget fields")
    if any(type(actual[name]) is not type(value) or actual[name] != value
           for name, value in zip(base, old_values, strict=True)):
        raise ValueError("VHS changed an original positional widget")
    if any(type(actual[name]) is not type(value) or actual[name] != value
           for name, value in dynamic.items()):
        raise ValueError("VHS added a non-default format control")
    if actual["videopreview"] != preview:
        raise ValueError("VHS changed the frontend preview state")
    new["widgets_values"] = list(old_values)
    new["widgets_values_named"] = dict(zip(base, old_values, strict=True))
    return [*dynamic, "videopreview"]


def normalize_known_ui_changes(original: dict, saved: dict, current_nodes: dict,
                               *, labels_sha256: str) -> tuple[dict, dict[str, dict]]:
    normalized = deepcopy(saved)
    before = {node["id"]: node for node in original["nodes"]}
    changes = {}
    for node in normalized["nodes"]:
        old = before.get(node["id"])
        if old is None:
            continue
        if node["type"] == "LanPaint_SamplerCustomAdvanced":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            if (isinstance(old_values, list) and len(old_values) == 5
                    and isinstance(new_values, list) and len(new_values) == 6):
                info = current_nodes[node["type"]]["info"]
                fields = []
                for group in ("required", "optional"):
                    for name in info["input_order"].get(group, []):
                        spec = info["input"][group][name]
                        options = spec[1] if len(spec) > 1 else {}
                        if ((isinstance(spec[0], list) or spec[0] in WIDGET_TYPES)
                                and not options.get("forceInput", False)):
                            fields.append(name)
                named = node.get("widgets_values_named") or {}
                if (_sha(ROOT.parent / "LanPaint/web/lanpaint_info.js") != LANPAINT_INFO_SHA256
                        or fields != LANPAINT_FIELDS or list(named) != [*fields, LANPAINT_BUTTON]
                        or new_values[-1] != "lanpaint_star_button"
                        or named[LANPAINT_BUTTON] != new_values[-1]
                        or any(not _same_json_default(value, new_values[index])
                               or not _same_json_default(value, named[fields[index]])
                               for index, value in enumerate(old_values))):
                    raise ValueError("LanPaint five controls or known info button changed")
                node["widgets_values"] = new_values[:5]
                node["widgets_values_named"] = {name: named[name] for name in fields}
                changes[str(node["id"])] = {"kind": "external_lanpaint_info_button_only",
                                             "source_sha256": LANPAINT_INFO_SHA256}
        if node["type"] == "MarkdownNote" and isinstance(old.get("widgets_values"), str):
            text = old["widgets_values"]
            if (_sha(ROOT / "web/legacy_markdown_note_widgets.js") != MARKDOWN_MIGRATION_SHA256
                    or node.get("widgets_values") != [text]
                    or node.get("widgets_values_named") != {"text": text}):
                raise ValueError("MarkdownNote scalar text changed during native import")
            changes[str(node["id"])] = {"kind": "markdown_scalar_wrapped_in_memory"}
        if node["type"] in TAIL_SEED_COUNTS:
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            count = TAIL_SEED_COUNTS[node["type"]]
            if (isinstance(old_values, list) and len(old_values) == count
                    and isinstance(new_values, list) and len(new_values) == count + 1):
                if (_sha(ROOT / "web/legacy_m0_widget_slots.js") != M0_WIDGET_SLOTS_SHA256
                        or new_values[:count] != old_values or new_values[count] != "fixed"
                        or list(named)[-1:] != ["control_after_generate"]
                        or named["control_after_generate"] != "fixed"):
                    raise ValueError(f"{node['type']} fixed seed changed during native import")
                node["widgets_values"] = new_values[:count]
                named.pop("control_after_generate")
                node["widgets_values_named"] = named
                changes[str(node["id"])] = {"kind": "legacy_fixed_seed_control_inserted"}
        if node["type"] == "MiniMaxH3MotionSegmentPlanT8Advanced":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            if (isinstance(old_values, list) and len(old_values) == 4
                    and isinstance(new_values, list) and len(new_values) == 5):
                if (_sha(ROOT / "web/legacy_m0_widget_slots.js") != M0_WIDGET_SLOTS_SHA256
                        or new_values != [*old_values[:2], "fixed", *old_values[2:]]
                        or list(named) != ["max_expanded_frames", "window_index",
                                              "control_after_generate", "handle_frames", "coverage"]
                        or list(named.values()) != new_values):
                    raise ValueError("Motion window-index seed control shifted old fields")
                node["widgets_values"] = list(old_values)
                named.pop("control_after_generate")
                node["widgets_values_named"] = named
                changes[str(node["id"])] = {"kind": "motion_fixed_window_index_inserted"}
        if node["type"] == "MiniMaxH3FlashVSRRestoreT8Advanced":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            if (isinstance(old_values, list) and len(old_values) == 4
                    and isinstance(new_values, list) and len(new_values) == 5):
                expected = [*old_values[:2], "fixed", *old_values[2:]]
                if (_sha(ROOT / "web/legacy_m0_widget_slots.js") != M0_WIDGET_SLOTS_SHA256
                        or not all(_same_json_default(a, b) for a, b in zip(
                            expected, new_values, strict=True))
                        or list(named) != ["scale", "seed", "control_after_generate",
                                              "color_fix", "release_policy"]
                        or list(named.values()) != new_values):
                    raise ValueError("FlashVSR seed control shifted old restore fields")
                node["widgets_values"] = list(old_values)
                named.pop("control_after_generate")
                node["widgets_values_named"] = named
                changes[str(node["id"])] = {"kind": "flashvsr_fixed_seed_control_inserted"}
        if node["type"] == "MiniMaxH3FaceRefineWindowExtractT8Advanced":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            if (isinstance(old_values, list) and len(old_values) == 1
                    and isinstance(new_values, list) and len(new_values) == 2):
                if (_sha(ROOT / "web/legacy_m0_widget_slots.js") != M0_WIDGET_SLOTS_SHA256
                        or not any(pin.get("name") == "window_index" and
                                       isinstance(pin.get("link"), int)
                                       for pin in old.get("inputs", []))
                        or new_values != [0, old_values[0]]
                        or named != {"window_index": 0, "pad_policy": old_values[0]}):
                    raise ValueError("Face connected window-index shifted pad policy")
                node["widgets_values"] = list(old_values)
                node["widgets_values_named"] = {"pad_policy": old_values[0]}
                changes[str(node["id"])] = {"kind": "face_connected_window_index_inserted"}
        if node["type"] == "MiniMaxH3NFERunContractT8Advanced":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            linked = ("conditioned_prompt", "media_map_json", "conditioning_report")
            if (isinstance(old_values, list) and len(old_values) == 1
                    and isinstance(new_values, list) and len(new_values) == 4):
                if (_sha(ROOT / "web/legacy_m0_widget_slots.js") != M0_WIDGET_SLOTS_SHA256
                        or not all(any(pin.get("name") == name and isinstance(pin.get("link"), int)
                                           for pin in old.get("inputs", [])) for name in linked)
                        or new_values != ["", "", "", old_values[0]]
                        or list(named) != [*linked, "hash_chunk_megabytes"]
                        or list(named.values()) != new_values):
                    raise ValueError("NFE connected widget slots changed during native import")
                node["widgets_values"] = list(old_values)
                node["widgets_values_named"] = {"hash_chunk_megabytes": old_values[0]}
                changes[str(node["id"])] = {"kind": "nfe_connected_string_slots_inserted"}
        if node["type"] == "SaveVideo":
            prior, latest = old.get("outputs", []), node.get("outputs", [])
            info = current_nodes[node["type"]]["info"]
            if ([(pin.get("name"), pin.get("type")) for pin in prior] ==
                    [("video_url", "STRING"), ("video", "VIDEO")]
                    and [(pin.get("name"), pin.get("type")) for pin in latest] ==
                    [("video", "VIDEO"), ("video", "VIDEO")]
                    and list(zip(info["output_name"], info["output"], strict=True)) ==
                    [("video", "VIDEO")]
                    and all(not pin.get("links") for pin in [*prior, *latest])):
                node["outputs"] = deepcopy(prior)
                changes[str(node["id"])] = {
                    "kind": "current_core_disconnected_savevideo_output_replacement"}
        if node["type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
            prior, latest = old.get("outputs", []), node.get("outputs", [])
            info = current_nodes[node["type"]]["info"]
            if (len(prior) == len(latest) == 2
                    and [(pin.get("name"), pin.get("type")) for pin in prior] ==
                    [("MODEL", "MODEL"), ("report_json", "STRING")]
                    and [(pin.get("name"), pin.get("type")) for pin in latest] ==
                    [("model", "MODEL"), ("report_json", "STRING")]
                    and list(zip(info["output_name"], info["output"], strict=True)) ==
                    [("model", "MODEL"), ("report_json", "STRING")]
                    and all((a.get("links") or []) == (b.get("links") or [])
                            for a, b in zip(prior, latest, strict=True))):
                node["outputs"][0]["name"] = "MODEL"
                changes[str(node["id"])] = {"kind": "lora_model_output_label_only"}
        old_string_output_name = {
            "PreviewAny": "output",
            "PrimitiveStringMultiline": "value",
        }.get(node["type"])
        if old_string_output_name is not None:
            prior, latest = old.get("outputs", []), node.get("outputs", [])
            info = current_nodes[node["type"]]["info"]
            if (len(prior) == len(latest) == 1
                    and (prior[0].get("name"), prior[0].get("type")) ==
                    (old_string_output_name, "STRING")
                    and (latest[0].get("name"), latest[0].get("type")) ==
                    ("STRING", "STRING")
                    and list(zip(info["output_name"], info["output"], strict=True)) ==
                    [("STRING", "STRING")]
                    and (prior[0].get("links") or []) == (latest[0].get("links") or [])):
                node["outputs"][0]["name"] = old_string_output_name
                changes[str(node["id"])] = {"kind": "core_string_output_label_only"}
        if node["type"] == "GetVideoComponents":
            prior, latest = old.get("outputs", []), node.get("outputs", [])
            info = current_nodes[node["type"]]["info"]
            if (len(prior) == 4 and len(latest) == 5
                    and [(item["name"], item["type"]) for item in prior] == [
                        ("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"),
                        ("bit_depth", "INT")]
                    and [(item["name"], item["type"]) for item in latest] == [
                        ("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"),
                        ("bit_depth", "COMBO"), ("color_space", "COMBO")]
                    and list(zip(info["output_name"], info["output"], strict=True)) == [
                        ("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"),
                        ("bit_depth", "COMBO"), ("color_space", "COMBO")]
                    and not prior[3].get("links") and not latest[3].get("links")
                    and not latest[4].get("links")
                    and all((a.get("links") or []) == (b.get("links") or [])
                            for a, b in zip(prior[:3], latest[:3], strict=True))):
                node["outputs"] = deepcopy(prior)
                changes[str(node["id"])] = {
                    "kind": "current_core_disconnected_video_metadata_outputs",
                    "fields": ["bit_depth", "color_space"]}
        expected_outputs = KNOWN_DISCONNECTED_OUTPUT_APPENDS.get(node["type"])
        old_outputs, saved_outputs = old.get("outputs", []), node.get("outputs", [])
        if (expected_outputs is not None and
                len(saved_outputs) > len(old_outputs)):
            info = current_nodes[node["type"]]["info"]
            if ([(item["name"], item["type"]) for item in old_outputs] !=
                    list(zip(info["output_name"][:len(old_outputs)],
                             info["output"][:len(old_outputs)], strict=True)) or
                    [(item["name"], item["type"]) for item in saved_outputs[len(old_outputs):]] !=
                    expected_outputs or
                    [(item["name"], item["type"]) for item in saved_outputs] !=
                    list(zip(info["output_name"], info["output"], strict=True)) or
                    any(item.get("links") for item in saved_outputs[len(old_outputs):])):
                raise ValueError("Browser added an unexpected or connected output")
            node["outputs"] = saved_outputs[:len(old_outputs)]
            changes[str(node["id"])] = {"kind": "current_schema_disconnected_outputs",
                                         "fields": [name for name, _ in expected_outputs]}
        if node["type"] == "VHS_VideoCombine" and isinstance(node.get("widgets_values"), dict):
            fields = _normalize_vhs(old, node, current_nodes[node["type"]]["info"])
            changes[str(node["id"])] = {"kind": "vhs_current_schema_defaults", "fields": fields}
        if node["type"] == "MiniMaxH3SpeechStudioT8":
            old_values, new_values = old.get("widgets_values"), node.get("widgets_values")
            named = node.get("widgets_values_named") or {}
            if (isinstance(old_values, list) and len(old_values) == 21
                    and isinstance(new_values, list) and len(new_values) == 22):
                if (_sha(ROOT / "web/legacy_speech_studio_widgets.js") != SPEECH_MIGRATION_SHA256
                        or list(named)[2:3] != ["control_after_generate"]
                        or new_values[2] != "fixed"
                        or new_values[:2] != old_values[:2]
                        or new_values[3:] != old_values[2:]):
                    raise ValueError("SpeechStudio old widget values shifted during native import")
                node["widgets_values"] = new_values[:2] + new_values[3:]
                named.pop("control_after_generate")
                node["widgets_values_named"] = named
                changes[str(node["id"])] = {"kind": "speech_studio_fixed_seed_control_inserted"}
        if node["type"] in ("MiniMaxH3AudioConditioningT8", "MiniMaxH3LongVideoConditioningT8"):
            old_values = old.get("widgets_values") or []
            new_values = node.get("widgets_values") or []
            named = node.get("widgets_values_named") or {}
            if "task_type" not in named or len(new_values) < len(old_values):
                continue
            index = list(named).index("task_type")
            if index >= len(old_values):
                raise ValueError("Task type widget moved beyond original values")
            canonical = old_values[index]
            display = new_values[index]
            if canonical != display:
                if (labels_sha256 != TASK_LABELS_SHA256
                        or TASK_LABELS.get(canonical) != display):
                    raise ValueError("Task type display label changed without known backend mapping")
                node["widgets_values"][index] = canonical
                node["widgets_values_named"]["task_type"] = canonical
                changes[str(node["id"])] = {"kind": "task_type_display_only",
                                             "canonical": canonical, "display": display}
    return normalized, changes


def schema_default_appends(original: dict, saved: dict, current_nodes: dict) -> dict[int, list]:
    before = {node["id"]: node for node in original["nodes"]}
    after = {node["id"]: node for node in saved["nodes"]}
    if before.keys() != after.keys():
        raise ValueError("Browser changed node IDs before widget-default review")
    appended = {}
    for node_id, old in before.items():
        new = after[node_id]
        old_values = old.get("widgets_values") or []
        new_values = new.get("widgets_values") or []
        if len(new_values) <= len(old_values):
            continue
        info = current_nodes[new["type"]]["info"]
        if new["type"] in NON_SERIALIZING_PANELS:
            source, source_sha256, panel = NON_SERIALIZING_PANELS[new["type"]]
            fields = []
            for group in ("required", "optional"):
                for name in info["input_order"].get(group, []):
                    spec = info["input"][group][name]
                    options = spec[1] if len(spec) > 1 else {}
                    if ((isinstance(spec[0], list) or spec[0] in WIDGET_TYPES)
                            and not options.get("forceInput", False)):
                        fields.append(name)
            named = new.get("widgets_values_named") or {}
            if (_sha(ROOT / source) != source_sha256 or len(old_values) != len(fields)
                    or list(named) != [*fields, panel]
                    or len(new_values) != len(old_values) + 1
                    or not all(_same_json_default(a, b) for a, b in zip(
                        old_values, new_values[:len(old_values)], strict=True))
                    or new_values[-1] != "" or named[panel] != ""):
                raise ValueError(f"Browser nonserializing panel changed: {node_id}")
            appended[node_id] = [""]
            continue
        if new["type"] in READABLE_REPORT_TYPES:
            required = info["input_order"].get("required", [])
            optional = info["input_order"].get("optional", [])
            named = new.get("widgets_values_named") or {}
            if (_sha(ROOT / "web/readable_audio.js") != READABLE_AUDIO_SHA256
                    or optional or len(old_values) != len(required)
                    or list(named) != [*required, "t8_readable_report"]
                    or len(new_values) != len(old_values) + 1
                    or new_values[:len(old_values)] != old_values
                    or new_values[-1] != named["t8_readable_report"]
                    or new_values[-1] != ""
                    or any(info["input"]["required"][field][0] not in WIDGET_TYPES
                           for field in required)):
                raise ValueError(f"Browser readable-only report changed: {node_id}")
            appended[node_id] = [""]
            continue
        if new["type"] == "MiniMaxH3StudioTimelineT8Advanced":
            named = new.get("widgets_values_named") or {}
            if (_sha(ROOT / "web/studio_timeline_v2.js") != TIMELINE_UI_SHA256
                    or len(new_values) != len(old_values) + 1
                    or list(named)[-1:] != ["t8_timeline_preview"]
                    or new_values[-1] != named["t8_timeline_preview"] or new_values[-1] != ""):
                raise ValueError("Timeline nonserializing UI preview changed")
            appended[node_id] = [""]
            continue
        if new["type"] in NON_SERIALIZING_BUTTONS:
            source, source_sha256, buttons = NON_SERIALIZING_BUTTONS[new["type"]]
            required = info["input_order"].get("required", [])
            optional = info["input_order"].get("optional", [])
            widgets = [field for field in required
                       if info["input"]["required"][field][0] in WIDGET_TYPES
                       and not info["input"]["required"][field][1].get("forceInput", False)]
            named = new.get("widgets_values_named") or {}
            if (_sha(ROOT / source) != source_sha256 or optional
                    or len(old_values) != len(widgets)
                    or list(named) != [*widgets, *buttons]
                    or len(new_values) != len(old_values) + len(buttons)
                    or new_values[len(old_values):] != [None] * len(buttons)):
                raise ValueError(f"Browser nonserializing buttons changed: {node_id}")
            appended[node_id] = [None] * len(buttons)
            continue
        if new["type"] in ("LoadImage", "LoadImageMask", "LoadAudio", "LoadVideo", "SaveVideo"):
            appended[node_id] = _core_ui_default_appends(old, new, info)
            continue
        order = info["input_order"]
        fields = []
        for group in ("required", "optional"):
            for name in order.get(group, []):
                spec = info["input"].get(group, {})[name]
                options = spec[1] if len(spec) > 1 else {}
                if (not (isinstance(spec[0], list) or spec[0] in WIDGET_TYPES)
                        or options.get("forceInput", False)):
                    continue
                fields.append(name)
                if options.get("control_after_generate") is True:
                    fields.append("control_after_generate")
        if list((new.get("widgets_values_named") or {}).keys()) != fields or len(new_values) != len(fields):
            raise ValueError(f"Browser widget names do not match current schema: {node_id}")
        added_fields = fields[len(old_values):]
        if not added_fields or any(name not in order.get("optional", []) for name in added_fields):
            raise ValueError(f"Browser added a non-optional widget: {node_id}")
        for name, value in zip(added_fields, new_values[len(old_values):], strict=True):
            spec = info["input"]["optional"][name]
            if not _same_json_default(spec[1].get("default"), value):
                raise ValueError(f"Browser changed an optional widget default: {node_id}.{name}")
        appended[node_id] = new_values[len(old_values):]
    return appended


def audit_pair(root: Path, profile: Path, baseline: dict, current: dict,
               original_relative: str, saved_name: str) -> dict:
    root, profile = root.resolve(), profile.resolve()
    original = (root / original_relative).resolve()
    expected = baseline["files"].get(original_relative)
    if (expected is None or not original.is_relative_to(root) or not original.is_file()
            or _sha(original) != expected):
        raise ValueError("Original graph is not the unchanged M0 frozen file")
    if Path(saved_name).name != saved_name or not saved_name.endswith(".json"):
        raise ValueError("Saved graph must be a bare JSON filename")
    saved = (profile / "user/default/workflows" / saved_name).resolve()
    if not saved.is_relative_to(profile) or not saved.is_file():
        raise ValueError("Native browser save is absent from the isolated profile")
    before = json.loads(original.read_bytes())
    after = json.loads(saved.read_bytes())
    if not (isinstance(before, dict) and isinstance(before.get("nodes"), list)
            and isinstance(after, dict) and isinstance(after.get("nodes"), list)):
        raise ValueError("Expected frontend workflow JSON on both sides")
    live_nodes = {node["id"]: node for node in current["nodes"]}
    labels_path = root / "web/task_type_labels.js"
    labels_sha256 = _sha(labels_path) if labels_path.is_file() else ""
    normalized, ui_changes = normalize_known_ui_changes(
        before, after, live_nodes, labels_sha256=labels_sha256)
    for node in before["nodes"]:
        if node["type"] == "VHS_VideoCombine" and isinstance(node.get("widgets_values"), dict):
            node["widgets_values"] = list(node["widgets_values"].values())
        if node["type"] == "MarkdownNote" and isinstance(node.get("widgets_values"), str):
            node["widgets_values"] = [node["widgets_values"]]
    appends = schema_default_appends(before, normalized, live_nodes)
    audit = audit_roundtrip(before, normalized, appended_widgets=appends)
    return {"original": original_relative, "original_sha256": expected,
            "saved": saved.relative_to(profile).as_posix(), "saved_sha256": _sha(saved),
            "semantic_audit": audit,
            "ui_normalizations": ui_changes,
            "task_type_labels_sha256": labels_sha256,
            "appended_optional_defaults": {str(node_id): values for node_id, values in appends.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--core-base-url", help="Explicit isolated 127.0.0.1 Core for installed UI schemas")
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("FROZEN_RELATIVE_JSON", "BROWSER_SAVED_FILENAME"))
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    output = options.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new private artifacts output")
    baseline = json.loads(options.baseline.read_text(encoding="utf8"))
    current = json.loads(options.current.read_text(encoding="utf8"))
    if baseline.get("schema") != SCHEMA or current.get("schema") != SCHEMA:
        parser.error("Expected compatible frozen/current M0 snapshots")
    core_schema_hashes = {}
    if options.core_base_url:
        try:
            core = fetch_core_schemas(options.core_base_url)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            parser.error(f"Cannot read isolated Core schemas: {error}")
        existing = {node["id"] for node in current["nodes"]}
        if any(name in existing for name in core):
            parser.error("Core schema collides with a project node ID")
        current["nodes"].extend(core.values())
        core_schema_hashes = {name: hashlib.sha256(json.dumps(node["info"], sort_keys=True).encode()).hexdigest()
                              for name, node in core.items()}
    cases = []
    for relative, saved in options.pair:
        try:
            cases.append({"status": "pass", **audit_pair(
                ROOT, options.profile, baseline, current, relative, saved)})
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            cases.append({"status": "fail", "original": relative, "saved": saved,
                          "error_type": type(error).__name__, "reason": str(error)})
    report = {"schema": "t8.modular-sampling.legacy-browser-save.v1",
              "status": "native_browser_roundtrip_pass_not_execution_qualification"
              if all(case["status"] == "pass" for case in cases) else "fail",
              "baseline_sha256": _sha(options.baseline), "current_sha256": _sha(options.current),
              "isolated_core_schema_sha256": core_schema_hashes,
              "cases": cases,
              "qualification": "Observed native browser open/save/reopen is external to this file audit. "
                               "This verifies frozen-source SHA and saved nodes/named edges/widgets only; "
                               "not queue execution, media, numerical parity, GPU or human quality."}
    write_new(output, report)
    print(json.dumps({"status": report["status"], "cases": [
        {"original": case["original"], "status": case["status"],
         "semantic_audit": case.get("semantic_audit"), "reason": case.get("reason")}
        for case in cases]}, ensure_ascii=False))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
