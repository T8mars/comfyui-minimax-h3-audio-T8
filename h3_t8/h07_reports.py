"""Read-only H07 reference and continuity diagnostics; no prompt, tensor or queue mutation."""
from copy import deepcopy
import hashlib
import json
import math
import re


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def document(text, expected):
    if not isinstance(text, str) or len(text.encode("utf8")) > 65536:
        raise ValueError("Use bounded explicit JSON (at most 64 KiB)")
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    value = json.loads(text, parse_constant=reject)
    if type(value) is not expected:
        raise ValueError("JSON must be " + expected.__name__)
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        raise ValueError("Use explicit short ASCII role/prop IDs, not inferred names")
    return value


def reference_summary(media_map_json, roles_json="[]", *, sigmas=None, completed_quality_stage=None):
    media = document(media_map_json, dict)
    roles = document(roles_json, list)
    entries, warnings = {}, []
    for plural, kind in (("pictures", "Picture"), ("videos", "Video"), ("audios", "Audio")):
        group = media.get(plural, {})
        if type(group) is not dict:
            raise ValueError("Actual media map needs numbered label dictionaries")
        if set(group) != {str(i) for i in range(1, len(group)+1)}:
            raise ValueError("Actual media ordinals must be contiguous and one-based")
        for ordinal in range(1, len(group)+1):
            label = group[str(ordinal)]
            if not isinstance(label, str):
                raise ValueError("Media source label must be text")
            tag = f"{kind} {ordinal}"
            entries[tag] = dict(tag=tag, source_label=label, role_bindings=[], Subject="unknown", S="unknown")
    seen = set()
    for role in roles:
        if type(role) is not dict or not isinstance(role.get("tags"), list):
            raise ValueError("Roles need explicit role_id and tags list")
        name = identifier(role.get("role_id"))
        expected_labels = role.get("expected_labels", {})
        if type(expected_labels) is not dict or any(not isinstance(k, str) or not isinstance(v, str)
                                                   for k, v in expected_labels.items()):
            raise ValueError("Expected labels must be an explicit tag-to-source-label dictionary")
        for field in ("Subject", "S"):
            if field in role and (not isinstance(role[field], str) or not 1 <= len(role[field]) <= 256):
                raise ValueError("Subject/S must be explicit bounded text, not inferred objects")
        if name in seen:
            raise ValueError("Duplicate role ID")
        seen.add(name)
        for tag in role["tags"]:
            if not isinstance(tag, str):
                raise ValueError("Reference tags must be explicit text")
            if tag not in entries:
                warnings.append(dict(role_id=name, tag=tag, issue="not_in_actual_media_map"))
                continue
            expected = expected_labels.get(tag)
            if expected is not None and expected != entries[tag]["source_label"]:
                warnings.append(dict(role_id=name, tag=tag, issue="expected_source_label_differs",
                                     expected=expected, actual=entries[tag]["source_label"]))
            entries[tag]["role_bindings"].append(dict(role_id=name,
                Subject=role.get("Subject", "unknown"), S=role.get("S", "unknown"),
                scope="explicit_user_binding_not_voice_identity_proof"))
    schedule = dict(status="unknown", actual_NFE="unknown")
    if sigmas is not None:
        import torch
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not 2 <= sigmas.numel() <= 4096 or not bool(torch.isfinite(sigmas).all()):
            raise ValueError("SIGMAS must be a bounded finite one-dimensional tensor")
        cpu = sigmas.detach().cpu().contiguous()
        schedule.update(status="actual_SIGMAS_input_not_forward_counter", entries=cpu.numel(),
            interval_count=cpu.numel()-1, dtype=str(cpu.dtype), values=cpu.tolist(),
            sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
    completion = dict(status="unknown", planned_NFE="unknown", actual_NFE="unknown")
    if completed_quality_stage is not None:
        from .freevideo_quality.runtime import validate_stage
        receipt = validate_stage(completed_quality_stage)
        completion = dict(status="validated_typed_completed_quality_stage_not_human_quality", role=receipt["role"],
            planned_NFE=receipt["plan"]["nfe"], actual_NFE=receipt["completed_nfe"],
            receipt_sha256=completed_quality_stage.receipt_sha256, request_sha256=receipt["request_sha256"],
            producer=receipt["freevideo_revision"], clock=deepcopy(receipt["clock"]))
    return dict(schema="t8.h07.reference-execution-summary.v1", actual_media_map=deepcopy(media),
        media_map_scope="supplied_native_media_map_not_independently_encoded", references=list(entries.values()),
        source_audio_ordinal=media.get("source_audio_ordinal"), warnings=warnings,
        SIGMAS=schedule, completion=completion, inputs_changed=False, automatic_accept=False)


def continuity_lint(states_json, allowed_transfers_json="[]"):
    states, transfers = document(states_json, list), document(allowed_transfers_json, list)
    for event in transfers:
        if (type(event) is not dict or type(event.get("t")) not in (int, float)
                or not math.isfinite(event["t"]) or event["t"] < 0):
            raise ValueError("Transfers need explicit nonnegative global seconds")
        identifier(event.get("id"))
        for field in ("from", "to"):
            if field not in event:
                raise ValueError("Transfers need explicit from/to holders (or null)")
            if event[field] is not None:
                identifier(event[field])
    warnings, previous, last_time, lines = [], {}, -1., []
    for state in states:
        if type(state) is not dict or type(state.get("t")) not in (int, float) or not math.isfinite(state["t"]) or state["t"] < 0 or state["t"] <= last_time:
            raise ValueError("Declare strictly increasing nonnegative GLOBAL seconds")
        if not isinstance(state.get("props"), list):
            raise ValueError("Each explicit state needs a props list")
        now, occupied, t = {}, {}, state["t"]
        for prop in state["props"]:
            if type(prop) is not dict:
                raise ValueError("Prop declarations must be objects")
            name, kind = identifier(prop.get("id")), identifier(prop.get("kind"))
            if name in now:
                warnings.append(dict(t=t, id=name, issue="duplicate_prop_ID_in_one_state"))
            holder, hand = prop.get("holder"), prop.get("hand")
            if holder is not None:
                identifier(holder)
            if hand not in (None, "left", "right", "both", "unknown"):
                raise ValueError("Hand is left/right/both/unknown/null; no guessed left-right conversion")
            if holder is None or hand in (None, "unknown"):
                warnings.append(dict(t=t, id=name, issue="holder_or_hand_unknown"))
            for side in (["left", "right"] if hand == "both" else [hand] if hand in ("left", "right") else []):
                if holder is not None:
                    key = holder, side
                    if key in occupied and occupied[key] != name:
                        warnings.append(dict(t=t, id=name, other=occupied[key], issue="same_hand_multiple_props",
                                             role_id=holder, hand=side))
                    occupied[key] = name
            old = previous.get(name)
            if old and old["kind"] != kind:
                warnings.append(dict(t=t, id=name, issue="kind_changed_for_stable_ID", before=old["kind"], after=kind))
            if old and old.get("holder") != holder:
                permitted = any(type(event) is dict and event.get("id") == name and event.get("t") == t
                                and event.get("from") == old.get("holder") and event.get("to") == holder for event in transfers)
                if not permitted:
                    warnings.append(dict(t=t, id=name, issue="holder_transfer_not_explicitly_allowed",
                                         before=old.get("holder"), after=holder))
            now[name] = deepcopy(prop)
            lines.append(f"{t:g}s：{name}（kind={kind}），holder={holder or 'unknown'}，hand={hand or 'unknown'}。")
        previous.update(now)
        last_time = t
    return dict(schema="t8.h07.prop-continuity-lint.v1", declarations=deepcopy(states),
        allowed_transfers=deepcopy(transfers), warnings=warnings, conflict_count=len(warnings),
        template="[手工声明；不自动成为已确认末态，不保证生成几何锁定]\n" + "\n".join(lines),
        scope="read_only_declared_states_not_visual_detection", inputs_changed=False, automatic_accept=False)
