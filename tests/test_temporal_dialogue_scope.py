"""CPU mechanism checks; no model, speech recognition or quality acceptance."""
from dataclasses import replace

import pytest

from h3_audio_t8_pkg import chunked_two_pass_parity as parity
from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import temporal_dialogue_scope as scope
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import _dialogue_block_spans


GLOBAL = "蓝色房间，两人坐在桌边；白色杯子始终在左边。保留衣着、手部动作、环境声。"


def event(event_id, start, end, utterance="不敢。", speaker="woman"):
    return dict(event_id=event_id, start_seconds=start, end_seconds=end,
                utterance=utterance, speaker=speaker)


def windows(chunk=187, overlap=34, total_frames=294):
    tokens = legacy.tokens_for_frames(total_frames)
    segments, actual_frames = legacy.compute_temporal_segments(tokens, chunk, overlap)
    audio_length = round(actual_frames * 5 / 3) + 11
    bounds = [parity._audio_bounds(sf, ef, actual_frames, audio_length)
              for _, sf, _, ef in segments]
    return scope.window_descriptors(segments, bounds, actual_frames, audio_length)


def user_plan():
    return scope.build_dialogue_plan(GLOBAL, [
        event("line1", 6.2, 7.4, "我没看过。", "woman"),
        event("line2", 8.1, 9.2, "一次都没有？", "man"),
        event("line3", 10.1, 10.9, "不敢。", "woman"),
    ], 294)


@pytest.mark.parametrize(("chunk", "overlap"), [(187, 34), (187, 17), (136, 34), (136, 17)])
def test_actual_snapped_windows_omit_future_dialogue_even_in_first_window(chunk, overlap):
    plan = user_plan()
    compiled = [scope.compile_window_text(plan, window) for window in windows(chunk, overlap)]
    onsets = [item.event_id for text in compiled for item in text.dialogue if item.state == "onset"]
    assert onsets == ["line1", "line2", "line3"]
    assert "不敢。" not in compiled[0].prompt
    for text in compiled:
        assert GLOBAL in text.prompt
        assert len(_dialogue_block_spans(text.prompt)) == len(text.dialogue)
        for item in text.dialogue:
            original = next(e for e in plan.events if e.event_id == item.event_id)
            assert original.end_frame > text.window.owned_start_frame
            assert original.start_frame < text.window.end_frame
            assert original.utterance in text.prompt[item.prompt_char_start:item.prompt_char_end]
            assert item.local_start_frame + text.window.start_frame == original.start_frame
            assert item.local_end_frame + text.window.start_frame == original.end_frame
    assert plan == user_plan()


def test_crossing_sentence_keeps_full_text_and_global_interval_without_new_onset():
    descriptors = windows()
    seam = descriptors[0].end_frame / 24
    plan = scope.build_dialogue_plan(GLOBAL, [event("crossing", seam - .4, seam + .4)], 294)
    first, second = [scope.compile_window_text(plan, w) for w in descriptors]
    assert first.dialogue[0].state == "onset"
    assert second.dialogue[0].state == "continuation"
    assert "不敢。" in first.prompt and "不敢。" in second.prompt
    assert first.dialogue[0].event_id == second.dialogue[0].event_id
    assert first.dialogue[0].global_end_frame == second.dialogue[0].global_end_frame
    assert "already-started" in second.prompt


def test_sentence_starting_exactly_on_ownership_boundary_only_enters_next_window():
    descriptors = windows()
    start = descriptors[0].end_frame / 24
    plan = scope.build_dialogue_plan(GLOBAL, [event("edge", start, start + .5)], 294)
    first, second = [scope.compile_window_text(plan, w) for w in descriptors]
    assert first.dialogue == () and first.omitted_event_ids == ("edge",)
    assert second.dialogue[0].state == "onset"


def test_finished_speech_is_omitted_but_render_overlap_keeps_local_performance():
    plan = scope.build_dialogue_plan(GLOBAL, [event("line1", 6.2, 7.4, "我没看过。")], 294,
        performance_events=[{"event_id": "cup", "start_seconds": 6.2, "end_seconds": 7.5,
                             "cue": "女子拿着白杯转头，杯子不变化，神态连续。"}])
    first, second = [scope.compile_window_text(plan, w) for w in windows(overlap=34)]
    assert first.dialogue and not second.dialogue
    assert second.performance_event_ids == ("cup",)
    assert "拿着白杯转头" in second.prompt and "我没看过。" not in second.prompt
    assert GLOBAL in second.prompt and "<d>" not in second.prompt
    changed = replace(plan, performance_events=(replace(plan.performance_events[0], cue="换杯"),))
    with pytest.raises(ValueError, match="SHA changed"):
        scope.compile_window_text(changed, windows()[1])


def test_performance_does_not_enter_a_future_or_already_ended_render_window():
    plan = scope.build_dialogue_plan(GLOBAL, [], 294, performance_events=[
        {"event_id": "early", "start_seconds": 0, "end_seconds": 1, "cue": "点头"},
        {"event_id": "late", "start_seconds": 10, "end_seconds": 11, "cue": "抬起白杯"}])
    first, second = [scope.compile_window_text(plan, w) for w in windows()]
    assert first.performance_event_ids == ("early",)
    assert second.performance_event_ids == ("late",)


@pytest.mark.parametrize("bad", [
    [{"event_id": "x", "start_seconds": 0, "end_seconds": 1, "cue": "<d>不敢</d>"}],
    [{"event_id": "x", "start_seconds": True, "end_seconds": 1, "cue": "拿杯"}],
    [{"event_id": "x", "start_seconds": 0, "end_seconds": 20, "cue": "拿杯"}],
])
def test_performance_requires_explicit_valid_non_dialogue_cues(bad):
    with pytest.raises(ValueError):
        scope.build_dialogue_plan(GLOBAL, [], 294, performance_events=bad)


def test_legacy_h16_audio_padding_remains_source_tail_not_sampled_final_padding():
    segments, frames = legacy.compute_temporal_segments(87, 187, 34)
    bounds = [(round(sf * 5 / 3), min(501, round(ef * 5 / 3))) for _, sf, _, ef in segments]
    descriptors = scope.window_descriptors(segments, bounds, frames, 501, "retain_source_tail")
    assert descriptors[-1].audio_stop == 490 and descriptors[-1].audio_length == 501
    scope.compile_window_text(user_plan(), descriptors[-1])
    with pytest.raises(ValueError, match="clock/padding"):
        scope.window_descriptors(segments, bounds, frames, 501, "include_last")


@pytest.mark.parametrize("count", [0, 1])
def test_zero_one_event_still_compiles_explicit_window_scope(count):
    plan = scope.build_dialogue_plan(GLOBAL, [event("only", 10.1, 10.9)][:count], 294)
    first, second = [scope.compile_window_text(plan, w) for w in windows()]
    assert first.dialogue == () and GLOBAL in first.prompt
    assert len(second.dialogue) == count
    assert first.sha256 != second.sha256


def test_same_utterance_different_events_are_not_text_deduplicated():
    plan = scope.build_dialogue_plan(GLOBAL, [event("one", 1, 2), event("two", 10, 11)], 294)
    compiled = [scope.compile_window_text(plan, w) for w in windows()]
    assert [[e.event_id for e in c.dialogue] for c in compiled] == [["one"], ["two"]]


def test_single_window_all_dialogue_stays_and_media_labels_are_not_rewritten():
    plan = scope.build_dialogue_plan("<Image 1> 女子，<Audio 1>保持音色。", [
        event("speech", 1, 2, "今天不敢。", "<Audio 1> woman")], 175)
    text = scope.compile_window_text(plan, windows(total_frames=175)[0])
    assert "<Image 1>" in text.prompt and "<Audio 1>" in text.prompt
    assert "今天不敢。" in text.prompt and text.dialogue[0].state == "onset"


@pytest.mark.parametrize("bad", [
    [event("x", float("nan"), 2)], [event("x", 1, float("inf"))],
    [event("x", -1, 2)], [event("x", 1, 1)], [event("x", 1, 20)],
    [event("x", 1, 1.0001)], [event("x", True, 2)],
    [event("same", 1, 2), event("same", 3, 4)],
    [event("x", 1, 2, "<d>nested</d>")], [event("x", 1, 2, "<D bad>")],
])
def test_invalid_structured_dialogue_rejected_before_native_encoding(bad):
    with pytest.raises(ValueError):
        scope.build_dialogue_plan(GLOBAL, bad, 294)


def test_global_dialogue_and_unknown_fields_are_not_silently_stripped():
    with pytest.raises(ValueError, match="explicit timed"):
        scope.build_dialogue_plan("global <d>不敢</d>", [], 294)
    with pytest.raises(ValueError, match="exactly"):
        scope.build_dialogue_plan(GLOBAL, [{**event("x", 1, 2), "speeker": "woman"}], 294)


def test_plan_prompt_and_quantized_time_tampering_are_rejected():
    plan = user_plan()
    for bad in (replace(plan, global_prompt=GLOBAL + "change"),
                replace(plan, events=(replace(plan.events[0], start_frame=1), *plan.events[1:]))):
        with pytest.raises(ValueError, match="SHA changed"):
            scope.compile_window_text(bad, windows()[0])
    text = scope.compile_window_text(plan, windows()[0])
    with pytest.raises(ValueError, match="identity changed"):
        scope.validate_compiled_window(plan, replace(text, prompt=text.prompt + "change"))


def test_padded_audio_last_window_and_complete_readonly_prefix_identity():
    plan, descriptors = user_plan(), windows()
    first, second = [scope.compile_window_text(plan, w) for w in descriptors]
    assert descriptors[-1].audio_stop == descriptors[-1].audio_length
    assert descriptors[-1].audio_length > round(descriptors[-1].total_frames * 5 / 3)
    receipt = scope.bind_published_window(plan, first, "a" * 64)
    scope.verify_previous_window(plan, second.window, receipt, "a" * 64)
    final = scope.bind_published_window(plan, second, "b" * 64, receipt, "a" * 64)
    assert final.previous_receipt_sha256 == receipt.sha256
    assert final.published_audio_sha256 == "b" * 64
    with pytest.raises(ValueError, match="actual audio prefix"):
        scope.verify_previous_window(plan, second.window, receipt, "c" * 64)
    with pytest.raises(ValueError, match="legacy cache"):
        scope.verify_previous_window(plan, second.window, None, "a" * 64)


def test_resume_rejects_changed_event_policy_and_receipt_content():
    plan, descriptors = user_plan(), windows()
    first = scope.compile_window_text(plan, descriptors[0])
    receipt = scope.bind_published_window(plan, first, "a" * 64)
    changed = scope.build_dialogue_plan(GLOBAL + " changed", [event("new", 1, 2)], 294)
    with pytest.raises(ValueError, match="policy, order"):
        scope.verify_previous_window(changed, descriptors[1], receipt, "a" * 64)
    with pytest.raises(ValueError, match="SHA changed"):
        scope.verify_previous_window(plan, descriptors[1], replace(receipt, published_audio_sha256="b" * 64), "b" * 64)


def test_guessed_gapped_windows_wrong_audio_clock_and_wrong_timeline_fail_closed():
    with pytest.raises(ValueError, match="without gaps"):
        scope.window_descriptors([(0, 0, 55, 187), (60, 204, 87, 294)],
                                 [(0, 312), (340, 501)], 294, 501)
    with pytest.raises(ValueError, match="native absolute"):
        scope.window_descriptors([(0, 0, 87, 294)], [(0, 500)], 294, 501)
    window = windows()[0]
    with pytest.raises(ValueError, match="native clocks"):
        scope.compile_window_text(user_plan(), replace(window, audio_stop=window.audio_stop - 1))
    with pytest.raises(ValueError, match="source timeline"):
        scope.compile_window_text(scope.build_dialogue_plan(GLOBAL, [], 300), window)
