from __future__ import annotations

import pytest

from h3_audio_t8_pkg.prompt_tags import canonicalize_media_tags, prepare_prompt


def test_canonicalizes_all_supported_tag_spellings():
    prompt = "Image 1 follows <Image 2>; Video #1 reacts to Audio1 and <Audio 2>."
    assert canonicalize_media_tags(prompt) == (
        "<Picture 1> follows <Picture 2>; <Video 1> reacts to <Audio 1> and <Audio 2>."
    )


def test_primary_audio_is_remapped_after_video_soundtracks():
    prompt, warnings = prepare_prompt(
        "<Audio 1> drives <Picture 1>",
        {"pictures": 1, "videos": 2, "audios": 3},
        source_audio_ordinal=3,
        prompt_primary_audio_ordinal=1,
        strict=True,
    )
    assert prompt == "<Audio 3> drives <Picture 1>"
    assert warnings == []


def test_strict_mode_rejects_explicit_disconnected_tags():
    with pytest.raises(ValueError, match="Audio 2"):
        prepare_prompt("Use <Audio 2>", {"pictures": 0, "videos": 0, "audios": 0})


def test_plain_numbered_media_prose_is_not_a_fatal_tag():
    prompt, warnings = prepare_prompt(
        "The edit compares Video 2 with Audio 3.",
        {"pictures": 0, "videos": 0, "audios": 0},
        strict=True,
    )
    assert prompt == "The edit compares Video 2 with Audio 3."
    assert len(warnings) == 2


def test_legacy_zero_based_and_single_media_ordinals_are_unambiguous():
    prompt, warnings = prepare_prompt(
        "Use <Image 0>, Picture 4, and <Audio 7>.",
        {"pictures": 1, "videos": 0, "audios": 1},
        prompt_primary_audio_ordinal=0,
        strict=True,
    )
    assert prompt == "Use <Picture 1>, <Picture 1>, and <Audio 1>."
    assert len(warnings) == 3


def test_non_strict_mode_returns_warnings():
    prompt, warnings = prepare_prompt(
        "Video 3", {"pictures": 0, "videos": 1, "audios": 0}, strict=False
    )
    assert prompt == "<Video 1>"
    assert len(warnings) == 1


def test_non_strict_explicit_disconnected_tag_is_demoted_to_text():
    prompt, warnings = prepare_prompt(
        "Use <Video 3>", {"pictures": 0, "videos": 0, "audios": 0}, strict=False
    )
    assert prompt == "Use Video 3"
    assert len(warnings) == 1
