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


def test_strict_mode_rejects_disconnected_tags():
    with pytest.raises(ValueError, match="Audio 2"):
        prepare_prompt("Use Audio 2", {"pictures": 0, "videos": 0, "audios": 1})


def test_non_strict_mode_returns_warnings():
    prompt, warnings = prepare_prompt(
        "Video 3", {"pictures": 0, "videos": 1, "audios": 0}, strict=False
    )
    assert prompt == "<Video 3>"
    assert len(warnings) == 1
