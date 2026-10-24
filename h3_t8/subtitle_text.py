"""Pure SRT/WebVTT text rendering; no audio processing, models or downloads."""
from __future__ import annotations

from html import escape


def subtitle_timestamp(seconds, separator=","):
    # Keep the historical Speech rounding convention, including Fraction input.
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render_subtitles(cues, format="srt", *, literal_text=False):
    """Render caller-ordered cues. Optional literal mode does not accept markup.

    Speech retains its existing raw-text output. Director literal captions escape
    markup and put a zero-width space on visually blank lines, preventing a user
    paragraph break from becoming a new SRT/VTT record. Source text stays intact
    in the separate caption manifest.
    """
    if format not in {"srt", "vtt"}:
        raise ValueError("字幕格式必须是 srt 或 vtt")
    blocks = [] if format == "srt" else ["WEBVTT", ""]
    for index, cue in enumerate(cues, 1):
        text = cue["text"]
        if literal_text:
            text = escape(text.replace("\r\n", "\n").replace("\r", "\n"), quote=False)
            text = "\n".join(line if line else "\u200b" for line in text.split("\n"))
        separator = "," if format == "srt" else "."
        if format == "srt":
            blocks.append(str(index))
        blocks.extend([
            f"{subtitle_timestamp(cue['start_seconds'], separator)} --> "
            f"{subtitle_timestamp(cue['end_seconds'], separator)}",
            text, "",
        ])
    result = "\n".join(blocks)
    return result + "\n" if format == "vtt" and not cues else result
