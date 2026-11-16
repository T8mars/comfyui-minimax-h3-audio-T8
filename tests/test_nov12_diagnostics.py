from __future__ import annotations

import pytest

from h3_audio_t8_pkg import environment_audit as audit
from h3_audio_t8_pkg import prompt_provider_advanced as provider


class Direct:
    def audio_scale(self):
        return self.model_sampling.audio_scale


class Fallback:
    def audio_scale(self):
        return getattr(self.model_sampling, "audio_scale", 1.0)


class RequiredGetattr:
    def audio_scale(self):
        return getattr(self.model_sampling, "audio_scale")


class Unfamiliar:
    def audio_scale(self):
        # model_sampling.audio_scale in a comment is not an access.
        return 1.0


@pytest.mark.parametrize("method,protocol,reads,required", [
    (None, "legacy_dual_clock", False, False),
    (Direct.audio_scale, "flow_av", True, True),
    (Fallback.audio_scale, "flow_av", True, False),
    (RequiredGetattr.audio_scale, "flow_av", True, True),
    (Unfamiliar.audio_scale, "unknown", None, None),
])
def test_audio_scale_read_and_property_requirement_are_distinct(method, protocol, reads, required):
    result = audit._audio_scale_access_contract(method)
    assert result == {"core_protocol": protocol,
                      "core_reads_model_sampling_audio_scale": reads,
                      "core_requires_model_sampling_audio_scale": required}


def test_uninspectable_audio_method_is_unknown(monkeypatch):
    monkeypatch.setattr(audit, "_safe_source", lambda _: None)
    result = audit._audio_scale_access_contract(Fallback.audio_scale)
    assert result["core_protocol"] == "unknown"
    assert result["core_requires_model_sampling_audio_scale"] is None


@pytest.mark.parametrize("reason", ["length", "stop", "content_filter", "tool_calls"])
def test_empty_response_has_safe_reported_diagnostics(reason):
    payload = {"choices": [{"finish_reason": reason, "message": {
        "content": None, "refusal": "PRIVATE_REFUSAL", "reasoning_content": "PRIVATE_REASONING",
        "tool_calls": [{"function": {"arguments": "PRIVATE_ARGUMENTS"}}]}}],
        "usage": {"completion_tokens": 100, "total_tokens": 150, "prompt_tokens": 50,
                  "secret": "PRIVATE_USAGE"}}
    with pytest.raises(ValueError) as error:
        provider._response_text("openai_compatible", payload)
    text = str(error.value)
    assert f"finish_reason={reason}" in text
    assert "completion_tokens=100" in text
    assert "reasoning_content_present=true" in text
    assert "tool_calls_present=true" in text
    assert "PRIVATE" not in text
    assert "increase" not in text


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [None]},
                                    {"choices": [{"message": None}]}])
def test_missing_envelope_is_not_a_success_or_a_diagnostic_crash(payload):
    with pytest.raises(ValueError, match="unsupported response envelope.*finish_reason=unknown"):
        provider._response_text("openai_compatible", payload)


def test_unknown_reason_and_invalid_usage_never_echo_arbitrary_values():
    payload = {"choices": [{"finish_reason": "PRIVATE_REASON", "message": {"content": ""}}],
               "usage": {"completion_tokens": float("nan"), "prompt_tokens": "PRIVATE", "total_tokens": True}}
    with pytest.raises(ValueError) as error:
        provider._response_text("openai_compatible", payload)
    text = str(error.value)
    assert "finish_reason=unknown" in text
    assert "PRIVATE" not in text and "tokens=" not in text


def test_valid_text_parts_are_unchanged():
    assert provider._response_text("openai_compatible", {"choices": [{"message": {
        "content": [{"type": "text", "text": "原文"}, {"type": "text", "text": " unchanged"}]}}]}) == "原文 unchanged"
