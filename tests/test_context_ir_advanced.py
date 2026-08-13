from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.context_ir_advanced import (
    CONTEXT_IR_SCHEMA,
    build_context_ir,
    context_ir_creative_brief,
    validate_context_ir,
)
from h3_audio_t8_pkg.nodes_context_ir_advanced import CONTEXT_IR_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.nodes_context_ir_advanced import (
    MiniMaxH3ContextIRPromptCompilerT8Advanced,
)


def _args(**overrides):
    values = {
        "provider_mode": "validate_local",
        "local_context_ir_json": json.dumps(
            {
                "context": {
                    "scene": "A verified rain-soaked station",
                    "camera": "slow push-in",
                    "continuity": "keep the red umbrella",
                }
            }
        ),
        "creative_brief": "A commuter waits in the station",
        "exact_dialogue": "We leave now.",
        "audio_transcript": "",
        "endpoint": "https://example.invalid/v1/chat/completions",
        "provider_model": "vision-model",
        "api_key_env": "T8_TEST_PROVIDER_KEY",
        "confirm_external_upload": False,
        "maximum_visual_frames": 8,
        "maximum_image_edge": 768,
        "jpeg_quality": 85,
        "timeout_seconds": 120.0,
        "maximum_response_bytes": 262144,
        "reference_images": None,
    }
    values.update(overrides)
    return values


def test_local_ir_is_safe_default_and_compiler_compatible():
    result, report = build_context_ir(**_args())
    assert result["schema"] == CONTEXT_IR_SCHEMA
    assert result["exact_dialogue"] == "We leave now."
    assert report["network_used"] is False
    assert report["external_media_uploaded"] is False
    assert result["provenance"]["api_key_serialized"] is False
    assert context_ir_creative_brief(result)["camera"] == "slow push-in"


@pytest.mark.parametrize(
    "payload",
    [
        {"context": {"scene": "x", "model_path": "C:/secret/model.safetensors"}},
        {"context": {"scene": "x"}, "node_id": 42},
        {"context": {"scene": "x", "exact_dialogue": "Changed words"}},
    ],
)
def test_remote_control_or_dialogue_rewrite_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_context_ir(
            payload,
            source_mode="openai_compatible_visual",
            exact_dialogue="Original words",
            provider_model="test",
            visual_frame_count=0,
            audio_transcript_supplied=False,
        )


def test_external_provider_requires_confirmation_before_key_or_network(monkeypatch):
    called = False

    def fail_call(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(
        "h3_audio_t8_pkg.context_ir_advanced._call_openai_compatible",
        fail_call,
    )
    with pytest.raises(ValueError, match="confirm_external_upload"):
        build_context_ir(
            **_args(provider_mode="openai_compatible_visual")
        )
    assert called is False


def test_external_provider_samples_frames_and_never_serializes_key(monkeypatch):
    monkeypatch.setenv("T8_TEST_PROVIDER_KEY", "secret-value-not-for-report")
    observed = {}

    def fake_call(**kwargs):
        observed.update(kwargs)
        return json.dumps(
            {
                "context": {
                    "scene": "two sampled views of the station",
                    "lighting": "cool overhead light",
                },
                "uncertainties": ["motion between sampled frames is unknown"],
            }
        )

    monkeypatch.setattr(
        "h3_audio_t8_pkg.context_ir_advanced._call_openai_compatible",
        fake_call,
    )
    images = torch.linspace(0, 1, 6 * 32 * 48 * 3).reshape(6, 32, 48, 3)
    result, report = build_context_ir(
        **_args(
            provider_mode="openai_compatible_visual",
            confirm_external_upload=True,
            maximum_visual_frames=3,
            reference_images=images,
        )
    )
    assert len(observed["images"]) == 3
    assert all(value.startswith("data:image/jpeg;base64,") for value in observed["images"])
    packed = json.dumps({"result": result, "report": report})
    assert "secret-value-not-for-report" not in packed
    assert report["sampled_visual_frame_indices"] == [0, 2, 5]
    assert report["raw_audio_uploaded"] is False
    assert result["remote_control_authority"] is False


def test_cleartext_remote_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("T8_TEST_PROVIDER_KEY", "secret")
    with pytest.raises(ValueError, match="loopback"):
        build_context_ir(
            **_args(
                provider_mode="openai_compatible_visual",
                confirm_external_upload=True,
                endpoint="http://remote.example/v1/chat/completions",
            )
        )


def test_context_ir_nodes_are_isolated_and_safe_by_default():
    schemas = [node.define_schema() for node in CONTEXT_IR_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3ContextIRProviderT8Advanced",
        "MiniMaxH3ContextIRPromptCompilerT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    provider_inputs = {item.id: item for item in schemas[0].inputs}
    assert provider_inputs["provider_mode"].default == "validate_local"
    assert provider_inputs["confirm_external_upload"].default is False


def test_context_ir_compiler_node_matches_schema_keyword_order():
    context_ir, _report = build_context_ir(**_args())
    result = MiniMaxH3ContextIRPromptCompilerT8Advanced.execute(
        context_ir=context_ir,
        prompt="A restrained close-up",
        backend="minimax_h3",
        duration_seconds=5.167,
        aspect_ratio="16:9",
        dialogue="We leave now.",
        negative_prompt="plastic skin",
        strict_exact_dialogue=True,
        cast=None,
        sound_canvas=None,
        cast_ids="",
    )
    assert "We leave now." in result[1]
