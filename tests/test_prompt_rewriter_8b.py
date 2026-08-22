from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.nodes_prompt_rewriter_8b_advanced import (
    MiniMaxH3PromptRewriter8BT8Advanced,
    MiniMaxH3PromptRewriterUnloadT8Advanced,
)
from h3_audio_t8_pkg import prompt_rewriter_8b as rewriter


def test_task_aliases_and_training_message_order_are_exact():
    assert rewriter.normalize_task("T2VA — 文生音视频") == "t2av"
    assert rewriter.normalize_task("FL2VA") == "fl2av"
    messages = rewriter.build_messages("让她挥手", "FL2VA", "16:9", 5)
    content = messages[1]["content"]
    assert [item["type"] for item in content] == [
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert "Picture 1 — exact first frame" in content[0]["text"]
    assert "Picture 2 — exact final frame" in content[2]["text"]
    assert content[-1]["text"].endswith("original_prompt: 让她挥手")


@pytest.mark.parametrize(
    ("task", "first", "last", "expected"),
    [
        ("T2VA", None, None, 0),
        ("I2VA", torch.zeros((1, 8, 8, 3)), None, 1),
        ("L2VA", None, torch.zeros((1, 8, 8, 3)), 1),
        (
            "FL2VA",
            torch.zeros((1, 8, 8, 3)),
            torch.ones((1, 8, 8, 3)),
            2,
        ),
    ],
)
def test_reference_images_follow_task_geometry(task, first, last, expected):
    images = rewriter._ordered_images(task, first, last)
    assert len(images) == expected
    assert all(image.mode == "RGB" for image in images)


def test_wrong_reference_image_count_fails_closed():
    with pytest.raises(ValueError, match="exactly 1"):
        rewriter._ordered_images("I2VA", None, None)
    with pytest.raises(ValueError, match="exactly 0"):
        rewriter._ordered_images("T2VA", torch.zeros((1, 8, 8, 3)), None)


def test_structured_output_parser_preserves_all_three_sections():
    text = (
        "alignment line\n\n"
        "integrated_multimodal_description: [Shot 1] A woman waves.\n"
        "overall_soundscape: Quiet room tone.\n"
        "non_diegetic_music: N/A"
    )
    assert rewriter.parse_rewritten_prompt(text) == (
        "[Shot 1] A woman waves.",
        "Quiet room tone.",
        "N/A",
        [],
    )


def test_error_path_releases_loaded_model_even_when_cache_was_requested(monkeypatch):
    released = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return "rendered"

        def __call__(self, **kwargs):
            return {"input_ids": torch.zeros((1, 2), dtype=torch.long)}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            raise RuntimeError("synthetic generation failure")

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessor,
        Qwen3VLForConditionalGeneration=FakeModel,
    )
    fake_peft = SimpleNamespace(PeftModel=FakeModel)

    monkeypatch.setattr(rewriter, "check_runtime_dependencies", lambda: {})
    monkeypatch.setattr(rewriter, "resolve_model_source", lambda value, kind, allow: value)
    monkeypatch.setattr(rewriter, "_memory_snapshot", lambda: {})
    monkeypatch.setattr(rewriter, "_peft_torchao_compatibility", lambda: nullcontext(None))
    monkeypatch.setattr(
        rewriter,
        "_release_model",
        lambda model: released.append(model) or ["released"],
    )
    monkeypatch.setattr(
        rewriter.importlib,
        "import_module",
        lambda name: fake_transformers if name == "transformers" else fake_peft,
    )
    rewriter._MODEL_CACHE.clear()
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        rewriter.rewrite_prompt_8b(
            prompt="A woman waves",
            task="T2VA",
            resolution="16:9",
            duration=5,
            base_model_path="base",
            adapter_path="adapter",
            load_policy="cpu_only",
            dtype="bfloat16",
            decoding="greedy",
            seed=42,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            min_image_pixels=1024,
            max_image_pixels=4096,
            unload_after_generate=False,
            free_comfy_models_before_load=False,
            allow_hub_download=False,
        )
    assert len(released) == 1
    assert rewriter._MODEL_CACHE == {}


def test_node_schema_keeps_safe_unload_defaults():
    schema = MiniMaxH3PromptRewriter8BT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["unload_after_generate"].default is True
    assert inputs["free_comfy_models_before_load"].default is True
    assert inputs["allow_hub_download"].default is False
    assert inputs["task"].default == "T2VA — 文生音视频"
    assert inputs["max_new_tokens"].default == 1024
    unload_schema = MiniMaxH3PromptRewriterUnloadT8Advanced.define_schema()
    assert unload_schema.is_output_node is True


def test_old_optional_torchao_is_ignored_only_inside_peft_dispatch(monkeypatch):
    def probe():
        return "original"

    fake_dispatch = SimpleNamespace(is_torchao_available=probe)
    real_import = rewriter.importlib.import_module

    monkeypatch.setattr(
        rewriter.importlib.metadata,
        "version",
        lambda name: "0.15.0" if name == "torchao" else "1.0.0",
    )
    monkeypatch.setattr(
        rewriter.importlib,
        "import_module",
        lambda name: fake_dispatch
        if name == "peft.tuners.lora.torchao"
        else real_import(name),
    )
    with rewriter._peft_torchao_compatibility() as note:
        assert fake_dispatch.is_torchao_available() is False
        assert "torchao_0.15.0" in note
    assert fake_dispatch.is_torchao_available is probe


def test_processor_only_mm_token_type_ids_are_not_forwarded_to_generate(monkeypatch):
    generated_kwargs = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, *args, **kwargs):
            return "rendered"

        def __call__(self, **kwargs):
            return {
                "input_ids": torch.zeros((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 2), dtype=torch.long),
            }

        def decode(self, *args, **kwargs):
            return (
                "integrated_multimodal_description: [Shot 1] A woman waves.\n"
                "overall_soundscape: Quiet room tone.\n"
                "non_diegetic_music: N/A"
            )

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            generated_kwargs.update(kwargs)
            return torch.zeros((1, 3), dtype=torch.long)

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeProcessor,
        Qwen3VLForConditionalGeneration=FakeModel,
    )
    fake_peft = SimpleNamespace(PeftModel=FakeModel)
    monkeypatch.setattr(rewriter, "check_runtime_dependencies", lambda: {})
    monkeypatch.setattr(rewriter, "resolve_model_source", lambda value, kind, allow: value)
    monkeypatch.setattr(rewriter, "_memory_snapshot", lambda: {})
    monkeypatch.setattr(rewriter, "_release_model", lambda model: [])
    monkeypatch.setattr(rewriter, "_peft_torchao_compatibility", lambda: nullcontext(None))
    monkeypatch.setattr(
        rewriter.importlib,
        "import_module",
        lambda name: fake_transformers if name == "transformers" else fake_peft,
    )
    rewriter._MODEL_CACHE.clear()
    output = rewriter.rewrite_prompt_8b(
        prompt="A woman waves",
        task="T2VA",
        resolution="16:9",
        duration=5,
        base_model_path="base",
        adapter_path="adapter",
        load_policy="cpu_only",
        dtype="bfloat16",
        decoding="greedy",
        seed=42,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.8,
        min_image_pixels=1024,
        max_image_pixels=4096,
        unload_after_generate=True,
        free_comfy_models_before_load=False,
        allow_hub_download=False,
    )
    assert "mm_token_type_ids" not in generated_kwargs
    assert "attention_mask" in generated_kwargs
    assert "removed_processor_only_mm_token_type_ids" in output[4]
