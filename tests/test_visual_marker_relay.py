"""Actual native temporal spans remain externally owned; tiny CPU encoders only."""
import json

from helpers import FakeAudioVAE, FakeVideoVAE
from test_prompt_relay_advanced import NativeLikeFakeClip, _plan
from test_visual_marker_binding import material
from h3_audio_t8_pkg.nodes_visual_marker import MiniMaxH3VisualMarkerRelayConditioningEXPT8
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_binding
from h3_audio_t8_pkg.temporal_dialogue_encoding import validate_native_text_recipe


def run(recipe, plan, **overrides):
    args = dict(model=object(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                prompt_relay_plan=plan, width=64, height=64, task_type="auto", audio_mode="native",
                audio_denoise_strength=1., add_source_as_reference=False, prompt_primary_audio_ordinal=1,
                strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
                execution_mode="report_only", query_chunk_rows=64)
    args.update(overrides)
    result = MiniMaxH3VisualMarkerRelayConditioningEXPT8.execute(recipe, **args)
    return result, args


def test_relay_plan_owns_actions_dialogue_only_static_actual_ordinal_legend_added():
    clean, _, recipe = material()
    plan, *_ = _plan(global_prompt="Keep the room and one actor.",
                     local_prompts="Walk once to the right.\n<d>你好，今天真不错。</d>")
    before = json.dumps(plan, sort_keys=True)
    result, args = run(recipe, plan, first_frame=clean)
    text, report, native, receipt = result[4], json.loads(result[6]), result[7], json.loads(result[8])
    assert result[0] is args["model"]
    assert text.startswith(plan["compiled_prompt"] + "\n\n") and "<Picture 2>" in receipt["static_legend"]
    assert text.count("Walk once to the right.") == 1 and text.count("<d>") == 1
    assert "walk to the destination and stop there" not in text
    assert recipe.base_prompt not in text and before == json.dumps(plan, sort_keys=True)
    assert not receipt["actions_appended_as_global"] and receipt["binding_verified"]
    assert receipt["text_owner"] == "external_compiled_plan_plus_static_legend"
    from h3_audio_t8_pkg.visual_marker_binding import bind_marker_relay_plan, insert_marker_reference
    _, selected = insert_marker_reference(recipe, [], external_plan_text=True)
    derived = bind_marker_relay_plan(plan, selected, receipt, text)
    assert derived["events"] == plan["events"] and derived["plan_hash"] != plan["plan_hash"]
    assert report["marker_source_plan_hash"] == plan["plan_hash"] and report["plan_hash"] == derived["plan_hash"]
    tokens = args["clip"].tokenize(text, **native.tokenize_kwargs)
    binding = build_prompt_relay_binding(args["clip"], derived, text, result[1], tokens)
    assert binding["text_len"] == report["text_len"] and len(binding["events"]) == 2
    for event, original in zip(binding["events"], plan["events"], strict=True):
        start, end = event["text_key_start"]-4, event["text_key_end"]-4
        assert bytes(entry[0] for entry in tokens["qwen3vl_32b"][0][start+4:end+4]).decode() == text[original["prompt_char_start"]:original["prompt_char_end"]]
    validate_native_text_recipe(native)


def test_manual_relay_marker_does_not_rewrite_external_plan_text():
    from h3_audio_t8_pkg.visual_marker import prepare_marker_prompt
    _, _, recipe = material()
    manual = prepare_marker_prompt(recipe.plan, "Unused original marker prompt.", mode="manual_original")[3]
    plan, *_ = _plan(local_prompts="A single walk, once.")
    result, _ = run(manual, plan, execution_mode="apply_exp")
    assert result[4] == plan["compiled_prompt"]
    assert json.loads(result[6])["status"] == "passthrough_single_event"
    assert not json.loads(result[8])["generated_legend_binding_verified"]


def test_derived_marker_plan_cannot_bless_noncanonical_or_arbitrary_text():
    import pytest
    from h3_audio_t8_pkg.visual_marker_binding import bind_marker_relay_plan, insert_marker_reference
    _, _, recipe = material()
    plan, *_ = _plan(local_prompts="Walk once.")
    result, _ = run(recipe, plan)
    _, request = insert_marker_reference(recipe, [], external_plan_text=True)
    with pytest.raises(ValueError, match="beyond its exact static legend"):
        bind_marker_relay_plan(plan, request, json.loads(result[-1]), result[4] + "unexpected event")


def test_relay_split_is_explicit_and_new_schema_does_not_mutate_original():
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
    _, _, recipe = material()
    plan, *_ = _plan(local_prompts="Walk once.")
    result, _ = run(recipe, plan, reference_policy="clean_vae_marked_qwen")
    receipt = json.loads(result[-1])
    assert receipt["actual_vae_rgb"] != receipt["actual_qwen_rgb"]
    assert receipt["policy"] == "clean_vae_marked_qwen"
    schema = MiniMaxH3VisualMarkerRelayConditioningEXPT8.define_schema()
    original = MiniMaxH3PromptRelayConditioningT8Advanced.define_schema()
    assert schema.is_experimental and len(schema.outputs) == 9 and len(original.outputs) == 7
    assert "prompt_recipe" in [item.id for item in schema.inputs]
    assert "prompt_recipe" not in [item.id for item in original.inputs]
