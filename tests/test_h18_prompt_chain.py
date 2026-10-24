"""Formal Bridge/Relay recipe + budget/role append + native token IDs at both stages.

Qwen/VAE weights and H3 projection/forward are explicit CPU boundary doubles.
Plan, budget, native tokenization, Bridge math, binding, layout, stage attachment,
the installed Relay wrapper and routed PyTorch attention are real.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import torch
import pytest
from comfy.patcher_extension import WrappersMP
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
import h3_audio_t8_pkg
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.prompt_budget_advanced import compile_prompt_budget
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    project_prompt_relay_plan_to_long_video_window,
    build_prompt_relay_long_video_conditioning,
    repair_long_video_layout,
)
from h3_audio_t8_pkg.long_video_dual_model_stages import bind_stage_conditioning
from h3_audio_t8_pkg import semantic_bridge as sb
from helpers import FakeVideoVAE, FakeAudioVAE, plugin_widget_map
from test_prompt_relay_long_video_advanced import (
    _model_patcher,
    _allow_fixture_core_contract,
)
from test_semantic_bridge import weight_file as _bridge_fixture, config

bridge_weights = pytest.fixture(name="weight_file", scope="module")(
    _bridge_fixture.__wrapped__
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL = (
    ROOT
    / "examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json"
)


def test_budget_to_plan_preserves_global_and_event_payload_whitespace():
    raw = " \t<Picture 1>中的人保持衣着，灯光稳定。\r\n🙂 𠀀  "
    budget = compile_prompt_budget(raw, 7000, 0, 1, 0, 0, "[]", False, False, False)[0]
    assert budget == raw
    local = " \t她说 <d>[Korean] 안녕</d>。  "
    plan = relay.build_prompt_relay_plan(
        budget, local, 73, "auto_equal", "", "paper_v1", 0.1, False, False
    )[0]
    assert plan["global_prompt"] == raw
    assert plan["events"][0]["local_prompt"] == local
    assert plan["compiled_prompt"] == "Global scene: " + raw + "\nEvent 1: " + local


class NativeTokenClip:
    def __init__(self):
        self.tokenizer = MiniMaxH3Tokenizer()
        self.observed = []

    def tokenize(self, text, **kwargs):
        tokens = self.tokenizer.tokenize_with_weights(text, **kwargs)
        self.observed.append((text, tokens, kwargs))
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        entries = tokens["qwen3vl_32b"][0]
        # Core real tokenizer presentation entries, unexpanded 1-cell vision double.
        tags = torch.tensor([0 if isinstance(x[0], dict) else 1 for x in entries])
        embedded = (
            torch.arange(len(entries), dtype=torch.float32)
            .view(1, -1, 1)
            .expand(1, -1, 5120)
            .clone()
            / 100
        )
        return [[embedded, {"minimax_token_tags": tags}]]


def test_formal_dual_recipe_full_cpu_chain_and_stage_pairing(
    weight_file,
    tmp_path,
    monkeypatch,
):
    _exercise_chain(weight_file, tmp_path, monkeypatch)


def _exercise_chain(weight_file, tmp_path, monkeypatch, alphas=(0.1, 0.2)):
    _allow_fixture_core_contract(monkeypatch)
    classes = {
        c.define_schema().node_id: c
        for c in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    }
    graph = json.loads(FORMAL.read_text(encoding="utf-8"))
    by_type = {n["type"]: n for n in graph["nodes"]}
    node = by_type["MiniMaxH3PromptRelayPlanT8Advanced"]
    plan_values = plugin_widget_map(node, classes[node["type"]])
    dual_node = by_type["MiniMaxH3DualModelLongVideoEXPT8"]
    dual = plugin_widget_map(dual_node, classes[dual_node["type"]])
    assert (
        dual["prompt_relay_mode"] == "apply_exp"
        and dual["coarse_steps"] == dual["refine_steps"] == 4
    )
    assert dual["render_window_frames"] == 124 and dual["context_frames"] == 22
    bridge_nodes = [
        n for n in graph["nodes"] if n["type"] == "MiniMaxH3SemanticBridgeConfigT8"
    ]
    assert len(bridge_nodes) == 2
    assert all(
        plugin_widget_map(n, classes[n["type"]])["enabled"] for n in bridge_nodes
    )
    clip = NativeTokenClip()
    # Fixed explicit regression text, not a rewrite or edit of the accepted formal workflow.
    raw = " \t<Picture 1>同一个人，稳定剧院灯光；🙂 𠀀\r\n不要字幕  "
    assignments = json.dumps(
        [{"subject_id": "singer", "picture_ordinal": 1, "role": "visual_identity"}]
    )
    budget = compile_prompt_budget(
        raw, 7000, 0, 1, 0, 0, assignments, True, False, True, clip
    )
    assert budget[1] and budget[0].startswith(raw + "\n\nReference role bindings:")
    global_text = budget[0]
    local = " \t她说 <d>[Korean] 안녕</d>。  \n她闭嘴微笑举手，只有环境声。  "
    values = {
        k: plan_values[k]
        for k in ("math_profile", "epsilon", "allow_gaps", "allow_overlaps")
    }
    plan = relay.build_prompt_relay_plan(
        global_text, local, 209, "seconds", f"0-3\n3-{209 / 24}", **values
    )[0]
    assert plan["compiled_prompt"].count("Reference role bindings:") == 1
    assert plan["events"][0]["local_prompt"] == local.split("\n")[0]
    # Legacy preserve_all keeps both text keys with explicit temporal penalty, not hard omission.
    assert (
        "안녕" in plan["compiled_prompt"]
        and plan["events"][0]["end_frame_exclusive"] == 72
    )
    observations = []
    for segment, start, end, context_frames in ((0, 0, 124, 0), (1, 124, 192, 22)):
        projected = project_prompt_relay_plan_to_long_video_window(
            plan, segment, 124, context_frames, start / 24, end / 24
        )[0]
        for stage, width, height, alpha in (
            ("LOW", 64, 32, alphas[0]),
            ("HIGH", 128, 64, alphas[1]),
        ):
            context = (
                {"schema": 1, "empty": True}
                if segment == 0
                else {
                    "empty": False,
                    "schema": 1,
                    "video_tail": torch.zeros(1, 24, 7, height // 16, width // 16),
                    "audio_tail": torch.zeros(1, 32, 2, 37),
                    "metadata": {
                        "source_segment_index": 0,
                        "max_context_frames": 22,
                        "audio_overhang": 0.0,
                    },
                }
            )
            model = _model_patcher()
            # Fixture Core model identity is explicit; actual apply_exp owner/router is installed.
            result = bind_stage_conditioning(
                model,
                build_prompt_relay_long_video_conditioning,
                clip=clip,
                video_vae=FakeVideoVAE(),
                audio_vae=FakeAudioVAE(),
                context=context,
                prompt_relay_plan=projected,
                segment_index=segment,
                context_frames=context_frames,
                context_audio="video_only",
                width=width,
                height=height,
                length=124,
                task_type="auto",
                audio_mode="native",
                audio_denoise_strength=0.35,
                add_source_as_reference=False,
                prompt_primary_audio_ordinal=0,
                strict_prompt_tags=True,
                ref_image_size="match",
                reference_video_policy="official_2_to_15s",
                execution_mode="apply_exp",
                query_chunk_rows=64,
                ref_images={"ref_image_0": torch.zeros(1, 32, 64, 3)},
                semantic_bridge=config(weight_file, alpha=alpha, device="cpu"),
            )
            bound, condition, latent, _, conditioned, media_map, report_json = result
            report = json.loads(report_json)
            assert (
                conditioned == projected["compiled_prompt"] == plan["compiled_prompt"]
            )
            receipt = condition[0][1][sb.RECEIPT_KEY]
            assert receipt["alpha"] == alpha
            ids, _ = relay._prompt_token_ids(clip, conditioned)
            tokens = clip.observed[-1][1]
            assert [t[0] for t in tokens["qwen3vl_32b"][0][-len(ids) :]] == ids
            binding = relay.build_prompt_relay_binding(
                clip, projected, conditioned, condition, tokens
            )
            assert binding["semantic_bridge_receipts"] == [receipt["receipt_sha256"]]
            assert (
                binding["prompt_token_sha256"]
                == hashlib.sha256(relay._canonical_json(ids).encode()).hexdigest()
            )
            assert condition[0][0].shape[-1] == 5120
            installed = relay.prompt_relay_model_contract(bound)
            binding = installed["binding"]
            assert (
                report["status"] == "applied_exp"
                and report["attention_patch_installed"]
            )
            assert condition[0][1][relay.PROMPT_RELAY_BINDING_KEY] == binding
            assert (
                condition[0][1]["model_conds"][relay.PROMPT_RELAY_PAYLOAD_KEY].cond
                == installed["binding_hash"]
            )
            video, audio = latent["samples"].unbind()
            keyframes = condition[0][1].get("minimax_keyframes", [])
            refs = condition[0][1].get("minimax_refs", [])
            layout = relay.build_packed_layout(
                binding["text_len"],
                video.shape[2],
                video.shape[3],
                video.shape[4],
                audio.shape[-1],
                keyframes=keyframes,
                refs=refs,
                frame_count=124,
            )
            layout = repair_long_video_layout(layout, keyframes, refs, 124)
            options = dict(bound.model_options["transformer_options"])
            all_wrappers = tuple(
                w
                for group in bound.wrappers[WrappersMP.DIFFUSION_MODEL].values()
                for w in group
            )
            wrapper = bound.wrappers[WrappersMP.DIFFUSION_MODEL][
                relay.PROMPT_RELAY_WRAPPER_KEY
            ][0]

            class ProjectionBoundary:
                wrappers = all_wrappers

                def __call__(self, _x, _t, _context, transformer_options, **_kwargs):
                    q = torch.zeros(1, 1, layout.seq_len, 8)
                    v = (
                        torch.arange(layout.seq_len, dtype=torch.float32)
                        .view(1, 1, -1, 1)
                        .expand_as(q)
                    )
                    return transformer_options["optimized_attention_override"](
                        relay.attention_module.attention_pytorch,
                        q,
                        q,
                        v,
                        1,
                        skip_reshape=True,
                        transformer_options=transformer_options,
                    )

            for _ in range(4):
                out = wrapper(
                    ProjectionBoundary(),
                    (video, audio),
                    torch.ones(1),
                    condition[0][0],
                    options,
                    minimax_payload={"layout": layout},
                    **{relay.PROMPT_RELAY_PAYLOAD_KEY: installed["binding_hash"]},
                )
                assert bool(torch.isfinite(out).all())
                assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
            stage_counts = relay.prompt_relay_model_contract(bound)["execution_counts"]
            assert stage_counts == {
                "completed_forwards": 4,
                "routed_attention_calls": 4,
            }
            event = projected["events"][0]
            assert (
                event["global_start_frame"] == 0
                and event["global_end_frame_exclusive"] == 72
            )
            coordinates = torch.tensor(
                [event["midpoint"], event["midpoint"] + event["window"] * 3]
            )
            bias = relay.prompt_relay_penalty(coordinates, event)
            assert (
                bias[0] == 0 and bias[1] > bias[0]
            )  # Positive penalty is subtracted in the actual attention bias.
            observations.append(
                {
                    "segment": segment,
                    "stage": stage,
                    "width": width,
                    "height": height,
                    "global_utf8_hex": global_text.encode().hex(),
                    "final_utf8_hex": conditioned.encode().hex(),
                    "token_ids": ids,
                    "binding": binding,
                    "bridge_receipt": receipt,
                    "media_map": json.loads(media_map),
                    "projected_events": projected["events"],
                    "stage_report": report,
                    "installed_owner_execution_counts": stage_counts,
                }
            )
    assert (
        observations[0]["token_ids"]
        == observations[1]["token_ids"]
        == observations[2]["token_ids"]
        == observations[3]["token_ids"]
    )
    if alphas[0] != alphas[1]:
        assert (
            observations[0]["bridge_receipt"]["output_sha256"]
            != observations[1]["bridge_receipt"]["output_sha256"]
        )
    assert (
        observations[0]["binding"]["binding_hash"]
        != observations[1]["binding"]["binding_hash"]
    )
    (tmp_path / "h18-chain-trace.json").write_text(
        json.dumps(
            {
                "formal_sha256": hashlib.sha256(FORMAL.read_bytes()).hexdigest(),
                "original_global_utf8_hex": raw.encode().hex(),
                "original_local_utf8_hex": local.encode().hex(),
                "bridge_weight_sha256": sb.file_sha(weight_file),
                "bridge_alphas": list(alphas),
                "scope": "CPU boundary doubles for Qwen/VAE/H3 projections; real native tokenizer, Bridge, apply_exp bindings and installed Relay wrapper/router: four calls per LOW/HIGH per segment. Not H3 diffusion/GPU/human quality qualification.",
                "observations": observations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
