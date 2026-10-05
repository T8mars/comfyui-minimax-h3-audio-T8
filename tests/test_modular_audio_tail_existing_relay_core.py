"""Existing Relay tails + EAV: full graph versus cold process and decoded media.

Real Core/Relay/EAV/sampling/storage/gates; seeded tiny random H3, explicit CLIP,
content-dependent VAE and resource doubles. Not pretrained/quality evidence.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import av
import comfy.model_management
import comfy_extras.nodes_custom_sampler as custom
from comfy_extras.nodes_video import CreateVideo, SaveVideo
from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly
from comfy_api.latest import io
import folder_paths
import pytest
import torch
import torch.nn.functional as functional

import h3_audio_t8_pkg
from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from helpers import FakeVideoVAE, FakeAudioVAE
from test_modular_audio_refine_full_resume_core import (
    RuntimeAudit, RuntimeSetup, TinyCandidateCLIP, TinyRefineUNET,
    ObservedRefineBypass, make_tiny_refine_lora,
)
from test_modular_audio_refine_cold_process import CaptureFrozenFirstPass
from test_modular_audio_refine_resume_all import CaptureLongDelivery, TARGET
from test_modular_audio_tail_effect_saved_core import CaptureTailEffects, SAVED, _one
from tools.build_formal_audio_tail_effect_workflows import generated
from tools import build_modular_audio_tail_effect_workflows as builder
from tools.build_modular_audio_refine_resume_all import CHECKPOINT_LOAD, CHECKPOINT_SAVE

ROOT = Path(__file__).resolve().parents[1]


class ContentVideoVAE(FakeVideoVAE):
    def decode(self, latent):
        frames = 1 if latent.shape[2] == 1 else ((latent.shape[2] - 2) // 5) * 17 + 5
        value = functional.interpolate(latent.float().mean(1, keepdim=True),
            size=(frames, latent.shape[3] * 16, latent.shape[4] * 16), mode="trilinear", align_corners=False)
        return torch.sigmoid(value).permute(0, 2, 3, 4, 1).expand(-1, -1, -1, -1, 3).contiguous()


class ContentAudioVAE(FakeAudioVAE):
    def decode(self, latent):
        value = functional.interpolate(latent.float().mean(1), size=latent.shape[-1] * 800,
                                       mode="linear", align_corners=False)
        return (.2 * torch.tanh(value)).transpose(1, 2).contiguous()


class ContentVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VAELoader", inputs=[io.String.Input("vae_name")],
                         outputs=[io.Vae.Output("vae")])

    @classmethod
    def execute(cls, vae_name):
        return io.NodeOutput(ContentAudioVAE() if "audio" in vae_name else ContentVideoVAE())


def _sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _prefix(graph, prefix):
    result = {}
    for key, node in deepcopy(graph).items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                node["inputs"][name] = [prefix + str(value[0]), value[1]]
        result[prefix + key] = node
    return result


def _graphs(route, mode, strength, checkpoint, full, media_prefix):
    long = route == "long"
    kind = "Long_Video_Prompt_Relay" if long else "Prompt_Relay"
    stem = f"2026-08-29_H3_Audio_Refine_{kind}_Turbo8_Advanced_EXP"
    expected = generated()
    from tools.workflow_paths import legacy_workflow_path
    public = next(path for path in expected if legacy_workflow_path(path).name ==
                  f"S26_{stem}_resume_audio_Separate_EXP_eav_TailEffects.json")
    frontend = json.loads(public.read_text(encoding="utf8"))
    saved_name = legacy_workflow_path(public).name
    assert frontend == expected[public] == json.loads((SAVED / saved_name).read_text(encoding="utf8"))
    tail = json.loads((SAVED / saved_name.replace(".json", ".api.json")).read_text(encoding="utf8"))
    freeze = json.loads((TARGET / stem / "freeze_video.api.json").read_text(encoding="utf8"))
    for graph in (freeze, tail):
        graph[_one(graph, builder.RELAY_PLAN)]["inputs"].update(length=22, time_ranges="0-6\n7-13\n14-21")
        cond = _one(graph, "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced" if long else builder.RELAY_COND)
        graph[cond]["inputs"].update(width=128, height=128, query_chunk_rows=64)
        if long:
            graph[_one(graph, "MiniMaxH3LongVideoPlannerT8")]["inputs"].update(
                chain_id="existing_relay_effects_fixture", new_duration_seconds=.75, minimum_render_frames=22)
    freeze[_one(freeze, "LoraLoaderBypassModelOnly")]["inputs"]["strength_model"] = 0.
    save = _one(freeze, CHECKPOINT_SAVE)
    freeze[save]["inputs"]["confirm_save"] = True
    freeze["capture-freeze"] = {"class_type": "CaptureFrozenFirstPass", "inputs": {
        "latent": [save, 0], "status": [save, 1], "checkpoint_path": [save, 2],
        "file_sha256": [save, 3], "manifest_json": [save, 4]}}
    load = _one(tail, CHECKPOINT_LOAD)
    if not full:
        tail[load]["inputs"].update(checkpoint_path=checkpoint[2], expected_manifest_json=checkpoint[4],
                                    expected_file_sha256=checkpoint[3])
    if long:
        tail[_one(tail, "MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8")]["inputs"]["expected_video_frame_count"] = 22
        tail[_one(tail, "MiniMaxH3OutputTrimT8")]["inputs"]["duration_seconds"] = 22 / 24
    else:
        tail[_one(tail, "LoraLoaderBypassModelOnly")]["inputs"]["strength_model"] = strength
    gate = _one(tail, "MiniMaxH3AudioRefineQualityGateT8Advanced")
    if not long:
        tail[gate]["inputs"]["video_frame_count"] = 22
    assert tail[gate]["inputs"]["accept_candidate"] is False
    config = _one(tail, builder.CONFIG)
    tail[config]["inputs"].update(mode=mode, tau=.25, start_video_progress=0., end_video_progress=1.)
    assert tail[config]["inputs"]["g_hard_limit"] == 1.5
    sampler, eav = _one(tail, "SamplerCustomAdvanced"), _one(tail, builder.AUDIT)
    cond = _one(tail, "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced" if long else builder.RELAY_COND)
    tail["capture-tail"] = {"class_type": "CaptureTailEffects", "inputs": {
        "original": [load, 0], "candidate": [sampler, 0], "selected": [gate, 0],
        "decision": [gate, 3], "relay_report": [cond, 6],
        "guider_report": [_one(tail, builder.GUIDER), 1], "eav_report": [eav, 1]}}
    if long:
        delivery = _one(tail, "MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced")
        tail["capture-long"] = {"class_type": "CaptureLongDelivery", "inputs": {
            "original": [load, 0], "continuation": [delivery, 0],
            "delivery": [delivery, 1], "report_json": [delivery, 2]}}
    # Keep the saved selected delivery path, including long delivery and trim.
    selected_save = "35" if long else "31"
    for node in tail.values():
        if node["class_type"] == "SaveVideo":
            label = "selected" if node is tail[selected_save] else "candidate"
            node["inputs"]["filename_prefix"] = media_prefix + "_" + label
    create = tail[tail[selected_save]["inputs"]["video"][0]]
    trim = deepcopy(tail[_one(tail, "MiniMaxH3OutputTrimT8")]) if long else None
    for label in (("original", "candidate") if long else ("original",)):
        decoder = tail[gate]["inputs"][label + "_audio"][0]
        if long:
            # Compare the same delivery interval. The selected long graph
            # already trims audio samples before AAC encoding; previews must
            # not encode the untrimmed tail and compare it with that output.
            tail["trim-" + label] = deepcopy(trim)
            tail["trim-" + label]["inputs"].update(frames=[decoder, 0], audio=[decoder, 1])
            decoder = "trim-" + label
        tail["create-" + label] = deepcopy(create)
        tail["create-" + label]["inputs"].update(images=[decoder, 0], audio=[decoder, 1])
        tail["save-" + label] = deepcopy(tail[selected_save])
        tail["save-" + label]["inputs"].update(video=["create-" + label, 0], filename_prefix=media_prefix + "_" + label)
    if full:
        # Baseline consumes the actual in-memory final AV; cold graph retains
        # strict disk Load. No substitute Load class is registered.
        for node in tail.values():
            for name, value in node["inputs"].items():
                if value == [load, 0]:
                    node["inputs"][name] = ["freeze-" + save, 0]
        del tail[load]
    graph = {**(_prefix(freeze, "freeze-") if full else {}), **_prefix(tail, "tail-")}
    outputs = [key for key, node in graph.items() if node["class_type"] in
        {"SaveVideo", "CaptureTailEffects", "CaptureLongDelivery", "CaptureFrozenFirstPass",
         "MiniMaxH3LongVideoContextSaveT8"}]
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) == (2 if full else 1)
    return graph, outputs


def _media(root, prefix):
    result = {}
    for label in ("original", "candidate", "selected"):
        files = list(root.rglob(prefix + "_" + label + "_*.mp4"))
        assert len(files) == 1, files
        with av.open(str(files[0])) as container:
            assert len(container.streams.video) == len(container.streams.audio) == 1
            frames = list(container.decode(video=0))
            assert len(frames) == 22
            video = b"".join(frame.to_ndarray(format="rgb24").tobytes() for frame in frames)
        with av.open(str(files[0])) as container:
            samples = list(container.decode(audio=0))
            assert samples
            audio = b"".join(frame.to_ndarray().tobytes() for frame in samples)
        result[label] = {"video": hashlib.sha256(video).hexdigest(),
                         "audio": hashlib.sha256(audio).hexdigest()}
    assert result["selected"] == result["original"]
    assert result["candidate"]["audio"] != result["original"]["audio"]
    return result


def _run(root, route, mode, strength, checkpoint=None, *, full=False):
    # The project's compatibility shim is also named nodes.py; bind the real
    # Core before execution imports it, in both parent pytest and cold child.
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(ROOT.parents[1]))
        import execution
        import nodes
    assert Path(nodes.__file__).resolve() == ROOT.parents[1] / "nodes.py"

    prefix = "full" if full else "cold"
    graph, outputs = _graphs(route, mode, strength, checkpoint, full, prefix)
    root.mkdir(parents=True, exist_ok=True)
    lora = root / "fixture_refine_lora.safetensors"
    if full and strength:
        make_tiny_refine_lora(lora)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoint_nodes.folder_paths, "get_output_directory", lambda: str(root))
        patch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
        patch.setattr(custom.latent_preview, "prepare_callback", lambda *_a, **_k: lambda *_v: None)
        if strength:
            assert lora.is_file()
            patch.setattr(folder_paths, "get_full_path_or_raise", lambda category, _name:
                str(lora) if category == "loras" else pytest.fail("Unexpected fixture asset lookup"))
        classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
        for cls in [*classes, TinyRefineUNET, TinyCandidateCLIP, ContentVAELoader, RuntimeAudit,
                    RuntimeSetup, custom.SamplerCustomAdvanced, custom.RandomNoise, custom.BasicGuider,
                    LoraLoaderBypassModelOnly, CreateVideo, SaveVideo, CaptureTailEffects,
                    CaptureFrozenFirstPass, CaptureLongDelivery]:
            key = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
            patch.setitem(nodes.NODE_CLASS_MAPPINGS, key, cls)
        patch.setitem(nodes.NODE_CLASS_MAPPINGS, "LoraLoaderBypassModelOnly", ObservedRefineBypass)
        for cls in (CaptureTailEffects, CaptureFrozenFirstPass, CaptureLongDelivery, ObservedRefineBypass):
            cls.observations.clear()
        TinyRefineUNET.calls.clear()
        calls = []
        previous_route = relay.route_prompt_relay_attention

        def observed(*args, **kwargs):
            calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
            return previous_route(*args, **kwargs)

        patch.setattr(relay, "route_prompt_relay_attention", observed)
        executor = execution.PromptExecutor(SimpleNamespace(client_id=None, last_node_id=None,
            sockets_metadata={}, send_sync=lambda *_a, **_k: None), cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))
        executor.execute(graph, "existing-relay-effects-" + prefix, execute_outputs=outputs)
        assert executor.success, executor.status_messages
        # Core's input-signature cache shares identical loader settings inside
        # the full graph. Verify both actual sampler stages below, not a second
        # identical weight load that normal ComfyUI deliberately avoids.
        assert len(TinyRefineUNET.calls) == 1
        assert len(calls) == (12 if full else 4)
        assert len(CaptureTailEffects.observations) == 1
        captured = CaptureTailEffects.observations[0]
        assert captured["selected"] is captured["original"]
        assert captured["decision"] == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
        assert json.loads(captured["relay_report"])["status"] == "applied_exp"
        assert json.loads(captured["guider_report"])["external_positive"] is False
        eav = json.loads(captured["eav_report"])
        assert eav["status"] == {"disabled": "disabled_identity", "report_only": "observed_report_only",
                                 "apply_exp": "observed_apply_exp"}[mode]
        if mode != "disabled":
            assert eav["completed_forwards"] == eav["relay_attention_calls"] == 4
            assert eav["clock_match"] is True
        tensors = {label: [_sha(t) for t in captured[label]["samples"].unbind()]
                   for label in ("original", "candidate", "selected")}
        original_video, original_audio = captured["original"]["samples"].unbind()
        candidate_video, candidate_audio = captured["candidate"]["samples"].unbind()
        assert torch.allclose(original_video, candidate_video, atol=1e-5, rtol=0)
        assert not torch.equal(original_audio, candidate_audio)
        if route == "long":
            assert len(CaptureLongDelivery.observations) == 1
            original, continuation, delivery, report = CaptureLongDelivery.observations[0]
            assert [_sha(t) for t in continuation["samples"].unbind()] == tensors["original"]
            assert [_sha(t) for t in delivery["samples"].unbind()] == tensors["original"]
            assert json.loads(report)["segment_index"] == 0
            assert json.loads(report)["candidate_selected"] is False
        if full:
            assert len(CaptureFrozenFirstPass.observations) == 1
            frozen = CaptureFrozenFirstPass.observations[0]
            assert frozen[1] == "SAVED_VERIFIED"
            checkpoint = [None, *frozen[1:]]
            assert [_sha(t) for t in frozen[0]["samples"].unbind()] == tensors["original"]
        expected_bypass = ([(0., 0)] if full else []) + ([(strength, int(bool(strength)))] if route == "plain" else [])
        assert sorted(ObservedRefineBypass.observations) == sorted(expected_bypass)
        assert not torch.cuda.is_initialized()
        return {"pid": os.getpid(), "tensors": tensors, "checkpoint": checkpoint,
                "media": _media(root, prefix), "relay_calls": len(calls),
                "eav_status": eav["status"], "eav_report": eav,
                "model_loads": len(TinyRefineUNET.calls), "lora_injections": expected_bypass,
                "cuda_initialized": False}


def _cold(root, route, mode, strength, checkpoint):
    code = '''
import json, os, runpy, sys
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root), str(root.parents[1])]
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
torch.set_num_threads(2)
runpy.run_path('tests/conftest.py')
os.chdir(root.parents[1])
from test_modular_audio_tail_existing_relay_core import _run
payload = json.load(sys.stdin)
result = _run(Path(payload['root']), payload['route'], payload['mode'], payload['strength'], payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        input=json.dumps({"root": str(root), "route": route, "mode": mode,
                          "strength": strength, "checkpoint": checkpoint}),
        capture_output=True, text=True, timeout=180, check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))


def _archive(root, case, full, cold):
    destination = os.environ.get("T8_TAIL_RELAY_TEST_EVIDENCE")
    if not destination:
        return
    target = (Path(destination).resolve() / case).resolve()
    if not target.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Test evidence must stay under private project artifacts")
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for pattern in ("full_*.mp4", "cold_*.mp4", "*.h3latent.safetensors", "*.context.safetensors"):
        for source in root.rglob(pattern):
            copied = target / source.relative_to(root)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)
            files[copied.relative_to(target).as_posix()] = hashlib.sha256(copied.read_bytes()).hexdigest()
    with (target / "evidence.json").open("x", encoding="utf8") as stream:
        json.dump({"case": case, "full": full, "cold": cold, "files_sha256": files,
            "scope": "Seeded tiny CPU, content-dependent VAE/CLIP/resource doubles; long segment zero only. "
                     "Exact candidate/selected AV and decoded media parity, not trained quality."},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)


@pytest.mark.parametrize("route,strength", [("plain", 0.), ("plain", .7), ("long", 0.)])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_existing_relay_full_generation_and_cold_effect_tail_match_latents_and_media(tmp_path, route, strength, mode):
    full = _run(tmp_path, route, mode, strength, full=True)
    checkpoint = tmp_path / "MiniMaxH3/latent_checkpoints" / full["checkpoint"][2]
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    context_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.context.safetensors")}
    cold = _cold(tmp_path, route, mode, strength, full["checkpoint"])
    assert cold["pid"] != os.getpid() and cold["relay_calls"] == 4
    assert full["tensors"] == cold["tensors"]
    assert full["media"] == cold["media"]
    assert cold["cuda_initialized"] is False
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert context_before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.context.safetensors")}
    _archive(tmp_path, f"{mode}-{route}-lora-{strength}", full, cold)


def test_fixture_decoders_are_content_dependent_not_constant_media():
    video = ContentVideoVAE()
    audio = ContentAudioVAE()
    assert not torch.equal(video.decode(torch.zeros(1, 24, 7, 2, 2)), video.decode(torch.ones(1, 24, 7, 2, 2)))
    assert not torch.equal(audio.decode(torch.zeros(1, 32, 2, 37)), audio.decode(torch.ones(1, 32, 2, 37)))


def test_long_comparison_media_keeps_the_original_trim_on_all_three_paths():
    graph, _ = _graphs("long", "apply_exp", 0., None, True, "test")
    selected = graph["tail-37"]["inputs"]
    for label in ("original", "candidate"):
        actual = graph["tail-trim-" + label]["inputs"]
        for field in ("duration_seconds", "start_seconds", "fps"):
            assert actual[field] == selected[field]
        assert graph["tail-create-" + label]["inputs"]["audio"] == ["tail-trim-" + label, 1]


def test_evidence_archive_refuses_external_target_and_existing_case(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path / "project")
    monkeypatch.setenv("T8_TAIL_RELAY_TEST_EVIDENCE", str(tmp_path / "outside"))
    with pytest.raises(ValueError, match="private project"):
        _archive(tmp_path / "source", "case", {}, {})
    destination = tmp_path / "project/artifacts/evidence"
    (destination / "case").mkdir(parents=True)
    monkeypatch.setenv("T8_TAIL_RELAY_TEST_EVIDENCE", str(destination))
    with pytest.raises(FileExistsError):
        _archive(tmp_path / "source", "case", {}, {})
