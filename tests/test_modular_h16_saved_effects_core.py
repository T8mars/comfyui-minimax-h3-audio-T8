"""Execute the saved H16 Relay/EAV freeze/resume pair under real CPU Core.

Only heavyweight upstream assets and the media terminal are replaced in an
in-memory copy. The saved candidate API files and production effect, window,
and checkpoint nodes remain unchanged.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from comfy_api.latest import io
import folder_paths
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling import h16_nodes
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.h16_nodes import NODES as H16_NODES
from h3_audio_t8_pkg.modular_sampling.h16_native_source_nodes import (
    MiniMaxH3H16VerifiedNativeSourceEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import PROMPT_RELAY_BINDING_KEY

from helpers import FakeAudioVAE, FakeVideoVAE
from test_modular_h16_relay import _case
from test_modular_h16_stages import _no_op_lift
from test_prompt_relay_advanced import NativeLikeFakeClip
from tools.build_modular_h16_effect_storage_workflow import TARGET, candidate_pair
from tools.build_modular_chunked_v5_storage_workflow import _prune_api


LABELS = ("01_freeze_after_window_2_relay_eav",
          "02_resume_windows_3_to_6_relay_eav_DRAFT")
HASHES = {
    LABELS[0]: "00ed46f56a7c8d79540bb14c8b69d28d5e2cf38d8967c63572105cc977c2ab7b",
    LABELS[1]: "7a1da7291e8caa7c74748206b7b4d5b03c59c972878918537ba630a600247d1e",
}


class TinyH16EffectInputs(io.ComfyNode):
    case = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("learned_av"), io.Model.Output("model"),
            io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
            io.Noise.Output("noise"), io.Clip.Output("clip"),
            io.Vae.Output("video_vae"), io.Vae.Output("audio_vae"),
            io.Image.Output("first_frame"),
        ])

    @classmethod
    def execute(cls):
        raw_model, sampler, sigmas = cls.case[:3]
        source, noise = cls.case[7], cls.case[9]
        return io.NodeOutput(source, raw_model, sampler, sigmas, noise,
                             NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE(),
                             torch.zeros((1, 256, 256, 3)))


class TinyH16Learned(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
                         inputs=[io.Latent.Input("source")], outputs=[
                             io.Latent.Output("learned_av"),
                             io.Int.Output("width"), io.Int.Output("height")])

    @classmethod
    def execute(cls, source):
        return io.NodeOutput(source, 256, 256)


class TinyH16Reconcile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
                         inputs=[io.Latent.Input("learned_latent")],
                         outputs=[io.Latent.Output("first_pass_latent")])

    @classmethod
    def execute(cls, learned_latent):
        return io.NodeOutput(learned_latent)


class TinyH16SamplingSetup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[
            io.Model.Input("model"), io.Sampler.Input("sampler"),
            io.Sigmas.Input("sigmas")], outputs=[
                io.Model.Output("model"), io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas")])

    @classmethod
    def execute(cls, model, sampler, sigmas):
        return io.NodeOutput(model, sampler, sigmas)


class BoundedCoordinateNoise:
    seed = 12345

    def generate_noise(self, latent):
        video, audio = latent["samples"].unbind()
        normalized = torch.arange(video.numel(), dtype=video.dtype,
                                  device=video.device).reshape(video.shape) / video.numel()
        return type(latent["samples"])((normalized, torch.ones_like(audio)))


class TinyH16FreezeReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.String.Input(name) for name in (
                             "native_status", "native_path", "native_sha",
                             "native_manifest", "window_path", "window_sha",
                             "audit_0", "audit_1", "audit_2")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                window_path, window_sha, audit_0, audit_1, audit_2):
        cls.seen.append({
            "status": native_status,
            "receipt": (native_path, native_sha, native_manifest,
                        window_path, window_sha),
            "audits": tuple(json.loads(report) for report in
                            (audit_0, audit_1, audit_2)),
        })
        return io.NodeOutput(native_status)


class TinyH16ResumeReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent"),
                                 io.Conditioning.Input("projected_positive"),
                                 *[io.String.Input(f"audit_{index}")
                                   for index in range(3, 7)]],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, projected_positive,
                audit_3, audit_4, audit_5, audit_6):
        video, audio = av_latent["samples"].unbind()
        cls.seen.append({
            "av_identity": (_input_identity(video), _input_identity(audio)),
            "relay_task": projected_positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"],
            "audits": tuple(json.loads(report) for report in
                            (audit_3, audit_4, audit_5, audit_6)),
        })
        return io.NodeOutput("observed")


def _saved(label):
    path = TARGET / f"{label}.api.json"
    raw = path.read_bytes()
    expected = candidate_pair()[0 if label.startswith("01") else 1]
    from tools.build_modular_h16_storage_workflow import candidate_api

    graph = json.loads(raw)
    assert graph == candidate_api(expected, freeze=label.startswith("01"))
    assert hashlib.sha256(raw).hexdigest() == HASHES[label]
    return graph


def _graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["100"] = {"class_type": "TinyH16EffectInputs", "inputs": {}}
    if not resume:
        graph["13"] = {"class_type": "TinyH16Learned",
                       "inputs": {"source": ["100", 0]}}
    learned_source = graph["15"]["inputs"]["learned_latent"]
    graph["15"] = {"class_type": "TinyH16Reconcile", "inputs": {
        "learned_latent": learned_source}}
    graph["16"] = {"class_type": "TinyH16SamplingSetup", "inputs": {
        "model": ["100", 1], "sampler": ["100", 2], "sigmas": ["100", 3]}}
    graph["29"]["inputs"]["noise"] = ["100", 4]
    graph["52"]["inputs"].update(
        model=["16", 0], clip=["100", 5], video_vae=["100", 6],
        audio_vae=["100", 7], first_frame=["100", 8],
        width=256, height=256,
    )
    for node in graph.values():
        if node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8":
            node["inputs"]["noise"] = ["100", 4]
    if resume:
        graph["88"]["inputs"].update(
            checkpoint_path=receipt[0], expected_file_sha256=receipt[1],
            expected_manifest_json=receipt[2],
        )
        graph["89"]["inputs"].update(
            artifact_path=receipt[3], artifact_sha256=receipt[4],
        )
        graph["101"] = {"class_type": "TinyH16ResumeReceipt", "inputs": {
            "av_latent": ["80", 0], "projected_positive": ["56", 1],
            **{f"audit_{index}": [str(71 + 3 * (index - 3)), 1]
               for index in range(3, 7)},
        }}
    else:
        graph["88"]["inputs"]["confirm_save"] = True
        graph["89"]["inputs"]["confirm_save"] = True
        graph["101"] = {"class_type": "TinyH16FreezeReceipt", "inputs": {
            "native_status": ["88", 1], "native_path": ["88", 2],
            "native_sha": ["88", 3], "native_manifest": ["88", 4],
            "window_path": ["89", 2], "window_sha": ["89", 3],
            **{f"audit_{index}": [str(62 + 3 * index), 1]
               for index in range(3)},
        }}
    graph = _prune_api(graph, ("101",))
    if resume:
        assert "12" not in graph and "13" not in graph
        assert not {"32", "35", "38", "53", "54", "55", "61", "64", "67"} & set(graph)
    else:
        assert "41" not in graph
    return graph


def _register(monkeypatch):
    import nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, *H16_NODES,
                MiniMaxH3H16VerifiedNativeSourceEXPT8,
                MiniMaxH3StageEAVConfigEXPT8,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                TinyH16EffectInputs, TinyH16Learned, TinyH16Reconcile,
                TinyH16SamplingSetup, TinyH16FreezeReceipt, TinyH16ResumeReceipt):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)


def _executor():
    import execution

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                    asset_manager=SimpleNamespace(enabled=False))


def _numeric_effect_case(monkeypatch):
    # Match the bounded latent scale used by the H16 effect math tests. Raw
    # coordinate noise reaches six figures and can round out tiny-model gain.
    case = list(_case(monkeypatch, with_frame=True, length=124))
    source = dict(case[7])
    video, audio = source["samples"].unbind()
    source["samples"] = type(source["samples"])((video / 100000, audio / 100000))
    case[7] = source
    case[9] = BoundedCoordinateNoise()
    return tuple(case)


def _cold_resume(output_root, receipt):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
        _register(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        patch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
        TinyH16EffectInputs.case = _numeric_effect_case(patch)
        TinyH16ResumeReceipt.seen.clear()
        graph = _graph(_saved(LABELS[1]), resume=True, receipt=receipt)
        executor = _executor()
        executor.execute(graph, "h16-effects-cold-resume", execute_outputs=["101"])
        assert executor.success, executor.status_messages
        assert not torch.cuda.is_initialized()
        return TinyH16ResumeReceipt.seen[-1]


def test_saved_h16_effect_pair_freezes_then_resumes_only_later_windows(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    TinyH16EffectInputs.case = _numeric_effect_case(monkeypatch)
    calls = []
    original = h16_nodes.sample_h16_pass2
    piece_hashes = []
    piece_masks = []
    original_piece = legacy.sample_piece

    def observed_piece(*args, **kwargs):
        mask = args[0]["noise_mask"].tensors[0]
        piece_masks.append((int(torch.count_nonzero(mask)), int(mask.numel()),
                            float(mask.max())))
        sampled = original_piece(*args, **kwargs)
        piece_hashes.append(_input_identity(sampled.tensors[0]))
        return sampled

    def observed(*args, **kwargs):
        calls.append(int(args[4].index))
        return original(*args, **kwargs)

    monkeypatch.setattr(h16_nodes, "sample_h16_pass2", observed)
    monkeypatch.setattr(legacy, "sample_piece", observed_piece)
    TinyH16FreezeReceipt.seen.clear()
    freeze = _graph(_saved(LABELS[0]), resume=False)
    executor = _executor()
    executor.execute(freeze, "h16-effect-freeze", execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [0, 1, 2]
    frozen = TinyH16FreezeReceipt.seen[-1]
    assert frozen["status"] == "SAVED_VERIFIED"
    for index, audit in enumerate(frozen["audits"]):
        assert audit["h16_window_index"] == index
        assert audit["status"] == "observed_report_only"
        assert audit["relay_attention_calls"] > 0
    receipt = frozen["receipt"]
    assert all(receipt)

    calls.clear()
    TinyH16ResumeReceipt.seen.clear()
    bad_receipt = (*receipt[:4], "0" * 64)
    rejected = _graph(_saved(LABELS[1]), resume=True, receipt=bad_receipt)
    executor = _executor()
    executor.execute(rejected, "h16-effect-reject-window-sha", execute_outputs=["101"])
    assert not executor.success
    assert calls == []
    assert not TinyH16ResumeReceipt.seen

    for missing in ("expected_manifest_json", "expected_file_sha256"):
        unverified = _graph(_saved(LABELS[1]), resume=True, receipt=receipt)
        unverified["88"]["inputs"][missing] = ""
        executor = _executor()
        executor.execute(unverified, f"h16-effect-reject-missing-{missing}",
                         execute_outputs=["101"])
        assert not executor.success
        assert calls == []
        assert not TinyH16ResumeReceipt.seen

    resume = _graph(_saved(LABELS[1]), resume=True, receipt=receipt)
    executor = _executor()
    executor.execute(resume, "h16-effect-resume", execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [3, 4, 5, 6]
    result = TinyH16ResumeReceipt.seen[-1]
    assert result["relay_task"] == "t2va"
    for index, audit in enumerate(result["audits"], start=3):
        assert audit["h16_window_index"] == index
        assert audit["status"] == "observed_report_only"
        assert audit["relay_attention_calls"] > 0

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['h16-effects-cold','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['h16-effects-cold']
runpy.run_path('tests/conftest.py')
from test_modular_h16_saved_effects_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1:6]),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output_root), *receipt],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold["av_identity"] == list(result["av_identity"])
    assert cold["relay_task"] == result["relay_task"]
    assert cold["audits"] == list(result["audits"])

    native_file = output_root / "MiniMaxH3/latent_checkpoints" / receipt[0]
    window_file = output_root / "MiniMaxH3/h16_window_artifacts" / receipt[3]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()

    # Only the final external EAV node is edited. Earlier frozen windows and
    # resumed windows 3–5 must not be sampled again by Core's graph cache.
    executor.execute(deepcopy(resume), "h16-effect-resume-repeat",
                     execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [3, 4, 5, 6]

    applied = deepcopy(resume)
    assert applied["78"]["class_type"] == "MiniMaxH3StageEAVConfigEXPT8"
    applied["78"]["inputs"].update(mode="apply_exp")
    executor.execute(applied, "h16-effect-final-window-apply",
                     execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [3, 4, 5, 6, 6]
    changed = TinyH16ResumeReceipt.seen[-1]
    assert changed["audits"][:3] == result["audits"][:3]
    last_audit = changed["audits"][3]
    assert last_audit["h16_window_index"] == 6
    assert last_audit["status"] == "observed_apply_exp", last_audit
    assert last_audit["relay_attention_calls"] > 0
    assert last_audit["feta"]["active_forward_count"] > 0
    assert last_audit["feta"]["g_max"] > 1.0, last_audit["feta"]
    assert piece_masks[-1][0] > 0 and piece_masks[-1][2] > 0
    assert piece_hashes[-1] != piece_hashes[-2]
    assert changed["av_identity"][0] != result["av_identity"][0]

    executor.execute(deepcopy(applied), "h16-effect-final-window-apply-repeat",
                     execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [3, 4, 5, 6, 6]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()
