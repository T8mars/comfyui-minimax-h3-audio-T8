"""Execute the saved S27 learned-LATENT tail with Core's real PromptExecutor.

Only the heavy checkpoint/adapter/model/audio-VAE sources are CPU stand-ins.  The
saved Bind -> Stage Sample -> Certified Audit -> original-H3-audio wiring is used
as authored; this is not pretrained LTX or media-output qualification.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import comfy.samplers
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
import torch

from h3_audio_t8_pkg.h3_ltx_adapter_runtime import MODEL_SHA, MODEL_REVISION, SOURCE_REVISION
from h3_audio_t8_pkg.h3_ltx_latent_contract import convert_video_latent
from h3_audio_t8_pkg.modular_sampling import ltx_latent_sample_nodes
from h3_audio_t8_pkg.modular_sampling.ltx_latent_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.ltx_latent_sample_nodes import NODES as SAMPLE_NODES
from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
from h3_audio_t8_pkg.nodes import MiniMaxH3OutputTrimT8
from h3_audio_t8_pkg.nodes_sol_engine_h3_super_advanced import TAEHV
from tools.build_modular_ltx_latent_workflow import build_prompt, STEM


ROOT = Path(__file__).resolve().parents[1]
SAVED = (ROOT / "artifacts/development/modular-sampling-m4-ltx-latent-20260924"
         / "candidate-v3" / (STEM + ".api.json"))


class TinyLearnedAdapter:
    def convert(self, video, *, pixel_frames, **_kwargs):
        frames = (pixel_frames - 1 + 7) // 8 + 1
        return torch.ones(video.shape[0], 128, frames,
                          video.shape[-2] // 2, video.shape[-1] // 2)


class ObservedLTXGuider(comfy.samplers.CFGGuider):
    calls = []

    def __init__(self, model, source, steps):
        super().__init__(model)
        self.source = source
        self.steps = steps

    def sample(self, _noise, *_args, callback=None, **_kwargs):
        self.calls.append(self.steps)
        for step in range(self.steps):
            callback(step)
        return self.source["samples"] + 0.1


class TinyNoise:
    seed = 42

    def generate_noise(self, latent):
        return torch.zeros_like(latent["samples"])


class SavedTailFixture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8TestSavedLTXLearnedInputs",
            inputs=[io.Int.Input("revision")],
            outputs=[io.Latent.Output("h3_latent"),
                     io.Latent.Output("original_h3_av"),
                     io.Latent.Output("ltx_video_latent"),
                     io.String.Output("adapter_report_json"),
                     io.Model.Output("model"), io.Noise.Output("noise"),
                     io.Guider.Output("guider"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"),
                     io.String.Output("setup_report_json"),
                     io.Vae.Output("audio_vae"),
                     io.Float.Output("output_fps")])

    @classmethod
    def execute(cls, revision):
        video = torch.ones(1, 24, 22, 4, 6)
        audio = torch.ones(1, 32, 2, 122)
        source = {"samples": NestedTensor((video, audio)), "noise_mask": "H3 only"}
        ltx, original, report = convert_video_latent(
            source, adapter=TinyLearnedAdapter(), source_frames=73,
            frame_policy="exact")
        report.update(model_sha256=MODEL_SHA, model_revision=MODEL_REVISION,
                      source_revision=SOURCE_REVISION)
        model = type("TinyLTXModel", (), {"model_options": {}})()
        _, sigmas, _, setup = sol.setup_ltx_stage2_refiner(model, enabled=False)
        guider = ObservedLTXGuider(model, ltx, 3 if revision == 0 else 2)
        sampler = comfy.samplers.KSAMPLER(lambda *_args, **_kwargs: None)
        return io.NodeOutput(source, original, ltx, json.dumps(report, sort_keys=True),
                             model, TinyNoise(), guider, sampler, sigmas, setup,
                             object(), 24.0)


class TinyTAEHVLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SolEngineTAEHVLoaderT8Advanced",
                         inputs=[io.String.Input("model_name")],
                         outputs=[TAEHV.Output("taehv")])

    @classmethod
    def execute(cls, model_name):
        assert model_name == "taeltx2_3_wide.pth"
        return io.NodeOutput(object())


class TinyTAEHVDecode(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3SolEngineTAEHVDecodeT8Advanced",
            inputs=[io.Latent.Input("latent"), TAEHV.Input("taehv"),
                    io.String.Input("execution_mode"), io.String.Input("precision")],
            outputs=[io.Image.Output("frames"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, latent, taehv, execution_mode, precision):
        assert taehv is not None
        assert execution_mode == "auto_official" and precision == "bf16_official"
        assert tuple(latent["samples"].shape) == (1, 128, 10, 2, 3)
        cls.calls.append(tuple(latent["samples"].shape))
        levels = torch.linspace(0.2, 0.8, 73).reshape(73, 1, 1, 1)
        frames = levels.expand(73, 64, 96, 3).clone()
        return io.NodeOutput(frames, json.dumps({"source": "explicit_cpu_taehv_double"}))


def _saved_tail():
    saved = json.loads(SAVED.read_text(encoding="utf8"))
    assert saved == build_prompt(certified_sample=True)
    graph = {key: deepcopy(saved[key]) for key in ("12", "13", "14", "16")}
    graph["90"] = {"class_type": SavedTailFixture.define_schema().node_id,
                   "inputs": {"revision": 0}}
    fixture_slots = {"h3_latent": 0, "original_h3_av": 1,
                     "ltx_video_latent": 2, "adapter_report_json": 3,
                     "model": 4, "noise": 5, "guider": 6, "sampler": 7,
                     "sigmas": 8, "setup_report_json": 9, "audio_vae": 10}
    for node_id in ("12", "14", "16"):
        for name, link in graph[node_id]["inputs"].items():
            if isinstance(link, list) and link[0] in {"1", "2", "5", "10", "11", "15"}:
                graph[node_id]["inputs"][name] = ["90", fixture_slots[name]]
    assert graph["13"] == saved["13"]
    assert graph["14"]["inputs"]["candidate_latent"] == ["13", 0]
    assert graph["14"]["inputs"]["sample_proof"] == ["13", 2]
    return graph


def _saved_media_tail():
    saved = json.loads(SAVED.read_text(encoding="utf8"))
    graph = _saved_tail()
    graph.update({key: deepcopy(saved[key]) for key in ("17", "18", "19", "20", "21")})
    for node_id in ("19", "20"):
        assert graph[node_id]["inputs"]["fps"] == ["2", 3]
        graph[node_id]["inputs"]["fps"] = ["90", 11]
    assert graph["18"] == saved["18"]
    assert graph["19"]["inputs"]["audio"] == ["16", 0]
    assert graph["20"]["inputs"]["audio"] == ["19", 1]
    assert graph["21"] == saved["21"]
    return graph


def test_saved_s27_tail_executes_real_core_and_rejects_incomplete_sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes

    for cls in (SavedTailFixture, *SOURCE_NODES, *SAMPLE_NODES):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.sample.fix_empty_latent_channels",
                        lambda _model, latent, *_args: latent)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.latent_preview.prepare_callback",
                        lambda *_args: lambda *_items: None)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.model_management.intermediate_device",
                        lambda: torch.device("cpu"))
    decoded_audio = []

    def decode_h3_audio(_vae, latent):
        decoded_audio.append(tuple(latent["samples"].shape))
        return {"waveform": torch.ones(1, 2, 100), "sample_rate": 32000}

    monkeypatch.setattr("comfy_extras.nodes_audio.vae_decode_audio", decode_h3_audio)
    audits = []
    original_audit = ltx_latent_sample_nodes.audit_completed_ltx_latent_stage

    def record_audit(*args):
        result = original_audit(*args)
        audits.append(json.loads(result[3]))
        return result

    monkeypatch.setattr(ltx_latent_sample_nodes, "audit_completed_ltx_latent_stage",
                        record_audit)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    graph = _saved_tail()
    ObservedLTXGuider.calls.clear()
    executor.execute(deepcopy(graph), "s27-saved-tail-cpu", execute_outputs=["14", "16"])
    assert executor.success, executor.status_messages
    assert ObservedLTXGuider.calls == [3]
    assert len(audits) == 1
    assert decoded_audio == [(1, 32, 2, 122)]
    assert audits[0]["sampler_execution_proven"] is True
    assert audits[0]["portable_cache_reuse_authorized"] is False
    executor.execute(deepcopy(graph), "s27-saved-tail-cache-cpu", execute_outputs=["14", "16"])
    assert executor.success, executor.status_messages
    assert ObservedLTXGuider.calls == [3]
    assert len(audits) == 1
    assert decoded_audio == [(1, 32, 2, 122)]

    incomplete = deepcopy(graph)
    incomplete["90"]["inputs"]["revision"] = 1
    executor.execute(incomplete, "s27-saved-tail-missing-step-cpu",
                     execute_outputs=["14", "16"])
    assert not executor.success
    assert ObservedLTXGuider.calls == [3, 2]
    assert len(audits) == 1
    assert decoded_audio == [(1, 32, 2, 122)]


def test_saved_s27_tail_reaches_native_av_media_terminal(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import av
    import execution
    import folder_paths
    import nodes
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    for cls in (SavedTailFixture, TinyTAEHVLoader, TinyTAEHVDecode,
                MiniMaxH3OutputTrimT8, CreateVideo, SaveVideo,
                *SOURCE_NODES, *SAMPLE_NODES):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.sample.fix_empty_latent_channels",
                        lambda _model, latent, *_args: latent)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.latent_preview.prepare_callback",
                        lambda *_args: lambda *_items: None)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.model_management.intermediate_device",
                        lambda: torch.device("cpu"))
    sample_count = round(73 / 24 * 32000)
    tone = torch.sin(torch.arange(sample_count).float() * (2 * torch.pi * 440 / 32000))
    monkeypatch.setattr("comfy_extras.nodes_audio.vae_decode_audio",
                        lambda _vae, _latent: {"waveform": tone.repeat(1, 2, 1) * 0.1,
                                               "sample_rate": 32000})
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    TinyTAEHVDecode.calls.clear()
    ObservedLTXGuider.calls.clear()
    executor.execute(_saved_media_tail(), "s27-saved-media-tail-cpu", execute_outputs=["21"])
    assert executor.success, executor.status_messages
    assert ObservedLTXGuider.calls == [3]
    assert TinyTAEHVDecode.calls == [(1, 128, 10, 2, 3)]
    media = list(output.rglob("learned_latent_ltx_candidate_*.mp4"))
    assert len(media) == 1
    with av.open(str(media[0])) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name == "h264"
        assert float(container.streams.video[0].average_rate) == 24.0
        assert container.streams.audio[0].codec_context.name == "aac"
        assert container.streams.audio[0].codec_context.sample_rate == 32000
        frames = list(container.decode(video=0))
        assert len(frames) == 73
        assert all((frame.width, frame.height) == (96, 64) for frame in frames)
        container.seek(0)
        audio = list(container.decode(audio=0))
        assert audio and any(frame.to_ndarray().any() for frame in audio)
