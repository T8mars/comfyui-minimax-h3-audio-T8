"""Solar experimental nodes; existing loaders/samplers stay explicit."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from . import solar_structural as structural
from . import solar_structural_sampling as runtime
from .core import nested_av_parts
from .reference_package import canonical
from .source_audio_plan import sample_at


CATEGORY = "T8/MiniMax H3/Structure/Experimental"
COMPATIBILITY_NOTE = (
    "User-selected models, LoRA order/strength and callable patches are retained; "
    "compatibility and media quality of their combination remain unqualified."
)


def silence_path(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Select an encoded silence safetensors file")
    path = Path(name)
    if path.is_absolute():
        return path
    root = (Path(folder_paths.models_dir) / "solar").resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Relative silence paths must stay inside models/solar; select an absolute file otherwise")
    return path


def _source_track(audio, source_range):
    if audio is None:
        return None, {"status": "not_supplied", "audio_driving": False}
    if not isinstance(audio, dict):
        raise ValueError("Source soundtrack must be an AUDIO mapping")
    rate = audio.get("sample_rate")
    if type(rate) is not int or rate <= 0:
        raise ValueError("Source soundtrack sample_rate must be a positive integer")
    pcm = structural.finite_tensor(audio.get("waveform"), "source soundtrack")
    if pcm.ndim != 3 or pcm.shape[0] != 1 or pcm.shape[1] not in (1, 2):
        raise ValueError("Source soundtrack requires [1, mono-or-stereo, samples]")
    start, stop = (sample_at(frame, rate) for frame in source_range)
    if stop <= start or stop > pcm.shape[-1]:
        raise ValueError("Source soundtrack cannot cover the selected global frame interval; no implicit padding")
    selected = {**audio, "waveform": pcm[..., start:stop].clone()}
    return selected, {
        "status": "original_source_slice", "source_samples": [start, stop],
        "output_samples": stop - start, "sample_rate": rate,
        "resampled": False, "padded": False, "audio_driving": False,
        "origin": "input AUDIO sample zero corresponds to input structure frame zero",
        "delivery": "separate mux; the chosen saver may encode/normalize the final track",
    }


class MiniMaxH3SolarStructureEncodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Solar · 结构与原声区间 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="显式24fps结构重渲染。沿用已选CLIP/VAE，实际长度向下对齐17n+5；"
                        "可选首帧是同一源时刻的外观参考。原声只精确切片供最终保存，不驱动采样。",
            inputs=[io.Clip.Input("clip"), io.Vae.Input("video_vae"), io.Image.Input("structure"),
                    io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                    io.Int.Input("width", default=832, min=32, step=32),
                    io.Int.Input("height", default=480, min=32, step=32),
                    io.Int.Input("length", default=158, min=5),
                    io.Int.Input("source_start", default=0, min=0),
                    io.Float.Input("fps", default=24., min=.001, advanced=True,
                                   tooltip="当前结构时钟要求24fps；非24fps素材先显式转换。"),
                    io.Image.Input("first_frame", optional=True),
                    io.Vae.Input("structure_vae", optional=True, advanced=True),
                    io.Int.Input("first_frame_source_index", default=-1, min=-1,
                                 optional=True, advanced=True,
                                 tooltip="-1表示坐标未声明；声明时须等于source_start，不检验视觉身份。"),
                    io.Audio.Input("source_audio", optional=True,
                                   tooltip="与输入structure同一零点的完整原声；仅最终保存使用。")],
            outputs=[io.Conditioning.Output("positive"), io.Latent.Output("av_latent"),
                     io.Int.Output("actual_frames"), io.Audio.Output("final_audio"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, clip, video_vae, structure, prompt, width, height, length,
                source_start=0, fps=24., first_frame=None, structure_vae=None,
                first_frame_source_index=-1, source_audio=None):
        index = None if first_frame_source_index == -1 else first_frame_source_index
        positive, latent, report = structural.encode_structure(
            clip, video_vae, structure, prompt, width, height, length,
            first_frame=first_frame, structure_vae=structure_vae,
            source_start=source_start, first_frame_source_index=index, fps=fps)
        audio, soundtrack = _source_track(source_audio, report["source_range"])
        report = {**report, "source_soundtrack": soundtrack, "quality_accepted": False,
                  "candidate_registered": True}
        return io.NodeOutput(positive, latent, report["frames"], audio, canonical(report))


class MiniMaxH3SolarSilenceSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Solar · 静音采样作用域 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="默认关闭，输入MODEL原样直通。开启时读取真实编码静音并委托现有采样作用域；"
                        "不加载LoRA、不改sampler/步数/sigma、不生成最终原声。0或部分mask行保留原输入音频。",
            inputs=[io.Model.Input("model"), io.Boolean.Input("enabled", default=False),
                    io.String.Input("silence_file", default="",
                                    tooltip="models/solar下的文件名或明确选择的绝对路径；不自动下载。"),
                    io.String.Input("tensor_key", default="silence_170f", advanced=True),
                    io.Int.Input("seed", default=0, min=0, max=2**64-1,
                                 tooltip="建议与RandomNoise连接同一seed；不改RandomNoise自身。"),
                    io.Latent.Input("av_latent", optional=True, advanced=True,
                                    tooltip="可选：开启时提前检查实际AV形状和静音长度；关闭时不读取。")],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, enabled=False, silence_file="", tensor_key="silence_170f", seed=0,
                av_latent=None):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if not enabled:
            return io.NodeOutput(model, canonical({
                "status": "disabled_identity", "requested_enabled": False,
                "input_model_unchanged": True, "existing_effects_removed": False,
                "quality_accepted": False, "candidate_registered": True}))
        silence, identity = structural.load_silence(silence_path(silence_file), key=tensor_key)
        if av_latent is not None:
            video, audio = nested_av_parts(av_latent)
            structural.finite_tensor(video, "Solar video latent")
            structural.finite_tensor(audio, "Solar audio latent")
            if video.shape[1] != 24 or tuple(audio.shape[1:3]) != (32, 2):
                raise ValueError("Solar AV latent requires video channels=24 and audio channels=32, stereo=2")
            if audio.shape[-1] > silence.shape[-1]:
                raise ValueError(f"Selected silence covers {silence.shape[-1]} audio ticks, actual clip needs {audio.shape[-1]}")
        selected, report = runtime.configure_silence(model, silence, seed=seed, identity=identity)
        report = {**report, "requested_enabled": True, "compatibility_warning": COMPATIBILITY_NOTE,
                  "sample_result_binding_proven": False, "quality_accepted": False,
                  "candidate_registered": True}
        return io.NodeOutput(selected, canonical(report))

    @classmethod
    def fingerprint_inputs(cls, enabled=False, **_kwargs):
        # Re-read the explicitly selected small silence asset on every enabled
        # queue, including replacement at the same path with preserved stat data.
        return float("nan") if enabled else "disabled_identity"


class MiniMaxH3SolarAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Solar · 实际运行报告 (EXP/T8)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="after_sampling仅建立执行顺序。显示MODEL最近请求实际调用/音画时钟与静音替换、"
                        "保留行数；不认证输入latent来源、已加载LoRA、成片质量或通用GPU收益。",
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, after_sampling):
        text = canonical({**runtime.audit_silence(model), "sample_result_binding_proven": False,
                          "quality_accepted": False, "candidate_registered": True})
        return io.NodeOutput(text, ui={"text": [text]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3SolarStructureEncodeEXPT8, MiniMaxH3SolarSilenceSetupEXPT8, MiniMaxH3SolarAuditEXPT8]
