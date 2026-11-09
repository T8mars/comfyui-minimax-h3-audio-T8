"""Optional official Topaz postprocessing. No discovery/installation at import."""
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

import folder_paths
from comfy_api.latest import io, InputImpl


CATEGORY = 'T8/MiniMax H3/Post FX/Topaz'
TopazRuntime = io.Custom('T8_OFFICIAL_TOPAZ_RUNTIME')
TOPAZ_REGULAR_MODEL_IDS = [
    'iris-3', 'prob-4', 'rhea-1', 'nyx-3', 'ahq-12', 'amq-13', 'alq-13',
    'thf-4', 'thd-3', 'ghq-5', 'gcg-5', 'ganim-1', 'nxf-1', 'nxl-1',
]
TOPAZ_MODEL_GUIDE = (
    'iris-3 Iris：真人脸、低清人像；prob-4 Proteus：通用且参数最全；'
    'rhea-1 Rhea：中等质量细节恢复；nyx-3/nxf-1/nxl-1 Nyx：强降噪；'
    'ahq-12/amq-13/alq-13 Artemis：分别用于高/中/低质量素材；'
    'thf-4 Theia Fidelity：偏忠实；thd-3 Theia Detail：偏细节；'
    'ghq-5 Gaia HQ：高质量实拍/CG；gcg-5 Gaia CG：CG、锯齿和摩尔纹；'
    'ganim-1 Gaia Animation：动画。当前正式定义中Gaia Animation通常仅2x、Nyx Fast/XL仅1x、'
    'Nyx通常为1x/2x；其他常用模型通常有1x/2x/4x。不同安装版本以环境报告为准，'
    '不存在的倍率不是漏下载。')
TOPAZ_FI_MODEL_IDS = ['apo-8', 'apf-2', 'chr-2', 'chf-3', 'aion-1']
TOPAZ_FI_GUIDE = ('apo-8 Apollo：高质量通用，适合非线性运动；apf-2 Apollo Fast：速度优先；'
    'chr-2 Chronos：通用高质量变帧率；chf-3 Chronos Fast：速度优先；'
    'aion-1 Aion：新版通用插帧。请先在正式Topaz软件内下载对应模型。')


def _parameter_overrides(mode, auto_estimate_frames, preblur, noise, details, halo,
                         sharpen, compression, add_noise, grain, grain_size,
                         keep_color, input_blend, parameters_json,
                         supported_model_parameters=None):
    try:
        custom = json.loads(parameters_json)
    except json.JSONDecodeError as error:
        raise ValueError('高级参数JSON格式无效；普通使用请保持 {}') from error
    if not isinstance(custom, dict):
        raise ValueError('高级参数JSON必须是对象，例如 {}')
    if mode == 'model_defaults':
        values = {}
    elif mode == 'auto_estimate':
        values = {'estimate': auto_estimate_frames}
    elif mode == 'manual':
        requested = {'preblur': preblur, 'noise': noise, 'details': details,
            'halo': halo, 'blur': sharpen, 'compression': compression}
        supported = (set(requested) if supported_model_parameters is None
            else {str(name).lower() for name in supported_model_parameters})
        values = {'estimate': 0, **{name: value for name, value in requested.items()
            if name in supported}, 'prenoise': add_noise, 'grain': grain,
            'gsize': grain_size, 'kcolor': int(keep_color), 'blend': input_blend}
    else:
        raise ValueError('未知 Topaz 参数模式')
    values.update(custom)
    return values


def _source_root(directory):
    if not isinstance(directory, str):
        raise ValueError('Topaz 输入规整输出目录必须是本地路径字符串')
    if directory.strip():
        root = Path(directory.strip())
        if not root.is_absolute() or root.parent == root:
            raise ValueError('自定义输出目录必须是绝对路径且不能是盘符根')
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve(strict=True)
    root = Path(folder_paths.get_output_directory()) / 'MiniMaxH3-Topaz-Source'
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve(strict=True)


def _free_destination(root, source):
    candidate = root / f'{source.stem}_cfr{source.suffix}'
    index = 2
    while candidate.exists():
        candidate = root / f'{source.stem}_cfr{index}{source.suffix}'
        index += 1
    return candidate


def _timing_report(timing):
    if not timing:
        return None
    return {'timescale': timing['timescale'], 'frames': timing['frames'],
        'duration_ticks': timing['duration_ticks'],
        'deltas': [[count, delta] for count, delta in timing['entries']]}


class MiniMaxH3TopazEnvironmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazEnvironmentEXPT8',
            display_name='Official Topaz · 环境检查 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='检查正式安装、签名、滤镜，并在报告中列出模型ID、各倍率候选权重与可用参数。文件存在不代表能运行。不会安装、下载或运行增强；星光需独立官方 Neuroserver 与有效权限。',
            inputs=[io.String.Input('install_directory', default='', tooltip='正式 Topaz Video 程序目录，不是模型目录。'),
                io.String.Input('model_definitions_directory', default='', tooltip='正式安装对应的 models JSON 定义目录。'),
                io.String.Input('model_data_directory', default='', tooltip='正式 Topaz 配置的模型数据目录，可能与定义目录不同。')],
            outputs=[TopazRuntime.Output('topaz_runtime'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, install_directory, model_definitions_directory, model_data_directory):
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import audit_installation
        if not all(isinstance(p, str) and p.strip() for p in
                (install_directory, model_definitions_directory, model_data_directory)):
            raise ValueError('请填写正式 Topaz 程序、模型定义和模型数据目录。没有安装 Topaz 不影响其他 T8 节点。')
        runtime = OfficialTopaz(install_directory, model_definitions_directory, model_data_directory)
        report = audit_installation(runtime)
        handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
            'definitions': str(runtime.definitions), 'data': str(runtime.data)}
        return io.NodeOutput(handle, json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazSourcePrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazSourcePrepareEXPT8',
            display_name='Official Topaz · 输入时间轴规整 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='送入正式Topaz前修正视频时间轴。拼接或转码时只要有一帧的时长写错，其后每一帧的时间戳都会错位，'
                '正式Topaz会在增强开始前直接拒绝整片。默认先只改写MP4/MOV容器里的帧时长表：编码帧数据与音轨逐字节不动，'
                '无画质损失。确实是可变帧率、必须重新规整时才用FFmpeg重编码为严格恒定帧率。输入本来就合规时原样透传，不产生新文件。',
            inputs=[io.Video.Input('source_video'),
                io.Combo.Input('repair_mode', options=['auto', 'lossless', 'reencode'], default='auto',
                    display_name='规整方式',
                    tooltip='auto：先无损改写时长表，不适用才重编码。lossless：只允许无损改写，不适用直接报错不重编码。'
                        'reencode：一律用FFmpeg重编码。普通使用保持auto。'),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='可选输出目录。留空写入ComfyUI/output/MiniMaxH3-Topaz-Source。')],
            outputs=[io.Video.Output('normalized_video'), io.String.Output('saved_path'),
                io.String.Output('report_json')])

    @classmethod
    def execute(cls, source_video, repair_mode='auto', output_directory=''):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from . import mp4_timing
        from .dlss_fi_backend.entry import file_video_path
        source = file_video_path(source_video, InputImpl.VideoFromFile)
        throw_exception_if_processing_interrupted()
        timing = mp4_timing.video_timing(source)
        report = {'source': str(source), 'container_timing': _timing_report(timing)}
        constant = mp4_timing.constant_delta(timing) if timing else None
        if constant:
            report.update(status='already_constant_rate', fps=str(mp4_timing.frame_rate(timing, constant)))
            return io.NodeOutput(source_video, str(source),
                json.dumps(report, ensure_ascii=False, indent=2))
        if timing is None:
            raise ValueError('无损规整只支持MP4/MOV容器。本片是其它容器，请先用其他节点转成MP4再送入正式Topaz。')
        destination = _free_destination(_source_root(output_directory), source)
        delta = mp4_timing.repair_delta(timing)
        if delta and repair_mode != 'reencode':
            mp4_timing.apply_delta(source, destination, timing, delta)
            report.update(status='lossless_timing_rewrite', saved_path=str(destination),
                fps=str(mp4_timing.frame_rate(timing, delta)), rewritten_delta=delta,
                original_deltas=report['container_timing']['deltas'],
                drift_ticks=timing['duration_ticks'] - timing['frames'] * delta,
                note='只改写容器帧时长表；编码帧数据与音轨逐字节未改动')
            return io.NodeOutput(InputImpl.VideoFromFile(str(destination)), str(destination),
                json.dumps(report, ensure_ascii=False, indent=2))
        if repair_mode == 'lossless':
            raise ValueError('这个视频不是单一帧时长写错，只改写容器时长表无法修好。'
                '请把规整方式改为auto或reencode。')
        from .ffmpeg_utils import resolve_ffmpeg
        fps = mp4_timing.frame_rate(timing, delta or mp4_timing.dominant_delta(timing))
        command = [resolve_ffmpeg(), '-hide_banner', '-nostdin', '-n', '-i', str(source),
            '-fps_mode', 'cfr', '-r', str(fps), '-frames:v', str(timing['frames']),
            '-c:v', 'hevc_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '16', '-b:v', '0',
            '-video_track_timescale', str(timing['timescale']), '-c:a', 'copy', str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf8',
            errors='replace', stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or not destination.is_file():
            raise RuntimeError('FFmpeg 重新规整时间轴失败：' + result.stderr.strip()[-500:])
        throw_exception_if_processing_interrupted()
        report.update(status='reencode_constant_rate', saved_path=str(destination), fps=str(fps),
            command=[str(part) for part in command], note='已用FFmpeg重编码为严格恒定帧率，音轨直接复制')
        return io.NodeOutput(InputImpl.VideoFromFile(str(destination)), str(destination),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazVideoEXPT8',
            display_name='Official Topaz · 视频高清增强 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='正式tvai_up高清增强；默认自动接收常见8/10/16-bit SDR并统一输出高质量8-bit H.264 MP4。兼容音轨直接保留，其他音轨自动转AAC；无需用户选择格式或再接“保存视频”。不插帧、不自动下载；Topaz软件无需保持打开。',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_REGULAR_MODEL_IDS, default='iris-3',
                    display_name='高清模型（用途见提示）', tooltip=TOPAZ_MODEL_GUIDE),
                io.Combo.Input('scale', options=['1x', '2x', '4x'], default='2x', tooltip='不改变帧率；需在正式 Topaz 中准备对应倍率的权重。'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True,
                    tooltip='Topaz可使用的显存比例，不是启动门槛。默认0.80；显存紧张可试0.60。'),
                io.String.Input('parameters_json', default='{}', multiline=True, advanced=True,
                    display_name='高级参数JSON（通常保持{}）',
                    tooltip='仅供精确覆盖独立控件或未来参数；普通用户不要填写。'),
                io.Combo.Input('size_mode', options=['scale', 'target_dimensions'], default='scale',
                    optional=True, advanced=True, tooltip='scale 使用上方倍率。target_dimensions 使用下方宽高，忽略倍率；需保持原比例、1至4倍，不裁剪或拉伸。'),
                io.Int.Input('target_width', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；填写32至8192之间的偶数。'),
                io.Int.Input('target_height', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='仅自定义尺寸模式生效；宽高须同时填写并保持原视频比例。'),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='可选母版输出目录。留空写入ComfyUI/output/MiniMaxH3-Topaz；磁盘不足时可填写另一个本地目录。'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='仅用于列表中尚未收录的新正式模型ID；填写后覆盖model_id。'),
                io.Combo.Input('output_profile', options=['delivery_h264', 'delivery_hevc_main10', 'lossless_master'],
                    default='delivery_h264', optional=True, advanced=True,
                    display_name='高级输出策略（普通用户保持默认）',
                    tooltip='默认delivery_h264自动接收常见8/10/16-bit SDR并统一输出8-bit H.264，无需按原片位深选择。Main10和lossless_master只供明确需要保留位深或审计的高级用户。HDR仍需先转为SDR。节点已直接保存，不要再接SaveVideo。'),
                io.Combo.Input('parameter_mode', options=['model_defaults', 'auto_estimate', 'manual'],
                    default='model_defaults', optional=True, display_name='参数模式',
                    tooltip='model_defaults使用模型默认；auto_estimate让Topaz抽帧估计；manual使用下方独立参数。'),
                io.Int.Input('auto_estimate_frames', default=8, min=1, max=100, optional=True,
                    advanced=True, display_name='自动分析帧数', tooltip='仅auto_estimate生效；默认8较稳定，复杂长片可试12至20，越多启动越慢。'),
                io.Float.Input('preblur', default=0., min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='抗锯齿(-) / 去模糊(+)',
                    tooltip='负值偏抗锯齿/摩尔纹，正值偏去模糊；例如-0.20或+0.20。仅manual生效。'),
                io.Float.Input('noise', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='降噪', tooltip='例如0.20轻度、0.45中度；过高会蜡化皮肤。仅manual生效。'),
                io.Float.Input('details', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='恢复细节', tooltip='例如0.25自然、0.50较强；过高可能生成假纹理。仅manual生效。'),
                io.Float.Input('halo', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='去光晕', tooltip='例如0.20轻度、0.50用于明显边缘光晕；过高会损失边缘。仅manual生效。'),
                io.Float.Input('sharpen', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='锐化', tooltip='内部映射Topaz blur参数；例如0.15轻锐、0.35较锐。仅manual生效。'),
                io.Float.Input('compression', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='去压缩伪影', tooltip='例如0.25轻度网压、0.55明显块效应。仅manual生效。'),
                io.Float.Input('add_noise', default=0., min=0., max=.1, step=.005, optional=True,
                    advanced=True, display_name='输入预加噪', tooltip='默认0；例如0.01可帮助过度平滑素材保留纹理。仅manual生效。'),
                io.Float.Input('grain', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='输出颗粒', tooltip='默认0；例如0.10轻颗粒、0.20明显胶片感。仅manual生效。'),
                io.Float.Input('grain_size', default=0., min=0., max=5., step=.1, optional=True,
                    advanced=True, display_name='颗粒尺寸', tooltip='仅输出颗粒大于0时有意义；1.0细颗粒，2.0较粗。'),
                io.Boolean.Input('keep_color', default=True, optional=True, advanced=True,
                    display_name='模型色彩校正', tooltip='默认true；关闭只建议用于色彩对比测试。'),
                io.Float.Input('input_blend', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='混合原片', tooltip='0=完全增强，1=完全原片；0.10至0.20可减弱过度锐化或蜡感。'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='额外推理实例',
                    tooltip='0最稳。1在本机60帧仅由11.5秒降到10.5秒，收益小但占用更多显存；16GB卡不建议更高。'),
                io.Int.Input('gpu_device_index', default=0, min=0, max=15, optional=True,
                    advanced=True, display_name='Topaz GPU序号',
                    tooltip='单卡保持0；多卡时选择正式Topaz所使用的GPU序号。')],
            outputs=[io.Video.Output('enhanced_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, scale, vram_fraction=.8, parameters_json='{}',
                size_mode='scale', target_width=0, target_height=0, output_directory='', custom_model_id='',
                output_profile='delivery_h264', parameter_mode='model_defaults', auto_estimate_frames=8,
                preblur=0., noise=.5, details=.5, halo=.5, sharpen=.5, compression=.5,
                add_noise=0., grain=0., grain_size=0., keep_color=True, input_blend=0., engine_instances=0,
                gpu_device_index=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_regular
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('Connect the official Topaz environment node')
        if scale not in ('1x', '2x', '4x') or not isinstance(parameters_json, str) or len(parameters_json) > 8192:
            raise ValueError('Invalid Topaz scale or parameter JSON')
        if not isinstance(custom_model_id, str):
            raise ValueError('Custom Topaz model ID must be a string')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        _, definition = runtime.model(model_id)
        declared = {str(item.get('name', '')).lower() for item in definition.get('parameters', [])}
        parameters = _parameter_overrides(parameter_mode, auto_estimate_frames, preblur,
            noise, details, halo, sharpen, compression, add_noise, grain, grain_size,
            keep_color, input_blend, parameters_json, supported_model_parameters=declared)
        manual_controls = {'preblur', 'noise', 'details', 'halo', 'blur', 'compression'}
        parameter_audit = {'mode': parameter_mode,
            'ignored_unsupported_model_controls': sorted(manual_controls - declared)
                if parameter_mode == 'manual' else []}
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz requires an untrimmed, uncropped local file VIDEO. Save edited/frame-batch VIDEO first.') from error
        if not isinstance(output_directory, str):
            raise ValueError('Topaz output directory must be a local path string')
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('Custom Topaz output directory must be an absolute non-root path')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            output = Path(folder_paths.get_output_directory()).resolve(strict=True)
            root = output / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('Topaz output directory must be a real local directory')
        progress_bar = ProgressBar(100)
        path, report = run_regular(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, width=target_width if size_mode == 'target_dimensions' else None,
            height=target_height if size_mode == 'target_dimensions' else None,
            scale=int(scale[0]), size_mode=size_mode,
            output_profile=output_profile,
            instances=engine_instances,
            parameter_audit=parameter_audit,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            device=gpu_device_index, vram=vram_fraction, parameters=parameters,
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazFrameInterpolationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazFrameInterpolationEXPT8',
            display_name='Official Topaz · 视频插帧 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='独立正式tvai_fi插帧：2x/4x提高帧率但保持时长和尺寸；默认自动接收常见SDR位深并保存高质量8-bit H.264 MP4，非兼容音轨自动转AAC。不是慢动作，不与高清模型混用，不需要保持Topaz软件开启。',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_FI_MODEL_IDS, default='apo-8',
                    display_name='插帧模型（用途见提示）', tooltip=TOPAZ_FI_GUIDE),
                io.Combo.Input('multiplier', options=['2x', '4x'], default='2x',
                    display_name='帧率倍数', tooltip='2x：24→48、30→60；4x：24→96。时长不变。'),
                io.Float.Input('duplicate_threshold', default=.01, min=-.01, max=.2, step=.005,
                    advanced=True, display_name='重复帧检测阈值',
                    tooltip='0或负值关闭重复帧检测；过高可能误删正常静止帧。一般保持0.01。'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='留空写入ComfyUI/output/MiniMaxH3-Topaz；也可填写其他本地盘。'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='仅用于列表未收录的正式插帧模型ID；填写后覆盖model_id。'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='额外推理实例', tooltip='0最稳；增加会占更多显存。'),
                io.Combo.Input('output_profile', options=['delivery_h264', 'delivery_hevc_main10'],
                    default='delivery_h264', optional=True, advanced=True,
                    display_name='高级输出策略（普通用户保持默认）',
                    tooltip='默认delivery_h264自动接收常见8/10/16-bit SDR并统一输出8-bit H.264，无需按原片位深选择。Main10仅供明确需要保持10-bit的高级用户。HDR仍需先转为SDR。'),
                io.Int.Input('gpu_device_index', default=0, min=0, max=15, optional=True,
                    advanced=True, display_name='Topaz GPU序号', tooltip='单卡保持0；多卡选择Topaz使用的GPU。')],
            outputs=[io.Video.Output('interpolated_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, multiplier,
                duplicate_threshold=.01, vram_fraction=.8, output_directory='', custom_model_id='',
                engine_instances=0, output_profile='delivery_h264', gpu_device_index=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_interpolation
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('请连接 Official Topaz 环境检查节点')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz插帧需要未裁剪的本地文件VIDEO；帧批请先保存成视频。') from error
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('自定义Topaz输出目录必须是绝对的非磁盘根目录')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            root = Path(folder_paths.get_output_directory()).resolve(strict=True) / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('Topaz输出目录必须是真实本地目录')
        progress_bar = ProgressBar(100)
        path, report = run_interpolation(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, multiplier=int(multiplier[0]),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            device=gpu_device_index, vram=vram_fraction, instances=engine_instances,
            duplicate_threshold=duplicate_threshold, output_profile=output_profile,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


TOPAZ_NODE_CLASSES = [MiniMaxH3TopazEnvironmentEXPT8, MiniMaxH3TopazSourcePrepareEXPT8,
                      MiniMaxH3TopazVideoEXPT8, MiniMaxH3TopazFrameInterpolationEXPT8]
