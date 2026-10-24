"""Real Core extras registration/schema validation, not a complete prepare.

Only the actual public _build import/registration fragment is executed. A
small owned WAV is a path/schema fixture: no LoadAudio.execute, audio decode,
MODEL, server, controller, full preparation or trained weights are executed.
"""
import ast
import asyncio
from copy import deepcopy
from pathlib import Path
import uuid
import wave

import execution
import folder_paths
import nodes as core_nodes
import torch
from comfy_extras import nodes_audio, nodes_custom_sampler, nodes_preview_any, nodes_video


def test_actual_build_registers_real_LoadAudio_and_validates_owned_audio_graph(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/h21_control_probe.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    build, = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == "_build"]
    imports = [row for row in build.body if isinstance(row, ast.ImportFrom)
               and row.module is not None and row.module.startswith("comfy_extras")]
    loop, = [row for row in build.body if isinstance(row, ast.For)
             and isinstance(row.iter, ast.Tuple)]
    assert [(row.module, [alias.name for alias in row.names]) for row in imports] == [
        ("comfy_extras", ["nodes_custom_sampler"]),
        ("comfy_extras.nodes_video", ["LoadVideo", "GetVideoComponents"]),
        ("comfy_extras.nodes_preview_any", ["PreviewAny"]),
        ("comfy_extras.nodes_audio", ["LoadAudio"]),
    ]
    expected = {
        "BasicGuider": nodes_custom_sampler.BasicGuider,
        "RandomNoise": nodes_custom_sampler.RandomNoise,
        "SamplerCustomAdvanced": nodes_custom_sampler.SamplerCustomAdvanced,
        "LoadVideo": nodes_video.LoadVideo,
        "GetVideoComponents": nodes_video.GetVideoComponents,
        "PreviewAny": nodes_preview_any.PreviewAny,
        "LoadAudio": nodes_audio.LoadAudio,
    }
    names = [row.attr if isinstance(row, ast.Attribute) else row.id for row in loop.iter.elts]
    assert names == list(expected)
    assert ast.dump(loop.body[0], include_attributes=False) == ast.dump(ast.parse(
        "core_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls").body[0], include_attributes=False)
    assert len(loop.body) == 1 and not loop.orelse

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    waveform = tmp_path / "owned-registration-only.wav"
    with wave.open(str(waveform), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(32000)
        stream.writeframes(b"\0\0\0\0" * 320)
    registry = {name: value for name, value in core_nodes.NODE_CLASS_MAPPINGS.items() if name not in expected}
    monkeypatch.setattr(core_nodes, "NODE_CLASS_MAPPINGS", registry)

    # Independent old registration oracle: exactly the captured old six-class
    # list, with the real audio import/entry absent. Validator is never faked.
    old_imports = deepcopy([row for row in imports if row.module != "comfy_extras.nodes_audio"])
    old_loop = deepcopy(loop)
    old_loop.iter.elts = old_loop.iter.elts[:-1]
    old_fragment = ast.Module(body=[*old_imports, old_loop], type_ignores=[])
    exec(compile(old_fragment, str(source), "exec"), {"core_nodes": core_nodes})
    before = dict(registry)
    assert "LoadAudio" not in before
    graph = {
        "1": dict(class_type="LoadAudio", inputs=dict(audio=waveform.name)),
        "2": dict(class_type="PreviewAny", inputs=dict(source=["1", 0])),
    }
    missing = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), graph, None))
    assert missing[0] is False and missing[1]["type"] == "missing_node_type"
    assert missing[1]["extra_info"]["class_type"] == "LoadAudio"

    fragment = ast.Module(body=[*imports, loop], type_ignores=[])
    exec(compile(fragment, str(source), "exec"), {"core_nodes": core_nodes})
    assert set(registry) - set(before) == {"LoadAudio"}
    assert all(registry[name] is value for name, value in before.items())
    assert all(registry[name] is value for name, value in expected.items())
    schema = registry["LoadAudio"].define_schema()
    assert schema.node_id == "LoadAudio"
    assert [value.id for value in schema.inputs] == ["audio"]
    assert len(schema.outputs) == 1 and schema.outputs[0].io_type == "AUDIO"
    assert Path(nodes_audio.__file__).resolve() == Path(folder_paths.__file__).parent / "comfy_extras/nodes_audio.py"
    valid = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), graph, None))
    assert valid[0] is True and valid[3] == {} and set(valid[2]) == {"2"}
    invalid = deepcopy(graph)
    invalid["1"]["inputs"]["audio"] = "missing-registration-only.wav"
    rejected = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), invalid, None))
    assert rejected[0] is False and rejected[3]
    assert "Invalid audio file" in str(rejected)
    assert not torch.cuda.is_initialized()
