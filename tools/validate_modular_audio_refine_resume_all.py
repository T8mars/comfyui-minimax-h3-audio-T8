"""Validate and save all S26 video-freeze/audio-tail pairs with current CPU Core.

No graph is queued. Source examples and the older single Relay pair are not
overwritten; outputs live under a new private artifacts version.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_modular_audio_refine_resume_all import (
    CHECKPOINT_LOAD, CHECKPOINT_SAVE, ROOT, TARGET, pairs_from_current_sources,
)
from tools.build_modular_audio_refine_workflows import split_api

CORE = ROOT.parents[1]


async def validate_pairs(saved_dir=None, *, pairs=None):
    import execution
    import nodes

    for name in ('nodes_custom_sampler.py', 'nodes_video.py', 'nodes_preview_any.py',
                 'nodes_lora_debug.py'):
        if not await nodes.load_custom_node(str(CORE / 'comfy_extras' / name),
                                            module_parent='comfy_extras'):
            raise RuntimeError('Required Core module failed: ' + name)
    if not await nodes.load_custom_node(str(CORE / 'custom_nodes/ComfyUI-ClipProj')):
        raise RuntimeError('Required ClipProj module failed')
    spec = importlib.util.spec_from_file_location(
        '_t8_modular_audio_refine_all_resume_core', ROOT / '__init__.py',
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    for cls in await package.comfy_entrypoint().get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    pairs = pairs_from_current_sources() if pairs is None else pairs
    needed = {node['type'] for pair in pairs.values() for graph in pair
              for node in graph['nodes']}
    info = {}
    for kind in needed - {'MarkdownNote', 'VHS_VideoCombine'}:
        cls = nodes.NODE_CLASS_MAPPINGS.get(kind)
        if cls is None:
            raise ValueError('Current Core lacks ' + kind)
        info[kind] = cls.GET_NODE_INFO_V1() if hasattr(cls, 'GET_NODE_INFO_V1') else cls.INPUT_TYPES()
    results, candidates = {}, {}
    for path, pair in pairs.items():
        for phase, graph in zip(('freeze_video', 'resume_audio'), pair, strict=True):
            key = path.stem + '/' + phase
            try:
                api = split_api(graph, info)
                if saved_dir is not None:
                    directory = saved_dir / path.stem
                    if (json.loads((directory / (phase + '.json')).read_text(encoding='utf8')) != graph or
                            json.loads((directory / (phase + '.api.json')).read_text(encoding='utf8')) != api):
                        raise ValueError('Saved frontend/API differs from current source or Core schema')
                checked = await execution.validate_prompt('audio-refine-all-' + key, api, None)
                kinds = [node['type'] for node in graph['nodes']]
                types = {str(node['id']): node['type'] for node in graph['nodes']}
                validated_outputs = {types[node_id] for node_id in checked[2]}
                samplers = kinds.count('SamplerCustomAdvanced')
                valid = bool(checked[0]) and not bool(checked[3])
                valid &= kinds.count(CHECKPOINT_SAVE if phase == 'freeze_video' else CHECKPOINT_LOAD) == 1
                valid &= samplers >= 1 if phase == 'freeze_video' else samplers == 1
                if phase == 'resume_audio':
                    valid &= 'MiniMaxH3AudioRefineQualityGateT8Advanced' in validated_outputs
                    valid &= any(kind.endswith('StageAuditEXPT8') for kind in validated_outputs)
                results[key] = {'core_valid': bool(valid), 'frontend_nodes': len(graph['nodes']),
                                'api_nodes': len(api), 'samplers': samplers,
                                'validated_outputs': sorted(validated_outputs), 'result': checked}
                candidates[key] = graph, api
            except (KeyError, TypeError, ValueError) as error:
                results[key] = {'core_valid': False, 'conversion_error': str(error)}
    return results, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, default=TARGET)
    parser.add_argument('--verify-candidate-dir', type=Path)
    args = parser.parse_args()
    report_path, candidate_dir = args.report.resolve(), args.candidate_dir.resolve()
    verify_dir = args.verify_candidate_dir.resolve() if args.verify_candidate_dir else None
    if (not report_path.is_relative_to(ROOT / 'artifacts') or
            not candidate_dir.is_relative_to(ROOT / 'artifacts') or
            (verify_dir is not None and not verify_dir.is_relative_to(ROOT / 'artifacts')) or
            report_path.exists() or (verify_dir is None and candidate_dir.exists())):
        parser.error('Use new private report and candidate paths under artifacts')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0',
                      OMP_NUM_THREADS='2')
    sys.path[:0] = [str(CORE), str(ROOT)]
    sys.argv = [sys.argv[0], '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(2)
    results, candidates = asyncio.run(validate_pairs(verify_dir))
    success = (len(results) == 20 and all(item['core_valid'] for item in results.values())
               and not torch.cuda.is_initialized())
    report = {'schema': 't8.modular-sampling.audio-refine-all-cold-resume.v1',
              'status': 'pass' if success else 'fail', 'cases': results,
              'queued': False, 'browser_used': False,
              'saved_candidate_verified': verify_dir is not None,
              'cuda_initialized': torch.cuda.is_initialized(),
              'boundary': 'Core schema only; media, real weights, browser, cancellation and sound need separate qualification'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    if success and verify_dir is None:
        candidate_dir.mkdir(parents=True)
        for key, (frontend, api) in candidates.items():
            directory = candidate_dir / key.split('/')[0]
            directory.mkdir(exist_ok=True)
            phase = key.split('/')[1]
            (directory / (phase + '.json')).write_text(
                json.dumps(frontend, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            (directory / (phase + '.api.json')).write_text(
                json.dumps(api, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': report['status'], 'pairs': len(results) // 2,
                      'failed': [name for name, item in results.items() if not item['core_valid']]},
                     ensure_ascii=False))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
