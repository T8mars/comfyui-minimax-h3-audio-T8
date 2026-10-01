"""Every implemented sampled Audio Refine entrance has a source-bound cold pair."""

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import comfy.model_management
import comfy.nested_tensor
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_api.latest import io
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter import LoRAAdapter
import pytest
import torch

from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg import audio_refine_advanced as refine_runtime
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.audio_refine_storage_nodes import (
    MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
    MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
    MiniMaxH3AudioRefineContextCommitGateEXPT8,
)
from h3_audio_t8_pkg.long_video import (
    load_context_state, patch_long_video_model, save_context_state,
)
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import save_native_h3_av_checkpoint
from h3_audio_t8_pkg.nodes import (
    MiniMaxH3AVDecodeT8, MiniMaxH3AudioConditioningT8,
    MiniMaxH3DualClockSamplerT8,
)
from h3_audio_t8_pkg.nodes_long_video_exp import (
    MiniMaxH3LongVideoContextLoadT8, MiniMaxH3LongVideoContextSaveT8,
    MiniMaxH3LongVideoPlannerT8,
)
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import (
    MiniMaxH3LearnedLatentUpscaleT8Advanced,
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
)
from h3_audio_t8_pkg.nodes_detail_sampling_advanced import (
    MiniMaxH3TwoPassDetailMixerT8Advanced,
)
from h3_audio_t8_pkg.nodes_enhance_a_video_advanced import (
    MiniMaxH3EnhanceAVideoT8Advanced, MiniMaxH3EnhanceAVideoAuditT8Advanced,
)
from h3_audio_t8_pkg.nodes_audio_refine_advanced import (
    MiniMaxH3AudioRefineCompatibilityPlanT8Advanced,
    MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
    MiniMaxH3AudioRefineDualClockSetupT8Advanced,
    MiniMaxH3AudioRefineDualModelSetupT8Advanced,
    MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced,
    MiniMaxH3AudioRefineModelRouteT8Advanced,
    MiniMaxH3AudioRefinePhase2PlanT8Advanced,
    MiniMaxH3AudioRefinePlanT8Advanced,
    MiniMaxH3AudioRefineQualityGateT8Advanced,
)
from h3_audio_t8_pkg.nodes_external_compatibility_advanced import (
    MiniMaxH3ClipProjCompatibilityAuditT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import (
    MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
    MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
)
from test_modular_audio_refine_full_resume_core import (
    CaptureFullResume, RuntimeAudit, RuntimeSetup, TinyCandidateCLIP,
    TinyRefineUNET, TinyRefineVAE, _native_av, _required_for,
)
from test_modular_audio_refine_cold_process import CaptureFrozenFirstPass
from test_audio_refine_advanced import _runtime, _turbo4_metadata
from tools.build_modular_audio_refine_resume_all import (
    CHECKPOINT_GUARD, CHECKPOINT_LOAD, CHECKPOINT_SAVE, CONTEXT_COMMIT_GATE,
    RELAY_SOURCE,
    TARGET, _last_video_sampler,
    pairs_from_current_sources,
)
from tools.build_modular_h16_storage_workflow import Draft


class TinyReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LoadImage', inputs=[io.String.Input('image')],
                         outputs=[io.Image.Output('image'), io.Mask.Output('mask')])

    @classmethod
    def execute(cls, image):
        assert image
        return io.NodeOutput(torch.full((1, 128, 128, 3), 0.5),
                             torch.zeros((1, 128, 128)))


class CaptureEAVFreezeAudit(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='CaptureEAVFreezeAudit', is_output_node=True,
                         inputs=[io.String.Input('eav_report')],
                         outputs=[io.String.Output('status')])

    @classmethod
    def execute(cls, eav_report):
        report = json.loads(eav_report)
        cls.observations.append(report)
        return io.NodeOutput(report['status'])


class TinyFullDepthEAVUNET(TinyRefineUNET):
    """Cheap H3 width with all 50 DiT attention blocks required by EAV Audit."""

    calls = []

    @classmethod
    def execute(cls, unet_name, weight_dtype):
        import comfy.latent_formats
        import comfy.model_base
        import comfy.model_patcher
        import comfy.ops
        import comfy.supported_models_base

        cls.calls.append((unet_name, weight_dtype))

        class Config(comfy.supported_models_base.BASE):
            latent_format = comfy.latent_formats.MiniMaxH3AV
            unet_extra_config = {}
            sampling_settings = {'shift': 12., 'audio_shift': 3.}
            custom_operations = comfy.ops.disable_weight_init

        config = Config(dict(hidden_size=24, num_layers=50,
                             token_refiner_num_layers=0,
                             num_attention_heads=3, attention_head_dim=128,
                             ffn_hidden_size=32, text_dim=8,
                             timestep_input_dim=4,
                             time_embed_hidden_size=24, time_embed_dim=24,
                             rope_inv_freq_len=16, gate_compress=True,
                             dtype=torch.float32))
        base = comfy.model_base.MiniMaxH3(config, device=torch.device('cpu'))
        generator = torch.Generator().manual_seed(26091603)
        with torch.no_grad():
            for value in base.parameters():
                value.copy_(torch.randn(value.shape, generator=generator) * .03)
            base.diffusion_model.rope.inv_freq.fill_(1.)
        return io.NodeOutput(comfy.model_patcher.ModelPatcher(
            base, torch.device('cpu'), torch.device('cpu')))


class TinyClipProjApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ClipProjApply',
                         inputs=[io.Clip.Input('clip'),
                                 io.String.Input('projection')],
                         outputs=[io.Clip.Output('clip')])

    @classmethod
    def execute(cls, clip, projection):
        assert projection.endswith('.safetensors')
        return io.NodeOutput(clip)


class TinyCompatCLIP(TinyCandidateCLIP):
    @classmethod
    def execute(cls, clip_name, type, device):
        assert type in {'minimax', 'boogu'}
        return super().execute(clip_name, 'minimax', device)


class TinyTurbo4Lora(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='LoraLoaderModelOnly',
                         inputs=[io.Model.Input('model'), io.String.Input('lora_name'),
                                 io.Float.Input('strength_model')],
                         outputs=[io.Model.Output('model')])

    @classmethod
    def execute(cls, model, lora_name, strength_model):
        assert 'turbo_4' in lora_name and strength_model == 1.0
        patched = model.clone()
        key = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
        weight = model.model.state_dict()[key]
        out_dim, in_dim = int(weight.shape[0]), int(weight.numel() // weight.shape[0])
        up = torch.full((out_dim, 1), 0.001)
        down = torch.full((1, in_dim), 0.001)
        adapter = LoRAAdapter(set(), (up, down, 1.0, None, None, None))
        assert patched.add_patches({key: adapter}) == [key]
        patched.set_attachments('lora_metadata', _turbo4_metadata())
        cls.observations.append((key, len(patched.patches[key])))
        return io.NodeOutput(patched)


class RuntimeDualClockSetup(MiniMaxH3AudioRefineDualClockSetupT8Advanced):
    @classmethod
    def execute(cls, plan, model, positive, av_latent):
        result = refine_runtime.setup_audio_refine(
            plan=plan, model=model, positive=positive, av_latent=av_latent,
            runtime_snapshot_fn=_runtime)
        return io.NodeOutput(result.model, result.noise, result.guider,
                             result.sampler, result.sigmas, result.latent,
                             result.report_json)


class RuntimeDualModelSetup(MiniMaxH3AudioRefineDualModelSetupT8Advanced):
    @classmethod
    def execute(cls, plan, refine_model, positive, av_latent):
        result = refine_runtime.setup_audio_refine_dual_model(
            plan=plan, refine_model=refine_model, positive=positive,
            av_latent=av_latent, runtime_snapshot_fn=_runtime)
        return io.NodeOutput(result.model, result.noise, result.guider,
                             result.sampler, result.sigmas, result.latent,
                             result.report_json)


class CaptureLongDelivery(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='CaptureLongDelivery', is_output_node=True,
                         inputs=[io.Latent.Input('original'),
                                 io.Latent.Input('continuation'),
                                 io.Latent.Input('delivery'),
                                 io.String.Input('report_json')],
                         outputs=[io.String.Output('report_json')])

    @classmethod
    def execute(cls, original, continuation, delivery, report_json):
        cls.observations.append((original, continuation, delivery, report_json))
        return io.NodeOutput(report_json)


def test_cold_long_video_patch_authentication_requires_bound_declared_patch():
    from h3_audio_t8_pkg.audio_refine_advanced import _authenticated_compat_runtime

    class MiniMaxH3(torch.nn.Module):
        def extra_conds(self, **_kwargs):
            return {}

    base = MiniMaxH3()
    plain = ModelPatcher(base, load_device=torch.device('cpu'),
                         offload_device=torch.device('cpu'))
    declared = patch_long_video_model(plain)
    assert 'extra_conds' not in base.__dict__
    assert _authenticated_compat_runtime(declared, [], 'long_video')[
        'long_video_patch_version'] == 1
    with pytest.raises(RuntimeError, match='authenticated Long Video MODEL'):
        _authenticated_compat_runtime(plain, [], 'long_video')

    spoofed = plain.clone()
    fake = lambda **_kwargs: {}  # noqa: E731
    fake._t8_long_video_patch_version = 1
    spoofed.add_object_patch('extra_conds', fake)
    with pytest.raises(RuntimeError, match='authenticated Long Video MODEL'):
        _authenticated_compat_runtime(spoofed, [], 'long_video')

    base.extra_conds = declared.object_patches['extra_conds']
    assert _authenticated_compat_runtime(declared, [], 'long_video')[
        'long_video_patch_version'] == 1
    base.extra_conds = lambda **_kwargs: {}
    with pytest.raises(RuntimeError, match='authenticated Long Video MODEL'):
        _authenticated_compat_runtime(declared, [], 'long_video')


def _source(graph, target, field):
    nodes = {node['id']: node for node in graph['nodes']}
    links = {link[0]: link for link in graph['links']}
    item = next(item for item in nodes[target]['inputs'] if item['name'] == field)
    return links[item['link']][1:3]


def test_all_ten_saved_pairs_freeze_last_video_and_resume_only_audio_tail():
    pairs = pairs_from_current_sources()
    assert len(pairs) == 10
    for path, (freeze, resume) in pairs.items():
        for phase, graph in (('freeze_video', freeze), ('resume_audio', resume)):
            saved = TARGET / path.stem / (phase + '.json')
            assert json.loads(saved.read_text(encoding='utf8')) == graph
            ids = {node['id'] for node in graph['nodes']}
            assert len(ids) == len(graph['nodes'])
            assert all(link[1] in ids and link[3] in ids for link in graph['links'])
        source = json.loads(path.read_text(encoding='utf8'))
        last_video = _last_video_sampler(Draft(source))['id']
        save = next(node for node in freeze['nodes'] if node['type'] == CHECKPOINT_SAVE)
        if 'EAV' in path.name:
            audit = next(node for node in freeze['nodes']
                         if node['type'] == 'MiniMaxH3EnhanceAVideoAuditT8Advanced')
            assert _source(freeze, save['id'], 'av_latent') == [audit['id'], 0]
            assert _source(freeze, audit['id'], 'av_latent') == [last_video, 0]
        else:
            assert _source(freeze, save['id'], 'av_latent') == [last_video, 0]
        assert sum(node['type'] == 'SamplerCustomAdvanced' for node in resume['nodes']) == 1
        assert last_video not in {node['id'] for node in resume['nodes']}
        assert any(node['type'] == CHECKPOINT_LOAD for node in resume['nodes'])
        assert any(node['type'] == 'MiniMaxH3AudioRefineQualityGateT8Advanced'
                   for node in resume['nodes'])
        if path != RELAY_SOURCE:
            guard = next(node for node in resume['nodes']
                         if node['type'] == CHECKPOINT_GUARD)
            load = next(node for node in resume['nodes']
                        if node['type'] == CHECKPOINT_LOAD)
            assert _source(resume, guard['id'], 'av_latent') == [load['id'], 0]
            assert [(link[3], link[4]) for link in resume['links']
                    if link[1:3] == [load['id'], 0]] == [
                        (guard['id'], next(index for index, item in enumerate(guard['inputs'])
                                           if item['name'] == 'av_latent'))]
            source_gate = next(node for node in source['nodes']
                               if node['type'] == 'MiniMaxH3AudioRefineQualityGateT8Advanced')
            assert guard['widgets_values'] == [source_gate['widgets_values'][1]]
            gate = next(node for node in resume['nodes']
                        if node['type'] == 'MiniMaxH3AudioRefineQualityGateT8Advanced')
            assert _source(resume, gate['id'], 'video_frame_count') == [guard['id'], 3]
            conditioning = next((node for node in resume['nodes']
                                 if node['type'] == 'MiniMaxH3AudioConditioningT8'), None)
            if conditioning is None:
                assert 'Long_Video_Prompt_Relay' in path.name
            else:
                for field, slot in (('width', 1), ('height', 2), ('length', 3)):
                    assert _source(resume, conditioning['id'], field) == [guard['id'], slot]
        if 'Long_Video_Prompt_Relay' in path.name:
            context_save = next(node for node in freeze['nodes']
                                if node['type'] == 'MiniMaxH3LongVideoContextSaveT8')
            context_commit = next(node for node in freeze['nodes']
                                  if node['type'] == CONTEXT_COMMIT_GATE)
            planner = next(node for node in freeze['nodes']
                           if node['type'] == 'MiniMaxH3LongVideoPlannerT8')
            assert _source(freeze, context_save['id'], 'av_latent') == [save['id'], 0]
            assert _source(freeze, context_save['id'], 'save_context') == [
                context_commit['id'], 0]
            assert _source(freeze, context_commit['id'], 'checkpoint_status') == [save['id'], 1]
            assert _source(freeze, context_commit['id'], 'planned_save_context') == [
                planner['id'], 8]
            assert sum(node['type'] == 'SamplerCustomAdvanced'
                       for node in freeze['nodes']) == 1
            assert not any(node['type'] == 'MiniMaxH3LongVideoContextSaveT8'
                           for node in resume['nodes'])
            old_relay_ids = {node['id'] for node in source['nodes']
                             if 'PromptRelay' in node['type']}
            new_relay = [node for node in resume['nodes']
                         if 'PromptRelay' in node['type']]
            assert len(new_relay) == 3
            assert old_relay_ids.isdisjoint({node['id'] for node in new_relay})
            assert all('REFINE ONLY' in node['title'] for node in new_relay)


def test_strict_frozen_load_appends_geometry_without_changing_existing_slots(tmp_path, monkeypatch):
    video = torch.zeros(1, 24, 7, 8, 10)
    audio = torch.zeros(1, 32, 2, 37)
    latent = {'samples': comfy.nested_tensor.NestedTensor((video, audio))}
    checkpoint = save_native_h3_av_checkpoint(
        latent, tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    output = MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8.execute(
        checkpoint[2], checkpoint[4], checkpoint[3]).result
    assert output[1:4] == ('MATCH_EXTERNAL', True, 'audio_refine_firstpass')
    assert output[8:] == (160, 128)
    assert torch.equal(output[0]['samples'].unbind()[0], video)
    with pytest.raises(ValueError, match='SHA|checksum|hash'):
        MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8.execute(
            checkpoint[2], checkpoint[4], '0' * 64)
    with pytest.raises(ValueError, match='frame count'):
        MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8.execute(output[0], 124)
    guarded = MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8.execute(output[0], 22).result
    assert guarded[0] is output[0] and guarded[1:] == (160, 128, 22)


def test_long_context_only_commits_after_verified_frozen_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    latent = _native_av()
    store = tmp_path / 'MiniMaxH3' / 'latent_checkpoints'
    preview = save_native_h3_av_checkpoint(
        latent, store, filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=False)
    assert preview[1] == 'NOT_SAVED'
    assert MiniMaxH3AudioRefineContextCommitGateEXPT8.execute(
        preview[1], True).result == (False,)
    skipped = save_context_state(latent, 'audio_refine_long_context_test', 0,
                                 save_context=False)
    assert skipped[0] == '' and not list(tmp_path.rglob('*segment_00000*.safetensors'))
    committed = save_native_h3_av_checkpoint(
        latent, store, filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert committed[1] == 'SAVED_VERIFIED'
    assert MiniMaxH3AudioRefineContextCommitGateEXPT8.execute(
        committed[1], True).result == (True,)
    assert MiniMaxH3AudioRefineContextCommitGateEXPT8.execute(
        committed[1], False).result == (False,)
    with pytest.raises(ValueError, match='verified frozen checkpoint'):
        MiniMaxH3AudioRefineContextCommitGateEXPT8.execute('UNKNOWN', True)
    path, _ = save_context_state(committed[0], 'audio_refine_long_context_test', 0,
                                 save_context=True)
    assert Path(path).is_file()
    context, present, _ = load_context_state('audio_refine_long_context_test', 1)
    assert present and context['metadata']['source_segment_index'] == 0
    original_video, original_audio = latent['samples'].unbind()
    assert torch.equal(context['video_tail'], original_video)
    assert torch.equal(context['audio_tail'], original_audio)


def _run_eav_cold_resume(tmp_path, monkeypatch, checkpoint):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    candidate = next(TARGET.glob('*_Audio_Refine_EAV_Turbo8_*/resume_audio.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph['35']['inputs'].update(checkpoint_path=checkpoint[2],
                                 expected_manifest_json=checkpoint[4],
                                 expected_file_sha256=checkpoint[3])
    guard_id = next(node_id for node_id, node in graph.items()
                    if node['class_type'] == CHECKPOINT_GUARD)
    graph[guard_id]['inputs']['expected_video_frame_count'] = 22
    assert graph['6']['inputs']['width'] == [guard_id, 1]
    assert graph['6']['inputs']['height'] == [guard_id, 2]
    assert graph['6']['inputs']['length'] == [guard_id, 3]
    assert graph['28']['inputs']['video_frame_count'] == [guard_id, 3]
    graph['200'] = {'class_type': 'CaptureFullResume', 'inputs': {
        'original': ['35', 0], 'candidate': ['34', 0], 'selected': ['28', 0],
        'decision': ['28', 3], 'route_report': ['20', 3],
        'conditioning_report': ['6', 5]}}
    graph = _required_for(graph, '200')
    assert not {'7', '10', '13', '14'} & set(graph)
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE, RuntimeAudit,
                MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
                MiniMaxH3AudioRefineCompatibilityPlanT8Advanced, RuntimeSetup,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
                MiniMaxH3AVDecodeT8, MiniMaxH3AudioConditioningT8,
                CaptureFullResume, *STAGE_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFullResume.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-eav-cold-resume-cpu', execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFullResume.observations) == 1
    loaded, candidate_av, selected, decision, route_report, _ = (
        CaptureFullResume.observations[0])
    assert decision == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert json.loads(route_report)['decision'] == 'ALLOW'
    assert selected is loaded
    loaded_video, loaded_audio = loaded['samples'].unbind()
    candidate_video, candidate_audio = candidate_av['samples'].unbind()
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    assert not torch.cuda.is_initialized()
    return {'child_pid': os.getpid(), 'route': 'EAV_Turbo8',
            'model_loads': len(TinyRefineUNET.calls),
            'loaded_video_shape': list(loaded_video.shape),
            'loaded_audio_shape': list(loaded_audio.shape),
            'loaded_video_sha256': hashlib.sha256(
                loaded_video.detach().contiguous().numpy().tobytes()).hexdigest(),
            'loaded_audio_sha256': hashlib.sha256(
                loaded_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'candidate_audio_sha256': hashlib.sha256(
                candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'decision': decision}


def test_saved_eav_cold_resume_executes_audio_tail_without_original_video_pass(
        tmp_path, monkeypatch):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    assert _run_eav_cold_resume(tmp_path, monkeypatch, checkpoint)['route'] == 'EAV_Turbo8'


def _run_three_stage_cold_resume(tmp_path, monkeypatch, route, checkpoint,
                                 *, expected_frames=22, expected_canvas=128):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    candidate = next(TARGET.glob(f'*_Audio_Refine_{route}_*/resume_audio.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    guard_id = next(node_id for node_id, node in graph.items()
                    if node['class_type'] == CHECKPOINT_GUARD)
    assert graph['14']['inputs']['width'] == [guard_id, 1]
    assert graph['14']['inputs']['height'] == [guard_id, 2]
    graph['43']['inputs'].update(checkpoint_path=checkpoint[2],
                                 expected_manifest_json=checkpoint[4],
                                 expected_file_sha256=checkpoint[3])
    graph[guard_id]['inputs']['expected_video_frame_count'] = expected_frames
    assert graph['14']['inputs']['length'] == [guard_id, 3]
    assert graph['36']['inputs']['video_frame_count'] == [guard_id, 3]
    graph['200'] = {'class_type': 'CaptureFullResume', 'inputs': {
        'original': ['43', 0], 'candidate': ['42', 0], 'selected': ['36', 0],
        'decision': ['36', 3], 'route_report': ['28', 3],
        'conditioning_report': ['14', 5]}}
    graph = _required_for(graph, '200')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                TinyReferenceImage, MiniMaxH3AudioConditioningT8,
                RuntimeAudit, MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
                MiniMaxH3AudioRefineCompatibilityPlanT8Advanced, RuntimeSetup,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
                MiniMaxH3AVDecodeT8, CaptureFullResume, *STAGE_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFullResume.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-' + route + '-cold-resume-cpu', execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFullResume.observations) == 1
    loaded, candidate_av, selected, decision, route_report, conditioning_report = (
        CaptureFullResume.observations[0])
    assert decision == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert json.loads(route_report)['decision'] == 'ALLOW'
    assert f'canvas={expected_canvas}x{expected_canvas}' in conditioning_report
    assert f'frames={expected_frames}' in conditioning_report
    assert selected is loaded
    loaded_video, loaded_audio = loaded['samples'].unbind()
    candidate_video, candidate_audio = candidate_av['samples'].unbind()
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    assert not torch.cuda.is_initialized()
    return {'child_pid': os.getpid(), 'route': route,
            'loaded_video_shape': list(loaded_video.shape),
            'loaded_audio_shape': list(loaded_audio.shape),
            'loaded_video_sha256': hashlib.sha256(
                loaded_video.detach().contiguous().numpy().tobytes()).hexdigest(),
            'loaded_audio_sha256': hashlib.sha256(
                loaded_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'candidate_audio_sha256': hashlib.sha256(
                candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'model_loads': len(TinyRefineUNET.calls), 'decision': decision}


@pytest.mark.parametrize('route', ('Learned_TwoPass_Final8', 'PDD_Ref2VA_4Plus4'))
def test_saved_three_stage_cold_resume_uses_authenticated_geometry_and_audio_tail(
        tmp_path, monkeypatch, route):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    receipt = _run_three_stage_cold_resume(tmp_path, monkeypatch, route, checkpoint)
    assert receipt['route'] == route


PLAIN_COMPAT_ROUTES = [
    ('PDD_Ref2VA8', '33', '34', '6', '26', '22', '18'),
    ('Turbo8_Plus_Refine4', '37', '38', '8', '25', '21', '17'),
]


def _run_plain_compat_cold_resume(tmp_path, monkeypatch, route, checkpoint):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    (_, load_id, guard_id, conditioning_id, quality_id, candidate_id,
     route_id) = next(spec for spec in PLAIN_COMPAT_ROUTES if spec[0] == route)
    candidate = next(TARGET.glob(f'*_Audio_Refine_{route}_*/resume_audio.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph[load_id]['inputs'].update(checkpoint_path=checkpoint[2],
                                   expected_manifest_json=checkpoint[4],
                                   expected_file_sha256=checkpoint[3])
    graph[guard_id]['inputs']['expected_video_frame_count'] = 22
    assert [graph[conditioning_id]['inputs'][field]
            for field in ('width', 'height', 'length')] == [
                [guard_id, 1], [guard_id, 2], [guard_id, 3]]
    assert graph[quality_id]['inputs']['video_frame_count'] == [guard_id, 3]
    graph['200'] = {'class_type': 'CaptureFullResume', 'inputs': {
        'original': [load_id, 0], 'candidate': [candidate_id, 0],
        'selected': [quality_id, 0], 'decision': [quality_id, 3],
        'route_report': [route_id, 3],
        'conditioning_report': [conditioning_id, 5]}}
    graph = _required_for(graph, '200')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCompatCLIP, TinyRefineVAE,
                TinyReferenceImage, TinyClipProjApply,
                MiniMaxH3ClipProjCompatibilityAuditT8Advanced,
                MiniMaxH3AudioConditioningT8,
                RuntimeAudit, MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
                MiniMaxH3AudioRefineCompatibilityPlanT8Advanced, RuntimeSetup,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
                MiniMaxH3AVDecodeT8, CaptureFullResume, *STAGE_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFullResume.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-' + route + '-cold-resume-cpu', execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFullResume.observations) == 1
    loaded, candidate_av, selected, decision, route_report, conditioning_report = (
        CaptureFullResume.observations[0])
    assert decision == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert json.loads(route_report)['decision'] == 'ALLOW'
    assert 'canvas=128x128' in conditioning_report
    assert 'frames=22' in conditioning_report
    assert selected is loaded
    loaded_video, loaded_audio = loaded['samples'].unbind()
    candidate_video, candidate_audio = candidate_av['samples'].unbind()
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    assert not torch.cuda.is_initialized()
    return {'child_pid': os.getpid(), 'route': route,
            'model_loads': len(TinyRefineUNET.calls),
            'loaded_video_shape': list(loaded_video.shape),
            'candidate_audio_sha256': hashlib.sha256(
                candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'decision': decision}


@pytest.mark.parametrize('route', [spec[0] for spec in PLAIN_COMPAT_ROUTES])
def test_saved_plain_compat_cold_resume_uses_frozen_geometry_and_only_audio_tail(
        tmp_path, monkeypatch, route):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    result = _run_plain_compat_cold_resume(tmp_path, monkeypatch, route, checkpoint)
    assert result['route'] == route


TURBO4_PHASE2_ROUTES = [
    ('Phase2_Base_Refine4', '37', '38', '8', '25', '21', ['17', 3]),
    ('Phase2_Same_Turbo4', '37', '38', '8', '25', '21', ['17', 3]),
    ('Turbo4_Plus_Refine4', '36', '37', '6', '24', '18', ['15', 2]),
]


def _run_turbo4_and_phase2_cold_resume(tmp_path, monkeypatch, route, checkpoint):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    (_, load_id, guard_id, conditioning_id, quality_id, candidate_id,
     report_source) = next(spec for spec in TURBO4_PHASE2_ROUTES if spec[0] == route)
    candidate = next(TARGET.glob(f'*_Audio_Refine_{route}_*/resume_audio.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph[load_id]['inputs'].update(checkpoint_path=checkpoint[2],
                                   expected_manifest_json=checkpoint[4],
                                   expected_file_sha256=checkpoint[3])
    graph[guard_id]['inputs']['expected_video_frame_count'] = 22
    assert [graph[conditioning_id]['inputs'][field]
            for field in ('width', 'height', 'length')] == [
                [guard_id, 1], [guard_id, 2], [guard_id, 3]]
    assert graph[quality_id]['inputs']['video_frame_count'] == [guard_id, 3]
    graph['200'] = {'class_type': 'CaptureFullResume', 'inputs': {
        'original': [load_id, 0], 'candidate': [candidate_id, 0],
        'selected': [quality_id, 0], 'decision': [quality_id, 3],
        'route_report': report_source,
        'conditioning_report': [conditioning_id, 5]}}
    graph = _required_for(graph, '200')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCompatCLIP, TinyRefineVAE,
                TinyClipProjApply, TinyTurbo4Lora,
                MiniMaxH3ClipProjCompatibilityAuditT8Advanced,
                MiniMaxH3AudioConditioningT8, MiniMaxH3DualClockSamplerT8,
                RuntimeAudit, MiniMaxH3AudioRefineModelRouteT8Advanced,
                MiniMaxH3AudioRefinePhase2PlanT8Advanced,
                MiniMaxH3AudioRefinePlanT8Advanced,
                RuntimeDualModelSetup, RuntimeDualClockSetup,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
                MiniMaxH3AVDecodeT8, CaptureFullResume, *STAGE_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFullResume.observations.clear()
    TinyRefineUNET.calls.clear()
    TinyTurbo4Lora.observations.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-' + route + '-cold-resume-cpu', execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert TinyTurbo4Lora.observations == [
        ('diffusion_model.blocks.0.attn.qkv_proj.weight', 1)]
    assert len(CaptureFullResume.observations) == 1
    loaded, candidate_av, selected, decision, route_report, conditioning_report = (
        CaptureFullResume.observations[0])
    assert decision == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert json.loads(route_report)['decision'] == 'ALLOW'
    assert 'canvas=128x128' in conditioning_report
    assert 'frames=22' in conditioning_report
    assert selected is loaded
    loaded_video, loaded_audio = loaded['samples'].unbind()
    candidate_video, candidate_audio = candidate_av['samples'].unbind()
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    assert not torch.cuda.is_initialized()
    return {'child_pid': os.getpid(), 'route': route,
            'model_loads': len(TinyRefineUNET.calls),
            'lora_patches': len(TinyTurbo4Lora.observations),
            'loaded_video_shape': list(loaded_video.shape),
            'loaded_audio_shape': list(loaded_audio.shape),
            'candidate_audio_sha256': hashlib.sha256(
                candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'decision': decision}


@pytest.mark.parametrize('route', [spec[0] for spec in TURBO4_PHASE2_ROUTES])
def test_saved_turbo4_and_phase2_cold_resume_runs_only_audio_tail(
        tmp_path, monkeypatch, route):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    result = _run_turbo4_and_phase2_cold_resume(tmp_path, monkeypatch, route, checkpoint)
    assert result['route'] == route
    assert result['model_loads'] == 1
    assert result['lora_patches'] == 1


@pytest.mark.parametrize('route', [spec[0] for spec in TURBO4_PHASE2_ROUTES])
def test_saved_turbo4_and_phase2_tail_cold_resumes_in_new_core_process(tmp_path, route):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_turbo4_and_phase2_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_turbo4_and_phase2_cold_resume(
        Path(payload['output_root']), patch, payload['route'], payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path), 'route': route,
               'checkpoint': [None, None, checkpoint[2], checkpoint[3], checkpoint[4]]}
    child = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['route'] == route
    assert result['model_loads'] == 1
    assert result['lora_patches'] == 1
    assert result['loaded_video_shape'] == [1, 24, 7, 8, 8]
    assert result['loaded_audio_shape'] == [1, 32, 2, 37]
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert len(result['candidate_audio_sha256']) == 64
    assert not torch.cuda.is_initialized()


def test_pdd_three_stage_tail_cold_resumes_in_new_core_process(tmp_path):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_three_stage_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_three_stage_cold_resume(
        Path(payload['output_root']), patch, payload['route'], payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path), 'route': 'PDD_Ref2VA_4Plus4',
               'checkpoint': [None, None, checkpoint[2], checkpoint[3], checkpoint[4]]}
    child = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['route'] == 'PDD_Ref2VA_4Plus4'
    assert result['loaded_video_shape'] == [1, 24, 7, 8, 8]
    assert result['model_loads'] == 1
    assert len(result['candidate_audio_sha256']) == 64
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('route', (
    'EAV_Turbo8', 'Learned_TwoPass_Final8', 'PDD_Ref2VA8',
    'Turbo8_Plus_Refine4',
))
def test_saved_remaining_audio_tail_cold_resumes_in_new_core_process(tmp_path, route):
    checkpoint = save_native_h3_av_checkpoint(
        _native_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import (
    _run_eav_cold_resume, _run_plain_compat_cold_resume,
    _run_three_stage_cold_resume,
)
import pytest
payload = json.load(sys.stdin)
route = payload['route']
with pytest.MonkeyPatch.context() as patch:
    args = (Path(payload['output_root']), patch, payload['checkpoint'])
    if route == 'EAV_Turbo8':
        result = _run_eav_cold_resume(*args)
    elif route == 'Learned_TwoPass_Final8':
        result = _run_three_stage_cold_resume(args[0], args[1], route, args[2])
    else:
        result = _run_plain_compat_cold_resume(args[0], args[1], route, args[2])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path), 'route': route,
               'checkpoint': [None, None, checkpoint[2], checkpoint[3], checkpoint[4]]}
    child = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['route'] == route
    assert result['model_loads'] == 1
    assert result['loaded_video_shape'] == [1, 24, 7, 8, 8]
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert len(result['candidate_audio_sha256']) == 64
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('route,conditioning_id,save_id', (
    ('Turbo8_Plus_Refine4', '8', '35'),
    ('Phase2_Base_Refine4', '8', '35'),
    ('Phase2_Same_Turbo4', '8', '35'),
    ('Turbo4_Plus_Refine4', '6', '34'),
))
def test_saved_simple_turbo_freeze_graph_then_new_core_only_refines(
        tmp_path, monkeypatch, route, conditioning_id, save_id):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    candidate = next(TARGET.glob(f'*_Audio_Refine_{route}_*/freeze_video.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph[conditioning_id]['inputs'].update(width=128, height=128, length=22)
    graph[save_id]['inputs']['confirm_save'] = True
    graph['200'] = {'class_type': 'CaptureFrozenFirstPass', 'inputs': {
        'latent': [save_id, 0], 'status': [save_id, 1],
        'checkpoint_path': [save_id, 2], 'file_sha256': [save_id, 3],
        'manifest_json': [save_id, 4]}}
    graph = _required_for(graph, '200')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCompatCLIP, TinyRefineVAE,
                TinyClipProjApply, TinyTurbo4Lora,
                MiniMaxH3ClipProjCompatibilityAuditT8Advanced,
                MiniMaxH3AudioConditioningT8, MiniMaxH3DualClockSamplerT8,
                custom_sampler_nodes.BasicGuider, custom_sampler_nodes.RandomNoise,
                custom_sampler_nodes.SamplerCustomAdvanced,
                checkpoint_nodes.MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                CaptureFrozenFirstPass):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFrozenFirstPass.observations.clear()
    TinyRefineUNET.calls.clear()
    TinyTurbo4Lora.observations.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-' + route + '-saved-freeze-cpu', execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert TinyTurbo4Lora.observations == [
        ('diffusion_model.blocks.0.attn.qkv_proj.weight', 1)]
    assert len(CaptureFrozenFirstPass.observations) == 1
    latent, status, path, file_sha256, manifest_json = (
        CaptureFrozenFirstPass.observations[0])
    assert status == 'SAVED_VERIFIED'
    assert len(file_sha256) == 64
    assert json.loads(manifest_json)['checkpoint_id'] == 'audio_refine_firstpass'
    video, audio = latent['samples'].unbind()
    assert list(video.shape) == [1, 24, 7, 8, 8]
    assert list(audio.shape) == [1, 32, 2, 37]
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import (
    _run_plain_compat_cold_resume, _run_turbo4_and_phase2_cold_resume,
)
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    if payload['route'] == 'Turbo8_Plus_Refine4':
        result = _run_plain_compat_cold_resume(
            Path(payload['output_root']), patch, payload['route'],
            payload['checkpoint'])
    else:
        result = _run_turbo4_and_phase2_cold_resume(
            Path(payload['output_root']), patch, payload['route'],
            payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path), 'route': route,
               'checkpoint': [None, None, path, file_sha256, manifest_json]}
    child = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['route'] == route
    assert result['model_loads'] == 1
    assert result['loaded_video_shape'] == list(video.shape)
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert len(result['candidate_audio_sha256']) == 64
    assert not torch.cuda.is_initialized()


def test_saved_learned_freeze_graph_runs_real_3d_upscaler(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly

    candidate = next(TARGET.glob('*_Audio_Refine_Learned_TwoPass_Final8_*/'
                                 'freeze_video.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph['5']['inputs']['strength_model'] = 0.0
    # Keep the published 3D network and checkpoint, but use the smallest
    # valid H3 video grid so a real CPU forward is practical in this suite.
    graph['7']['inputs'].update(width=32, height=32, length=5)
    graph['14']['inputs']['length'] = 5
    graph['13']['inputs'].update(precision='fp32', release_policy='clear_after')
    graph['41']['inputs']['confirm_save'] = True
    graph['200'] = {'class_type': 'CaptureFrozenFirstPass', 'inputs': {
        'latent': ['41', 0], 'status': ['41', 1],
        'checkpoint_path': ['41', 2], 'file_sha256': ['41', 3],
        'manifest_json': ['41', 4]}}
    graph = _required_for(graph, '200')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 2
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                TinyReferenceImage, LoraLoaderBypassModelOnly,
                MiniMaxH3AudioConditioningT8, MiniMaxH3DualClockSamplerT8,
                MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
                MiniMaxH3LearnedLatentUpscaleT8Advanced,
                MiniMaxH3TwoPassLatentReconcileT8Advanced,
                MiniMaxH3TwoPassDetailMixerT8Advanced,
                custom_sampler_nodes.BasicGuider, custom_sampler_nodes.RandomNoise,
                custom_sampler_nodes.SamplerCustomAdvanced,
                checkpoint_nodes.MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                CaptureFrozenFirstPass):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFrozenFirstPass.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-learned-saved-freeze-real-3d-cpu',
                     execute_outputs=['200'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFrozenFirstPass.observations) == 1
    latent, status, path, file_sha256, manifest_json = (
        CaptureFrozenFirstPass.observations[0])
    assert status == 'SAVED_VERIFIED'
    assert path and len(file_sha256) == 64
    assert json.loads(manifest_json)['checkpoint_id'] == 'audio_refine_firstpass'
    video, audio = latent['samples'].unbind()
    assert list(video.shape) == [1, 24, 2, 4, 4]
    assert list(audio.shape[:3]) == [1, 32, 2]
    assert audio.shape[-1] > 0
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert not torch.cuda.is_initialized()
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_three_stage_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_three_stage_cold_resume(
        Path(payload['output_root']), patch, 'Learned_TwoPass_Final8',
        payload['checkpoint'], expected_frames=5, expected_canvas=64)
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path),
               'checkpoint': [None, None, path, file_sha256, manifest_json]}
    child = subprocess.run([sys.executable, '-c', code],
                           cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['route'] == 'Learned_TwoPass_Final8'
    assert result['loaded_video_shape'] == list(video.shape)
    assert result['loaded_audio_shape'] == list(audio.shape)
    assert result['loaded_video_sha256'] == hashlib.sha256(
        video.detach().contiguous().numpy().tobytes()).hexdigest()
    assert result['loaded_audio_sha256'] == hashlib.sha256(
        audio.detach().contiguous().numpy().tobytes()).hexdigest()
    assert result['model_loads'] == 1
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'


def test_saved_eav_freeze_graph_audits_real_first_pass_before_cold_audio_tail(
        tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly

    candidate = next(TARGET.glob('*_Audio_Refine_EAV_Turbo8_*/freeze_video.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph['6']['inputs'].update(width=128, height=128, length=22)
    graph['18']['inputs']['strength_model'] = 0.0
    # The saved tau=4 candidate is intentionally fail-closed at g>1.5 on
    # random tiny weights; keep the same apply_exp path with a mild test tau.
    graph['13']['inputs']['tau'] = 0.2
    graph['33']['inputs']['confirm_save'] = True
    graph['200'] = {'class_type': 'CaptureFrozenFirstPass', 'inputs': {
        'latent': ['33', 0], 'status': ['33', 1],
        'checkpoint_path': ['33', 2], 'file_sha256': ['33', 3],
        'manifest_json': ['33', 4]}}
    graph['201'] = {'class_type': 'CaptureEAVFreezeAudit',
                    'inputs': {'eav_report': ['14', 1]}}
    graph = _required_for(graph, '200', '201')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyFullDepthEAVUNET, TinyCandidateCLIP, TinyRefineVAE,
                LoraLoaderBypassModelOnly, MiniMaxH3AudioConditioningT8,
                MiniMaxH3DualClockSamplerT8, MiniMaxH3EnhanceAVideoT8Advanced,
                MiniMaxH3EnhanceAVideoAuditT8Advanced,
                custom_sampler_nodes.BasicGuider, custom_sampler_nodes.RandomNoise,
                custom_sampler_nodes.SamplerCustomAdvanced,
                checkpoint_nodes.MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                CaptureFrozenFirstPass, CaptureEAVFreezeAudit):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFrozenFirstPass.observations.clear()
    CaptureEAVFreezeAudit.observations.clear()
    TinyFullDepthEAVUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-eav-saved-freeze-cpu', execute_outputs=['200', '201'])
    assert executor.success, executor.status_messages
    assert len(TinyFullDepthEAVUNET.calls) == 1
    assert len(CaptureFrozenFirstPass.observations) == 1
    assert len(CaptureEAVFreezeAudit.observations) == 1
    eav_report = CaptureEAVFreezeAudit.observations[0]
    assert eav_report['config']['mode'] == 'apply_exp'
    assert eav_report['status'] == 'apply_exp_verified'
    assert eav_report['model_forward_count'] == 8
    assert eav_report['attention_calls_per_active_forward'] == [50] * 8
    latent, status, path, file_sha256, manifest_json = (
        CaptureFrozenFirstPass.observations[0])
    assert status == 'SAVED_VERIFIED'
    assert path and len(file_sha256) == 64
    video, audio = latent['samples'].unbind()
    assert list(video.shape) == [1, 24, 7, 8, 8]
    assert list(audio.shape) == [1, 32, 2, 37]
    assert not torch.cuda.is_initialized()
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_eav_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_eav_cold_resume(
        Path(payload['output_root']), patch, payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path),
               'checkpoint': [None, None, path, file_sha256, manifest_json]}
    child = subprocess.run([sys.executable, '-c', code],
                           cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['route'] == 'EAV_Turbo8'
    assert result['model_loads'] == 1
    assert result['loaded_video_shape'] == list(video.shape)
    assert result['loaded_audio_shape'] == list(audio.shape)
    assert result['loaded_video_sha256'] == hashlib.sha256(
        video.detach().contiguous().numpy().tobytes()).hexdigest()
    assert result['loaded_audio_sha256'] == hashlib.sha256(
        audio.detach().contiguous().numpy().tobytes()).hexdigest()
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'


@pytest.mark.parametrize('confirm_save', (False, True))
def test_saved_long_relay_freeze_graph_commits_context_only_after_verified_av(
        tmp_path, monkeypatch, confirm_save):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly

    candidate = next(TARGET.glob('*_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_*/'
                                 'freeze_video.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph['5']['inputs'].update(length=22, time_ranges='0-6\n7-13\n14-21')
    graph['6']['inputs'].update(chain_id='s26_long_freeze_tiny',
                                new_duration_seconds=0.75,
                                minimum_render_frames=22)
    graph['9']['inputs'].update(width=128, height=128, query_chunk_rows=64)
    graph['10']['inputs']['strength_model'] = 0.0
    graph['39']['inputs']['confirm_save'] = confirm_save
    graph['200'] = {'class_type': 'CaptureFrozenFirstPass', 'inputs': {
        'latent': ['39', 0], 'status': ['39', 1],
        'checkpoint_path': ['39', 2], 'file_sha256': ['39', 3],
        'manifest_json': ['39', 4]}}
    graph = _required_for(graph, '200', '15')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    assert graph['15']['inputs']['save_context'] == ['40', 0]
    assert graph['40']['inputs']['checkpoint_status'] == ['39', 1]
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
                MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
                MiniMaxH3LongVideoPlannerT8, MiniMaxH3LongVideoContextLoadT8,
                MiniMaxH3LongVideoContextSaveT8, LoraLoaderBypassModelOnly,
                MiniMaxH3DualClockSamplerT8, custom_sampler_nodes.BasicGuider,
                custom_sampler_nodes.RandomNoise,
                custom_sampler_nodes.SamplerCustomAdvanced,
                checkpoint_nodes.MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3AudioRefineContextCommitGateEXPT8,
                CaptureFrozenFirstPass):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFrozenFirstPass.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-long-freeze-' + str(confirm_save),
                     execute_outputs=['200', '15'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFrozenFirstPass.observations) == 1
    latent, status, path, file_sha256, manifest_json = (
        CaptureFrozenFirstPass.observations[0])
    video, audio = latent['samples'].unbind()
    assert list(video.shape) == [1, 24, 7, 8, 8]
    assert list(audio.shape) == [1, 32, 2, 37]
    context_files = list(tmp_path.rglob('*.context.safetensors'))
    if confirm_save:
        assert status == 'SAVED_VERIFIED'
        assert path and len(file_sha256) == 64
        assert json.loads(manifest_json)['checkpoint_id'] == 'audio_refine_firstpass'
        assert len(context_files) == 1
        context, present, _ = load_context_state('s26_long_freeze_tiny', 1)
        assert present
        assert torch.equal(context['video_tail'], video)
        assert torch.equal(context['audio_tail'], audio)
        code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_long_relay_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_long_relay_cold_resume(
        Path(payload['output_root']), patch, payload['checkpoint'],
        expected_video_frames=22,
        expected_video_sha256=payload['video_sha256'],
        expected_audio_sha256=payload['audio_sha256'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
        payload = {
            'output_root': str(tmp_path),
            'checkpoint': [None, None, path, file_sha256, manifest_json],
            'video_sha256': hashlib.sha256(
                video.detach().contiguous().numpy().tobytes()).hexdigest(),
            'audio_sha256': hashlib.sha256(
                audio.detach().contiguous().numpy().tobytes()).hexdigest(),
        }
        child = subprocess.run([sys.executable, '-c', code],
                               cwd=Path(__file__).resolve().parents[1],
                               input=json.dumps(payload), capture_output=True,
                               text=True, timeout=90, check=False)
        assert child.returncode == 0, child.stderr
        result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                                 if line.startswith('RESULT=')))
        assert result['child_pid'] != os.getpid()
        assert result['loaded_video_shape'] == list(video.shape)
        assert result['loaded_audio_shape'] == list(audio.shape)
        assert result['loaded_video_sha256'] == payload['video_sha256']
        assert result['loaded_audio_sha256'] == payload['audio_sha256']
        assert result['model_loads'] == 1
        assert result['segment_index'] == 0
        assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    else:
        assert status == 'NOT_SAVED'
        assert not path and not context_files
    assert not torch.cuda.is_initialized()


def _long_tiny_frozen_av():
    video = torch.linspace(-0.1, 0.1, 24 * 37 * 8 * 8).reshape(1, 24, 37, 8, 8)
    audio = torch.linspace(-0.1, 0.1, 32 * 2 * 207).reshape(1, 32, 2, 207)
    return {'samples': comfy.nested_tensor.NestedTensor((video, audio))}


def _run_long_relay_cold_resume(tmp_path, monkeypatch, checkpoint,
                                expected_video_frames=124, expected_video_sha256=None,
                                expected_audio_sha256=None):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes

    candidate = next(TARGET.glob('*_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_*/'
                                 'resume_audio.api.json'))
    graph = deepcopy(json.loads(candidate.read_text(encoding='utf8')))
    graph['44']['inputs'].update(checkpoint_path=checkpoint[2],
                                 expected_manifest_json=checkpoint[4],
                                 expected_file_sha256=checkpoint[3])
    graph['6']['inputs']['chain_id'] = 's26_audio_refine_long_relay_tiny'
    if expected_video_frames == 22:
        graph['6']['inputs'].update(chain_id='s26_long_freeze_tiny',
                                    new_duration_seconds=0.75,
                                    minimum_render_frames=22)
        graph['41']['inputs'].update(length=22, time_ranges='0-6\n7-13\n14-21')
    graph['45']['inputs']['expected_video_frame_count'] = expected_video_frames
    graph['43']['inputs'].update(width=128, height=128, query_chunk_rows=64)
    graph['200'] = {'class_type': 'CaptureFullResume', 'inputs': {
        'original': ['44', 0], 'candidate': ['40', 0], 'selected': ['32', 0],
        'decision': ['32', 3], 'route_report': ['24', 3],
        'conditioning_report': ['43', 6]}}
    graph['201'] = {'class_type': 'CaptureLongDelivery', 'inputs': {
        'original': ['44', 0], 'continuation': ['36', 0],
        'delivery': ['36', 1], 'report_json': ['36', 2]}}
    graph = _required_for(graph, '200', '201')
    assert sum(node['class_type'] == 'SamplerCustomAdvanced'
               for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, 'get_output_directory',
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, 'intermediate_device',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, 'prepare_callback',
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                MiniMaxH3LongVideoPlannerT8, MiniMaxH3LongVideoContextLoadT8,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
                MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
                RuntimeAudit, MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
                MiniMaxH3AudioRefineCompatibilityPlanT8Advanced, RuntimeSetup,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
                MiniMaxH3AVDecodeT8, CaptureFullResume, CaptureLongDelivery,
                *STAGE_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, 'define_schema') else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    CaptureFullResume.observations.clear()
    CaptureLongDelivery.observations.clear()
    TinyRefineUNET.calls.clear()
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                        send_sync=lambda *_args, **_kwargs: None),
        cache_args={'ram': 0., 'ram_inactive': 0.},
        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, 's26-long-relay-cold-resume-cpu', execute_outputs=['200', '201'])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureFullResume.observations) == 1
    assert len(CaptureLongDelivery.observations) == 1
    loaded, candidate_av, selected, decision, route_report, conditioning_report = (
        CaptureFullResume.observations[0])
    assert decision == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert json.loads(route_report)['decision'] == 'ALLOW'
    assert json.loads(conditioning_report)['status'] == 'applied_exp'
    assert selected is loaded
    candidate_video, candidate_audio = candidate_av['samples'].unbind()
    original_loaded, continuation, delivery, report = CaptureLongDelivery.observations[0]
    loaded_video, loaded_audio = original_loaded['samples'].unbind()
    if expected_video_frames == 124:
        video, audio = _long_tiny_frozen_av()['samples'].unbind()
        assert torch.equal(loaded_video, video)
        assert torch.equal(loaded_audio, audio)
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    loaded_video_sha256 = hashlib.sha256(
        loaded_video.detach().contiguous().numpy().tobytes()).hexdigest()
    loaded_audio_sha256 = hashlib.sha256(
        loaded_audio.detach().contiguous().numpy().tobytes()).hexdigest()
    if expected_video_sha256 is not None:
        assert loaded_video_sha256 == expected_video_sha256
    if expected_audio_sha256 is not None:
        assert loaded_audio_sha256 == expected_audio_sha256
    continuation_video, continuation_audio = continuation['samples'].unbind()
    delivery_video, delivery_audio = delivery['samples'].unbind()
    assert torch.equal(continuation_video, loaded_video)
    assert torch.equal(continuation_audio, loaded_audio)
    assert torch.equal(delivery_video, loaded_video)
    assert torch.equal(delivery_audio, loaded_audio)
    delivery_report = json.loads(report)
    assert delivery_report['segment_index'] == 0
    assert delivery_report['candidate_selected'] is False
    assert delivery_report['next_segment_input'] == 'continuation_av_latent_only'
    assert not torch.cuda.is_initialized()
    return {'child_pid': os.getpid(), 'model_loads': len(TinyRefineUNET.calls),
            'loaded_video_shape': list(loaded_video.shape),
            'loaded_audio_shape': list(loaded_audio.shape),
            'loaded_video_sha256': loaded_video_sha256,
            'loaded_audio_sha256': loaded_audio_sha256,
            'candidate_audio_sha256': hashlib.sha256(
                candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'continuation_audio_sha256': hashlib.sha256(
                continuation_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            'decision': decision, 'segment_index': delivery_report['segment_index']}


def test_saved_long_relay_cold_resume_runs_audio_tail_and_preserves_continuation(
        tmp_path, monkeypatch):
    checkpoint = save_native_h3_av_checkpoint(
        _long_tiny_frozen_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    result = _run_long_relay_cold_resume(tmp_path, monkeypatch, checkpoint)
    assert result['model_loads'] == 1
    assert result['segment_index'] == 0


def test_saved_long_relay_tail_cold_resumes_in_new_core_process(tmp_path):
    checkpoint = save_native_h3_av_checkpoint(
        _long_tiny_frozen_av(), tmp_path / 'MiniMaxH3' / 'latent_checkpoints',
        filename_prefix='audio_refine_firstpass',
        checkpoint_id='audio_refine_firstpass', confirm_save=True)
    assert checkpoint[1] == 'SAVED_VERIFIED'
    code = '''
import json, os, runpy, sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_resume_all import _run_long_relay_cold_resume
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_long_relay_cold_resume(Path(payload['output_root']), patch,
                                         payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {'output_root': str(tmp_path),
               'checkpoint': [None, None, checkpoint[2], checkpoint[3], checkpoint[4]]}
    child = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines()
                             if line.startswith('RESULT=')))
    assert result['child_pid'] != os.getpid()
    assert result['model_loads'] == 1
    assert result['loaded_video_shape'] == [1, 24, 37, 8, 8]
    assert result['loaded_audio_shape'] == [1, 32, 2, 207]
    assert result['decision'] == 'ABSTAIN_HUMAN_REVIEW_REQUIRED'
    assert len(result['candidate_audio_sha256']) == 64
    assert len(result['continuation_audio_sha256']) == 64
    assert result['segment_index'] == 0
    assert not torch.cuda.is_initialized()
