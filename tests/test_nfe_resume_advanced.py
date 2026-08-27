from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

import comfy.model_sampling
import comfy.nested_tensor

from h3_audio_t8_pkg.nfe_resume_advanced import (
    NFE_RESUME_METADATA_KEY,
    _runtime_signature,
    fingerprint_nfe_resume_checkpoint,
    read_nfe_resume_checkpoint,
    sample_minimax_h3_dual_clock_euler_resumable,
    setup_nfe_resume_sampling,
)
from h3_audio_t8_pkg.nodes_nfe_resume_advanced import (
    MiniMaxH3NFEResumeSamplerT8Advanced,
)
from h3_audio_t8_pkg.sampling import (
    native_flow_sigmas,
    sample_minimax_h3_dual_clock_euler,
)


class RuntimeModel:
    def __init__(self, noise, latent_image, *, fail_on_call=None):
        self.noise = noise.clone()
        self.latent_image = latent_image.clone()
        self.sigmas = None
        self.inner_model = None
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, x, sigma, **_kwargs):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected sampler interruption")
        scale = sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
        return x * (0.2 + 0.1 * scale) + 0.05


class _SignaturePatcher:
    def __init__(self):
        self.model = object()
        self.object_patches = {"model_sampling": object()}
        self.patches = {"diffusion_model.block.0.weight": [(1.0, None)]}


class _SignatureGuider:
    def __init__(self):
        self.model_patcher = _SignaturePatcher()
        self.cfg = 1.0


class _SignatureModel:
    def __init__(self):
        self.inner_model = _SignatureGuider()


def test_runtime_signature_binds_patch_keys_and_value_types_without_serializing_values():
    model = _SignatureModel()
    first = _runtime_signature(
        model,
        {
            "model_options": {
                "wrapper": 1,
                "transformer_options": {"patches_replace": {}, "sample_sigmas": object()},
            }
        },
    )
    second = _runtime_signature(
        model,
        {
            "model_options": {
                "wrapper": [],
                "transformer_options": {"patches_replace": [], "sample_sigmas": object()},
            }
        },
    )

    assert first["model_option_keys"] == second["model_option_keys"]
    assert first["model_option_types"] != second["model_option_types"]
    assert first["transformer_option_types"] != second["transformer_option_types"]
    assert "sample_sigmas" not in first["transformer_option_types"]
    assert first["object_patch_types"]["model_sampling"] == "builtins.object"
    original_digest = first["weight_patch_keys_sha256"]
    model.inner_model.model_patcher.patches = {
        "diffusion_model.block.1.weight": [(1.0, None)]
    }
    changed = _runtime_signature(model, {"model_options": {}})
    assert changed["weight_patch_key_count"] == 1
    assert changed["weight_patch_keys_sha256"] != original_digest


def _checkpoint_config(
    tmp_path: Path,
    full_sigmas: torch.Tensor,
    *,
    resume_state=None,
    write_enabled=True,
):
    return {
        "full_sigmas": full_sigmas,
        "resume_state": resume_state,
        "write_enabled": write_enabled,
        "target": tmp_path / "state.h3nfe.safetensors",
        "session_id": (
            resume_state["payload"]["session_id"]
            if resume_state is not None
            else "0123456789abcdef0123456789abcdef"
        ),
        "model_contract_id": "base=sha256:model; lora=none; wrappers=none",
        "run_contract_sha256": "A" * 64,
        "max_chunk_bytes": 1024 * 1024,
        "allow_replace_existing": resume_state is not None,
    }


def _sample(
    model,
    x,
    sigmas,
    config,
    *,
    denoise_mask=None,
):
    return sample_minimax_h3_dual_clock_euler_resumable(
        model,
        x,
        sigmas,
        extra_args={"seed": 42, "denoise_mask": denoise_mask, "model_options": {}},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
        checkpoint_config=config,
    )


def test_disabled_resumable_loop_is_bit_exact_with_stable_dual_clock_euler(tmp_path):
    sigmas = native_flow_sigmas(4, 12.0)
    noise = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
    latent = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    start = noise.clone()
    stable = sample_minimax_h3_dual_clock_euler(
        RuntimeModel(noise, latent),
        start.clone(),
        sigmas,
        extra_args={"seed": 42},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
    )
    config = _checkpoint_config(
        tmp_path,
        sigmas,
        write_enabled=False,
    )
    resumed_loop = _sample(RuntimeModel(noise, latent), start.clone(), sigmas, config)
    assert torch.equal(resumed_loop, stable)
    assert not config["target"].exists()


def test_interrupted_run_resumes_from_last_atomic_boundary_bit_exactly(tmp_path):
    sigmas = native_flow_sigmas(4, 12.0)
    noise = torch.tensor([[0.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
    latent = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    denoise_mask = torch.tensor([[1.0, 0.5, 0.25, 0.75]], dtype=torch.float32)
    start = noise.clone()
    reference = _sample(
        RuntimeModel(noise, latent),
        start.clone(),
        sigmas,
        _checkpoint_config(tmp_path / "reference", sigmas, write_enabled=False),
        denoise_mask=denoise_mask,
    )

    interrupted_config = _checkpoint_config(tmp_path, sigmas)
    with pytest.raises(RuntimeError, match="injected sampler interruption"):
        _sample(
            RuntimeModel(noise, latent, fail_on_call=3),
            start.clone(),
            sigmas,
            interrupted_config,
            denoise_mask=denoise_mask,
        )
    state = read_nfe_resume_checkpoint(tmp_path, "state.h3nfe.safetensors")
    assert state["payload"]["completed_steps"] == 2
    assert state["payload"]["remaining_steps"] == 2
    assert state["payload"]["has_denoise_mask"] is True
    assert len(state["file_sha256"]) == 64
    assert fingerprint_nfe_resume_checkpoint(
        tmp_path, "state.h3nfe.safetensors"
    ) == state["file_sha256"]

    resume_config = _checkpoint_config(
        tmp_path,
        sigmas,
        resume_state=state,
        write_enabled=False,
    )
    output = _sample(
        RuntimeModel(torch.zeros_like(noise), torch.zeros_like(latent)),
        torch.full_like(start, 999.0),
        sigmas[2:],
        resume_config,
    )
    assert torch.equal(output, reference)


def test_resume_rejects_seed_change_before_running_model(tmp_path):
    sigmas = native_flow_sigmas(2, 12.0)
    noise = torch.ones((1, 4), dtype=torch.float32)
    latent = torch.zeros_like(noise)
    config = _checkpoint_config(tmp_path, sigmas)
    model = RuntimeModel(noise, latent, fail_on_call=2)
    with pytest.raises(RuntimeError, match="injected sampler interruption"):
        _sample(model, noise.clone(), sigmas, config)
    state = read_nfe_resume_checkpoint(tmp_path, "state.h3nfe.safetensors")
    resumed_model = RuntimeModel(noise, latent)
    with pytest.raises(ValueError, match="resume seed"):
        sample_minimax_h3_dual_clock_euler_resumable(
            resumed_model,
            noise.clone(),
            sigmas[1:],
            extra_args={"seed": 43, "model_options": {}},
            disable=True,
            video_values=2,
            packed_values=4,
            shift_video=12.0,
            shift_audio=3.0,
            audio_velocity_is_raw=True,
            checkpoint_config=_checkpoint_config(
                tmp_path,
                sigmas,
                resume_state=state,
                write_enabled=False,
            ),
        )
    assert resumed_model.calls == 0


def test_valid_safetensors_payload_tamper_is_detected(tmp_path):
    sigmas = native_flow_sigmas(2, 12.0)
    noise = torch.ones((1, 4), dtype=torch.float32)
    latent = torch.zeros_like(noise)
    _sample(
        RuntimeModel(noise, latent),
        noise.clone(),
        sigmas,
        _checkpoint_config(tmp_path, sigmas),
    )
    target = tmp_path / "state.h3nfe.safetensors"
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    tensors["state_x"][0, 0] += 1
    save_file(tensors, str(target), metadata=metadata)
    assert NFE_RESUME_METADATA_KEY in metadata
    with pytest.raises(ValueError, match="digest mismatch"):
        read_nfe_resume_checkpoint(tmp_path, target.name)


class FakeModelConfig:
    sampling_settings = {"shift": 1.0, "multiplier": 1000}


class FakeBaseModel:
    model_config = FakeModelConfig()


class FakeModelPatcher:
    def __init__(self):
        self.model = FakeBaseModel()
        self.model_options = {}
        self.objects = {
            "model_sampling": comfy.model_sampling.ModelSamplingDiscreteFlow(
                self.model.model_config
            )
        }

    def clone(self):
        cloned = FakeModelPatcher.__new__(FakeModelPatcher)
        cloned.model = self.model
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.objects = self.objects.copy()
        return cloned

    def get_model_object(self, name):
        return self.objects[name]

    def add_object_patch(self, name, value):
        self.objects[name] = value


def _av_latent():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def test_setup_is_disabled_by_default_and_checkpoint_mode_requires_consent(tmp_path):
    result = setup_nfe_resume_sampling(
        FakeModelPatcher(),
        _av_latent(),
        steps=4,
        shift_video=12.0,
        shift_audio=3.0,
        mode="disabled",
        checkpoint_path="state.h3nfe.safetensors",
        model_contract_id="",
        run_contract_json="{}",
        confirm_checkpoint_write=False,
        allow_replace_existing=False,
        hash_chunk_megabytes=8,
        storage_root=tmp_path,
    )
    assert result[3:5] == ("DISABLED", "")
    assert len(result[2]) == 5
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(ValueError, match="confirm_checkpoint_write"):
        setup_nfe_resume_sampling(
            FakeModelPatcher(),
            _av_latent(),
            steps=4,
            shift_video=12.0,
            shift_audio=3.0,
            mode="checkpoint_each_step",
            checkpoint_path="state.h3nfe.safetensors",
            model_contract_id="sha256:model",
            run_contract_json=json.dumps({"prompt": "test"}),
            confirm_checkpoint_write=False,
            allow_replace_existing=False,
            hash_chunk_megabytes=8,
            storage_root=tmp_path,
        )


def test_paths_contracts_and_node_schema_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        setup_nfe_resume_sampling(
            FakeModelPatcher(),
            _av_latent(),
            steps=4,
            shift_video=12.0,
            shift_audio=3.0,
            mode="checkpoint_each_step",
            checkpoint_path="../state.h3nfe.safetensors",
            model_contract_id="sha256:model",
            run_contract_json=json.dumps({"prompt": "test"}),
            confirm_checkpoint_write=True,
            allow_replace_existing=False,
            hash_chunk_megabytes=8,
            storage_root=tmp_path,
        )
    with pytest.raises(ValueError, match="non-empty JSON"):
        setup_nfe_resume_sampling(
            FakeModelPatcher(),
            _av_latent(),
            steps=4,
            shift_video=12.0,
            shift_audio=3.0,
            mode="checkpoint_each_step",
            checkpoint_path="state.h3nfe.safetensors",
            model_contract_id="sha256:model",
            run_contract_json="{}",
            confirm_checkpoint_write=True,
            allow_replace_existing=False,
            hash_chunk_megabytes=8,
            storage_root=tmp_path,
        )

    schema = MiniMaxH3NFEResumeSamplerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["mode"].default == "disabled"
    assert inputs["confirm_checkpoint_write"].default is False
    assert inputs["allow_replace_existing"].default is False
    assert [output.id for output in schema.outputs] == [
        "model",
        "sampler",
        "sigmas",
        "status",
        "checkpoint_path",
        "report_json",
    ]


def test_completed_boundary_resumes_bit_exactly_in_a_new_process(tmp_path):
    worker = Path(__file__).with_name("multiprocess_nfe_resume_worker.py")
    store = tmp_path / "store"
    handoff = tmp_path / "handoff.json"
    env = os.environ.copy()
    comfy_root = Path(__file__).resolve().parents[3]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(comfy_root) + (os.pathsep + existing if existing else "")
    saved = subprocess.run(
        [sys.executable, str(worker), "save", str(store), str(handoff)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    resumed = subprocess.run(
        [sys.executable, str(worker), "resume", str(store), str(handoff)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    saved_result = json.loads(saved.stdout.strip().splitlines()[-1])
    resumed_result = json.loads(resumed.stdout.strip().splitlines()[-1])
    assert saved_result["completed_steps"] == 2
    assert resumed_result["completed_steps"] == 2
    assert resumed_result["bit_exact"] is True
    assert resumed_result["save_pid"] == saved_result["save_pid"]
    assert resumed_result["resume_pid"] != resumed_result["save_pid"]
    assert resumed_result["file_sha256"] == saved_result["file_sha256"]


def test_frontend_workflow_is_importable_documented_and_safe_by_default():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-23_H3_Dual_Clock_NFE_Checkpoint_Resume_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    sampler = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3NFEResumeSamplerT8Advanced"
    )
    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    contract = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3NFERunContractT8Advanced"
    )
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sampler["widgets_values"][3] == "disabled"
    assert sampler["widgets_values"][7:9] == [False, False]
    run_contract_input = next(
        item for item in sampler["inputs"] if item["name"] == "run_contract_json"
    )
    run_contract_slot = sampler["inputs"].index(run_contract_input)
    contract_link = run_contract_input["link"]
    assert contract_link is not None
    assert any(
        link[:5] == [
            contract_link,
            contract["id"],
            0,
            sampler["id"],
            run_contract_slot,
        ]
        and link[5] == "STRING"
        for link in workflow["links"]
    )
    expected_contract_sources = {
        (conditioning["id"], 0, contract["id"], 0, "CONDITIONING"),
        (conditioning["id"], 3, contract["id"], 1, "STRING"),
        (conditioning["id"], 4, contract["id"], 2, "STRING"),
        (conditioning["id"], 5, contract["id"], 3, "STRING"),
    }
    actual_contract_sources = {
        (link[1], link[2], link[3], link[4], link[5])
        for link in workflow["links"]
        if link[3] == contract["id"]
    }
    assert actual_contract_sources == expected_contract_sources
    notes = [node for node in nodes.values() if node["type"] == "MarkdownNote"]
    assert len(notes) >= 5
    combined = "\n".join(str(node["widgets_values"]) for node in notes)
    assert "dual_clock_euler + native_flow" in combined
    assert "逐位一致" in combined
    assert "不恢复" in combined
