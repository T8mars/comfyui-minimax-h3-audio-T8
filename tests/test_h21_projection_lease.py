"""Actual shared-network lock ordering, without mutating active R1 selection."""
import threading

import torch

from h3_audio_t8_pkg.h3_projection_chunks import (
    ProjectionPlan, projection_scope, shared_network_lease,
)
from test_h21_projection_model import model, sample


def test_lease_inside_selected_projection_does_not_suspend_active_plan():
    bare = model()
    network = bare.model.diffusion_model
    with projection_scope(network, ProjectionPlan(rows=5)) as observed:
        with shared_network_lease(network) as acquired:
            assert acquired
            sample(bare)
    assert observed.calls["norm_qkv.fused"]["max_rows"] <= 5
    assert observed.calls["norm_qkv.fused"]["calls"] > 0
    assert not torch.cuda.is_initialized()


def test_lease_only_has_no_forward_writes_and_nested_projection_is_reentrant():
    bare = model()
    network = bare.model.diffusion_model
    before = {name: vars(module).get("forward") for name, module in network.named_modules()}
    with shared_network_lease(network) as acquired:
        assert acquired
        assert {name: vars(module).get("forward") for name, module in network.named_modules()} == before
        with projection_scope(network, ProjectionPlan(rows=7)) as observed:
            sample(bare)
        assert observed.calls["norm_qkv.fused"]["max_rows"] <= 7
    assert {name: vars(module).get("forward") for name, module in network.named_modules()} == before


def test_shared_lease_blocks_other_cooperating_thread_then_releases_after_exception():
    network = model().model.diffusion_model
    entered, finished, errors = threading.Event(), threading.Event(), []

    def run():
        entered.set()
        try:
            with shared_network_lease(network) as acquired:
                assert acquired
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    try:
        with shared_network_lease(network):
            thread = threading.Thread(target=run)
            thread.start()
            assert entered.wait(2)
            assert not finished.wait(.05)
            raise RuntimeError("actual lease owner exception")
    except RuntimeError as error:
        assert str(error) == "actual lease owner exception"
    thread.join(3)
    assert finished.is_set() and not errors and not thread.is_alive()
