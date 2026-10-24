"""Independent re-selection contracts: foreign wrapper order must not change."""

import pytest
import torch
from comfy.patcher_extension import CallbacksMP, WrappersMP

from h3_audio_t8_pkg import h3_projection_chunks as projection
from h3_audio_t8_pkg import projection_chunks_advanced as projected
from h3_audio_t8_pkg import int8_row_chunk as int8
from h3_audio_t8_pkg import int8_row_chunk_advanced as quantized
from test_h21_int8_row_chunk import actual_int8_model
from test_h21_projection_model import sample


@pytest.mark.parametrize("kind", ["projection", "int8"])
def test_reselect_preserves_real_foreign_diffusion_wrapper_order(tmp_path, kind):
    feature, plan, active = (
        (projected, projection.ProjectionPlan, projection._ACTIVE)
        if kind == "projection" else
        (quantized, int8.Int8RowPlan, int8._ACTIVE)
    )
    with torch.inference_mode():
        original = actual_int8_model(tmp_path)
        selected, _ = feature.configure_projection_chunks(original, plan(rows=7)) if kind == "projection" else feature.configure_int8_row_chunks(original, plan(rows=7))
        seen = []

        def foreign(executor, *args, **kwargs):
            seen.append(bool(active.get()))
            return executor(*args, **kwargs)

        selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, feature.KEY, foreign)
        sample(selected)
        assert seen == [True, True]
        seen.clear()
        replaced, _ = feature.configure_projection_chunks(selected, plan(rows=11)) if kind == "projection" else feature.configure_int8_row_chunks(selected, plan(rows=11))
        sample(replaced)
        assert seen == [True, True], "Re-selection moved a user wrapper outside the original scope"
        assert selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, feature.KEY)[1] is foreign
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["projection", "int8"])
def test_reselect_keeps_foreign_lifecycle_callbacks_at_exact_relative_slots(kind):
    from test_fast_h3_v2_core_sampler import model

    feature, plan, configure = (
        (projected, projection.ProjectionPlan, projected.configure_projection_chunks)
        if kind == "projection" else
        (quantized, int8.Int8RowPlan, quantized.configure_int8_row_chunks)
    )
    selected, _ = configure(model(), plan(rows=7))

    class Foreign:
        __hash__ = None

        def __eq__(self, other):
            raise AssertionError("Owner inspection must use identity, never user equality")

        def __call__(self, *args, **kwargs):
            pass

    before, after = Foreign(), Foreign()
    for kind in (CallbacksMP.ON_PRE_RUN, CallbacksMP.ON_CLEANUP):
        bucket = selected.callbacks[kind][feature.KEY]
        bucket.insert(0, before)
        bucket.append(after)
    replaced, _ = configure(selected, plan(rows=11))
    for kind in (CallbacksMP.ON_PRE_RUN, CallbacksMP.ON_CLEANUP):
        bucket = replaced.callbacks[kind][feature.KEY]
        assert len(bucket) == 3 and bucket[0] is before and bucket[2] is after
        assert bucket[1] is not selected.callbacks[kind][feature.KEY][1]
