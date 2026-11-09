import json

import kornia.color
import pytest
import torch

from h3_audio_t8_pkg.masked_color_stabilize import stabilize_masked_color
from h3_audio_t8_pkg.nodes_masked_color_stabilize import (
    MiniMaxH3MaskedColorStabilizeEXPT8,
)


def rgb(lab):
    return kornia.color.lab_to_rgb(lab.permute(0, 3, 1, 2), clip=False).permute(
        0, 2, 3, 1
    )


def lab(image):
    return kornia.color.rgb_to_lab(image[..., :3].permute(0, 3, 1, 2)).permute(
        0, 2, 3, 1
    )


def fixture():
    source = torch.zeros(4, 16, 16, 3)
    source[..., 0] = torch.arange(50.0, 54.0)[:, None, None]
    source = rgb(source)
    generated = torch.zeros(4, 16, 16, 3)
    generated[..., 0] = torch.tensor([32.0, 58.0, 35.0, 55.0])[:, None, None]
    generated[..., 1] = torch.tensor([5.0, 17.0, 8.0, 12.0])[:, None, None]
    generated[..., 2] = torch.tensor([-15.0, -8.0, -23.0, -19.0])[:, None, None]
    # Retain a measurable within-frame fabric-like gradient.
    generated[..., 0] += torch.linspace(-2, 2, 16)[None, None, :]
    generated = rgb(generated)
    reference = torch.zeros(1, 16, 16, 3)
    reference[..., 0] = 45
    reference[..., 1] = 12
    reference[..., 2] = -22
    mask = torch.zeros(4, 16, 16)
    mask[:, 3:13, 3:13] = 1
    return generated, source, mask, rgb(reference)


def test_known_flicker_removed_without_frame_blending_or_background_changes():
    generated, source, mask, reference = fixture()
    before = generated.clone()
    source_before = source.clone()
    mask_before = mask.clone()
    output, report = stabilize_masked_color(generated, source, mask, reference)
    values = lab(output)[:, 3:13, 3:13].mean((1, 2))
    torch.testing.assert_close(
        values[:, 0], torch.arange(45.0, 49.0), atol=0.003, rtol=0
    )
    torch.testing.assert_close(
        values[:, 1:], torch.tensor([[12.0, -22.0]]).expand(4, 2), atol=0.003, rtol=0
    )
    old_detail = lab(generated)[:, 3:13, 3:13]
    new_detail = lab(output)[:, 3:13, 3:13]
    torch.testing.assert_close(
        new_detail - new_detail.mean((1, 2), keepdim=True),
        old_detail - old_detail.mean((1, 2), keepdim=True),
        atol=0.005,
        rtol=0,
    )
    assert torch.equal(output[mask == 0], generated[mask == 0])
    assert (
        torch.equal(generated, before)
        and torch.equal(source, source_before)
        and torch.equal(mask, mask_before)
    )
    receipt = json.loads(report)
    assert receipt["applied"] and not receipt["temporal_frame_blending"]
    assert receipt["operation"] == "explicit_generated_RGB_postprocess"
    assert receipt["RGB_gamut_clipped_values"] == 0
    assert not receipt["sampling_or_model_quality_pass_inferred"]


def test_disabled_and_empty_masks_return_original_object_without_colour_roundtrip():
    generated, source, mask, reference = fixture()
    result, report = stabilize_masked_color(
        generated, source, mask, reference, strength=0
    )
    assert result is generated and not json.loads(report)["applied"]
    result, report = stabilize_masked_color(generated, source, mask * 0, reference)
    assert result is generated and not json.loads(report)["applied"]


def test_empty_reference_is_reported_as_bypass_not_fake_success():
    generated, source, mask, reference = fixture()
    mask[0] = 0
    result, report = stabilize_masked_color(generated, source, mask, reference)
    assert result is generated and json.loads(report)["reason"].startswith(
        "reference_frame_mask_empty"
    )


def test_soft_masks_strength_and_extra_alpha_are_preserved():
    generated, source, mask, reference = fixture()
    original = torch.cat((generated, torch.full_like(generated[..., :1], 0.37)), dim=-1)
    corrected, _ = stabilize_masked_color(original, source, mask, reference)
    mixed, _ = stabilize_masked_color(
        original, source, mask * 0.5, reference, strength=0.5
    )
    torch.testing.assert_close(
        mixed[..., :3],
        generated * 0.75 + corrected[..., :3] * 0.25,
        atol=0.0001,
        rtol=0,
    )
    assert torch.equal(mixed[..., 3], original[..., 3])


def test_optional_constant_lighting_and_chroma_bypass():
    generated, source, mask, reference = fixture()
    result, _ = stabilize_masked_color(
        generated,
        source,
        mask,
        reference,
        follow_source_lighting=False,
        stabilize_chroma=False,
    )
    actual, original = lab(result), lab(generated)
    torch.testing.assert_close(
        actual[:, 3:13, 3:13, 0].mean((1, 2)),
        torch.full((4,), 45.0),
        atol=0.003,
        rtol=0,
    )
    torch.testing.assert_close(
        actual[:, 3:13, 3:13, 1:], original[:, 3:13, 3:13, 1:], atol=0.003, rtol=0
    )


def test_extreme_reference_adjustment_maps_gamut_without_nonfinite_or_geometry_change():
    generated, source, mask, reference = fixture()
    source[1] = 0.9  # Large actual source-lighting change, still normalized sRGB.
    result, raw = stabilize_masked_color(generated, source, mask, reference)
    report = json.loads(raw)
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    assert report["chroma_compressed_pixels"] > 0
    assert torch.equal(result[mask == 0], generated[mask == 0])


@pytest.mark.parametrize(
    "kind", ["shape", "nan", "HDR", "source_index", "target_index", "weight"]
)
def test_actual_input_errors_explain_contract(kind):
    generated, source, mask, reference = fixture()
    kwargs = {}
    if kind == "shape":
        source = source[:1]
    if kind == "nan":
        generated[0, 0, 0, 0] = float("nan")
    if kind == "HDR":
        generated[0, 0, 0, 0] = 2
    if kind == "source_index":
        kwargs["source_reference_frame"] = 4
    if kind == "target_index":
        kwargs["target_reference_index"] = 1
    if kind == "weight":
        kwargs["strength"] = float("nan")
    with pytest.raises(ValueError):
        stabilize_masked_color(generated, source, mask, reference, **kwargs)


def test_schema_explicitly_labels_postprocessing_and_model_independence():
    schema = MiniMaxH3MaskedColorStabilizeEXPT8.define_schema()
    assert schema.is_experimental and "显式后处理" in schema.display_name
    assert not any(item.id in ["model", "audio", "av_latent"] for item in schema.inputs)
    assert any(item.id == "strength" and item.default == 1 for item in schema.inputs)


def test_source_recolour_stops_local_shading_flicker_that_has_identical_global_means():
    generated, source, mask, reference = fixture()
    # Opposite local light changes have the same global mean in each frame.
    generated_lab = lab(generated)
    pattern = torch.tensor([[-12.0] * 8 + [12.0] * 8]).expand(16, 16)
    for i in range(4):
        generated_lab[i, ..., 0] = 45 + pattern * (1 if i % 2 else -1)
        generated_lab[i, ..., 1:] = torch.tensor([12.0, -22.0])
    generated = rgb(generated_lab)
    old, _ = stabilize_masked_color(generated, source, mask, reference)
    fixed, receipt = stabilize_masked_color(
        generated, source, mask, reference, detail_source="source_recolour"
    )
    old_lab, fixed_lab = lab(old), lab(fixed)
    assert abs(float(old_lab[1, 7, 5, 0] - old_lab[0, 7, 5, 0])) > 15
    # The real source illumination increases by one Lab unit per frame.
    torch.testing.assert_close(
        fixed_lab[1:, 3:13, 3:13] - fixed_lab[:-1, 3:13, 3:13],
        torch.tensor([1.0, 0.0, 0.0]).expand(3, 10, 10, 3),
        atol=0.004,
        rtol=0,
    )
    report = json.loads(receipt)
    assert (
        report["source_texture_recolour"] and not report["generated_texture_retained"]
    )
    assert not report["temporal_frame_blending"]
    torch.testing.assert_close(
        torch.tensor([i["Lab_offset"] for i in report["corrections"]]),
        torch.tensor(report["corrections"][0]["Lab_offset"]).expand(4, 3),
        atol=1e-5,
        rtol=0,
    )
    assert torch.equal(fixed[mask == 0], generated[mask == 0])


def test_source_recolour_preserves_real_source_detail_not_generated_replacement():
    generated, source, mask, reference = fixture()
    source_lab = lab(source)
    source_lab[..., 0] += torch.linspace(-3, 3, 16)[None, None, :]
    source = rgb(source_lab)
    before = source.clone()
    result, _ = stabilize_masked_color(
        generated, source, mask, reference, detail_source="source_recolour"
    )
    actual = lab(result)[:, 3:13, 3:13]
    detail = lab(source)[:, 3:13, 3:13]
    torch.testing.assert_close(
        actual - actual.mean((1, 2), keepdim=True),
        detail - detail.mean((1, 2), keepdim=True),
        atol=0.004,
        rtol=0,
    )
    assert torch.equal(source, before)
    # A different generated texture does not substitute for source cloth details.
    other, _ = stabilize_masked_color(
        generated.flip(2), source, mask, reference, detail_source="source_recolour"
    )
    torch.testing.assert_close(other[mask == 1], result[mask == 1], atol=1e-6, rtol=0)


def test_source_recolour_respects_lighting_and_chroma_controls():
    generated, source, mask, reference = fixture()
    result, _ = stabilize_masked_color(
        generated,
        source,
        mask,
        reference,
        detail_source="source_recolour",
        follow_source_lighting=False,
        stabilize_chroma=False,
    )
    actual = lab(result)[:, 3:13, 3:13]
    torch.testing.assert_close(
        actual[..., 0].mean((1, 2)), torch.full((4,), 45.0), atol=0.003, rtol=0
    )
    torch.testing.assert_close(
        actual[..., 1:], lab(generated)[:, 3:13, 3:13, 1:], atol=0.004, rtol=0
    )


def test_source_recolour_soft_mask_alpha_bypass_and_legacy_default():
    generated, source, mask, reference = fixture()
    original = torch.cat((generated, torch.full_like(generated[..., :1], 0.3)), dim=-1)
    full, _ = stabilize_masked_color(
        original, source, mask, reference, detail_source="source_recolour"
    )
    mixed, _ = stabilize_masked_color(
        original,
        source,
        mask * 0.5,
        reference,
        strength=0.5,
        detail_source="source_recolour",
    )
    torch.testing.assert_close(
        mixed[..., :3],
        torch.where(
            (mask > 0)[..., None],
            source * 0.75 + full[..., :3] * 0.25,
            generated,
        ),
        atol=0.0001,
        rtol=0,
    )
    alternate, _ = stabilize_masked_color(
        1 - original[..., :3],
        source,
        mask * 0.5,
        reference,
        strength=0.5,
        detail_source="source_recolour",
    )
    # Soft edges must not reintroduce the model's changing cloth colours.
    torch.testing.assert_close(
        alternate[mask > 0], mixed[..., :3][mask > 0], atol=1e-6, rtol=0
    )
    assert torch.equal(mixed[..., 3], original[..., 3])
    bypass, _ = stabilize_masked_color(
        original, source, mask, reference, strength=0, detail_source="source_recolour"
    )
    assert bypass is original
    old, _ = stabilize_masked_color(original, source, mask, reference)
    explicit, _ = stabilize_masked_color(
        original, source, mask, reference, detail_source="generated"
    )
    assert torch.equal(old, explicit)


def test_source_recolour_input_and_schema_are_explicit_not_silent_texture_substitution():
    generated, source, mask, reference = fixture()
    with pytest.raises(ValueError, match="detail_source"):
        stabilize_masked_color(
            generated, source, mask, reference, detail_source="automatic"
        )
    option = next(
        i
        for i in MiniMaxH3MaskedColorStabilizeEXPT8.define_schema().inputs
        if i.id == "detail_source"
    )
    assert option.optional and option.default == "generated"
