from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.nfe_run_contract_advanced import (
    NFE_RUN_CONTRACT_SCHEMA,
    compile_nfe_run_contract,
)
from h3_audio_t8_pkg.nodes_nfe_run_contract_advanced import (
    MiniMaxH3NFERunContractT8Advanced,
)


MEDIA_MAP = json.dumps(
    {
        "pictures": {"1": "first_frame"},
        "videos": {},
        "audios": {"1": "drive_audio (primary source)"},
        "source_audio_ordinal": 1,
    }
)


def _positive(*, offset: float = 0.0):
    return [
        [
            torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) + offset,
            {
                "pooled_output": torch.arange(8, dtype=torch.float32).reshape(1, 8),
                "minimax_refs": [
                    {
                        "kind": "audio",
                        "ref_audio_t": 4,
                        "audio_latent": torch.ones((1, 2, 4), dtype=torch.float16),
                    }
                ],
            },
        ]
    ]


def _compile(**overrides):
    values = {
        "positive": _positive(),
        "conditioned_prompt": "A woman speaks. <Audio 1>",
        "media_map_json": MEDIA_MAP,
        "conditioning_report": "task=I2VA\nframes=22\ncanvas=256x256",
        "hash_chunk_megabytes": 1,
    }
    values.update(overrides)
    return compile_nfe_run_contract(**values)


def test_contract_is_deterministic_canonical_and_chunk_size_invariant():
    first = _compile(hash_chunk_megabytes=1)
    second = _compile(hash_chunk_megabytes=8)

    assert first[:2] == second[:2]
    payload = json.loads(first[0])
    report = json.loads(first[2])
    assert payload["schema"] == NFE_RUN_CONTRACT_SCHEMA
    assert payload["conditioned_prompt"] == "A woman speaks. <Audio 1>"
    assert payload["positive_conditioning"]["tensor_count"] == 3
    assert payload["positive_conditioning"]["tensor_bytes"] > 0
    assert report["contract_sha256"] == first[1]
    assert len(first[1]) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"positive": _positive(offset=1.0)},
        {"conditioned_prompt": "A woman whispers. <Audio 1>"},
        {
            "media_map_json": json.dumps(
                {
                    "pictures": {"1": "last_frame"},
                    "videos": {},
                    "audios": {"1": "drive_audio (primary source)"},
                    "source_audio_ordinal": 1,
                }
            )
        },
        {"conditioning_report": "task=I2VA\nframes=39\ncanvas=256x256"},
    ],
)
def test_contract_hash_changes_when_bound_generation_content_changes(overrides):
    baseline = _compile()[1]
    assert _compile(**overrides)[1] != baseline


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"media_map_json": "not-json"}, "media_map_json is invalid JSON"),
        ({"media_map_json": "[]"}, "root must be a JSON object"),
        ({"positive": [[object(), {}]]}, "unsupported runtime object"),
        ({"positive": []}, "at least one conditioning entry"),
        (
            {"positive": [[[float("nan")], {}]]},
            "non-finite float",
        ),
    ],
)
def test_contract_rejects_ambiguous_or_unhashable_content(overrides, message):
    with pytest.raises(ValueError, match=message):
        _compile(**overrides)


def test_contract_rejects_cycles_in_conditioning_metadata():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="contains a cycle"):
        _compile(positive=[[torch.zeros((1, 1, 1)), {"cycle": cyclic}]])


def test_node_schema_is_append_only_experimental_and_safe():
    schema = MiniMaxH3NFERunContractT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3NFERunContractT8Advanced"
    assert schema.is_experimental is True
    assert [item.id for item in schema.inputs] == [
        "positive",
        "conditioned_prompt",
        "media_map_json",
        "conditioning_report",
        "hash_chunk_megabytes",
    ]
    assert [item.id for item in schema.outputs] == [
        "run_contract_json",
        "contract_sha256",
        "report_json",
    ]
    chunk = schema.inputs[4]
    assert chunk.default == 8
    assert chunk.min == 1
    assert chunk.max == 64
