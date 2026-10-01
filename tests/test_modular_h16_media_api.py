"""Saved H16 resume API must include the frontend's real AV media terminal."""

from copy import deepcopy
import json

import pytest

from tools.build_modular_h16_media_api import PAIRS, completed_resume_api, source_pair


@pytest.mark.parametrize("route", ["plain", "effects"])
def test_h16_media_api_preserves_original_sampling_graph(route):
    freeze_front, freeze_api, resume_front, resume_api = source_pair(route)
    config = PAIRS[route]
    freeze, resume = config["names"]
    for name, original in ((f"{freeze}.json", freeze_front),
                           (f"{freeze}.api.json", freeze_api),
                           (f"{resume}.json", resume_front)):
        path = config["target"] / name
        assert json.loads(path.read_bytes()) == original
        assert path.read_bytes() == (config["source"] / name).read_bytes()
    final = completed_resume_api(route)
    saved = json.loads((config["target"] / f"{resume}.api.json").read_bytes())
    assert saved == final
    assert {key: value for key, value in final.items() if key not in {"20", "21"}} == resume_api
    assert final["20"]["inputs"] == {
        "av_latent": [str(config["final_av"][0]), 0],
        "video_vae": ["1", 0], "audio_vae": ["2", 0]}
    assert final["21"]["inputs"]["images"] == ["20", 0]
    assert final["21"]["inputs"]["audio"] == ["20", 1]
    assert final["21"]["inputs"]["save_output"] is True
    assert not any(value["class_type"] == "SamplerCustomAdvanced"
                   for value in final.values())


def test_h16_media_api_rejects_tampered_decode_link(monkeypatch):
    from tools import build_modular_h16_media_api as builder

    saved = builder.source_pair

    def modified(route):
        freeze_front, freeze_api, resume_front, resume_api = saved(route)
        resume_front = deepcopy(resume_front)
        node = next(node for node in resume_front["nodes"] if node["id"] == 20)
        node["inputs"][0]["link"] = None
        return freeze_front, freeze_api, resume_front, resume_api

    monkeypatch.setattr(builder, "source_pair", modified)
    with pytest.raises((KeyError, ValueError, TypeError)):
        builder.completed_resume_api("plain")
