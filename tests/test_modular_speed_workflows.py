"""SPEED full-chain and explicit later-stage candidate dependencies."""

import pytest

from tools import build_modular_speed_workflow as builder


@pytest.mark.parametrize("stages,effects,resume", [
    (2, False, None), (2, True, None),
    (2, False, 1), (2, True, 1),
    (3, False, 1), (3, False, 2), (3, True, None), (3, True, 1), (3, True, 2),
])
def test_speed_split_and_resume_graphs_prune_old_stage_execution(stages, effects, resume):
    graph = builder.split_graph(stages, with_effects=effects, resume_stage=resume)
    assert graph["82"]["class_type"] == "SaveVideo"
    models = [node for node in graph.values() if node["class_type"] == "UNETLoader"]
    samplers = [node for node in graph.values()
                if node["class_type"] == "MiniMaxH3SPEEDStageSampleEXPT8"]
    assert len(models) == len(samplers) == stages - (resume or 0)
    if resume is None:
        assert not any(node["class_type"] == "MiniMaxH3SPEEDStageLoadEXPT8" for node in graph.values())
    else:
        assert graph["5"]["class_type"] == "MiniMaxH3SPEEDStageLoadEXPT8"
        assert graph[str(12 + 12 * resume)]["inputs"]["previous_spec"] == ["5", 1]
        assert graph[str(15 + 12 * resume)]["inputs"]["completed_stage"] == ["5", 0]
        assert all(str(10 + 12 * index) not in graph for index in range(resume))
    saves = [node for node in graph.values()
             if node["class_type"] == "MiniMaxH3SPEEDStageSaveEXPT8"]
    assert len(saves) == stages - (resume or 0) - 1
