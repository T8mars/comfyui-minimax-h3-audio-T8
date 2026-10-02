"""S29 public RF workflows keep independent explicit BASE and RESTART stages."""
import json

import pytest

from tools import build_formal_rf_split_workflows as formal


@pytest.mark.parametrize("path", sorted(formal.generated()))
def test_s29_public_graph_matches_builder_and_separates_rf_stages(path):
    graph = formal.generated()[path]
    assert json.loads(path.read_text(encoding="utf8")) == graph
    kinds = [node["type"] for node in graph["nodes"]]
    assert kinds.count("MiniMaxH3RFRestartStageSetupEXPT8") == 1
    if "resume_effects" in path.name:
        assert "MiniMaxH3StageLoadEXPT8" in kinds
        assert "MiniMaxH3RFBaseStageSetupEXPT8" not in kinds
    else:
        assert kinds.count("MiniMaxH3RFBaseStageSetupEXPT8") == 1
    assert "MiniMaxH3RFHandoffEXPT8" in kinds
    assert "MiniMaxH3StageSamplerEXPT8" in kinds
