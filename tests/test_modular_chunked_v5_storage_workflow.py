"""Freeze/resume drafts expose only the requested v5 stages and exact receipts."""
import ast
import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from build_modular_chunked_v5_storage_workflow import build_pair  # noqa: E402
from build_modular_chunked_v5_workflow import TEMPLATE  # noqa: E402
from h3_audio_t8_pkg.chunked_two_pass_upscale_advanced import compute_temporal_segments  # noqa: E402
from h3_audio_t8_pkg.modular_sampling import node_classes  # noqa: E402
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage_nodes import NODES as v1_storage_nodes  # noqa: E402
from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage_nodes import NODES as v5_storage_nodes  # noqa: E402


def _ancestors(graph, roots):
    seen, pending = set(roots), list(roots)
    while pending:
        node = graph[pending.pop()]
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                source = str(value[0])
                if source not in seen:
                    seen.add(source)
                    pending.append(source)
    return seen


def test_v5_storage_pair_has_no_hidden_sampling_and_keeps_source_bytes():
    original_sha = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    pair = build_pair()
    freeze_ui, freeze_api = pair["01_freeze_window_0"]
    resume_ui, resume_api = pair["02_resume_window_1_DRAFT"]
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == original_sha
    for ui, api in ((freeze_ui, freeze_api), (resume_ui, resume_api)):
        frontend_types = {str(node["id"]): node["type"] for node in ui["nodes"]}
        assert {key: item["class_type"] for key, item in api.items()} == frontend_types
        links = {link[0]: link for link in ui["links"]}
        nodes = {node["id"]: node for node in ui["nodes"]}
        for link_id, source, source_slot, target, target_slot, _dtype in ui["links"]:
            assert link_id in nodes[source]["outputs"][source_slot]["links"]
            assert nodes[target]["inputs"][target_slot]["link"] == link_id
            assert link_id in links
    assert freeze_api["42"]["class_type"] == "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
    assert freeze_api["43"]["class_type"] == "MiniMaxH3ChunkedV5WindowSaveEXPT8"
    assert freeze_api["43"]["inputs"]["window_result"] == ["30", 1]
    assert freeze_api["42"]["inputs"]["confirm_save"] is False
    assert freeze_api["43"]["inputs"]["confirm_save"] is False
    assert "31" not in freeze_api
    assert sum(item["class_type"] == "SamplerCustomAdvanced"
               for item in freeze_api.values()) == 1
    assert set(_ancestors(freeze_api, ("42", "43", "38"))) == set(freeze_api)

    assert resume_api["42"]["class_type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
    assert resume_api["43"]["class_type"] == "MiniMaxH3ChunkedV5WindowLoadEXPT8"
    assert resume_api["43"]["inputs"]["expected_window_index"] == 0
    assert not resume_api["42"]["inputs"]["checkpoint_path"]
    assert not resume_api["43"]["inputs"]["artifact_path"]
    assert "12" not in resume_api and "30" not in resume_api
    assert resume_api["31"]["inputs"]["previous_result"] == ["43", 1]
    assert resume_api["35"]["inputs"]["previous_result"] == ["43", 1]
    assert resume_api["40"]["inputs"]["previous_result"] == ["43", 1]
    assert sum(item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"
               for item in resume_api.values()) == 1
    assert sum(item["class_type"] == "SamplerCustomAdvanced"
               for item in resume_api.values()) == 0
    assert set(_ancestors(resume_api, ("25", "41"))) == set(resume_api)
    for graph in (freeze_api, resume_api):
        assert sum(item["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
                   for item in graph.values()) == 1
        assert sum(item["class_type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"
                   for item in graph.values()) == 1
        assert len(json.dumps(graph)) > 1000


def test_v5_storage_registration_only_appends_to_frozen_module_order():
    # Frozen 516-node rolling source audit from before the two new IDs.
    old_v5_nodes = PROJECT / "h3_t8/modular_sampling/chunked_v5_nodes.py"
    assert hashlib.sha256(old_v5_nodes.read_bytes()).hexdigest() == (
        "6409f5beb0762ea22608e9ea0c385f81a4a112136b8d8a91a8fd8f556b6503a6"
    )
    registration = (PROJECT / "h3_t8/modular_sampling/__init__.py").read_bytes()
    new_import = b"    from .chunked_v5_storage_nodes import NODES as chunked_v5_storage_nodes\n"
    assert registration.count(new_import) == 1
    text = registration.decode("utf8")
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "node_classes")
    additions = {"legacy_pass_through_nodes", "pass_through_audit_nodes",
        "h16_native_source_nodes", "motion_storage_nodes", "ltx_latent_sample_nodes",
        "fast_h3_v2_continuation_nodes", "fast_h3_v2_job_nodes", "chunked_v5_storage_nodes"}
    lines = text.splitlines(keepends=True)
    prior = text
    removed = set()
    for statement in function.body[:-1]:
        assert isinstance(statement, ast.ImportFrom)
        chunk = "".join(lines[statement.lineno - 1:statement.end_lineno])
        extra_face = (statement.module == "face_nodes" and
                      any(item.name == "MiniMaxH3FaceRelayBindEXPT8" for item in statement.names))
        if statement.module in additions or extra_face:
            assert prior.count(chunk) == 1
            prior = prior.replace(chunk, "")
            removed.add("face_relay" if extra_face else statement.module)
    assert removed == additions | {"face_relay"}
    prefix, marker, _tail = prior.rpartition("*chunked_v1_storage_nodes")
    assert marker and not _tail.startswith("]")
    prior = prefix + marker + "]\n"
    # Still reconstruct the exact historical bytes: no updated baseline SHA,
    # reordered old import or arbitrary schema normalization is accepted.
    assert hashlib.sha256(prior.encode("utf8")).hexdigest() == (
        "ad4d49a70d4776de9f1b5e5d6564032661536f3bf5b994aea28264dede269b76"
    )
    returned = function.body[-1]
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.List)
    names = [ast.unparse(item) for item in returned.value.elts]
    start = names.index("*chunked_v5_storage_nodes")
    assert names[start - 1] == "*chunked_v1_storage_nodes"
    assert names[start + 1:] == ["MiniMaxH3H16VerifiedNativeSourceEXPT8",
        "*ltx_latent_sample_nodes", "MiniMaxH3MotionFrozenFirstPassLoadEXPT8",
        "*fast_h3_v2_continuation_nodes", "*fast_h3_v2_job_nodes",
        "MiniMaxH3FaceRelayBindEXPT8", "MiniMaxH3FaceLocalRelayBindEXPT8",
        "MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8",
        "MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8",
        "MiniMaxH3MotionStageAuditPassThroughEXPT8", "MiniMaxH3StageEAVAuditPassThroughEXPT8"]
    actual = node_classes()
    start = actual.index(v5_storage_nodes[0])
    assert actual[start - len(v1_storage_nodes):start] == v1_storage_nodes
    assert actual[start:start + len(v5_storage_nodes)] == v5_storage_nodes
    ids = [item.define_schema().node_id for item in actual]
    assert len(ids) == len(set(ids))


def test_four_window_pair_freezes_completed_first_three_and_resumes_only_last():
    original_sha = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    segments, total_frames = compute_temporal_segments(57, 85, 34)
    assert total_frames == 192
    assert segments == [(0, 0, 25, 85), (15, 51, 40, 136),
                        (30, 102, 55, 187), (45, 153, 57, 192)]

    pair = build_pair(4)
    freeze_ui, freeze = pair["01_freeze_window_2"]
    resume_ui, resume = pair["02_resume_window_3_DRAFT"]
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == original_sha
    for ui, graph in ((freeze_ui, freeze), (resume_ui, resume)):
        assert {str(node["id"]): node["type"] for node in ui["nodes"]} == {
            node_id: node["class_type"] for node_id, node in graph.items()}
        nodes = {node["id"]: node for node in ui["nodes"]}
        for link_id, source, source_slot, target, target_slot, _dtype in ui["links"]:
            assert link_id in nodes[source]["outputs"][source_slot]["links"]
            assert nodes[target]["inputs"][target_slot]["link"] == link_id
        assert graph["14"]["inputs"]["temporal_chunk_frames"] == 85
        assert graph["14"]["inputs"]["temporal_overlap_frames"] == 34

    freeze_windows = {node["inputs"]["window_index"] for node in freeze.values()
                      if node["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"}
    resume_windows = {node["inputs"]["window_index"] for node in resume.values()
                      if node["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"}
    assert freeze_windows == {0, 1, 2} and resume_windows == {3}
    assert sum(node["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"
               for node in freeze.values()) == 3
    assert sum(node["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"
               for node in resume.values()) == 1
    assert freeze["53"]["class_type"] == "MiniMaxH3ChunkedV5WindowSaveEXPT8"
    assert resume["53"]["class_type"] == "MiniMaxH3ChunkedV5WindowLoadEXPT8"
    assert resume["53"]["inputs"]["expected_window_index"] == 2
    assert not resume["53"]["inputs"]["artifact_path"]
    assert not resume["52"]["inputs"]["checkpoint_path"]
    assert "12" not in resume and "30" not in resume and "31" not in resume
    assert "32" not in resume
    assert sum(node["class_type"] == "SamplerCustomAdvanced"
               for node in resume.values()) == 0
    assert sum(node["class_type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8"
               for node in resume.values()) == 1


def test_fixed_window_pair_builder_preserves_saved_candidates():
    base = PROJECT / "artifacts/development/modular-sampling-m4-chunked-v5-storage-20260924"
    versions = {2: "candidate-v2", 3: "candidate-v4-three-windows",
                4: "candidate-v6-four-windows"}
    for count, directory in versions.items():
        for name, (frontend, api) in build_pair(count).items():
            for suffix, graph in ((".json", frontend), (".api.json", api)):
                saved = base / directory / f"{name}{suffix}"
                assert saved.read_text(encoding="utf8") == (
                    json.dumps(graph, ensure_ascii=False, indent=2) + "\n")

    expected_four = {
        "01_freeze_window_2.json": "1faa383d9e8cbb2a1d935d569cd847369190d684a4d8776c4b164eb6669c6d41",
        "01_freeze_window_2.api.json": "ec81a29932884e7bc6a556dde02c6afe07850462adad023209b17d78bbe47c37",
        "02_resume_window_3_DRAFT.json": "0508c270c81de4d79d173a39ca8dd7b2bdaac6bcbb026f39191d579cde893ed7",
        "02_resume_window_3_DRAFT.api.json": "6f79adfa04a598c71b5978805b447d6371f68e8912782e106b9cd5650018aa19",
    }
    for name, digest in expected_four.items():
        assert hashlib.sha256((base / versions[4] / name).read_bytes()).hexdigest() == digest
