"""One stdlib-only release regression: reviewed delta, topology and fail-closed gate."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import audit_modular_new_sampling as gate  # noqa: E402


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf8"))


class ReviewedINT8Delta(unittest.TestCase):
    def test_exact_reviewed_delta_keeps_unknown_calls_closed(self):
        ledger = read("tests/fixtures/h09_reviewed_sampling_1_98.json")
        old = read("tests/fixtures/" + ledger["previous_baseline"])
        new = read("tests/fixtures/" + ledger["new_baseline"])
        self.assertEqual(gate.BASELINE.name, ledger["new_baseline"])
        self.assertEqual(len(old["sites"]), 128)
        self.assertEqual(new["sites"][:128], old["sites"])
        self.assertEqual(len(new["sites"]), 131)
        self.assertEqual([list(site.values()) for site in new["sites"][128:]],
                         [entry["site"] for entry in ledger["entries"]])
        self.assertEqual({site["path"] for site in new["sites"][128:]},
                         {"h3_t8/freevideo_int8/nodes.py", "h3_t8/freevideo_int8/worker.py"})
        for relative, digest in ledger["runtime_file_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        graph = ledger["public_workflow"]
        self.assertEqual(hashlib.sha256((ROOT / graph).read_bytes()).hexdigest(),
                         ledger["public_workflow_sha256"])
        types, edges = gate._workflow_graph(ROOT, graph)
        for node_id, kind, effect_ids in zip(ledger["workflow_stage_ids"],
                ledger["independent_stage_nodes"], ((6, 7), (11, 12))):
            self.assertEqual(types[node_id], kind)
            for effect_id, effect_kind in zip(effect_ids,
                    ("MiniMaxH3FreeVideoINT8EAVEXPT8", "MiniMaxH3FreeVideoINT8PromptRelayEXPT8")):
                self.assertEqual(types[effect_id], effect_kind)
                self.assertTrue(gate._reaches(edges, effect_id, node_id))
        self.assertTrue(gate._reaches(edges, 8, 9))
        self.assertEqual(types[9], "MiniMaxH3FreeVideoINT8StageSaveEXPT8")
        self.assertEqual(types[14], "MiniMaxH3FreeVideoINT8StageSaveEXPT8")
        # StageSave is a side branch; the completed LOW also feeds the lift directly.
        self.assertTrue(gate._reaches(edges, 8, 10))
        self.assertTrue(gate._reaches(edges, 9, 13))
        self.assertTrue(gate._reaches(edges, 10, 13))
        self.assertTrue(gate._reaches(edges, 13, 14))
        self.assertIn("MiniMaxH3FreeVideoINT8StageLoadEXPT8", read("features.json")["nodes"])
        self.assertEqual(ledger["human_review_A2_status"], "pass")
        self.assertEqual((ledger["actual_LOW_NFE"], ledger["actual_HIGH_NFE"]), (8, 3))
        self.assertTrue(ledger["LOW_audio_preserved_exact"])
        self.assertTrue(ledger["actual_LOW_and_HIGH_stage_content_cold_loading_verified"])
        self.assertFalse(ledger["new_independent_cold_HIGH_GPU_run_claimed"])
        self.assertFalse(ledger["strict_FP8_INT8_speed_comparison_claimed"])
        live = gate.discover(ROOT)
        with patch.object(gate, "discover", return_value=live):
            result = gate.guard(ROOT, new)
            self.assertEqual(result["status"], "pass", result["issues"])
            # The exact seven-site Pika two-pass successor is checked by the
            # source gate; the historical H09 ledger is not recaptured.
            self.assertEqual(len(read("tests/fixtures/pika_two_pass_source.json")["reviewed_sites"]), 7)
            self.assertEqual((result["declared_baseline_sites"], result["current_sites"],
                              result["reviewed_absent_root_alias_sites"]), (131, 120, 21))
            before = gate.guard(ROOT, old)
            self.assertEqual((before["status"], before["new_sites"], before["removed_sites"]),
                             ("fail", 3, 0))
        # Same call hash in another path/symbol is not covered by this reviewed ledger.
        unknown = dict(live[-1], path="h3_t8/unreviewed.py", symbol="Unreviewed.execute")
        with patch.object(gate, "discover", return_value=[*live, unknown]):
            result = gate.guard(ROOT, new)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(i["kind"] == "new_sampling_site_requires_admission"
                                and i["site"]["path"] == unknown["path"] for i in result["issues"]))
        altered = [dict(site) for site in live]
        known = next(site for site in altered if site["path"] == "h3_t8/freevideo_int8/worker.py")
        known["call_sha256"] = "0" * 64
        with patch.object(gate, "discover", return_value=altered):
            result = gate.guard(ROOT, new)
            self.assertEqual((result["status"], result["new_sites"], result["removed_sites"]),
                             ("fail", 1, 1))


if __name__ == "__main__":
    unittest.main()
