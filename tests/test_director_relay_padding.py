"""Portable real Director -> Relay Plan CPU regression, no MODEL/server.

Run with the existing Core CPU test runner. An explicit candidate override is
optional; default always uses the public director_generation module. The seven
original event/draft/gap/overlap/global-only/percent assertions are preserved.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import unittest
import uuid

from h3_audio_t8_pkg import director_generation as public_generation
from h3_audio_t8_pkg import director_project, prompt_relay_advanced

HERE = Path(__file__).resolve().parent


# Independent pre-padding-fix oracle, copied from director_generation.py
# source SHA256 9a39e6646f7615a485830bc186c7a7b2342c0e3f63d548f26755a0117f9ce5e5.
# Its complete function AST is checked below, not imported from the code tested.
ORIGINAL_RELAY_PLAN_AST_SHA256 = "9f9cd9828483627dc8700ac5371593b075fb3bec000076a87241e60aec1b7c05"


def _relay_plan_inputs(shot: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the Director's advanced event list into native Relay inputs."""

    global_prompt = str(shot.get("global_prompt") or "").strip()
    local_prompt = str(shot.get("local_prompt") or "").strip()
    events = list(shot.get("events") or [])
    if not global_prompt:
        global_prompt = local_prompt or "Cinematic continuous shot with coherent subject and stable lighting."
    if str(shot.get("writing_mode")) != "advanced" or not events:
        local_prompts = ""
        time_ranges = ""
        timing_mode = "auto_equal"
    else:
        local_prompts = "\n".join(str(event["text"]) for event in events)
        time_ranges = "\n".join(
            f"{int(event['start_frame'])}-{int(event['end_frame']) - 1}" for event in events
        )
        timing_mode = "frames"
    return {
        "global_prompt": global_prompt,
        "local_prompts": local_prompts,
        "length": int(shot["time"]["aligned_frames"]),
        "timing_mode": timing_mode,
        "time_ranges": time_ranges,
        "math_profile": "paper_v1",
        "epsilon": 0.1,
        "allow_gaps": False,
        "allow_overlaps": False,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


class RelayPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="t8-relay-padding-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name).resolve()
        original_function = next(node for node in ast.parse(Path(__file__).read_text(encoding="utf8")).body
                                 if isinstance(node, ast.FunctionDef) and node.name == "_relay_plan_inputs")
        original_ast_sha = hashlib.sha256(ast.dump(original_function, include_attributes=False).encode("utf8")).hexdigest()
        assert original_ast_sha == ORIGINAL_RELAY_PLAN_AST_SHA256
        cls.original = SimpleNamespace(_relay_plan_inputs=_relay_plan_inputs)
        cls.project_api, cls.relay = director_project, prompt_relay_advanced
        selected = os.environ.get("T8_DIRECTOR_GENERATION_SOURCE")
        cls.generation = (load(public_generation.__name__ + "_padding_candidate", Path(selected).resolve(strict=True))
                          if selected else public_generation)
        cls.source = json.loads((HERE / "fixtures/director_relay_padding_project.json").read_bytes())
        package = public_generation.__package__
        modules = [public_generation, director_project, prompt_relay_advanced, cls.generation,
                   importlib.import_module(package + ".timing"),
                   importlib.import_module(package + ".keyframe_positions")]
        cls.bindings = {str(Path(module.__file__).resolve()): sha(module.__file__) for module in modules}
        cls.bindings[str(Path(__file__).resolve())] = sha(__file__)
        cls.bindings[str(HERE / "fixtures/director_relay_padding_project.json")] = sha(HERE / "fixtures/director_relay_padding_project.json")
        cls.before = {path: sha(path) for path in cls.bindings}
        assert cls.before == cls.bindings
        import torch
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        import torch
        after = {path: sha(path) for path in cls.bindings}
        write(cls.output / "source-check.json", dict(before=cls.before, after=after, sources_identical=cls.before == after,
            selected_source=cls.generation.__file__, CUDA_initialized=torch.cuda.is_initialized(), model_execution=False, HTTP_started=False))
        assert cls.before == after
        assert not torch.cuda.is_initialized()

    def compiled(self, bounds=(0, 60, 120), text=None, duration=5.):
        project = deepcopy(self.source)
        shot = project["doc"]["shots"][0]
        shot["writingMode"] = "advanced"
        shot["prompt"] = shot["simplePrompt"]
        shot["manualDuration"] = shot["duration"] = duration
        shot["events"] = [dict(id=str(uuid.UUID(int=500 + i)), start=start / 24, end=end / 24,
            text=(text or ["Opening hand motion.", "Later turn toward one another.", "Final 50% role_A 中文 literal."])[i])
            for i, (start, end) in enumerate(zip(bounds, bounds[1:]))]
        report = self.project_api.compile_project(project)
        self.assertTrue(report["ready"], report["errors"])
        return project, report["shots"][0]

    def test_actual_two_events_cover_generated_grid_without_mutating_visible_draft(self):
        project, shot = self.compiled()
        before = deepcopy(shot)
        inputs = self.generation._relay_plan_inputs(shot)
        write(self.output / "actual-two-event-inputs.json", dict(project=project, compiled=shot, relay_inputs=inputs))
        plan, _, length, _, _ = self.relay.build_prompt_relay_plan(**inputs)
        self.assertEqual(length, 124)
        self.assertEqual([(item["start_frame"], item["end_frame_exclusive"]) for item in plan["events"]], [(0, 60), (60, 124)])
        self.assertEqual(shot, before)
        self.assertEqual(project["doc"]["shots"][0]["events"][-1]["end"], 5)
        self.assertEqual(plan["events"][-1]["local_prompt"], "Later turn toward one another.")

    def test_three_segments_keep_last_literal_and_earlier_boundaries(self):
        _, shot = self.compiled((0, 40, 80, 120))
        before = deepcopy(shot)
        plan = self.relay.build_prompt_relay_plan(**self.generation._relay_plan_inputs(shot))[0]
        self.assertEqual([(item["start_frame"], item["end_frame_exclusive"]) for item in plan["events"]], [(0, 40), (40, 80), (80, 124)])
        self.assertEqual(plan["events"][-1]["local_prompt"], "Final 50% role_A 中文 literal.")
        self.assertEqual(shot, before)
        for item in plan["events"]:
            self.assertIn(item["local_prompt"], plan["compiled_prompt"][item["prompt_char_start"]:item["prompt_char_end"]])

    def test_visible_gap_not_filled_as_hidden_padding(self):
        _, shot = self.compiled((0, 60, 108))
        with self.assertRaisesRegex(ValueError, "complete target"):
            self.relay.build_prompt_relay_plan(**self.generation._relay_plan_inputs(shot))

    def test_visible_overlap_still_real_error(self):
        _, shot = self.compiled()
        shot["events"][1]["start_frame"] = 48
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.relay.build_prompt_relay_plan(**self.generation._relay_plan_inputs(shot))

    def test_already_aligned_grid_keeps_original_inputs(self):
        _, shot = self.compiled((0, 60, 124), duration=124 / 24)
        self.assertEqual(self.generation._relay_plan_inputs(shot), self.original._relay_plan_inputs(shot))
        plan = self.relay.build_prompt_relay_plan(**self.generation._relay_plan_inputs(shot))[0]
        self.assertEqual(plan["events"][-1]["end_frame_exclusive"], 124)

    def test_global_only_original_bypass_unchanged(self):
        project = deepcopy(self.source)
        shot = self.project_api.compile_project(project)["shots"][0]
        inputs = self.generation._relay_plan_inputs(shot)
        self.assertEqual(inputs, self.original._relay_plan_inputs(shot))
        plan = self.relay.build_prompt_relay_plan(**inputs)[0]
        self.assertEqual(plan["events"], [])
        self.assertEqual(plan["frame_count"], 124)

    def test_visible_percent_keyframe_clock_not_shifted_to_generated_percent(self):
        project, _ = self.compiled()
        item = project["doc"]["shots"][0]["middleKeyframes"]["items"][0]
        item["position_mode"], item["position"] = "percent", 50
        shot = self.project_api.compile_project(project)["shots"][0]
        before = deepcopy(shot["middle_keyframes"])
        self.assertEqual(before[0]["frame_index"], 60)
        inputs = self.generation._relay_plan_inputs(shot)
        plan = self.relay.build_prompt_relay_plan(**inputs)[0]
        self.assertEqual(shot["middle_keyframes"], before)
        percent = dict(inputs, timing_mode="percent", time_ranges=f"0-{100 * 60 / 124}\n{100 * 60 / 124}-100")
        other = self.relay.build_prompt_relay_plan(**percent)[0]
        self.assertEqual([(row["start_frame"], row["end_frame_exclusive"]) for row in other["events"]],
            [(row["start_frame"], row["end_frame_exclusive"]) for row in plan["events"]])
