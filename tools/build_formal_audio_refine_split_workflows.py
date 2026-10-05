"""Publish additive S26 video-freeze/audio-tail graphs from pinned old examples.

This builder never edits the ten original Audio Refine graphs or an existing
public split graph. Saved graphs remain EXP candidates, not quality acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from tools.build_modular_audio_refine_resume_all import pairs_from_current_sources
from tools.build_modular_audio_refine_workflows import ROOT, sources
from tools.workflow_paths import public_workflow_path


DESTINATION = ROOT / "examples/workflows/59-audio-refine-split"
SOURCE_SHA256 = {
    "2026-08-26_H3_Audio_Refine_Phase2_Base_Refine4_Advanced_EXP.json": "7A0ED429841C5528E5736520366B5E75AAB75760C0B1B6AB36E318BE284037E2",
    "2026-08-26_H3_Audio_Refine_Phase2_Same_Turbo4_Advanced_EXP.json": "F98B4FA2D657C6A9A0CC1ACD2031C70624C5B2EA49ED70B94C5E3C880116ECFA",
    "2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP.json": "047DC963096846CF0351C958D8D22AD29975E10288F13B8CC7B6C185E5526D46",
    "2026-08-29_H3_Audio_Refine_EAV_Turbo8_Advanced_EXP.json": "2266FB4525C167495D63F6092230077F65F69EBC5BF02F1D5368C0C338C9A969",
    "2026-08-29_H3_Audio_Refine_Learned_TwoPass_Final8_Advanced_EXP.json": "929A308A45E459F2597D247BB9CA68BA42828DB63CA7063D33D52AF3EC435436",
    "2026-08-29_H3_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_Advanced_EXP.json": "80B54BDAD6FEC25B558BD902A7E03FC5532D6FFC5A5C63D2B9776642B16BD88A",
    "2026-08-29_H3_Audio_Refine_PDD_Ref2VA_4Plus4_Advanced_EXP.json": "1DBD27635D34F7E7C3B450DB341CD4D4E88C55264F0AB31619DF94B006C2DB42",
    "2026-08-29_H3_Audio_Refine_PDD_Ref2VA8_Advanced_EXP.json": "6571A289B232F1808480CE4D1E32F6E72173B2FD40E6EB40F6F3A8FFD43C7E34",
    "2026-08-29_H3_Audio_Refine_Prompt_Relay_Turbo8_Advanced_EXP.json": "21EE0919271E24C8ADE562AEB5A242B7269E0C900DAB2454260C5356D8EA8DF1",
    "2026-08-29_H3_Audio_Refine_Turbo8_Plus_Refine4_Advanced_EXP.json": "0693C39058D44410E596EACB817F33CEAD5CE74C9ADD5D9DA7160F8388406E4B",
}
PHASES = ("freeze_video", "resume_audio")


def generated():
    paths = sources()
    if {path.name for path in paths} != set(SOURCE_SHA256):
        raise ValueError("S26 original sampled workflow inventory changed")
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != SOURCE_SHA256[path.name]:
            raise ValueError(f"S26 original workflow changed: {path}")
    pairs = pairs_from_current_sources()
    if set(pairs) != set(paths):
        raise ValueError("S26 generated pair inventory differs from old sources")
    return {
        public_workflow_path(DESTINATION / f"S26_{path.stem}_{phase}_Separate_EXP.json"): graph
        for path in paths
        for phase, graph in zip(PHASES, pairs[path], strict=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create only missing public EXP graphs")
    options = parser.parse_args()
    expected = generated()
    if len(expected) != 20:
        raise ValueError("S26 must have ten separate frozen-video/audio-tail pairs")
    pending = {}
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified public S26 graph: {path}")
        else:
            pending[path] = content
    if pending and not options.write:
        print(f"S26 verified {len(expected) - len(pending)} graphs; {len(pending)} missing (use --write)")
        return 1
    if pending:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for path, content in pending.items():
            path.write_bytes(content)
    print(f"S26 verified {len(expected)} graphs; created {len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
