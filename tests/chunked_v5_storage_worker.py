"""Fresh-process CPU probe for exact Chunked v5 window-0 load/only-window-1 resume."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
COMFY = PROJECT.parents[1]
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(PROJECT / "tests"))
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
comfy.cli_args.args.use_pytorch_cross_attention = True

PACKAGE_NAME = "h3_audio_t8_pkg"
spec = importlib.util.spec_from_file_location(
    PACKAGE_NAME, PROJECT / "__init__.py", submodule_search_locations=[str(PROJECT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE_NAME] = package
assert spec.loader is not None
spec.loader.exec_module(package)

from pytest import MonkeyPatch  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import torch  # noqa: E402
from h3_audio_t8_pkg.modular_sampling import chunked_v5  # noqa: E402
from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage import load_window  # noqa: E402
from h3_audio_t8_pkg.modular_sampling.results import _input_identity  # noqa: E402
from test_chunked_two_pass_parity import _plan  # noqa: E402
from test_modular_chunked_v5 import make_v5_harness  # noqa: E402


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: worker.py ROOT MANIFEST_PATH MANIFEST_SHA [WINDOW_COUNT]")
    window_count = int(sys.argv[4]) if len(sys.argv) == 5 else 2
    if window_count not in (2, 3, 4):
        raise ValueError("Only the verified 2/3/4-window probes are available")
    torch.set_num_threads(1)
    with MonkeyPatch.context() as patcher:
        calls, source, positive, noise, sampler, sigmas, plan = make_v5_harness(patcher)
        if window_count > 2:
            source = {"samples": comfy.nested_tensor.NestedTensor((
                torch.zeros(1, 24, 57, 2, 2), torch.zeros(1, 32, 2, 320),
            ))}
            plan = _plan(temporal_chunk_frames={3: 102, 4: 85}[window_count],
                         temporal_overlap_frames=34)
        lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
        prepared, _ = chunked_v5.prepare_standard_joint(
            source, lifted, receipt, plan, noise)
        _output, previous, report_json = load_window(
            source, lifted, prepared, plan, Path(sys.argv[1]), sys.argv[2],
            sys.argv[3], window_count - 2)
        if calls["sample"] != 0 or json.loads(report_json)["automatic_cache_reuse"]:
            raise AssertionError("Frozen window load sampled or claimed automatic cache reuse")
        final, _result, _report = chunked_v5.sample_standard_window(
            object(), positive, source, lifted, prepared, plan, noise, sampler,
            sigmas, window_count - 1, previous)
        video, audio = final["samples"].unbind()
        print(json.dumps({"loaded_index": previous.index, "sample_count": calls["sample"],
                          "video_identity": _input_identity(video),
                          "audio_identity": _input_identity(audio)}, sort_keys=True))


if __name__ == "__main__":
    main()
