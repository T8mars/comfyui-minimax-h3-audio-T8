"""Real inputs above the former arbitrary frame cap must reach Core consumers."""
from fractions import Fraction

import av
import numpy as np
import torch
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import source_conform as c


def test_conform_and_control_map_accept_actual_4102_frame_source():
    count = 4102  # Valid 17n+5, above the former 4096 policy cutoff.
    frames = torch.arange(count).float().div(count).reshape(count, 1, 1, 1).expand(-1, 32, 32, 3)
    output, audio, observed, _, mapping, _ = c.conform(frames, 24., 32, 32, count)
    control, lineage = c.derive_control(frames, "identity_rgb")
    mapped, report = c.map_control(mapping, control, lineage=lineage)
    assert observed == count and mapping.indices == tuple(range(count))
    # Core's unchanged Lanczos path quantizes RGB through 8-bit PIL, even when
    # dimensions match. Removing resource caps must retain that legacy path.
    torch.testing.assert_close(output, frames,rtol=0,atol=1 / 255 + 1e-7)
    assert torch.equal(mapped, output)
    assert audio["waveform"].shape[-1] == round(count / 24 * 32000)
    assert report["lineage_status"] == "owned_preprocess_verified"


def test_native_decode_clock_observes_all_4102_frames(tmp_path):
    path = tmp_path / "actual-4102.mp4"
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for index in range(4102):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index % 255, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    components, clock, report = c.decode_clock(InputImpl.VideoFromFile(str(path)))
    assert len(components.images) == len(clock.pts_seconds) == 4102
    assert clock.status == "observed_CFR"
    assert clock.pts_seconds[-1] - clock.pts_seconds[0] == Fraction(4101, 24)
    assert report["decoder"] == "native_Core_serial"
