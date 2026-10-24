"""Experimental thread-local Python profiler; never installs MODEL hooks.

This is a diagnostic, not a speed benchmark or portable owner. Activation is
explicit and current-thread only. It observes original frame locals/returns;
no helper, weight, argument, sigma, RNG or module callable is replaced.
"""
from __future__ import annotations

import hashlib
import sys


def tensor_record(value):
    import torch
    raw = value.detach().to(device="cpu").contiguous()
    return dict(shape=list(raw.shape), dtype=str(raw.dtype),
                sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())


class ConsumptionProfiler:
    def __init__(self, network, *, hyperflow_runtime, advanced_conditioning):
        self.network = network
        self.hyperflow = hyperflow_runtime
        self.advanced = advanced_conditioning
        self.records = []
        self.diagnostics = []
        self.stack = []
        self.prior = None
        self.installed = False

    def __enter__(self):
        import torch
        from comfy.ldm.minimax import model as native
        self.prior = sys.getprofile()
        if self.prior is not None:
            raise ValueError("Existing thread profiler retained; do not replace another observer")
        self.native = native
        self.forward_code = native.MiniMaxH3Model.forward.__code__
        self.cond_codes = {native.MiniMaxH3Model._cond_video_rows.__code__, self.advanced._condition_rows_with_augs.__code__}
        self.call_code = torch.nn.Module._call_impl.__code__
        self.installed = True
        sys.setprofile(self._profile)
        return self

    def __exit__(self, kind, value, traceback):
        sys.setprofile(self.prior)
        self.installed = False
        self.stack.clear()

    def _profile(self, frame, event, returned):
        try:
            self._observe(frame, event, returned)
        except Exception as error:
            # Diagnostics cannot turn an unknown executable owner into a
            # sampling ban. Qualification later requires no diagnostics.
            self.diagnostics.append(dict(type=type(error).__name__, message=str(error)))

    def _observe(self, frame, event, returned):
        code, local = frame.f_code, frame.f_locals
        if code is self.forward_code and local.get("self") is self.network:
            if event == "call":
                payload = local.get("minimax_payload") or {}
                layout = payload.get("layout")
                step = self.hyperflow.active_step()
                video = local["x"][0]
                record = dict(index=len(self.records), video_shape=list(video.shape),
                    timestep=tensor_record(local["timestep"]),
                    HF_owner=None if step is None else step.owner,
                    HF_absolute_interval=None if step is None else step.absolute_index,
                    layout_signature=None if layout is None else list(layout.signature),
                    layout_segments=None if layout is None else [list(row) for row in layout.segments],
                    keyframes=[dict(frame_index=int(row.get("t8_multikeyframe_frame_index", row["resolved_frame_index"])),
                                    latent=tensor_record(row["latent"])) for row in payload.get("keyframes", [])],
                    actual_cond_helper_return_observed=False, actual_projection_input_observed=False,
                    cond_return_equals_actual_projection=False, completed_forward=False)
                self.records.append(record)
                self.stack.append(dict(frame=frame, record=record, payload=payload, layout=layout,
                                       rows=None, row_parts=None))
            elif event == "return" and self.stack and self.stack[-1]["frame"] is frame:
                state = self.stack.pop()
                state["record"]["completed_forward"] = returned is not None
            return
        if not self.stack:
            return
        state = self.stack[-1]
        if code in self.cond_codes and local.get("self") is self.network and event == "return":
            if returned is None:
                return
            rows = local.get("rows")
            payload = local["payload"]
            if not isinstance(rows, list) or len(rows) != len(payload.get("cond_video_latents", [])):
                self.diagnostics.append(dict(type="coverage", message="Original helper locals did not expose actual per-condition rows"))
                return
            state["rows"] = returned.detach().to(device="cpu").clone()
            state["row_parts"] = [value.detach().to(device="cpu").clone() for value in rows]
            state["record"].update(actual_cond_helper_return_observed=True,
                original_cond_helper=code.co_qualname,
                actual_raw_condition_latents=[tensor_record(value) for value in payload["cond_video_latents"]],
                actual_helper_parts=[tensor_record(value) for value in rows], actual_helper_return=tensor_record(returned))
        elif (code is self.call_code and event == "call"
              and local.get("self") is self.network.video_patch_proj):
            import torch
            layout = state["layout"]
            if layout is None or state["rows"] is None:
                self.diagnostics.append(dict(type="coverage", message="Projection reached without the observed original condition helper/layout"))
                return
            actual = local["args"][0]
            condition = actual[~layout.img_update.to(actual.device)].detach().to(device="cpu")
            row = state["record"]
            row.update(actual_projection_input_observed=True, actual_projection_condition_rows=tensor_record(condition),
                       cond_return_equals_actual_projection=torch.equal(condition, state["rows"]))
            cond_segments = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
            positions = []
            text_len = int(layout.signature[0])
            offset = 0
            for index, keyframe in enumerate(row["keyframes"]):
                if index >= len(cond_segments) or index >= len(state["row_parts"]):
                    break
                start, stop = cond_segments[index]
                part = state["row_parts"][index]
                count = part.shape[0]
                chosen = condition[offset:offset + count]
                t = layout.position_ids[start:stop, 0].detach().to(device="cpu")
                # Replay the ACTUAL Advanced endpoint operation order, not
                # the algebraically equivalent (frame_count-1)*rescale. The
                # published repair intentionally preserves this FP64 rounding.
                frame_count = state["payload"].get("frame_count")
                if keyframe["frame_index"] == 0:
                    expected_t = float(text_len)
                elif frame_count is not None and keyframe["frame_index"] == int(frame_count) - 1:
                    spans = self.native.FRAME_PER_TOKEN
                    physical_count = sum(spans[i % len(spans)] for i in range(int(layout.signature[1])))
                    expected_t = float(text_len) + self.native.FRAME_RESCALE * physical_count - self.native.FRAME_RESCALE
                else:
                    expected_t = float(text_len) + self.native.FRAME_RESCALE * keyframe["frame_index"]
                positions.append(dict(frame_index=keyframe["frame_index"], segment=[start, stop],
                    row_count=count, projected_part=tensor_record(chosen), actual_helper_part=tensor_record(part),
                    actual_projected_part_equals_helper=torch.equal(chosen, part),
                    actual_physical_times=t.tolist(),
                    physical_time_matches_native_no_ref=(not state["payload"].get("refs")
                        and count == stop - start and bool(torch.all(t == expected_t)))))
                offset += count
            row["actual_keyframe_projection_segments"] = positions

    def report(self):
        return dict(records=self.records, diagnostics=self.diagnostics,
            complete_target_coverage=bool(self.records) and not self.diagnostics and all(
                row["completed_forward"] and row["actual_cond_helper_return_observed"]
                and row["actual_projection_input_observed"] and row["cond_return_equals_actual_projection"]
                and len(row.get("actual_keyframe_projection_segments", [])) == len(row["keyframes"])
                and all(part["actual_projected_part_equals_helper"] and part["physical_time_matches_native_no_ref"]
                        for part in row["actual_keyframe_projection_segments"])
                for row in self.records),
            installed=self.installed, module_hooks_installed=False, extra_DiT_or_reference_encodes=0,
            profiler_sync_copy_hash_overhead_measured=False, trained_GPU_qualified=False,
            portable_identity_inferred=False, speed_or_peak_memory_qualified=False)
