"""Private frame-only RGB/encoding/real-projection observer adapter.

All tensor work is observation of an already running selected request. No
encoder/model/helper is called for verification. No model is held beyond its
actual active frame: per-network observers use weak references. This module
imports no Torch/Core and grants no completion/quality by itself.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import weakref


def weak_profiler_type(base):
    class WeakProfiler(base):
        @property
        def network(self):
            return self._network_ref()

        @network.setter
        def network(self, value):
            self._network_ref = weakref.ref(value)
    return WeakProfiler


class RequestObservation:
    def __init__(self, selected, prompt_id, *, deps, assets):
        self.selected, self.prompt_id, self.deps = selected, prompt_id, deps
        self.assets = assets
        self.thread_ident = threading.get_ident()
        self.readers, self.encodings, self.models = [], [], {}
        self.reader_frames, self.keyframe_frames, self.vae_frames = {}, [], {}
        self.diagnostics = []
        self.vae_entries = 0
        self.WeakProfiler = weak_profiler_type(deps.profiler_base)

    def _tensor(self, value):
        return self.deps.tensor_record(value)

    def _diagnose(self, detail):
        self.diagnostics.append(detail)

    def _reader_return(self, frame, value):
        original = self.reader_frames.pop(frame, None)
        if original is None or value is None:
            return
        actual_path = Path(frame.f_locals["image_path"]).resolve(strict=True)
        selected = self.assets.get(str(actual_path))
        if selected is None:
            self._diagnose("Actual LoadImage return has no selected owned asset binding")
            return
        with actual_path.open("rb") as stream:
            actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha != selected["sha256"]:
            self._diagnose("Actual loaded image file differs from selected asset SHA")
            return
        self.readers.append(dict(actual_path=str(actual_path), file_sha256=actual_sha,
            asset_ids=list(selected["asset_ids"]), actual_image_argument=original,
            actual_RGB_return=self._tensor(value[0]), decoder_reexecuted=False))

    def _keyframe_call(self, frame):
        local = frame.f_locals
        incoming = self._tensor(local["image"][:1])
        matching = [row for row in self.readers if row["actual_RGB_return"] == incoming]
        self.keyframe_frames.append(dict(frame=frame, index=int(local["frame_index"]),
            width=int(local["width"]), height=int(local["height"]), source_RGB=incoming,
            loaded_sources=[dict(path=row["actual_path"], file_sha256=row["file_sha256"], asset_ids=row["asset_ids"])
                            for row in matching], vae_calls=[]))

    def _keyframe_return(self, frame, value):
        if not self.keyframe_frames or self.keyframe_frames[-1]["frame"] is not frame:
            self._diagnose("Actual keyframe encoder frame stack differs")
            return
        row = self.keyframe_frames.pop()
        row.pop("frame")
        if value is None:
            row.update(completed=False, actual_original_VAE_join=False)
            self.encodings.append(row)
            return
        resized, keyframe = value
        actual_rgb, latent = self._tensor(resized), self._tensor(keyframe["latent"])
        row.update(completed=True, resized_RGB=actual_rgb, returned_latent=latent,
            returned_frame_index=int(keyframe.get(self.deps.actual_frame_index_key, keyframe["resolved_frame_index"])))
        row["actual_original_VAE_join"] = bool(len(row["vae_calls"]) == 1
            and row["vae_calls"][0].get("actual_input_RGB") == actual_rgb
            and row["vae_calls"][0].get("actual_return_latent") == latent
            and row["returned_frame_index"] == row["index"])
        self.encodings.append(row)

    def _profiler(self, network):
        key = id(network)
        old = self.models.get(key)
        if old is not None:
            if old.network is not network:
                self._diagnose("Actual network identity was reused after its lifetime; no association invented")
                return None
            return old
        profiler = self.WeakProfiler(network, hyperflow_runtime=self.deps.hyperflow,
            advanced_conditioning=self.deps.advanced)
        # No __enter__/new setprofile or MODEL hook: the actual worker router
        # already owns this thread's profile. Configure the unchanged original
        # frame observer with precisely the same real code objects it uses.
        profiler.native = self.deps.native
        profiler.forward_code = self.deps.native.MiniMaxH3Model.forward.__code__
        profiler.cond_codes = {self.deps.native.MiniMaxH3Model._cond_video_rows.__code__,
                               self.deps.advanced._condition_rows_with_augs.__code__}
        profiler.call_code = self.deps.module_call_code
        profiler.installed = True
        self.models[key] = profiler
        return profiler

    def event(self, frame, kind, returned):
        if threading.get_ident() != self.thread_ident:
            self._diagnose("Request observer received a foreign execution thread")
            return
        code, local = frame.f_code, frame.f_locals
        if code is self.deps.load_image_code:
            if kind == "call":
                self.reader_frames[frame] = str(local["image"])
            elif kind == "return":
                self._reader_return(frame, returned)
        elif code is self.deps.keyframe_encode_code:
            if kind == "call":
                self._keyframe_call(frame)
            elif kind == "return":
                self._keyframe_return(frame, returned)
        elif code is self.deps.vae_encode_code:
            if kind == "call":
                self.vae_entries += 1
                if self.keyframe_frames:
                    self.vae_frames[frame] = (self.keyframe_frames[-1],
                        dict(actual_input_RGB=self._tensor(local["pixel_samples"])))
            elif kind == "return":
                selected = self.vae_frames.pop(frame, None)
                if selected is not None:
                    owner, value = selected
                    if returned is not None:
                        value["actual_return_latent"] = self._tensor(returned)
                    owner["vae_calls"].append(value)
        if code is self.deps.native.MiniMaxH3Model.forward.__code__ and kind == "call":
            self._profiler(local["self"])
        # Original observer catches unknown-frame diagnostics itself. Iterate
        # values without acquiring/holding any strong model reference.
        for profiler in tuple(self.models.values()):
            profiler._profile(frame, kind, returned)
        if code is self.deps.native.MiniMaxH3Model.forward.__code__ and kind == "call":
            profiler = self.models.get(id(local["self"]))
            if profiler is not None and profiler.records:
                profiler.records[-1]["actual_timestep_values"] = local["timestep"].detach().to(device="cpu").reshape(-1).tolist()

    def finish(self, frame, *, aborted):
        # Core's sync execute returns None on both ordinary and handled-error
        # paths. Never interpret a profiler `return` or aborted=False as success.
        executor = None if frame is None else frame.f_locals.get("self")
        actual_core_success = getattr(executor, "success", None)
        target_frame = self.selected["middle_frame_index"]
        observed = []
        for number, profiler in enumerate(self.models.values()):
            profiler.installed = False
            report = profiler.report()
            for row in report["records"]:
                selected = [keyframe for keyframe in row["keyframes"] if keyframe["frame_index"] == target_frame]
                matches = []
                for keyframe in selected:
                    joined = [item for item in self.encodings if item.get("completed")
                        and item["index"] == target_frame and item.get("returned_latent") == keyframe["latent"]]
                    valid = [item for item in joined if item["actual_original_VAE_join"]
                        and any(source["file_sha256"] == self.selected["middle_asset_sha256"]
                                for source in item["loaded_sources"])]
                    matches.append(dict(frame_index=target_frame, actual_encoder_join_count=len(valid),
                        source_file_SHA_matches=bool(valid), actual_RGB_VAE_latent_payload_join=bool(valid)))
                row["actual_middle_RGB_encoding_joins"] = matches
                row["actual_middle_enabled_contract_matches"] = bool(
                    len(selected) == 1 and len(matches) == 1 and matches[0]["actual_RGB_VAE_latent_payload_join"]
                    if self.selected["middle_enabled"] else not selected)
            observed.append(dict(actual_network_ordinal=number, weak_model_reference=True, report=report))
            profiler.stack.clear()
        rows = [row for model in observed for row in model["report"]["records"]]
        traces_complete = bool(rows) and not self.diagnostics and all(
            model["report"]["complete_target_coverage"] for model in observed)
        joins_complete = bool(rows) and not self.diagnostics and all(
            row["actual_middle_enabled_contract_matches"] for row in rows)
        result = dict(schema="t8.h21.D.actual-worker-consumption.v1", prompt_id=self.prompt_id,
            selected_case=self.selected["name"], request_thread_ident=self.thread_ident,
            actual_Core_executor_success_observed=actual_core_success, profiler_request_return_observed=True,
            profiler_return_not_authoritative_history_success=True, aborted=aborted,
            readers=self.readers, actual_keyframe_encodings=self.encodings, actual_VAE_encode_entries=self.vae_entries,
            networks=observed, diagnostics=list(self.diagnostics),
            complete_actual_forward_projection_coverage=traces_complete,
            complete_selected_middle_RGB_latent_projection_joins=joins_complete,
            no_extra_encoder_or_DiT_calls=True, no_MODEL_or_module_hooks=True,
            immutable_history_and_typed_stage_and_callbacks_still_required=True,
            actual_sampler_completion_or_trained_GPU_qualified=False,
            speed_or_peak_memory_qualified=False, human_quality_accepted=False)
        self.reader_frames.clear()
        self.keyframe_frames.clear()
        self.vae_frames.clear()
        self.models.clear()
        return result
