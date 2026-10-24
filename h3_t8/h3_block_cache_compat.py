"""Scoped call adaptation for an external cache's legacy H3 finalization path.

No external source, MODEL object or module forward is patched. The cache wrapper
still performs its own hit handling, prefetch cleanup and output unpacking.
"""

import inspect

from .h3_core_compat import call_h3_final_layer, cleanup_h3_prefetched_modules


def _legacy_prefetch_hit_type(wrapper):
    """Recognize only an observable legacy cleanup ABI, not arbitrary hooks."""
    import comfy.model_prefetch

    namespace = getattr(wrapper, "__globals__", {})
    hit_type = namespace.get("H3BlockCacheHit")
    cleanup = namespace.get("_cleanup_short_circuited_prefetch")
    if (namespace.get("PREFETCH_CLEANUP_TAKES_MODULE") is not False
            or not isinstance(hit_type, type) or not issubclass(hit_type, Exception)
            or getattr(cleanup, "__globals__", None) is not namespace
            or namespace.get("comfy") is None
            or getattr(namespace["comfy"], "model_prefetch", None) is not comfy.model_prefetch):
        return None
    signature = inspect.signature(comfy.model_prefetch.cleanup_prefetched_modules)
    try:
        signature.bind(object())
    except TypeError:
        try:
            signature.bind(object(), [])
        except TypeError:
            return None
        return hit_type
    return None


def _cleanup_owned_prefetch(model, excluded_queues):
    """Release this model's queued state; leave every foreign queue item intact."""
    import comfy.model_prefetch

    owned = {id(block) for block in model.blocks}
    for queue in reversed(comfy.model_prefetch.PREFETCH_QUEUES):
        if id(queue) in excluded_queues:
            continue
        retained = []
        matched = False
        for entry in queue:
            module = entry[1][0] if isinstance(entry, tuple) else entry
            if id(module) not in owned:
                retained.append(entry)
                continue
            matched = True
            if isinstance(entry, tuple):
                modules = entry[1][1]
                if modules is not None:
                    cleanup_h3_prefetched_modules(module, modules)
        if matched:
            queue[:] = retained or [None]
            # Core pushes nested/active queues last. Match the external owner's
            # single-queue extent; do not release an outer invocation's state.
            break


class _FinalizationView:
    def __init__(self, model, timestep, options):
        self._model = model
        self._timestep = timestep
        self._options = options
        self._prefetch_scope = None
        self._prefetch_complete = False

    def __getattr__(self, name):
        if name == "blocks":
            if self._prefetch_scope is not None:
                # The original external cleanup has already closed its malloc
                # scope before asking for blocks. Keep that ordering, then hide
                # only this view's completed scan so it cannot reach outer queues.
                excluded = self._prefetch_scope
                self._prefetch_scope = None
                _cleanup_owned_prefetch(self._model, excluded)
                self._prefetch_complete = True
            if self._prefetch_complete:
                return ()
        return getattr(self._model, name)

    def final_layer(self, x, t_emb, video_seg, audio_seg, *args, **kwargs):
        layer = self._model.final_layer
        if args or kwargs:
            # A newer external wrapper already supplies the Core interface.
            return layer(x, t_emb, video_seg, audio_seg, *args, **kwargs)
        options = self._options
        shifts = (
            float(options.get("minimax_h3_sigma_shift_video", self._model.sigma_shift_video)),
            float(options.get("minimax_h3_sigma_shift_audio", self._model.sigma_shift_audio)),
        )
        sigma = (self._timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        return call_h3_final_layer(
            layer, x, t_emb, video_seg, audio_seg, sigma=sigma,
            sample_sigmas=options.get("sample_sigmas"), shifts=shifts,
        )


class _ExecutorView:
    def __init__(self, executor, timestep, options, hit_type=None):
        self._executor = executor
        self._hit_type = hit_type
        self.class_obj = _FinalizationView(executor.class_obj, timestep, options)

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def __call__(self, *args, **kwargs):
        # Keep pre-existing queue objects alive to prevent identity reuse. Native
        # H3 creates this invocation's queue inside _forward, after this snapshot.
        existing = ()
        if self._hit_type is not None:
            import comfy.model_prefetch
            existing = tuple(comfy.model_prefetch.PREFETCH_QUEUES)
            self.class_obj._prefetch_scope = None
            self.class_obj._prefetch_complete = False
        try:
            return self._executor(*args, **kwargs)
        except BaseException as error:
            if self._hit_type is not None and isinstance(error, self._hit_type):
                self.class_obj._prefetch_scope = frozenset(map(id, existing))
            raise


def cache_finalization_executor(executor, timestep, options, *, cache_wrapper=None):
    if not hasattr(executor.class_obj, "final_layer"):
        # Non-H3 executors cannot reach the external H3 finalization path.
        return executor
    return _ExecutorView(executor, timestep, options, _legacy_prefetch_hit_type(cache_wrapper))
