from __future__ import annotations

import asyncio

from .long_video_background import BACKGROUND_JOBS, BackgroundJobError


_ROUTES_REGISTERED = False


def _runtime_memory_snapshot(*, reset_peak: bool = False) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    divisor = 1024 * 1024
    return {
        "cuda_available": True,
        "device_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mib": torch.cuda.memory_allocated(device) / divisor,
        "reserved_mib": torch.cuda.memory_reserved(device) / divisor,
        "max_allocated_mib": torch.cuda.max_memory_allocated(device) / divisor,
        "max_reserved_mib": torch.cuda.max_memory_reserved(device) / divisor,
        "peak_reset": bool(reset_peak),
    }


def register_long_video_background_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    try:
        from aiohttp import web
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        if server is None:
            return False
    except ImportError:
        return False

    routes = server.routes

    @routes.get("/minimax_h3_t8/long_video/background/{chain_id}")
    async def background_status(request):
        try:
            state = await asyncio.to_thread(
                BACKGROUND_JOBS.status, request.match_info["chain_id"]
            )
            return web.json_response(state)
        except (BackgroundJobError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)

    @routes.post("/minimax_h3_t8/long_video/background/{chain_id}/{action}")
    async def background_control(request):
        chain_id = request.match_info["chain_id"]
        action = request.match_info["action"]
        try:
            if action == "pause":
                state = await asyncio.to_thread(BACKGROUND_JOBS.pause, chain_id)
            elif action == "resume":
                # resume validates and queues a prompt on PromptServer.loop. Running this
                # synchronous manager call on that same aiohttp loop would deadlock while it
                # waits for the validation coroutine.
                state = await asyncio.to_thread(BACKGROUND_JOBS.resume, chain_id)
            elif action == "cancel":
                state = await asyncio.to_thread(BACKGROUND_JOBS.cancel, chain_id)
            else:
                return web.json_response(
                    {"error": "action must be pause, resume, or cancel"}, status=404
                )
            return web.json_response(state)
        except (BackgroundJobError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=409)

    @routes.get("/minimax_h3_t8/runtime_memory")
    async def runtime_memory(_request):
        try:
            return web.json_response(
                await asyncio.to_thread(_runtime_memory_snapshot)
            )
        except Exception as error:  # noqa: BLE001 - diagnostic route reports runtime failures.
            return web.json_response({"error": str(error)}, status=500)

    @routes.post("/minimax_h3_t8/runtime_memory/reset_peak")
    async def runtime_memory_reset(_request):
        running, _queued = server.prompt_queue.get_current_queue()
        if running:
            return web.json_response(
                {"error": "Cannot reset CUDA peak counters while a prompt is running"},
                status=409,
            )
        try:
            return web.json_response(
                await asyncio.to_thread(_runtime_memory_snapshot, reset_peak=True)
            )
        except Exception as error:  # noqa: BLE001 - diagnostic route reports runtime failures.
            return web.json_response({"error": str(error)}, status=500)

    _ROUTES_REGISTERED = True
    return True
