from __future__ import annotations

import asyncio

from .long_video_background import BACKGROUND_JOBS, BackgroundJobError


_ROUTES_REGISTERED = False


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

    _ROUTES_REGISTERED = True
    return True
