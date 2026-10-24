"""Inert owned HTTP/WS transport emitter, never Core/MODEL/GPU execution."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import web


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--_serve', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args._serve.read_bytes())
    root = Path(plan['run_root'])
    state = dict(posts=0, gets=0, ws_connections=0, clients={}, fixture_only=True)

    def evidence():
        # Fixture state only, never a product file or a completion receipt.
        (root / 'fixture-state.json').write_text(json.dumps({key: value for key, value in state.items() if key != 'clients'}), encoding='utf8')

    async def ready(request):
        return web.json_response({'fixture_only': True})

    async def ws(request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        client = request.query['clientId']
        state['clients'][client] = sock
        state['ws_connections'] += 1
        evidence()
        await sock.send_json(dict(type='status', data=dict(sid=client)))
        async for _ in sock:
            pass
        return sock

    async def post(request):
        value = await request.json()
        state['posts'] += 1
        state['POST_received_at'] = time.time()
        evidence()
        sock = state['clients'][value['client_id']]
        for kind, data in (
            ('execution_start', dict(prompt_id='fixture-prompt')),
            ('executing', dict(prompt_id='fixture-prompt', node='1')),
            ('execution_success', dict(prompt_id='fixture-prompt')),
        ):
            await sock.send_json(dict(type=kind, data=data))
        return web.json_response(dict(prompt_id='fixture-prompt', node_errors={}))

    async def history(request):
        state['gets'] += 1
        first = state['gets'] == 1
        state['last_GET_started_at'] = time.time()
        evidence()
        if first:
            # Exceeds the real unchanged OwnedServer.request read_timeout=2.
            await asyncio.sleep(2.4)
        state['last_GET_responded_at'] = time.time()
        evidence()
        return web.json_response({'fixture-prompt': dict(status=dict(status_str='success', completed=True), outputs={})})

    app = web.Application()
    app.router.add_get('/ready', ready)
    app.router.add_get('/ws', ws)
    app.router.add_post('/prompt', post)
    app.router.add_get('/history/{prompt}', history)
    web.run_app(app, host='127.0.0.1', port=plan['port'], print=None, handle_signals=False)


if __name__ == '__main__':
    main()
