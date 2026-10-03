"""Same-origin director services.

D1 owns the persistent project/asset contract. D2a–D2c add explicit,
validated native-generation routes that use Core's normal in-process queue.
D3 either compiles the four compatible H3 route patches or hands the user an
exact allow-listed native workflow; it never disguises a handoff as a queue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import wraps
from pathlib import Path
import time
import uuid
from copy import deepcopy

from .director_project import (
    ProjectConflict,
    ProjectStore,
    compile_project,
    contained,
    identity,
    new_project,
    validate_project,
    referenced_assets,
    atomic_json,
    sha,
)
from .director_generation import (
    build_director_generation_prompt,
    cancel_director_prompt,
    director_model_catalog,
    director_job_status,
    queue_director_prompt,
)
from .director_workflow_export import export_director_split_workflow
from .director_capabilities import inspect_director_capabilities
from .director_d3 import export_d3_package, handoff_d3_route, inspect_d3_routes
from .director_batch import (
    batch_status, capture_resources, create_batch, load_batch,
    retry_batch_item, verify_resources,
    batch_selection, selected_project, compile_batch_selection,
)

PREFIX = "/minimax_h3_t8/director"
_REGISTERED = False
_GENERATE_LOCK = asyncio.Lock()
_CORE_EPOCH = str(uuid.uuid4())
_FILM_EXPORT_LOCK = asyncio.Lock()
_FILM_EXPORT_TASKS = {}


def _record_deleted_queue_receipt(store, prompt_id):
    """Persist only a confirmed queue deletion, never a running interruption."""
    matches = []
    request_dir = contained(store.root, "requests")
    for path in request_dir.glob("*.json") if request_dir.is_dir() else ():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (isinstance(receipt, dict) and receipt.get("prompt_id") == prompt_id
                and receipt.get("state") == "queued" and receipt.get("terminal") is None):
            matches.append((path, receipt))
    if len(matches) != 1:
        return False
    path, receipt = matches[0]
    receipt["terminal"] = {"state": "cancelled", "outputs": {}}
    receipt["cancelled_at"] = time.time()
    atomic_json(path, receipt)
    return True


def _resolve_director_resource(folder, name):
    if folder == "hyperflow_selection":
        from .nodes_hyperflow_advanced import _resolve

        return _resolve(name)
    if folder == "semantic_bridge":
        from .nodes_semantic_bridge import resolve_model

        return resolve_model(name)
    import folder_paths

    path = folder_paths.get_full_path(folder, name)
    if not path:
        raise ValueError(f"冻结批次缺少所选模型：{folder}/{name}")
    return path


async def _submit_director_request_locked(store, project, shot_id, seed, request_id, client_id=None, *, built=None):
    """Exactly-once receipt and Core queue submission under _GENERATE_LOCK."""
    request_id = identity(request_id)
    shot_id = identity(shot_id)
    seed = int(seed)
    fingerprint = hashlib.sha256(json.dumps(
        {"project": project, "shot_id": shot_id, "seed": seed},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    receipt_path = contained(store.root, f"requests/{request_id}.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("fingerprint") != fingerprint:
            raise ValueError("同一个请求 ID 对应不同配置；请新建生成请求")
        if receipt.get("state") == "queued":
            return receipt["result"], 200
        return {"error": "提交状态尚未确认；请先按任务 ID 查询，不会自动重发", "prompt_id": receipt.get("prompt_id")}, 409
    if built is None:
        try:
            built = await asyncio.to_thread(
                build_director_generation_prompt, project, shot_id, store, seed=seed,
            )
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            # No receipt/reservation or queue operation exists at this point.
            # The client may release its pending ID only with this explicit proof;
            # ambiguous errors AFTER reservation must retain the original ID.
            return {"error": str(error), "submission_state": "not_submitted"}, 400
    prompt_id = str(uuid.uuid4())
    # Save the input AND compiled graph before queueing. Neither a later edit nor
    # an ambiguous queue response may replace this generation's configuration.
    snapshot = {
        "schema": "t8.director.generation_snapshot", "version": 1,
        "project": project, "shot_id": shot_id, "seed": seed,
        "prompt": built["prompt"], "report": built.get("report"),
        "recipe": built.get("recipe"), "sampling": built.get("sampling"),
        "d3_routes": built.get("d3_routes", []),
    }
    snapshot = deepcopy(snapshot)
    snapshot_fields = {"snapshot": snapshot, "snapshot_sha256": sha(snapshot)}
    atomic_json(receipt_path, {"state": "submitting", "fingerprint": fingerprint,
                               "prompt_id": prompt_id, "core_epoch": _CORE_EPOCH,
                               "project_id": project["id"], "shot_id": shot_id,
                               **snapshot_fields})
    try:
        prompt_id = await queue_director_prompt(built["prompt"], client_id, prompt_id=prompt_id)
    except Exception:
        # Unknown history cannot prove a queue write did not happen; retaining
        # the reservation prevents an automatic duplicate on a lost response.
        raise
    result = {
        "prompt_id": prompt_id, "recipe": built["recipe"],
        "d3_routes": built.get("d3_routes", []), "seed": built["seed"],
        "turbo_lora": built["turbo_lora"], "sampling": built["sampling"],
        "report": built["report"],
    }
    shot_source = next((shot for shot in project.get("doc", {}).get("shots", []) if shot.get("id") == shot_id), {})
    atomic_json(receipt_path, {
        "state": "queued", "fingerprint": fingerprint, "prompt_id": prompt_id,
        "core_epoch": _CORE_EPOCH,
        "project_id": project["id"], "shot_id": shot_id,
        "shot_rev": shot_source.get("rev"), "submitted_at": time.time(), "result": result,
        **snapshot_fields,
    })
    return result, 202


def director_result_snapshot(store, project_id, request_id):
    """Read one immutable source snapshot without inferring missing legacy data."""
    project_id, request_id = identity(project_id), identity(request_id)
    path = contained(store.root, f"requests/{request_id}.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("project_id") != project_id:
        raise ValueError("此版本不属于当前项目")
    snapshot = receipt.get("snapshot")
    if not isinstance(snapshot, dict):
        return {"complete": False, "reason": "旧版本未保存完整配置快照，不能精确还原"}
    if sha(snapshot) != receipt.get("snapshot_sha256"):
        raise ValueError("版本配置快照校验失败，不能用于还原")
    if (snapshot.get("project", {}).get("id") != project_id
            or snapshot.get("shot_id") != receipt.get("shot_id")):
        raise ValueError("版本配置快照身份不匹配")
    return {"complete": True, "sha256": receipt["snapshot_sha256"], "snapshot": snapshot}


def copy_version_project(store, project_id, request_id, new_project_id):
    """Explicitly save a new project; never overwrite the source or current draft."""
    new_project_id = identity(new_project_id)
    if new_project_id == identity(project_id):
        raise ValueError('版本草稿必须使用新项目身份，不能覆盖原工程')
    source = director_result_snapshot(store, project_id, request_id)
    if not source['complete']:
        raise ValueError(source['reason'])
    snapshot = source['snapshot']
    project = validate_project(snapshot['project'])
    selected = next((shot for shot in project['doc']['shots'] if shot['id'] == snapshot['shot_id']), None)
    if selected is None:
        raise ValueError('快照缺少生成镜头')
    seed = snapshot.get('seed')
    if type(seed) is not int or not 0 <= seed <= 2**53-1:
        raise ValueError('快照种子超出页面可精确保存范围；请保留原始快照，不舍入还原')
    reference = {'project_id': project_id, 'request_id': request_id, 'snapshot_sha256': source['sha256']}
    try:
        existing = store.load(new_project_id)
    except FileNotFoundError:
        existing = None
    if existing:
        if existing.get('versionSource') != reference:
            raise ValueError('此项目身份已用于不同内容，拒绝覆盖')
        return {'id': new_project_id, 'revision': existing['revision'], 'already_created': True}
    assets = {asset['id']: asset for asset in project['assets']}
    for asset_id in referenced_assets(project):
        original = assets.get(asset_id)
        actual = store.asset(asset_id, verify=True)
        if not original or any(original.get(key) != actual.get(key) for key in ('sha256', 'size', 'kind')):
            raise ValueError('快照素材身份已改变，不能静默替换后复制版本')
    project.update(id=new_project_id, revision=0, title=project['title'][:180]+' · 版本草稿',
                   current=selected['id'], versionSource=reference)
    selected['seed'] = seed
    for shot in project['doc']['shots']:
        shot.pop('adoptedResultId', None)
        shot.pop('filmTrim', None)
    saved = store.save(project, 0)
    return {'id': new_project_id, 'revision': saved['revision'], 'already_created': False}


def director_project_results(store, project_id, status_lookup=director_job_status, *, shot_ids=(), output_root=None):
    """Recover a project's queued and finished shots from durable request receipts.

    Core history may disappear after a restart, so terminal output metadata is
    copied into the receipt the first time it is observed.  The media remains
    in Core's output directory and is served by Core's normal /view route.
    """
    project_id = identity(project_id)
    from .director_bundle import imported_results
    records = imported_results(store, project_id, output_root)
    from .director_external import external_results
    records.extend(external_results(store, project_id, output_root))
    for path in (store.root / "requests").glob("*.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            result = receipt.get("result") or {}
            report = result.get("report") or {}
            owner = receipt.get("project_id") or report.get("project_id")
            if owner != project_id or receipt.get("state") != "queued":
                continue
            shot_id = receipt.get("shot_id") or (report.get("selection") or {}).get("shot_id")
            if not shot_id:
                continue
            prompt_id = identity(receipt["prompt_id"])
            terminal = receipt.get("terminal")
            if terminal is None:
                current = status_lookup(prompt_id)
                if current.get("state") in {"success", "error"}:
                    terminal = {"state": current["state"], "outputs": current.get("outputs") or {}}
                    receipt["terminal"] = terminal
                    atomic_json(path, receipt)
            records.append({
                "shot_id": shot_id,
                "prompt_id": prompt_id,
                "state": terminal["state"] if terminal else "pending",
                "outputs": terminal.get("outputs", {}) if terminal else {},
                "recipe": result.get("recipe", "director"),
                "submitted_at": receipt.get("submitted_at") or path.stat().st_mtime,
                "shot_rev": receipt.get("shot_rev"),
                "request_id": path.stem,
                "seed": result.get("seed"),
                "snapshot_available": isinstance(receipt.get("snapshot"), dict),
            })
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A damaged unrelated receipt must not hide the remaining films.
            continue
    if output_root is not None:
        # Older Director tasks predate request receipts. Their SafeAVSave prefix
        # is deterministic: T8_Director/<project UUID first 8>/<shot UUID first 8>.
        # Only recover files for shot identities supplied by this project page.
        root = Path(output_root).resolve()
        folder = root / "T8_Director" / project_id[:8]
        if not folder.resolve().is_relative_to(root):
            raise ValueError("成片目录越界")
        known_media = {
            (item.get("subfolder"), item.get("filename"))
            for record in records
            for output in record["outputs"].values()
            if isinstance(output, dict)
            for item in (output.get("images") or [])
            if isinstance(item, dict)
        }
        for raw_shot_id in shot_ids:
            shot_id = identity(raw_shot_id)
            for media in folder.glob(f"{shot_id[:8]}_*.mp4"):
                subfolder = f"T8_Director\\{project_id[:8]}"
                if (subfolder, media.name) in known_media:
                    continue
                records.append({
                    "shot_id": shot_id, "prompt_id": None, "state": "success",
                    "outputs": {"legacy": {"images": [{
                        "filename": media.name,
                        "subfolder": subfolder,
                        "type": "output",
                    }]}},
                    "recipe": "既有成片", "submitted_at": media.stat().st_mtime,
                    "shot_rev": None, "recovered_by": "project_and_shot_output_prefix",
                })
    records.sort(key=lambda item: (item["submitted_at"], item["prompt_id"] or ""), reverse=True)
    return {"project_id": project_id, "results": records}


def get_store():
    import folder_paths

    return ProjectStore(
        folder_paths.get_user_directory(), folder_paths.get_input_directory()
    )


def register_director_routes():
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        # CPU/schema tools do not necessarily have the Core server module initialized.
        return False
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return False
    routes = server.routes

    def guarded(function):
        @wraps(function)
        async def call(request):
            try:
                return await function(request)
            except ProjectConflict as error:
                return web.json_response(
                    {"error": str(error), "code": "revision_conflict"}, status=409
                )
            except FileNotFoundError:
                return web.json_response(
                    {"error": "项目或素材不存在，请重连；当前草稿不会被覆盖"},
                    status=404,
                )
            except (ValueError, KeyError, TypeError) as error:
                return web.json_response({"error": str(error)}, status=400)
            except Exception as error:
                import logging

                logging.exception("Director service failed")
                return web.json_response(
                    {
                        "error": f"保存／读取失败，可保留草稿后重试：{type(error).__name__}"
                    },
                    status=500,
                )

        return call

    @routes.get(PREFIX + "/projects")
    @guarded
    async def projects(_request):
        return web.json_response(
            {"projects": await asyncio.to_thread(get_store().list)}
        )

    @routes.get(PREFIX + "/projects/{project_id}")
    @guarded
    async def load(request):
        project = await asyncio.to_thread(
            get_store().load, request.match_info["project_id"]
        )
        return web.json_response(project)

    @routes.post(PREFIX + "/projects/{project_id}")
    @guarded
    async def save(request):
        body = await request.json()
        if identity(request.match_info["project_id"]) != body["project"]["id"]:
            raise ValueError("项目路径与内容身份不一致")
        result = await asyncio.to_thread(
            get_store().save, body["project"], body["expected_revision"]
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/compile")
    @guarded
    async def compile(request):
        body = await request.json()
        if "shot_ids" in body:
            result = await asyncio.to_thread(compile_batch_selection, body["project"], get_store(), body["shot_ids"])
            return web.json_response(result)
        result = await asyncio.to_thread(
            compile_project, body["project"], get_store(), shot_id=body.get("shot_id")
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/validate")
    @guarded
    async def validate(request):
        body = await request.json()
        return web.json_response({"project": validate_project(body["project"])})

    @routes.post(PREFIX + '/radar/operate')
    @guarded
    async def radar_operate(request):
        """Pure draft operation: no save/queue/provider/model or automatic adoption."""
        from .director_project import sha
        from .director_radar import apply_operation, candidate_status, evidence_status, evidence_time_map
        if request.content_length is not None and request.content_length > 2 * 1024**2:
            raise ValueError('证据/规则请求超过2MiB')
        body = await request.json()
        project = validate_project(body['project'])
        base_sha = sha(project)
        if 'base_json' in body:
            import hashlib
            import json
            from .director_project import canonical
            if not isinstance(body['base_json'], str) or canonical(json.loads(body['base_json'])) != canonical(project):
                raise ProjectConflict('请求原稿 JSON 与当前项目不一致')
            base_sha = hashlib.sha256(body['base_json'].encode('utf-8')).hexdigest()
        if body.get('base_sha256') != base_sha:
            raise ProjectConflict('请求草稿身份已改变，请重新打开证据/规则面板')
        store = get_store()
        try:
            saved = await asyncio.to_thread(store.load, project['id'])
        except FileNotFoundError:
            saved = None
        if saved is not None and saved['revision'] != project['revision']:
            raise ProjectConflict('服务端项目版本已变化，当前草稿不被覆盖')
        from .director_project import referenced_assets
        for aid in referenced_assets(project):
            actual = await asyncio.to_thread(store.asset, aid, verify=True)
            client = next((row for row in project['assets'] if row['id'] == aid), {})
            if actual['sha256'] != client.get('sha256'):
                raise ValueError('素材真实字节与草稿不一致，请重新登记')
        # Evidence imported in this operation can reference a registered library
        # asset that was not previously used by the shot.
        if body['operation'] == 'evidence_import':
            from .director_radar import _packet_media, seal_packet, verify_packet_timing, verify_audio_timing
            from .director_media import source_timing
            packet = seal_packet(body['value'])
            for source in _packet_media(packet):
                actual = await asyncio.to_thread(store.asset, source['asset_id'], verify=True)
                if actual['sha256'] != source['sha256']:
                    raise ValueError('证据来源真实 SHA 不符')
            source = await asyncio.to_thread(store.asset, packet['source']['asset_id'], verify=True)
            timing = await asyncio.to_thread(source_timing, contained(store.input_root, source['server_path']))
            verify_packet_timing(packet, timing)
            if (await asyncio.to_thread(store.asset, source['id'], verify=True))['sha256'] != source['sha256']:
                raise ValueError('测量期间证据来源发生变化')
            if packet.get('audio_source'):
                audio = await asyncio.to_thread(store.asset, packet['audio_source']['asset_id'], verify=True)
                audio_timing = await asyncio.to_thread(source_timing, contained(store.input_root, audio['server_path']))
                verify_audio_timing(packet['audio_source'], audio_timing)
                if (await asyncio.to_thread(store.asset, audio['id'], verify=True))['sha256'] != audio['sha256']:
                    raise ValueError('测量期间音频证据来源发生变化')
        result = apply_operation(project, body['shot_id'], body['operation'], body['value'])
        shot = next(row for row in result['doc']['shots'] if row['id'] == body['shot_id'])
        return web.json_response({'project': result, 'evidence_status': evidence_status(result, shot),
                                  'candidate_status': candidate_status(result, shot), 'queued': False, 'saved': False,
                                  'time_map': evidence_time_map(shot['sourceEvidence']['packet']) if shot.get('sourceEvidence') else None})

    @routes.get(PREFIX + '/radar/assets/{asset_id}/timing')
    @guarded
    async def radar_source_timing(request):
        from .director_media import source_timing
        store = get_store()
        asset = await asyncio.to_thread(store.asset, request.match_info['asset_id'], verify=True)
        if asset['kind'] != 'video':
            raise ValueError('源证据需要真实视频素材')
        timing = await asyncio.to_thread(source_timing, contained(store.input_root, asset['server_path']))
        if (await asyncio.to_thread(store.asset, asset['id'], verify=True))['sha256'] != asset['sha256']:
            raise ValueError('测量期间源视频改变')
        return web.json_response({'asset_id': asset['id'], 'sha256': asset['sha256'], **timing})

    @routes.post(PREFIX + "/export")
    @guarded
    async def export(request):
        body = await request.json()
        result = await asyncio.to_thread(
            compile_project, body["project"], get_store(), shot_id=body["shot_id"]
        )
        if not result["ready"]:
            return web.json_response(
                {
                    "error": "预检未通过，保留项目但不能导出可执行预检图",
                    "report": result,
                },
                status=422,
            )
        return web.json_response(
            {
                **export_preflight_workflow(body["project"], body["shot_id"]),
                "report": result,
            }
        )

    @routes.post(PREFIX + "/generate")
    @guarded
    async def generate(request):
        """Queue the validated D2a–D2c recipe selected by the current shot."""
        body = await request.json()
        store = get_store()
        request_id = identity(body.get("request_id") or str(uuid.uuid4()))
        async with _GENERATE_LOCK:
            result, status = await _submit_director_request_locked(
                store, body["project"], body["shot_id"],
                body.get("seed", 26091901), request_id, body.get("client_id"),
            )
        return web.json_response(result, status=status)

    @routes.get(PREFIX + "/batch-features")
    async def batch_features(_request):
        # UI files can refresh while an old Core still has its Python modules loaded.
        # Negotiate before accepting a subset so a legacy server cannot queue all shots.
        return web.json_response({"schema": "t8.director.batch_features.v1", "selection_version": 2})

    @routes.post(PREFIX + "/batches")
    @guarded
    async def start_batch(request):
        body = await request.json()
        store = get_store()
        project = body["project"]
        batch_id = identity(body["batch_id"])
        seed = int(body["seed"])
        shots, seeds = batch_selection(project, seed, body.get("shot_ids"), body.get("seed_map"))
        def same_request(existing):
            return (sha(existing["project"]) == sha(project)
                    and [item["shot_id"] for item in existing["items"]] == [shot["id"] for shot in shots]
                    and {item["shot_id"]: item["seed"] for item in existing["items"]} == seeds)
        async with _GENERATE_LOCK:
            try:
                existing = load_batch(store, batch_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not same_request(existing):
                    raise ValueError("批次身份已用于不同的项目或生成配置")
                return web.json_response({"id": existing["id"], "project_id": existing["project_id"], "fingerprint": existing["fingerprint"]})
        report = await asyncio.to_thread(compile_batch_selection, project, store, [shot["id"] for shot in shots])
        if not report["ready"]:
            return web.json_response({"error": "全部生成前检查未通过", "report": report}, status=422)
        async with _GENERATE_LOCK:
            try:
                existing = load_batch(store, batch_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not same_request(existing):
                    raise ValueError("批次身份已用于不同的项目或生成配置")
                return web.json_response({"id": existing["id"], "project_id": existing["project_id"], "fingerprint": existing["fingerprint"]})
            prepared = []
            for shot in shots:
                shot_project = {**project, "current": shot["id"]}
                prepared.append(await asyncio.to_thread(
                    build_director_generation_prompt, shot_project, shot["id"],
                    store, seed=seeds[shot["id"]],
                ))
            resources = await asyncio.to_thread(
                capture_resources, store, selected_project(project, shots), prepared, _resolve_director_resource,
                evidence_context=project,
            )
            batch = create_batch(store, batch_id, project, seed, prepared, resources,
                                 shot_ids=body.get("shot_ids"), seed_map=body.get("seed_map"))
        return web.json_response({"id": batch["id"], "project_id": batch["project_id"], "fingerprint": batch["fingerprint"]})

    @routes.get(PREFIX + "/batches/{batch_id}")
    @guarded
    async def read_batch(request):
        import folder_paths

        store = get_store()
        async with _GENERATE_LOCK:
            state = await asyncio.to_thread(
                batch_status, store, request.match_info["batch_id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
        return web.json_response(state)

    @routes.post(PREFIX + "/batches/{batch_id}/continue")
    @guarded
    async def continue_batch(request):
        import folder_paths

        body = await request.json()
        store = get_store()
        async with _GENERATE_LOCK:
            batch = load_batch(store, request.match_info["batch_id"])
            status = await asyncio.to_thread(
                batch_status, store, batch["id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
            index = status["next_index"]
            if index is None:
                return web.json_response({"complete": True, "batch": status})
            row = status["items"][index]
            if row["state"] != "not_submitted":
                return web.json_response({"error": "当前镜头任务尚未确认，不能重复提交", "batch": status}, status=409)
            item = batch["items"][index]
            project = {**batch["project"], "current": item["shot_id"]}
            await asyncio.to_thread(
                verify_resources, store, batch["resources"], _resolve_director_resource,
            )
            result, code = await _submit_director_request_locked(
                store, project, item["shot_id"], item["seed"],
                item["request_id"], body.get("client_id"), built=item["built"],
            )
        return web.json_response({**result, "shot_id": item["shot_id"], "batch_id": batch["id"]}, status=code)

    @routes.post(PREFIX + "/batches/{batch_id}/retry")
    @guarded
    async def retry_batch(request):
        import folder_paths

        body = await request.json()
        if body.get("confirm_abandoned") is not True:
            raise ValueError("请明确确认弃用旧尝试，不能由刷新页面自动重试")
        store = get_store()
        async with _GENERATE_LOCK:
            state = await asyncio.to_thread(
                batch_status, store, request.match_info["batch_id"], director_job_status,
                folder_paths.get_output_directory(), core_epoch=_CORE_EPOCH,
            )
            index = state["next_index"]
            if index is None:
                return web.json_response({"error": "批次全部完成，不可重试"}, status=409)
            row = state["items"][index]
            if not row["retry_available"]:
                return web.json_response({"error": "旧任务仍在当前 Core 中或提交状态未确认；不能重试", "batch": state}, status=409)
            batch = retry_batch_item(store, state["id"], index)
        return web.json_response({"batch_id": batch["id"], "index": index,
                                  "attempt": batch["items"][index]["attempt"],
                                  "previous_request_ids": batch["items"][index]["previous_request_ids"]})

    @routes.post(PREFIX + "/d3/compile")
    @guarded
    async def d3_compile(request):
        """Compile the selected D3 graph without submitting it to Core."""
        body = await request.json()
        built = await asyncio.to_thread(
            build_director_generation_prompt,
            body["project"],
            body["shot_id"],
            get_store(),
            seed=int(body.get("seed", 26091901)),
        )
        return web.json_response(
            {
                "schema": "t8.minimax_h3.director_d3_compiled_graph.v1",
                "recipe": built["recipe"],
                "d3_routes": built.get("d3_routes", []),
                "seed": built["seed"],
                "nodes": {
                    str(node_id): {
                        "class_type": node.get("class_type"),
                        "inputs": node.get("inputs", {}),
                    }
                    for node_id, node in built["prompt"].items()
                },
                "report": built["report"],
                "warning": "只编译图，不排队、不加载模型、不代表 GPU 或感知质量通过。",
            }
        )

    @routes.post(PREFIX + "/d3/editable-split-workflow")
    @guarded
    async def d3_editable_split_workflow(request):
        """Export an editable copy; never mutate the project or Core queue."""
        body = await request.json()
        result = await asyncio.to_thread(
            export_director_split_workflow,
            body["project"], body["shot_id"], get_store(),
            seed=int(body.get("seed", 26091901)),
        )
        return web.json_response(result)

    @routes.get(PREFIX + "/jobs/{prompt_id}")
    @guarded
    async def job(request):
        return web.json_response(
            director_job_status(request.match_info["prompt_id"])
        )

    @routes.get(PREFIX + "/results/{project_id}")
    @guarded
    async def results(request):
        import folder_paths

        shot_ids = request.query.getall("shot", [])
        if len(shot_ids) > 200:
            raise ValueError("一次最多查询 200 个镜头结果")
        async with _GENERATE_LOCK:
            records = await asyncio.to_thread(
                director_project_results, get_store(), request.match_info["project_id"],
                shot_ids=shot_ids, output_root=folder_paths.get_output_directory(),
            )
        return web.json_response(records)

    @routes.post(PREFIX + "/films/prepare")
    @guarded
    async def prepare_film_route(request):
        import folder_paths
        from .director_film import prepare_film

        body = await request.json()
        project = validate_project(body['project'])
        store = get_store()
        output = folder_paths.get_output_directory()
        async with _GENERATE_LOCK:
            records = await asyncio.to_thread(director_project_results, store, project['id'],
                                            shot_ids=[shot['id'] for shot in project['doc']['shots']], output_root=output)
        # Hashing/copying/decoding does not hold the generation submission lock.
        result = await asyncio.to_thread(prepare_film, store, project, records['results'], output)
        return web.json_response(result)

    @routes.post(PREFIX + '/bundles/prepare')
    @guarded
    async def bundle_prepare_route(request):
        import folder_paths
        from .director_bundle import prepare_bundle
        body = await request.json()
        project = validate_project(body['project'])
        store, output = get_store(), folder_paths.get_output_directory()
        records = await asyncio.to_thread(director_project_results, store, project['id'],
                                         shot_ids=[shot['id'] for shot in project['doc']['shots']], output_root=output)
        result = await asyncio.to_thread(prepare_bundle, store, project, records['results'], output,
                                         body.get('include_results', True), include_radar=body.get('include_radar', False))
        return web.json_response(result)

    @routes.post(PREFIX + '/bundles/{project_id}/{bundle_id}/build')
    @guarded
    async def bundle_build_route(request):
        from .director_bundle import build_bundle
        if (await request.json()).get('confirm_contents') is not True:
            raise ValueError('请先确认工程包媒体清单及私人内容')
        path = await asyncio.to_thread(build_bundle, get_store(), request.match_info['project_id'], request.match_info['bundle_id'])
        return web.json_response({'ready': True, 'bytes': path.stat().st_size})

    @routes.get(PREFIX + '/bundles/{project_id}/{bundle_id}/file')
    @guarded
    async def bundle_download_route(request):
        from .director_bundle import build_bundle
        store = get_store()
        # Download is read-only: an unbuilt package must be explicitly confirmed first.
        path = contained(store.root, f"bundle_exports/{identity(request.match_info['bundle_id'])}.zip")
        if not path.is_file():
            raise ValueError('工程包尚未确认生成')
        path = await asyncio.to_thread(build_bundle, store, request.match_info['project_id'], request.match_info['bundle_id'])
        return web.FileResponse(path, headers={'Content-Disposition': 'attachment; filename="director-project.zip"'})

    @routes.post(PREFIX + '/bundle-imports')
    @guarded
    async def bundle_upload_route(request):
        from .director_bundle import MAX_BYTES, MAX_JSON, inspect_bundle
        store = get_store()
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != 'file' or not part.filename or not part.filename.lower().endswith('.zip'):
            raise ValueError('请选择导演台工程ZIP文件')
        upload_id = str(uuid.uuid4())
        path = contained(store.root, f'bundle_imports/{upload_id}.zip')
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with path.open('xb') as stream:
                while chunk := await part.read_chunk(1024**2):
                    size += len(chunk)
                    if size > MAX_BYTES+MAX_JSON+4*1024**2:
                        raise ValueError('工程ZIP上传超过20GiB上限')
                    await asyncio.to_thread(stream.write, chunk)
            result = await asyncio.to_thread(inspect_bundle, store, upload_id)
            return web.json_response(result)
        except BaseException:
            # Remove only this request's incomplete/rejected upload, never existing projects.
            path.unlink(missing_ok=True)
            raise

    @routes.post(PREFIX + '/bundle-imports/{upload_id}/apply')
    @guarded
    async def bundle_import_route(request):
        import folder_paths
        from .director_bundle import import_bundle
        body = await request.json()
        if body.get('confirm_new_project') is not True:
            raise ValueError('请确认以全新工程导入')
        result = await asyncio.to_thread(import_bundle, get_store(), folder_paths.get_output_directory(),
                                         request.match_info['upload_id'], body['new_project_id'])
        return web.json_response(result)

    @routes.get(PREFIX + '/bundle-imports/{upload_id}')
    @guarded
    async def bundle_import_preview_route(request):
        from .director_bundle import import_preview
        result = await asyncio.to_thread(import_preview, get_store(), request.match_info['upload_id'])
        return web.json_response(result)

    @routes.get(PREFIX + "/films/{project_id}/{film_id}")
    @guarded
    async def film_manifest(request):
        from .director_film import load_film
        result = await asyncio.to_thread(load_film, get_store(), request.match_info['project_id'], request.match_info['film_id'])
        return web.json_response(result)

    @routes.get(PREFIX + "/films/{project_id}/{film_id}/media/{index}")
    @guarded
    async def film_media_route(request):
        from .director_film import film_media
        path = await asyncio.to_thread(film_media, get_store(), request.match_info['project_id'],
                                       request.match_info['film_id'], int(request.match_info['index']))
        return web.FileResponse(path)

    @routes.post(PREFIX + "/films/{project_id}/{film_id}/frames")
    @guarded
    async def film_frame_route(request):
        from .director_frame import extract_frame
        body = await request.json()
        result = await asyncio.to_thread(extract_frame, get_store(), request.match_info['project_id'],
                                         request.match_info['film_id'], body['index'], body['frame'])
        return web.json_response(result)

    @routes.post(PREFIX + "/films/{project_id}/{film_id}/exports/{job_id}")
    @guarded
    async def film_export_start(request):
        from .director_film_export import reserve_export, run_export
        body = await request.json()
        owner, film_id, job_id = (identity(request.match_info[key]) for key in ('project_id', 'film_id', 'job_id'))
        store = get_store()
        async with _FILM_EXPORT_LOCK:
            exists = contained(store.root, f'film_exports/{job_id}.json').exists()
            if not exists and any(not task.done() for task in _FILM_EXPORT_TASKS.values()):
                raise ValueError('已有整片导出在运行，请等它完成后再导出；没有提交重复编码')
            job, created = await asyncio.to_thread(reserve_export, store, owner, film_id, job_id, body)
            if created:
                async def work():
                    try:
                        await asyncio.to_thread(run_export, store, owner, film_id, job_id)
                    finally:
                        _FILM_EXPORT_TASKS.pop(job_id, None)
                _FILM_EXPORT_TASKS[job_id] = asyncio.create_task(work())
        return web.json_response(job)

    @routes.get(PREFIX + "/films/{project_id}/{film_id}/exports/{job_id}")
    @guarded
    async def film_export_progress(request):
        from .director_film_export import export_status
        args = tuple(request.match_info[key] for key in ('project_id', 'film_id', 'job_id'))
        job = await asyncio.to_thread(export_status, get_store(), *args)
        if job['state'] in {'queued', 'running'} and args[2] not in _FILM_EXPORT_TASKS:
            job = {**job, 'state': 'interrupted', 'error': 'Core 已重启或导出进程不在当前实例；未自动重新编码，请明确另起导出。'}
        return web.json_response(job)

    @routes.get(PREFIX + "/films/{project_id}/{film_id}/exports/{job_id}/file")
    @guarded
    async def film_export_download(request):
        from .director_film_export import export_file
        path = await asyncio.to_thread(export_file, get_store(), *(request.match_info[key] for key in ('project_id', 'film_id', 'job_id')))
        return web.FileResponse(path, headers={'Content-Disposition': 'attachment; filename="director-film.mp4"'})

    @routes.get(PREFIX + "/snapshots/{project_id}/{request_id}")
    @guarded
    async def generation_snapshot(request):
        result = await asyncio.to_thread(
            director_result_snapshot, get_store(), request.match_info["project_id"],
            request.match_info["request_id"],
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/snapshots/{project_id}/{request_id}/copy")
    @guarded
    async def copy_version(request):
        body = await request.json()
        result = await asyncio.to_thread(copy_version_project, get_store(), request.match_info['project_id'],
                                         request.match_info['request_id'], body['new_project_id'])
        return web.json_response(result)

    @routes.post(PREFIX + "/external-takes")
    @guarded
    async def register_external(request):
        from .director_external import register_external_take, registration_absent
        import folder_paths
        if request.content_length is not None and request.content_length > 2 * 1024**2:
            raise ValueError('外片登记请求超过2MiB')
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'project', 'shot_id', 'asset_id', 'take_id', 'label', 'provenance_category'}:
            raise ValueError('外片登记字段不完整，不接受模型/seed等虚构生成配置')
        project = validate_project(body['project'])
        store, output_root = get_store(), folder_paths.get_output_directory()
        value = {key: body[key] for key in ('shot_id', 'asset_id', 'take_id', 'label', 'provenance_category')}
        value.update(project_id=project['id'], expected_revision=project['revision'], project_sha256=sha(project))
        try:
            result = await asyncio.to_thread(register_external_take, store, value, output_root)
        except (ValueError, OSError) as error:
            absent = await asyncio.to_thread(registration_absent, store, project['id'], value['take_id'], output_root)
            return web.json_response({'error': str(error), 'registration_state': 'not_registered' if absent else 'unknown_or_registered'},
                                     status=409 if isinstance(error, ProjectConflict) else 400)
        return web.json_response(result)

    @routes.post(PREFIX + "/jobs/{prompt_id}/cancel")
    @guarded
    async def cancel(request):
        prompt_id = identity(request.match_info["prompt_id"])
        async with _GENERATE_LOCK:
            result = cancel_director_prompt(prompt_id)
            if result["deleted_from_queue"]:
                result["receipt_recorded"] = _record_deleted_queue_receipt(get_store(), prompt_id)
        return web.json_response(result)

    @routes.post(PREFIX + "/assets")
    @guarded
    async def upload(request):
        store = get_store()
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file" or not part.filename:
            raise ValueError("上传必须包含 file")
        asset_id = str(uuid.uuid4())
        suffix = Path(part.filename).suffix.lower()
        if suffix not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
        }:
            raise ValueError("请选择标准图片、视频或音频文件")
        path = contained(store.input_root, f"t8_director/{asset_id}/source{suffix}")
        path.parent.mkdir(parents=True, exist_ok=False)
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await part.read_chunk(1024 * 1024):
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise ValueError(
                            "单素材上传上限1GiB，请分段或压缩后重试；未截断保存"
                        )
                    await asyncio.to_thread(stream.write, chunk)
            asset = await asyncio.to_thread(
                store.register_asset, path, asset_id, part.filename
            )
            return web.json_response(asset, status=201)
        except BaseException:
            path.unlink(missing_ok=True)  # Only this incomplete, server-created upload.
            path.parent.rmdir()
            raise

    @routes.get(PREFIX + "/assets/{asset_id}")
    @guarded
    async def asset(request):
        store = get_store()
        asset = await asyncio.to_thread(store.asset, request.match_info["asset_id"])
        return web.FileResponse(contained(store.input_root, asset["server_path"]))

    @routes.get(PREFIX + "/assets/{asset_id}/input-preview")
    @guarded
    async def input_preview(request):
        store = get_store()
        prepared = await asyncio.to_thread(
            store.prepare_image,
            request.match_info["asset_id"],
            int(request.query["width"]),
            int(request.query["height"]),
        )
        return web.FileResponse(contained(store.input_root, prepared["server_path"]))

    @routes.get(PREFIX + "/ui")
    async def ui(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "index.html"
        )

    @routes.get(PREFIX + "/session.mjs")
    async def session(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "session.mjs"
        )

    @routes.get(PREFIX + "/sampling_ui.mjs")
    async def sampling_ui(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "sampling_ui.mjs"
        )

    @routes.get(PREFIX + "/radar_state.mjs")
    async def radar_state(_request):
        # session.mjs is also served at PREFIX, so its static relative import
        # must resolve here as well as under Core's /extensions directory.
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "radar_state.mjs"
        )

    @routes.get(PREFIX + "/default")
    async def default(_request):
        return web.json_response(new_project())

    @routes.get(PREFIX + "/capabilities")
    @guarded
    async def capabilities(_request):
        """Report D3 native entry points without pretending to queue them."""
        import nodes

        return web.json_response(
            inspect_director_capabilities(nodes.NODE_CLASS_MAPPINGS.keys())
        )

    @routes.get(PREFIX + "/models")
    @guarded
    async def models(_request):
        """Return only installed H3-compatible models for the Director selectors."""
        return web.json_response(director_model_catalog())

    @routes.get(PREFIX + "/d3/routes")
    @guarded
    async def d3_routes(_request):
        """List D3 native hand-off routes without touching a project or queue."""
        import nodes

        return web.json_response(inspect_d3_routes(node_ids=nodes.NODE_CLASS_MAPPINGS.keys()))

    @routes.post(PREFIX + "/d3/preflight")
    @guarded
    async def d3_preflight(request):
        """Preflight one saved Director shot before handing it to a D3 route."""
        body = await request.json()
        return web.json_response(
            await asyncio.to_thread(
                inspect_d3_routes,
                body.get("project"),
                body.get("shot_id"),
                get_store(),
                capability=body.get("capability"),
            )
        )

    @routes.post(PREFIX + "/d3/handoff")
    @guarded
    async def d3_handoff(request):
        """Return an exact allow-listed native workflow/README without queuing."""
        body = await request.json()
        result = await asyncio.to_thread(
            handoff_d3_route,
            str(body.get("capability", "")),
            body.get("file"),
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/d3/package")
    @guarded
    async def d3_package(request):
        """Export the current project plus an exact native route hand-off."""
        body = await request.json()
        result = await asyncio.to_thread(
            export_d3_package,
            str(body.get("capability", "")),
            body.get("project"),
            body.get("shot_id"),
            get_store(),
        )
        return web.json_response(result)

    _REGISTERED = True
    return True


def export_preflight_workflow(project, shot_id):
    """A real native CPU contract graph, not a mislabeled runnable GPU recipe."""
    from .director_project import canonical, validate_project

    project = validate_project(project)
    if shot_id not in {s["id"] for s in project["doc"]["shots"]}:
        raise ValueError("镜头身份不存在")
    values = [canonical(project), shot_id]
    api = {
        "1": {
            "class_type": "MiniMaxH3DirectorProjectT8",
            "inputs": {"project_json": values[0], "shot_id": shot_id},
            "_meta": {"title": "曜石导演台 · D1 CPU预检（不生成）"},
        }
    }
    workflow = {
        "id": str(uuid.uuid4()),
        "version": 0.4,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "MiniMaxH3DirectorProjectT8",
                "pos": [160, 140],
                "size": [520, 260],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": name, "type": kind, "links": None}
                    for name, kind in (
                        ("compiled_prompt", "STRING"),
                        ("width", "INT"),
                        ("height", "INT"),
                        ("length", "INT"),
                        ("media_map_json", "STRING"),
                        ("report_json", "STRING"),
                    )
                ],
                "properties": {
                    "Node name for S&R": "MiniMaxH3DirectorProjectT8",
                    "cnr_id": "minimax-h3-audio-t8",
                },
                "widgets_values": values,
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "t8_director": {
                "project_id": project["id"],
                "scope": "D1 CPU preflight only; not generation",
            }
        },
    }
    return {"workflow": workflow, "api_snapshot": api}
