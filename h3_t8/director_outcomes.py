"""Pure request-outcome validation, not a queue, tracker or new store."""
from copy import deepcopy
from collections import Counter
import json

from .director_project import contained, identity, sha


TERMINAL_STATES = frozenset({"success", "error", "cancelled"})


class AmbiguousPrompt(ValueError):
    pass


def scan_request_receipts(store):
    """One read-only observation snapshot; never hide storage IO failures.

    Malformed unrelated JSON has no usable identity. A readable dict's prompt
    still counts before any owner, submission-state or terminal filtering.
    The caller holds the existing observation lock; nothing is cached here.
    """
    request_dir = contained(store.root, "requests")
    try:
        paths = sorted(request_dir.iterdir())
    except FileNotFoundError:
        return {}, Counter()
    records, prompts = {}, Counter()
    for entry in paths:
        if entry.suffix.lower() != ".json":
            continue
        path = contained(store.root, entry.relative_to(store.root))
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, TypeError):
            receipt = None
        # Keep presence in this same snapshot even for a malformed record.
        # Batch observers must not reread it or mistake it for unsubmitted.
        records[path] = receipt
        if isinstance(receipt, dict):
            prompt_id = receipt.get("prompt_id")
            if isinstance(prompt_id, str):
                prompts[prompt_id] += 1
    return records, prompts


def require_unique_prompt(prompts, prompt_id):
    if prompts.get(prompt_id, 0) != 1:
        raise AmbiguousPrompt("任务 ID 对应的生成回执不唯一；没有猜测归属、写入终态或自动重试。")


def terminal_payload(observed):
    if not isinstance(observed, dict):
        return None
    state = observed.get("state")
    outputs = observed.get("outputs", {})
    if not isinstance(state, str) or state not in TERMINAL_STATES or not isinstance(outputs, dict):
        return None
    payload = {"state": state, "outputs": deepcopy(outputs)}
    if isinstance(observed.get("status"), dict):
        payload["status"] = deepcopy(observed["status"])
    return payload


def receipt_owner(receipt):
    """Validate known bindings; legacy receipts need no invented snapshot."""
    result = receipt.get("result")
    result = {} if result is None else result
    if not isinstance(result, dict):
        raise ValueError("生成回执结果形状损坏")
    report = result.get("report")
    report = {} if report is None else report
    if not isinstance(report, dict):
        raise ValueError("生成回执报告形状损坏")
    selection = report.get("selection")
    selection = {} if selection is None else selection
    if not isinstance(selection, dict):
        raise ValueError("生成回执镜头归属损坏")
    owners = [identity(value) for value in (receipt.get("project_id"), report.get("project_id")) if value is not None]
    shots = [identity(value) for value in (receipt.get("shot_id"), selection.get("shot_id")) if value is not None]
    if not owners or not shots or len(set(owners)) != 1 or len(set(shots)) != 1:
        raise ValueError("生成回执项目或镜头归属不一致")
    owner, shot = owners[0], shots[0]
    snapshot = receipt.get("snapshot")
    if snapshot is not None or receipt.get("snapshot_sha256") is not None:
        if not isinstance(snapshot, dict) or sha(snapshot) != receipt.get("snapshot_sha256"):
            raise ValueError("生成回执配置快照校验失败")
        project = snapshot.get("project")
        if (not isinstance(project, dict) or project.get("id") != owner or snapshot.get("shot_id") != shot):
            raise ValueError("生成回执配置快照归属不一致")
    return owner, shot
