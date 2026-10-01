"""Opt-in two-shot Director batch acceptance on two controller-owned Cores.

Default only describes the plan. --run --exclusive-gpu-ack is required to start
GPU work. Never connects to the user's Core, modifies its queue, or saves a
user project. Uses unique owned user/output directories across a hard restart.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/minimax_h3_t8/director"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def call(server, method, path, payload=None, *, expected=200):
    server.assert_port_owner()
    with requests.Session() as client:
        client.trust_env = False
        result = client.request(method, server.url + path, json=payload, timeout=(5, 600))
    accepted = {expected} if isinstance(expected, int) else set(expected)
    if result.status_code not in accepted:
        raise RuntimeError(f"{method} {path}: {result.status_code} {result.text[:3000]}")
    return result.json()


def assert_idle(server):
    queue = call(server, "GET", "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Owned queue is not empty at an acceptance boundary")
    return queue


def source_identity():
    paths = sorted([*(ROOT / "h3_t8").rglob("*.py"), *ROOT.glob("*.py")])
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def stop_servers(servers):
    """Attempt every owned cleanup, preserving failures instead of masking them."""
    stops, errors = [], []
    for server in reversed(servers):
        try:
            server.stop()
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
        stops.append(server.stop_receipt)
    return stops, errors


def build_server_command(base_command, shared, *values, **kwargs):
    result = base_command(*values, **kwargs)
    for flag, folder in (("--input-directory", "input"), ("--output-directory", "output"), ("--user-directory", "user")):
        result[result.index(flag) + 1] = str(shared / folder)
    # The existing helper places the nargs+ whitelist immediately before this
    # option. Add VHS to that whitelist, not to the configuration-file values.
    result.insert(result.index("--extra-model-paths-config"), "ComfyUI-VideoHelperSuite")
    result += ["--disable-comfy-compiler", "--use-pytorch-cross-attention"]
    return result


def configure_project(project, seconds, mp, *, sampling_mode="single"):
    project = deepcopy(project)
    project["title"] = "HR1 owned cross-process acceptance"
    project["doc"]["global"] = "Stable cinematic composition; natural ambience; no dialogue, music or subtitles."
    first = project["doc"]["shots"][0]
    first.update(mode="text", sound="native", first=None, last=None, refs=[], tray=[], audio=None,
                 name="Owned shot 1", simplePrompt="A candle flame flickers on a wooden desk.",
                 writingMode="simple", simpleInitialized=True, duration=seconds,
                 manualDuration=seconds, autoDuration=False, ratio="1:1", ownRatio="1:1")
    second = deepcopy(first)
    second.update(id=str(uuid.uuid4()), name="Owned shot 2", simplePrompt="A small brass bell sways gently on a wooden desk.")
    project["doc"].update(shots=[first, second], sharedRefs=[], sharedRatio=True, ratio="1:1")
    project["current"] = first["id"]
    project["doc"]["generation"].update(
        unet="minimax_h3_fl2va_int8_convrot.safetensors",
        clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
        audio_vae="minimax_h3_audio_vae_fp32.safetensors", resolution_mp=mp)
    if sampling_mode == "hyperflow":
        project["doc"]["generation"]["lora_mode"] = "none"
        project["doc"]["sampling"] = {
            "mode": "hyperflow", "variant": "single8", "output_mp": mp,
            "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors",
            "low_loras": [], "high_loras": [],
        }
    else:
        project["doc"]["sampling"] = {"mode": "single", "resolution_mp": mp, "lora_mode": "manual",
            "loras": [{"id": "accepted-turbo", "name": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors",
                       "strength": 1.0, "enabled": True}]}
    return project


def wait_shot(server, batch_id, index, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        state = call(server, "GET", PREFIX + "/batches/" + batch_id)
        row = state["items"][index]
        if row["state"] != last:
            print(json.dumps({"shot": index + 1, "state": row["state"]}), flush=True)
            last = row["state"]
        if row["state"] == "success":
            return state
        if row["state"] in {"error", "needs_review", "unknown"}:
            raise RuntimeError("Batch requires review: " + json.dumps(state))
        time.sleep(2)
    raise TimeoutError("Owned shot did not finish by deadline")


def selected_probe_project(project):
    """A deliberately incomplete unselected middle shot must never be prepared."""
    result = deepcopy(project)
    first, last = result['doc']['shots']
    middle = deepcopy(first)
    middle.update(id=str(uuid.uuid4()), name='UNSELECTED missing inputs', mode='ends',
                  sound='record', simplePrompt='', first=None, last=None, audio=None)
    first['seed'], last['seed'] = 0, 4294967295
    result['doc']['shots'] = [first, middle, last]
    return result, [last['id'], first['id']], {first['id']: 0, last['id']: 4294967295}


def check_selection_frozen(frozen, project, seed_map):
    expected = [shot['id'] for shot in project['doc']['shots'] if shot['id'] in seed_map]
    if (not frozen['schema'].endswith('.v2') or len(frozen['project']['doc']['shots']) != 3
            or [item['shot_id'] for item in frozen['items']] != expected
            or any(item['seed'] != seed_map[item['shot_id']] or item['built']['seed'] != item['seed'] for item in frozen['items'])):
        raise RuntimeError('v2 selection, project order or exact seed map changed')


def verify_media(output, row, built):
    import av

    details = []
    for evidence in row["media"]:
        path = (output / evidence["file"]).resolve(strict=True)
        if not path.is_relative_to(output.resolve()):
            raise RuntimeError("Media evidence escaped owned output")
        with av.open(str(path)) as container:
            video = container.streams.video[0]
            frames = sum(1 for _ in container.decode(video))
            shape = [video.width, video.height]
        with av.open(str(path)) as container:
            audio = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio))
            audio_seconds = samples / audio.rate
        shot = built["shot"]
        expected_frames = shot["time"]["delivery_trim_frames"]
        expected_shape = [shot["canvas"]["width"], shot["canvas"]["height"]]
        expected_seconds = shot["time"]["requested_seconds"]
        if frames != expected_frames or shape != expected_shape or abs(audio_seconds - expected_seconds) > .15:
            raise RuntimeError(f"Incomplete/mismatched AV delivery: {frames=} {shape=} {audio_seconds=}")
        actual_sha = digest(path)
        if actual_sha != evidence["sha256"]:
            raise RuntimeError("Media changed after batch verification")
        details.append({"file": str(path), "sha256": actual_sha, "frames": frames,
                        "width_height": shape, "audio_samples": samples, "audio_seconds": audio_seconds})
    if not details:
        raise RuntimeError("Successful shot has no full AV output")
    return details


def rejection_probes(server, batch_id, receipt_path, first_media, evidence):
    """Mutate only own receipt/media, preserve originals, require zero submissions."""
    receipt_bytes = receipt_path.read_bytes()
    media_path = Path(first_media[0]["file"])
    media_bytes = media_path.read_bytes()
    write_new(evidence / "pre-fault-sha.json", {"receipt": digest(receipt_path), "media": digest(media_path)})
    try:
        bad = json.loads(receipt_bytes)
        bad["fingerprint"] = "owned-probe-invalid-fingerprint"
        receipt_path.write_text(json.dumps(bad), encoding="utf-8")
        status = call(server, "GET", PREFIX + "/batches/" + batch_id)
        if status["items"][0]["state"] != "needs_review":
            raise RuntimeError("Bad receipt was accepted")
        rejected = call(server, "POST", PREFIX + "/batches/" + batch_id + "/continue", {}, expected=409)
        write_new(evidence / "bad-receipt-rejected.json", {"status": status, "continue": rejected, "queue": assert_idle(server)})
    finally:
        receipt_path.write_bytes(receipt_bytes)
    try:
        media_path.write_bytes(media_bytes + b"\nT8-owned-media-corruption-probe\n")
        status = call(server, "GET", PREFIX + "/batches/" + batch_id)
        if status["items"][0]["state"] != "needs_review":
            raise RuntimeError("Changed media was accepted")
        rejected = call(server, "POST", PREFIX + "/batches/" + batch_id + "/continue", {}, expected=409)
        write_new(evidence / "bad-media-rejected.json", {"status": status, "continue": rejected, "queue": assert_idle(server)})
    finally:
        media_path.write_bytes(media_bytes)
        receipt_path.write_bytes(receipt_bytes)
    if digest(media_path) != first_media[0]["sha256"]:
        raise RuntimeError("Owned fault probe failed to restore original media")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--exclusive-gpu-ack", action="store_true")
    parser.add_argument("--core", type=Path, default=ROOT.parents[1])
    parser.add_argument("--port", type=int, default=8871)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--mp", type=float, default=.2)
    parser.add_argument("--sampling", choices=("single", "hyperflow"), default="single")
    parser.add_argument("--selection-version", choices=(1, 2), type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "plan_only", "gpu_started": False,
            "plan": "Two owned Cores, fixed 2-shot AV batch, hard restart after verified shot 1, bad receipt/media rejection, continue shot 2 without recomputing shot 1."}))
        return 0
    if not args.exclusive_gpu_ack or os.name != "nt":
        raise RuntimeError("Requires explicit exclusive GPU acknowledgment and Windows hard-stop semantics")
    if not 0 < args.seconds <= 4 or not .2 <= args.mp <= .5:
        raise ValueError("This bounded acceptance allows at most 4 seconds / 0.5 MP")
    sys.path.insert(0, str(ROOT / "tools"))
    import run_progressive_pilot as pilot
    from vdn_probe_environment import probe_resource_config

    evidence = ROOT / "artifacts/development" / ("director-batch-restart-" + uuid.uuid4().hex[:10])
    evidence.mkdir(parents=True, exist_ok=False)
    shared = evidence / "state"
    for name in ("input", "output", "user"):
        (shared / name).mkdir(parents=True)
    sources = source_identity()
    write_new(evidence / "sources.json", sources)
    pilot.CORE = args.core.resolve(strict=True)
    base_command = pilot.server_command

    def command(*values, **kwargs):
        return build_server_command(base_command, shared, *values, **kwargs)

    pilot.server_command = command
    servers = []
    summary = {"status": "incomplete", "evidence": str(evidence), "core_runs": []}
    try:
        def start(number):
            if source_identity() != sources:
                raise RuntimeError("Source changed between Core runs; refusing mixed-code acceptance")
            phase = evidence / f"core-{number}"
            phase.mkdir()
            write_new(phase / "paths.json", probe_resource_config(pilot.CORE, ROOT))
            server = pilot.OwnedServer(phase, args.port, False)
            servers.append(server)
            server.start()
            pilot.wait_ready(server, lambda: None, seconds=240)
            assert_idle(server)
            return server

        first_core = start(1)
        project = configure_project(call(first_core, "GET", PREFIX + "/default"), args.seconds, args.mp,
                                    sampling_mode=args.sampling)
        selection = {}
        if args.selection_version == 2:
            features = call(first_core, 'GET', PREFIX+'/batch-features')
            if features.get('selection_version') != 2:
                raise RuntimeError('Core does not support v2 subset batches')
            project, shot_ids, seed_map = selected_probe_project(project)
            selection = {'shot_ids': shot_ids, 'seed_map': seed_map}
        write_new(evidence / "project.json", project)
        batch_id = str(uuid.uuid4())
        payload = {"batch_id": batch_id, "project": project, "seed": 26092201, **selection}
        created = call(first_core, "POST", PREFIX + "/batches", payload)
        if call(first_core, 'POST', PREFIX+'/batches', payload) != created:
            raise RuntimeError('Same batch request was not idempotent')
        if args.selection_version == 2:
            saved = call(first_core, 'POST', PREFIX+'/projects/'+project['id'], {'project': project, 'expected_revision': 0})
            edited = deepcopy(saved)
            edited['doc']['shots'].reverse()
            for shot in edited['doc']['shots']:
                shot['seed'] = 123456
            call(first_core, 'POST', PREFIX+'/projects/'+project['id'], {'project': edited, 'expected_revision': saved['revision']})
            write_new(evidence/'reordered-live-project.json', edited)
        submitted1 = call(first_core, "POST", PREFIX + "/batches/" + batch_id + "/continue", {}, expected=(200, 202))
        complete1 = wait_shot(first_core, batch_id, 0, args.timeout)
        batch_path = shared / "user/t8_director/batches" / (batch_id + ".json")
        frozen = json.loads(batch_path.read_text(encoding="utf-8"))
        if args.selection_version == 2:
            check_selection_frozen(frozen, project, seed_map)
        receipt1 = shared / "user/t8_director/requests" / (frozen["items"][0]["request_id"] + ".json")
        record1 = json.loads(receipt1.read_text(encoding="utf-8"))
        media1 = verify_media(shared / "output", complete1["items"][0], frozen["items"][0]["built"])
        summary.update(batch_id=batch_id, first_prompt_id=submitted1["prompt_id"], first_media=media1)
        summary["core_runs"].append({"pid": first_core.process.pid, "epoch": record1["core_epoch"]})
        write_new(evidence / "first-verified.json", {"batch": complete1, "receipt": record1, "media": media1})
        assert_idle(first_core)
        first_core.stop()  # TerminateProcess on Windows: no graceful Python shutdown.
        write_new(evidence / "core-1-stop.json", first_core.stop_receipt)

        second_core = start(2)
        if args.selection_version == 2:
            check_selection_frozen(json.loads(batch_path.read_text(encoding='utf-8')), project, seed_map)
            if record1['snapshot']['seed'] != 0:
                raise RuntimeError('First request did not snapshot exact seed zero')
        recovered = call(second_core, "GET", PREFIX + "/batches/" + batch_id)
        if recovered["next_index"] != 1 or recovered["items"][0]["prompt_id"] != submitted1["prompt_id"]:
            raise RuntimeError("Restart did not retain the verified first shot")
        if call(second_core, "GET", "/history/" + submitted1["prompt_id"]):
            raise RuntimeError("Expected a genuinely new process with empty old Core history")
        rejection_probes(second_core, batch_id, receipt1, media1, evidence)
        submitted2 = call(second_core, "POST", PREFIX + "/batches/" + batch_id + "/continue", {}, expected=(200, 202))
        if submitted2.get("shot_id") != frozen["items"][1]["shot_id"] or submitted2["prompt_id"] == submitted1["prompt_id"]:
            raise RuntimeError("Continue did not submit exactly the remaining second shot")
        complete2 = wait_shot(second_core, batch_id, 1, args.timeout)
        media2 = verify_media(shared / "output", complete2["items"][1], frozen["items"][1]["built"])
        receipt2 = shared / "user/t8_director/requests" / (frozen["items"][1]["request_id"] + ".json")
        record2 = json.loads(receipt2.read_text(encoding="utf-8"))
        if args.selection_version == 2 and record2['snapshot']['seed'] != 4294967295:
            raise RuntimeError('Restarted second request did not retain its frozen seed')
        if record2["core_epoch"] == record1["core_epoch"] or second_core.process.pid == first_core.process.pid:
            raise RuntimeError("Core PID/epoch did not change")
        history2 = call(second_core, "GET", "/history")
        if set(history2) != {submitted2["prompt_id"]}:
            raise RuntimeError("Unexpected extra tasks in restarted owned Core")
        if digest(Path(media1[0]["file"])) != media1[0]["sha256"] or json.loads(receipt1.read_text(encoding="utf-8"))["prompt_id"] != submitted1["prompt_id"]:
            raise RuntimeError("First-shot evidence changed during second generation")
        if not complete2["complete"]:
            raise RuntimeError("Two-shot batch did not become complete")
        summary["core_runs"].append({"pid": second_core.process.pid, "epoch": record2["core_epoch"]})
        summary.update(status="machine_pass_human_review_pending", selection_version=args.selection_version,
                       selection=selection, second_prompt_id=submitted2["prompt_id"],
                       second_media=media2, first_shot_recomputed=False, restarted_core_history=list(history2),
                       final_queue=assert_idle(second_core), final_batch=complete2)
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stops, stop_errors = stop_servers(servers)
        summary["owned_stops"] = stops
        if stop_errors:
            summary.update(status="failed_cleanup", cleanup_errors=stop_errors)
        pilot.server_command = base_command
        write_new(evidence / "result.json", summary)
        print(json.dumps({"evidence": str(evidence), "status": summary["status"]}), flush=True)
        if stop_errors and sys.exc_info()[0] is None:
            raise RuntimeError("Owned process cleanup failed; inspect result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
