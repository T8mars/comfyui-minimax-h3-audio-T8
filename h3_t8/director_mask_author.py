"""Explicit per-frame brush MASK drafts bound to an adopted, immutable film.
No tracking, sampling, audio edits, automatic region propagation or project edits.
"""

from copy import deepcopy
import base64
import io
import json
import math
import hashlib
import os
import uuid
import threading
import zipfile
import re
import shutil

from PIL import Image, ImageDraw

from .director_film import film_media, load_film
from .director_project import atomic_json, contained, file_sha, identity, sha

SCHEMA = "t8.director.mask_author.v1"
_LOCK = threading.RLock()


def _source(store, project_id, film_id, index):
    film = load_film(store, project_id, film_id)
    if type(index) is not int or not 0 <= index < len(film["entries"]):
        raise ValueError("请选择冻结整片中的有效镜头")
    entry = film["entries"][index]
    return film, entry, film_media(store, project_id, film_id, index)


def _binding(film, entry, index):
    return {
        "film_sha256": film["sha256"],
        "index": index,
        "shot_id": entry["shot_id"],
        "version_id": entry["version_id"],
        "media_sha256": entry["media_sha256"],
        "shot_name": entry["name"],
        "width": entry["media"]["width"],
        "height": entry["media"]["height"],
        "frames": entry["media"]["frames"],
        "fps": entry["media"]["fps"],
        "in_frame": entry["in_frame"],
        "out_frame": entry["out_frame"],
    }


def _path(store, uid):
    return contained(store.root, f"mask_authors/{identity(uid)}.json")


def load_author(store, project_id, film_id, index, author_id):
    film, entry, _ = _source(store, project_id, film_id, index)
    binding = _binding(film, entry, index)
    path = _path(store, author_id)
    if not path.exists():
        return {
            "schema": SCHEMA,
            "id": identity(author_id),
            "project_id": identity(project_id),
            "film_id": identity(film_id),
            "source": binding,
            "revision": 0,
            "strokes": {},
        }
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (
        saved.get("project_id") != project_id
        or saved.get("film_id") != film_id
        or saved.get("source") != binding
        or sha({k: v for k, v in saved.items() if k != "sha256"}) != saved.get("sha256")
    ):
        raise ValueError("遮罩来源或保存校验不一致；原稿保留，请回到对应采用版本")
    return saved


def _strokes(value, source):
    if not isinstance(value, dict):
        raise ValueError("逐帧遮罩必须是帧号到画笔列表的对象")
    result = {}
    for frame, strokes in value.items():
        if (
            not isinstance(frame, str)
            or not frame.isdecimal()
            or str(int(frame)) != frame
        ):
            raise ValueError("遮罩帧号必须是明确的源片整数帧")
        if not source["in_frame"] <= int(frame) < source["out_frame"] or not isinstance(
            strokes, list
        ):
            raise ValueError("遮罩帧超出该镜的采用范围")
        rows = []
        for stroke in strokes:
            if not isinstance(stroke, dict) or stroke.get("tool") not in (
                "paint",
                "erase",
            ):
                raise ValueError("画笔只能绘制或擦除")
            width, points = stroke.get("width"), stroke.get("points")
            if (
                type(width) not in (int, float)
                or not math.isfinite(width)
                or width <= 0
            ):
                raise ValueError("画笔宽度必须为正的有限值")
            if not isinstance(points, list) or not points:
                raise ValueError("画笔至少需要一个实际坐标")
            for point in points:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        type(x) not in (int, float)
                        or not math.isfinite(x)
                        or not 0 <= x <= 1
                        for x in point
                    )
                ):
                    raise ValueError("画笔坐标必须对应完整源图的0–1范围")
            rows.append(
                {"tool": stroke["tool"], "width": width, "points": deepcopy(points)}
            )
        result[frame] = rows
    return result


def save_author(store, project_id, film_id, index, author_id, body):
    with _LOCK:
        previous = load_author(store, project_id, film_id, index, author_id)
        if body.get("source") != previous["source"]:
            raise ValueError("采用版本、范围或源媒体已变；没有把旧遮罩套到新片")
        strokes = _strokes(body.get("strokes"), previous["source"])
        if (
            type(body.get("revision")) is int
            and body["revision"] == previous["revision"] - 1
            and strokes == previous["strokes"]
        ):
            return previous  # Explicit retry after an unconfirmed save reply.
        if (
            type(body.get("revision")) is not int
            or body["revision"] != previous["revision"]
        ):
            raise ValueError("遮罩已在别处更新；本稿未覆盖，请先保留并重开核对")
        saved = {**previous, "strokes": strokes, "revision": previous["revision"] + 1}
        saved.pop("sha256", None)
        saved["sha256"] = sha(saved)
        atomic_json(_path(store, author_id), saved)
        return saved


def preview_frame(store, project_id, film_id, index, frame):
    from .director_media import extract_image

    _, entry, path = _source(store, project_id, film_id, index)
    if type(frame) is not int or not entry["in_frame"] <= frame < entry["out_frame"]:
        raise ValueError("请选择采用范围内的实际帧")
    image = extract_image(path, frame)
    if (
        image.size != (entry["media"]["width"], entry["media"]["height"])
        or file_sha(path) != entry["media_sha256"]
    ):
        raise ValueError("源帧尺寸或媒体内容变化，不能继续绘制")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {
        "frame": frame,
        "media_sha256": entry["media_sha256"],
        "data_url": "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def rasterize(strokes, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for stroke in strokes:
        size = max(1, round(stroke["width"] * min(width, height)))
        points = [
            (round(x * (width - 1)), round(y * (height - 1)))
            for x, y in stroke["points"]
        ]
        color = 255 if stroke["tool"] == "paint" else 0
        if len(points) > 1:
            draw.line(points, fill=color, width=size, joint="curve")
        radius = size / 2
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return mask


def _source_rgb24_sha(path, source):
    """Stream the exact decoded source RGB24, independently of the thumbnail."""
    import av

    digest = hashlib.sha256()
    digest.update(
        f"RGB24:{source['frames']}:{source['height']}:{source['width']}".encode()
    )
    count = 0
    with av.open(str(path)) as media:
        for frame in media.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if tuple(rgb.shape) != (source["height"], source["width"], 3):
                raise ValueError("实际源RGB画布与MASK映射不一致")
            digest.update(rgb.tobytes())
            count += 1
    if count != source["frames"]:
        raise ValueError("实际源帧数与MASK映射不一致")
    return digest.hexdigest()


def export_author(store, project_id, film_id, index, author_id):
    """Full source frame count; untouched/occluded frames remain explicit zero MASKs."""
    with _LOCK:
        saved = load_author(store, project_id, film_id, index, author_id)
        if saved["revision"] == 0:
            raise ValueError("请先保存遮罩稿，再导出")
        path = contained(
            store.root, f"mask_exports/{identity(author_id)}/{saved['sha256']}.zip"
        )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            source = saved["source"]
            temp = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
            source_path = _source(store, project_id, film_id, index)[2]
            rgb_sha = _source_rgb24_sha(source_path, source)
            rows = []
            try:
                with zipfile.ZipFile(
                    temp, "x", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    for frame in range(source["frames"]):
                        image = rasterize(
                            saved["strokes"].get(str(frame), []),
                            source["width"],
                            source["height"],
                        )
                        data = io.BytesIO()
                        image.save(data, format="PNG")
                        raw = data.getvalue()
                        name = f"frames/{frame:06d}.png"
                        archive.writestr(name, raw)
                        rows.append(
                            {
                                "source_frame": frame,
                                "file": name,
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        )
                    manifest = {
                        "schema": "t8.authored_mask_sequence.v1",
                        "author_sha256": saved["sha256"],
                        "source": source,
                        "source_rgb24_sha256": rgb_sha,
                        "frame_map": rows,
                        "unpainted_frames": "explicit_zero",
                        "source_audio": "unchanged_not_included",
                        "automatic_tracking": False,
                    }
                    archive.writestr(
                        "manifest.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2),
                    )
                _source(
                    store, project_id, film_id, index
                )  # Never publish against changed source bytes.
                os.replace(temp, path)
                atomic_json(
                    path.with_suffix(".json"),
                    {"sha256": file_sha(path), "author_sha256": saved["sha256"]},
                )
            finally:
                temp.unlink(missing_ok=True)
        receipt = path.with_suffix(".json")
        if not receipt.is_file():
            raise ValueError("遮罩导出没有完整回执，原文件保留；请另存一版后重新导出")
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            result.get("sha256") != file_sha(path)
            or result.get("author_sha256") != saved["sha256"]
        ):
            raise ValueError("遮罩导出文件内容变化，拒绝冒充原稿")
        _publish_input_sequence(store, path, saved)
        return path


def _publish_input_sequence(store, path, saved):
    label = (
        re.sub(r'[<>:\\/?*"|\x00-\x1f]', "_", saved["source"]["shot_name"]).strip(" .")[
            :24
        ]
        or "shot"
    )
    name = f"mask-{label}-{saved['id'][:8]}-r{saved['revision']}-{saved['sha256'][:12]}.zip"
    target = contained(store.input_root, "t8_masks/" + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha(target) != file_sha(path):
            raise ValueError("同名输入MASK包内容已变，保留原文件；请另存新稿")
        return
    temp = target.with_name(target.name + "." + str(uuid.uuid4()) + ".tmp")
    try:
        shutil.copyfile(path, temp)
        os.link(
            temp, target
        )  # Publish only if absent; never overwrite a raced user file.
    finally:
        temp.unlink(missing_ok=True)


def export_file(store, project_id, film_id, index, author_id, digest=None):
    if digest is None:
        return export_author(store, project_id, film_id, index, author_id)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(x not in "0123456789abcdef" for x in digest)
    ):
        raise ValueError("遮罩版本SHA无效")
    film, entry, _ = _source(store, project_id, film_id, index)
    path = contained(store.root, f"mask_exports/{identity(author_id)}/{digest}.zip")
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt.get("author_sha256") != digest or receipt.get("sha256") != file_sha(
        path
    ):
        raise ValueError("遮罩导出内容或版本不一致")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if (
        manifest.get("source") != _binding(film, entry, index)
        or manifest.get("author_sha256") != digest
    ):
        raise ValueError("遮罩导出不属于当前冻结媒体")
    return path


def load_mask_sequence(archive_path, frames):
    """Consume the full source IMAGE and verify its decoded RGB24 before MASK output."""
    import numpy as np
    import torch

    if (
        not isinstance(frames, torch.Tensor)
        or frames.ndim != 4
        or frames.shape[-1] != 3
    ):
        raise ValueError("请连接完整原片RGB IMAGE序列")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        source, rows = manifest["source"], manifest["frame_map"]
        if (
            manifest.get("schema") != "t8.authored_mask_sequence.v1"
            or tuple(frames.shape)
            != (source["frames"], source["height"], source["width"], 3)
            or not isinstance(rows, list)
            or len(rows) != source["frames"]
        ):
            raise ValueError(
                "MASK包与实际源片的帧数/画布不一致；裁切缩放请共用Source Conform映射"
            )
        digest = hashlib.sha256()
        digest.update(
            f"RGB24:{source['frames']}:{source['height']}:{source['width']}".encode()
        )
        masks = []
        for i, frame in enumerate(frames.detach()):
            frame = frame.to(device="cpu")
            if (
                not torch.isfinite(frame).all()
                or (frame < 0).any()
                or (frame > 1).any()
            ):
                raise ValueError("源片RGB包含无效像素")
            digest.update(
                (frame * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
            )
            row = rows[i]
            if row.get("source_frame") != i or row.get("file") != f"frames/{i:06d}.png":
                raise ValueError("MASK帧映射顺序变化，不能猜测重排")
            data = archive.read(row["file"])
            if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                raise ValueError("MASK图片内容校验失败")
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "L" or image.size != (
                    source["width"],
                    source["height"],
                ):
                    raise ValueError("MASK图片画布或通道变化")
                masks.append(torch.from_numpy(np.array(image, copy=True)).float() / 255)
        if digest.hexdigest() != manifest.get("source_rgb24_sha256"):
            raise ValueError("完整源RGB24内容不匹配；请连接该遮罩绑定的原片或重新制作")
    return torch.stack(masks), manifest
