"""Exact decoded frame to a new input asset; no project mutation or sampling."""
import threading
import uuid

from .director_film import film_media, load_film
from .director_project import atomic_json, contained, file_sha

_LOCK = threading.RLock()


def extract_frame(store, project_id, film_id, index, frame_number):
    from .director_media import extract_image

    manifest = load_film(store, project_id, film_id)
    if type(index) is not int or not 0 <= index < len(manifest['entries']):
        raise ValueError('成片序号无效')
    entry = manifest['entries'][index]
    if type(frame_number) is not int or not 0 <= frame_number < entry['media']['frames']:
        raise ValueError('请选择成片内有效整数帧（从 0 开始）')
    source = film_media(store, project_id, film_id, index)
    digest = entry['media_sha256']
    provenance = {'media_sha256': digest, 'frame': frame_number, 'fps': entry['media']['fps']}
    aid = str(uuid.uuid5(uuid.UUID(project_id), f'frame-v1:{digest}:{frame_number}'))
    with _LOCK:
        receipt = contained(store.root, f'assets/{aid}.json')
        if receipt.exists():
            asset = store.asset(aid, verify=True)
            if 'source_frame' in asset and asset['source_frame'] != provenance:
                raise ValueError('已有取帧素材来源不一致，拒绝覆盖')
        else:
            asset = None
        if asset is None or 'source_frame' not in asset:
            image = extract_image(source, frame_number)
            if image.size != (entry['media']['width'], entry['media']['height']):
                raise ValueError('目标帧无法完整解码或尺寸不一致')
            if file_sha(source) != digest:
                raise ValueError('取帧期间冻结成片发生变化，拒绝使用')
            if asset is not None:
                # Upgrade only a verified legacy PNG whose decoded pixels match.
                from PIL import Image
                with Image.open(contained(store.input_root, asset['server_path'])) as previous:
                    if previous.size != image.size or previous.convert('RGB').tobytes() != image.convert('RGB').tobytes():
                        raise ValueError('旧取帧图片与来源帧不符，拒绝补写来源')
                asset = {**asset, 'source_frame': provenance}
                atomic_json(receipt, asset)
            else:
                asset = _register_frame(store, aid, frame_number, image, provenance)
        return {'asset': asset, 'project_id': project_id, 'film_id': film_id,
                'shot_id': entry['shot_id'], 'version_id': entry['version_id'],
                'media_sha256': digest, 'frame': frame_number,
                'seconds': frame_number/entry['media']['fps']}


def _register_frame(store, aid, frame_number, image, provenance):
    path = contained(store.input_root, f't8_director/{aid}/source.png')
    # Do not overwrite partial or manually altered material on retry.
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('已有未登记的取帧文件，保留原文件；请核对后再处理，未覆盖')
    with path.open('xb') as output:
        image.save(output, format='PNG')
    return store.register_asset(path, aid, f'成片帧-{frame_number:06d}.png', source_frame=provenance)
