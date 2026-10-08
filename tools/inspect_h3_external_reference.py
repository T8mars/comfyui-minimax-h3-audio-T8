"""Read-only bounded RefMod/RefLoRA header evidence, never a tensor loader.

Community encoded references are NOT native T8 reference packages. A header
does not authenticate an encoder, original RGB/PCM, normalization or license.
This tool never follows embedded paths, executes config, imports torch, applies
a LoRA, or writes/converts the input. Unknown tensor payload stays on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct


MAX_HEADER = 1024**2
MAX_TENSORS = 8192
MAX_MEMBERS = 256
MAX_FILE = 64 * 1024**3
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
    'U32': 4, 'I32': 4, 'U64': 8, 'I64': 8,
    'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
    'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8,
}


def plain_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result

    def reject(value):
        raise ValueError('Nonfinite JSON is unsupported: ' + value)

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('Malformed or excessive-depth JSON') from error


def _snapshot(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def inspect_header(path):
    path = Path(path)
    before = path.stat()
    if not path.is_file() or not 10 <= before.st_size <= MAX_FILE:
        raise ValueError('Missing or out-of-budget safetensors file')
    with path.open('rb') as stream:
        if _snapshot(os.fstat(stream.fileno())) != _snapshot(before):
            raise ValueError('File replaced before inspection')
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= MAX_HEADER or 8 + length > before.st_size:
            raise ValueError('Safetensors header exceeds its byte budget')
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError('Truncated safetensors header')
        header = plain_json(raw)
        if _snapshot(os.fstat(stream.fileno())) != _snapshot(before):
            raise ValueError('File changed during header inspection')
    if _snapshot(path.stat()) != _snapshot(before):
        raise ValueError('File replaced during header inspection')
    if type(header) is not dict or type(header.get('__metadata__')) is not dict:
        raise ValueError('Missing flat safetensors metadata')
    metadata = header['__metadata__']
    if any(type(k) is not str or type(v) is not str for k, v in metadata.items()):
        raise ValueError('Safetensors metadata must be a string map')
    descriptors = {k: v for k, v in header.items() if k != '__metadata__'}
    if not 1 <= len(descriptors) <= MAX_TENSORS:
        raise ValueError('Tensor descriptor count exceeds its budget')
    ranges, byte_counts = [], {}
    payload_bytes = before.st_size - 8 - length
    for name, item in descriptors.items():
        if (not name or type(item) is not dict
                or set(item) != {'dtype', 'shape', 'data_offsets'}):
            raise ValueError('Invalid tensor descriptor')
        shape, offsets = item['shape'], item['data_offsets']
        if (type(shape) is not list or len(shape) > 8
                or any(type(n) is not int or not 0 <= n <= 2**31 - 1 for n in shape)
                or type(offsets) is not list or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
                or type(item['dtype']) is not str or item['dtype'] not in DTYPE_BYTES):
            raise ValueError('Invalid shape, dtype or tensor offsets')
        size = math.prod(shape) * DTYPE_BYTES[item['dtype']]
        if size != offsets[1] - offsets[0]:
            raise ValueError('Tensor shape/dtype/bytes disagree')
        ranges.append(tuple(offsets))
        byte_counts[name] = size
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError('Tensor data has an overlap or hole')
        end = stop
    if end != payload_bytes:
        raise ValueError('Unclaimed trailing tensor data')

    meta = plain_json(metadata.get('refmod_meta', 'null'))
    if type(meta) is not dict or type(meta.get('_format_version')) is not int:
        raise ValueError('Not a declared supported community RefMod')
    version = meta['_format_version']
    if version == 4 and meta.get('kind') in ('image', 'video', 'audio'):
        members, keys = [meta], ['latent']
    elif version == 5 and meta.get('kind') == 'bundle':
        members = meta.get('members')
        if type(members) is not list or not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError('Invalid bounded RefMod member list')
        keys = ['ref_' + str(i) for i in range(len(members))]
    else:
        raise ValueError('Unsupported RefMod format; no implicit migration')
    observed_ref_keys = {k for k in descriptors if k.startswith('ref_') or k == 'latent'}
    if observed_ref_keys != set(keys):
        raise ValueError('Reference member indices and tensors disagree')
    rows = []
    for index, (member, key) in enumerate(zip(members, keys, strict=True)):
        if (type(member) is not dict or type(member.get('_format_version')) is not int
                or member['_format_version'] != 4
                or member.get('kind') not in ('image', 'video', 'audio')):
            raise ValueError('Nested or unknown reference member')
        kind = member['kind']
        item = descriptors[key]
        shape = item['shape']
        if item['dtype'] not in ('F16', 'BF16', 'F32'):
            raise ValueError('Reference latent requires a supported floating dtype')
        if kind == 'audio':
            if (len(shape) != 4 or shape[:3] != [1, 32, 2] or shape[3] <= 0
                    or member.get('sample_rate', 32000) != 32000
                    or type(member.get('sample_rate', 32000)) is not int):
                raise ValueError('Audio latent is not the declared native codec grid')
            t, tokens = shape[3], 2 * shape[3]
            dims = {'latent_t': t}
        else:
            if (len(shape) != 5 or shape[:2] != [1, 24] or any(n <= 0 for n in shape[2:])
                    or shape[3] % 2 or shape[4] % 2 or (kind == 'image' and shape[2] != 1)):
                raise ValueError('Visual latent is not the declared native patch grid')
            t, h, w = shape[2:]
            tokens = t * (h // 2) * (w // 2)
            dims = {'latent_t': t, 'latent_h': h, 'latent_w': w}
        for field, actual in dims.items():
            if field in member and (type(member[field]) is not int or member[field] != actual):
                raise ValueError('Member dimensions disagree with tensor header')
        rows.append({'ordinal': index + 1, 'kind': kind, 'tensor_key': key,
                     'shape': shape, 'dtype': item['dtype'], 'bytes': byte_counts[key],
                     'declared_grid_tokens_one_copy': tokens,
                     'serialized_config_present_not_executed': 'refmod_config' in member})

    other_count = len(descriptors) - len(keys)
    marker = plain_json(metadata.get('h3_hybrid', 'null'))
    if marker is not None:
        if (type(marker) is not dict or type(marker.get('version')) is not int
                or marker['version'] != 1 or type(marker.get('lora')) is not dict
                or type(marker['lora'].get('keys')) is not int
                or marker['lora']['keys'] != other_count
                or type(marker.get('refmod_count')) is not int
                or marker['refmod_count'] != len(rows) or other_count == 0):
            raise ValueError('Hybrid marker declares missing or mismatched weight half')
    return {
        'schema': 't8.community-reference.header-evidence.v1',
        'inspection_status': 'header_structure_verified_only',
        'format_version': version, 'hybrid_marker_present': marker is not None,
        'header_sha256': hashlib.sha256(raw).hexdigest(),
        'file_bytes': before.st_size, 'bytes_read': 8 + length,
        'payload_content_verified': False, 'tensor_loads': 0,
        'non_reference_tensor_count': other_count,
        'non_reference_bytes': sum(byte_counts[k] for k in descriptors if k not in keys),
        'members': rows, 'reference_bytes': sum(row['bytes'] for row in rows),
        'declared_grid_tokens_one_copy': sum(row['declared_grid_tokens_one_copy'] for row in rows),
        'sound_status': 'sound_incomplete_original_PCM_and_encoder_unverified'
                        if any(row['kind'] == 'audio' for row in rows) else 'no_audio_members',
        'conversion_status': 'unsupported_without_explicit_RGB_actual_producer_normalization_and_permission_evidence',
        'inference_ready': False, 'conversion_performed': False,
        'configuration_executed': False, 'embedded_paths_followed': False,
        'source_payload_modified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_header(args.file), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
