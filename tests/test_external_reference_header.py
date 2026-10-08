"""Bounded community-header inspection; stdlib only, no Core/models/CUDA."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import inspect_h3_external_reference as inspector  # noqa: E402


def member(kind='image', **extra):
    return {'_format_version': 4, 'kind': kind, **extra}


def descriptor(shape, dtype='F16', start=0):
    size = inspector.DTYPE_BYTES[dtype]
    for n in shape:
        size *= n
    return {'shape': shape, 'dtype': dtype, 'data_offsets': [start, start + size]}


def standalone():
    return {'__metadata__': {'refmod_meta': json.dumps(member())},
            'latent': descriptor([1, 24, 1, 2, 4])}


def bundle():
    visual = descriptor([1, 24, 2, 2, 4])
    audio = descriptor([1, 32, 2, 3], 'F32', visual['data_offsets'][1])
    return {'__metadata__': {'refmod_meta': json.dumps({
        '_format_version': 5, 'kind': 'bundle',
        'members': [member('video', latent_t=2, latent_h=2, latent_w=4),
                    member('audio', latent_t=3, sample_rate=32000)]})},
            'ref_0': visual, 'ref_1': audio}


class ExternalReferenceHeaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='t8-ref-header-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sample.safetensors'

    def write(self, header=None, *, raw=None, payload_size=None, prefix=None):
        if raw is None:
            raw = json.dumps(header, separators=(',', ':')).encode('utf8')
        if payload_size is None:
            payload_size = max(v['data_offsets'][1] for k, v in header.items()
                               if k != '__metadata__')
        self.path.write_bytes((prefix if prefix is not None else struct.pack('<Q', len(raw)))
                              + raw + b'\xff' * payload_size)
        return self.path

    def inspect(self, header):
        self.write(header)
        return inspector.inspect_header(self.path)

    def assert_invalid(self, header):
        self.write(header)
        with self.assertRaises(ValueError):
            inspector.inspect_header(self.path)

    def test_image_v4_header_only_and_file_unchanged(self):
        self.write(standalone())
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = inspector.inspect_header(self.path)
        self.assertEqual(result['members'][0]['declared_grid_tokens_one_copy'], 2)
        self.assertEqual(result['tensor_loads'], 0)
        self.assertFalse(result['payload_content_verified'])
        self.assertFalse(result['inference_ready'])
        self.assertFalse(result['conversion_performed'])
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_video_audio_bundle_declared_cost_not_sound_authentication(self):
        result = self.inspect(bundle())
        self.assertEqual(result['declared_grid_tokens_one_copy'], 10)
        self.assertEqual([r['kind'] for r in result['members']], ['video', 'audio'])
        self.assertEqual(result['sound_status'], 'sound_incomplete_original_PCM_and_encoder_unverified')

    def test_hybrid_retains_ordinary_weight_half_including_scalar(self):
        header = bundle()
        start = header['ref_1']['data_offsets'][1]
        header['lora.alpha'] = descriptor([], 'F32', start)
        header['__metadata__']['h3_hybrid'] = json.dumps({
            'version': 1, 'lora': {'keys': 1}, 'refmod_count': 2})
        result = self.inspect(header)
        self.assertTrue(result['hybrid_marker_present'])
        self.assertEqual(result['non_reference_tensor_count'], 1)
        self.assertEqual(result['non_reference_bytes'], 4)

    def test_embedded_config_paths_and_prompt_are_inert_not_reported(self):
        header = standalone()
        meta = member(refmod_config={'path': 'C:/sensitive/never-read',
                                    'code': '__import__("os").remove("x")'},
                      prompt='private text', producer='unproven claim')
        header['__metadata__']['refmod_meta'] = json.dumps(meta)
        result = self.inspect(header)
        text = json.dumps(result)
        self.assertNotIn('sensitive', text)
        self.assertNotIn('private text', text)
        self.assertNotIn('unproven claim', text)
        self.assertFalse(result['configuration_executed'])
        self.assertFalse(result['embedded_paths_followed'])
        self.assertTrue(result['members'][0]['serialized_config_present_not_executed'])

    def test_stream_reads_only_prefix_and_bounded_header(self):
        self.write(bundle())
        original = Path.open
        reads = []

        class Counted:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.wrapped.close()

            def fileno(self):
                return self.wrapped.fileno()

            def read(self, count):
                reads.append(count)
                return self.wrapped.read(count)

        def counted(path, *args, **kwargs):
            self.assertEqual(path, self.path)
            self.assertEqual(args, ('rb',))
            return Counted(original(path, *args, **kwargs))

        with patch.object(Path, 'open', counted):
            result = inspector.inspect_header(self.path)
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0], 8)
        self.assertEqual(sum(reads), result['bytes_read'])
        self.assertLess(result['bytes_read'], result['file_bytes'])

    def test_invalid_json_duplicate_and_nonfinite(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'\xff', b'{'):
            with self.subTest(raw=raw):
                self.write(raw=raw, payload_size=1)
                with self.assertRaises(ValueError):
                    inspector.inspect_header(self.path)

    def test_header_prefix_budget_and_truncation(self):
        for length in (0, 1, inspector.MAX_HEADER + 1, 2**63, 999):
            with self.subTest(length=length):
                self.write(raw=b'{}', payload_size=0, prefix=struct.pack('<Q', length))
                with self.assertRaises(ValueError):
                    inspector.inspect_header(self.path)

    def test_missing_file_and_nonfile(self):
        with self.assertRaises(FileNotFoundError):
            inspector.inspect_header(self.path)
        with self.assertRaises(ValueError):
            inspector.inspect_header(Path(self.directory.name))

    def test_duplicate_and_nonflat_metadata(self):
        for value in ([], None, {'refmod_meta': {}}, {'refmod_meta': True}):
            with self.subTest(value=value):
                header = standalone()
                header['__metadata__'] = value
                self.assert_invalid(header)
        header = standalone()
        header['__metadata__']['refmod_meta'] = '{"_format_version":4,"kind":"image","kind":"video"}'
        self.assert_invalid(header)

    def test_shape_dtype_and_offset_integrity(self):
        base = standalone()
        mutations = [
            {'shape': [True, 24, 1, 2, 4]}, {'shape': [1, 24, -1, 2, 4]},
            {'shape': [1] * 9}, {'dtype': 'unknown'}, {'dtype': ['F16']},
            {'data_offsets': [True, 384]}, {'data_offsets': [0, 383]},
            {'data_offsets': [-1, 383]}, {'data_offsets': [1, 385]},
            {'data_offsets': [384, 0]}, {'extra': 0},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                header = copy.deepcopy(base)
                header['latent'].update(mutation)
                self.assert_invalid(header)

    def test_overlap_holes_and_unclaimed_payload(self):
        for start in (0, 500):
            with self.subTest(start=start):
                header = bundle()
                header['ref_1'] = descriptor([1, 32, 2, 3], 'F32', start)
                self.assert_invalid(header)
        header = standalone()
        self.write(header, payload_size=header['latent']['data_offsets'][1] + 1)
        with self.assertRaisesRegex(ValueError, 'trailing'):
            inspector.inspect_header(self.path)

    def test_version_and_member_count_are_explicit_not_migrated(self):
        for meta in (member(_format_version=True), member(_format_version=6),
                     {'_format_version': 5, 'kind': 'bundle', 'members': []},
                     {'_format_version': 5, 'kind': 'bundle', 'members': [member()] * 257}):
            with self.subTest(meta_kind=meta['kind']):
                header = standalone()
                header['__metadata__']['refmod_meta'] = json.dumps(meta)
                self.assert_invalid(header)

    def test_member_indices_nested_and_dimensions(self):
        header = bundle()
        header['ref_2'] = header.pop('ref_1')
        self.assert_invalid(header)
        for mutation in ({'kind': 'bundle'}, {'latent_h': 4}, {'latent_t': True},
                         {'_format_version': True}):
            with self.subTest(mutation=mutation):
                header = bundle()
                meta = json.loads(header['__metadata__']['refmod_meta'])
                meta['members'][0].update(mutation)
                header['__metadata__']['refmod_meta'] = json.dumps(meta)
                self.assert_invalid(header)

    def test_visual_grid_and_floating_dtype(self):
        for shape, dtype in (([1, 24, 2, 2, 4], 'F16'), ([1, 24, 1, 3, 4], 'F16'),
                             ([1, 24, 1, 2, 4], 'I16'), ([2, 24, 1, 2, 4], 'F16'),
                             ([1, 24, 0, 2, 4], 'F16')):
            with self.subTest(shape=shape, dtype=dtype):
                header = standalone()
                header['latent'] = descriptor(shape, dtype)
                self.assert_invalid(header)

    def test_audio_grid_and_rate_not_filename_claim(self):
        for shape, rate in (([1, 32, 1, 3], 32000), ([1, 32, 2, 3], 44100),
                            ([1, 32, 2, 3], 32000.0), ([1, 32, 2, 3], True)):
            with self.subTest(shape=shape, rate=rate):
                header = {'__metadata__': {'refmod_meta': json.dumps(member('audio', sample_rate=rate))},
                          'latent': descriptor(shape, 'F32')}
                self.assert_invalid(header)

    def test_hybrid_missing_weight_half_and_count_mismatch(self):
        for marker in ({'version': 1, 'lora': {'keys': 0}, 'refmod_count': 2},
                       {'version': True, 'lora': {'keys': 1}, 'refmod_count': 2},
                       {'version': 1, 'lora': {'keys': True}, 'refmod_count': 2},
                       {'version': 1, 'lora': {'keys': 2}, 'refmod_count': 2},
                       {'version': 1, 'lora': {'keys': 1}, 'refmod_count': 1}):
            with self.subTest(marker=marker):
                header = bundle()
                if marker['lora']['keys'] != 0:
                    header['lora.alpha'] = descriptor([], 'F32', header['ref_1']['data_offsets'][1])
                header['__metadata__']['h3_hybrid'] = json.dumps(marker)
                self.assert_invalid(header)

    def test_replaced_file_is_not_authenticated(self):
        self.write(standalone())
        real_stat = self.path.stat()
        replacement = self.path.with_name('replacement.safetensors')
        replacement.write_bytes(self.path.read_bytes() + b'x')
        with patch.object(Path, 'stat', side_effect=[real_stat, real_stat, replacement.stat()]):
            with self.assertRaisesRegex(ValueError, 'replaced'):
                inspector.inspect_header(self.path)


if __name__ == '__main__':
    unittest.main()
