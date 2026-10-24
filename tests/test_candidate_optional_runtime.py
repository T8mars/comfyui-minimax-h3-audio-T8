"""Strict opt-in dependency boundary, not a substitute for real media execution."""
import hashlib
import json

import pytest

from tools.run_candidate_cpu_regression import (
    OPENCV_WHEEL_SHA256, private_opencv_inventory, private_opencv_stable, validate_private_opencv,
)


def fixture_runtime(project):
    target = project / 'artifacts/runtime/opencv-headless-4.10.0.84'
    target.mkdir(parents=True)
    (target / 'cv2.pyd').write_bytes(b'synthetic-inventory-only-not-a-runtime')
    receipt = {'source_report': 'synthetic-boundary-only', 'wheel_sha256': OPENCV_WHEEL_SHA256,
               'inventory': private_opencv_inventory(target)}
    (target / 'task-runtime-receipt.json').write_text(json.dumps(receipt), encoding='utf8')
    return target


def test_literal_target_inventory_and_receipt_binding_do_not_install(tmp_path):
    target = fixture_runtime(tmp_path)
    binding = validate_private_opencv(tmp_path, target)
    assert private_opencv_stable(binding)
    assert binding['wheel_sha256'] == OPENCV_WHEEL_SHA256
    assert binding['inventory'] == {'cv2.pyd': hashlib.sha256(
        b'synthetic-inventory-only-not-a-runtime').hexdigest()}
    assert set(p.name for p in target.iterdir()) == {'cv2.pyd', 'task-runtime-receipt.json'}


@pytest.mark.parametrize('damage', ['wrong_target', 'missing_receipt', 'oversized_receipt',
                                   'wrong_wheel', 'changed_file', 'extra_file', 'missing_file',
                                   'empty_inventory', 'unknown_receipt_field'])
def test_optional_runtime_refuses_wrong_identity_or_inventory(tmp_path, damage):
    target = fixture_runtime(tmp_path)
    receipt_path = target / 'task-runtime-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if damage == 'wrong_target':
        target = tmp_path
    elif damage == 'missing_receipt':
        receipt_path.unlink()
    elif damage == 'oversized_receipt':
        receipt_path.write_bytes(b' ' * (1024 * 1024 + 1))
    elif damage == 'wrong_wheel':
        receipt['wheel_sha256'] = '0' * 64
        receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    elif damage == 'changed_file':
        (target / 'cv2.pyd').write_bytes(b'changed')
    elif damage == 'extra_file':
        (target / 'extra.py').write_text('unreviewed', encoding='utf8')
    elif damage == 'missing_file':
        (target / 'cv2.pyd').unlink()
    elif damage == 'empty_inventory':
        receipt['inventory'] = {}
        receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    else:
        receipt['unknown'] = True
        receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    with pytest.raises(ValueError):
        validate_private_opencv(tmp_path, target)


@pytest.mark.parametrize('damage', ['file', 'receipt', 'missing_receipt', 'nested_extra_receipt'])
def test_runtime_mutation_is_detected_after_qualification(tmp_path, damage):
    target = fixture_runtime(tmp_path)
    binding = validate_private_opencv(tmp_path, target)
    if damage == 'missing_receipt':
        (target / 'task-runtime-receipt.json').unlink()
    elif damage == 'nested_extra_receipt':
        (target / 'extra').mkdir()
        (target / 'extra/task-runtime-receipt.json').write_bytes(b'unreviewed')
    else:
        path = target / ('cv2.pyd' if damage == 'file' else 'task-runtime-receipt.json')
        path.write_bytes(path.read_bytes() + b'\n')
    assert not private_opencv_stable(binding)
