"""CPU-only candidate regression with durable collection and per-phase receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid


OPENCV_WHEEL_SHA256 = 'afcf28bd1209dd58810d33defb622b325d3cbe49dcd7a43a902982c33e5fad05'


def private_opencv_inventory(target):
    inventory = {}
    for path in sorted(target.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(target):
            raise ValueError('Private optional runtime contains an escaping or linked path')
        if path.is_file() and path.suffix.lower() != '.pyc' and path != target / 'task-runtime-receipt.json':
            inventory[str(path.relative_to(target))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inventory


def validate_private_opencv(project, target):
    """An explicit, already qualified dependency; never install or mutate an env."""
    expected = project.resolve() / 'artifacts/runtime/opencv-headless-4.10.0.84'
    if target.is_symlink() or target.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError('Only the exact owned optional OpenCV runtime is supported')
    target = target.resolve(strict=True)
    receipt = target / 'task-runtime-receipt.json'
    if receipt.is_symlink() or not receipt.is_file() or not 0 < receipt.stat().st_size <= 1024 * 1024:
        raise ValueError('Private OpenCV receipt is missing or exceeds its limit')
    data = json.loads(receipt.read_text(encoding='utf8'))
    if (type(data) is not dict or set(data) != {'source_report', 'wheel_sha256', 'inventory'}
            or data['wheel_sha256'] != OPENCV_WHEEL_SHA256
            or type(data['inventory']) is not dict or not data['inventory']
            or private_opencv_inventory(target) != data['inventory']):
        raise ValueError('Private OpenCV wheel SHA or immutable inventory differs')
    return {'target': str(target), 'receipt': str(receipt), 'wheel_sha256': data['wheel_sha256'],
            'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
            'inventory': data['inventory']}


def activate_private_opencv(project, target):
    binding = validate_private_opencv(project, target)
    sys.path.insert(0, binding['target'])
    os.environ['PYTHONPATH'] = binding['target'] + os.pathsep + os.environ['PYTHONPATH']
    import cv2
    if (cv2.__version__ != '4.10.0' or not Path(cv2.__file__).resolve().is_relative_to(Path(binding['target']))):
        raise ValueError('Actual optional OpenCV import comes from an unqualified runtime')
    binding['module'] = cv2.__file__
    binding['version'] = cv2.__version__
    binding['build_sha256'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    return binding


def private_opencv_stable(binding):
    try:
        return (hashlib.sha256(Path(binding['receipt']).read_bytes()).hexdigest() == binding['receipt_sha256']
                and private_opencv_inventory(Path(binding['target'])) == binding['inventory'])
    except (OSError, ValueError):
        return False


def source_snapshot(project):
    paths = set(project.glob('*.py'))
    paths.update(project / name for name in ('pyproject.toml', 'meta.json', 'features.json', '.comfyignore'))
    for directory in ('h3_t8', 'tools', 'tests', 'web', 'examples'):
        paths.update(path for path in (project / directory).rglob('*')
                     if path.is_file() and path.suffix.lower() in {'.py', '.js', '.json', '.html'})
    return {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def isolated_tmp_path_plugin(pytest, output):
    """Opt-in directory boundary, not a test/codec retry or safety bypass.

    Each test gets a fresh short directory without pytest's numbered/current
    alias setup. Keep all inputs, including failures, and record exact paths.
    The default pytest fixture and tmp_path_factory remain unchanged unless
    this explicit runner option is selected.
    """
    root = Path(tempfile.mkdtemp(prefix='t8-cpu-reg-')).resolve()
    directories = []
    receipt = output / 'temporary-boundary.json'

    def save():
        receipt.write_text(json.dumps({'mode': 'isolated', 'root': str(root),
            'directories': directories, 'retained': True,
            'test_assertions_and_runtime_unchanged': True}, ensure_ascii=False, indent=2), encoding='utf8')

    save()

    class IndependentTemporaryDirectories:
        @pytest.fixture
        def tmp_path(self, request):
            directory = root / uuid.uuid4().hex[:12]
            directory.mkdir(mode=0o700)
            if directory.is_symlink() or not directory.resolve().is_relative_to(root):
                raise ValueError('Regression temporary directory escaped its owned root')
            directories.append({'test': request.node.nodeid, 'directory': str(directory)})
            save()
            return directory

    return IndependentTemporaryDirectories()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select', nargs='*', default=['tests'])
    parser.add_argument('--deselect', nargs='*', default=[],
                        help='Explicit integration cases outside this candidate; retained in the receipt')
    parser.add_argument('--tmp-path-mode', choices=('pytest', 'isolated'), default='pytest',
                        help='Explicit independent per-test directories; inputs/failures retained, assertions unchanged')
    parser.add_argument('--private-opencv', type=Path,
                        help='Explicit already qualified, hash-pinned task runtime; never install into the main env')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(project / 'artifacts'):
        raise ValueError('New task-owned artifacts directory required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                      PYTHONPATH=str(args.core.resolve()))
    sys.path[:0] = [str(project), str(args.core.resolve())]
    optional_runtime = (activate_private_opencv(project, args.private_opencv)
                        if args.private_opencv is not None else None)
    sys.argv = ['candidate-cpu-regression', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import torch
    import pytest
    torch.set_num_threads(2)
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError('Regression must not have CUDA available or initialized')
    output.mkdir(parents=True)
    if optional_runtime is not None:
        (output / 'optional-runtime.json').write_text(json.dumps(optional_runtime, indent=2), encoding='utf8')
    frozen = source_snapshot(project)
    (output / 'sources-before.json').write_text(json.dumps(frozen, indent=2), encoding='utf8')
    started = time.monotonic()
    counts = {}

    class Durable:
        def pytest_deselected(self, items):
            (output / 'deselected.json').write_text(json.dumps([item.nodeid for item in items],
                ensure_ascii=False, indent=2), encoding='utf8')

        def pytest_collection_finish(self, session):
            (output / 'collection.json').write_text(json.dumps([item.nodeid for item in session.items],
                ensure_ascii=False, indent=2), encoding='utf8')

        def pytest_runtest_logreport(self, report):
            row = {'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
                   'duration': report.duration}
            if report.failed or report.skipped:
                row['detail'] = str(report.longrepr)
            with (output / 'phases.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            key = report.when + ':' + report.outcome
            counts[key] = counts.get(key, 0) + 1

    plugins = [Durable()]
    if args.tmp_path_mode == 'isolated':
        plugins.append(isolated_tmp_path_plugin(pytest, output))
    code = pytest.main([*args.select, *['--deselect=' + item for item in args.deselect],
                        '-q', '--junitxml=' + str(output / 'results.xml')], plugins=plugins)
    after = source_snapshot(project)
    changed = sorted(key for key in frozen.keys() | after.keys() if frozen.get(key) != after.get(key))
    receipt = {'pytest_exit_code': int(code), 'phase_counts': counts,
               'cuda_initialized': torch.cuda.is_initialized(), 'elapsed_seconds': time.monotonic() - started,
               'sources_checked': len(frozen), 'changed_sources': changed,
               'scope': args.select, 'deselected': args.deselect,
               'tmp_path_mode': args.tmp_path_mode,
               'status': 'pass' if code == 0 else 'failed'}
    if receipt['cuda_initialized']:
        receipt['status'] = 'failed_cuda_initialized'
        code = 1
    if changed:
        receipt['status'] = 'failed_source_changed'
        code = 1
    if optional_runtime is not None:
        receipt['optional_runtime_stable'] = private_opencv_stable(optional_runtime)
        receipt['optional_runtime_receipt'] = 'optional-runtime.json'
        if not receipt['optional_runtime_stable']:
            receipt['status'] = 'failed_optional_runtime_changed'
            code = 1
    (output / 'terminal.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
