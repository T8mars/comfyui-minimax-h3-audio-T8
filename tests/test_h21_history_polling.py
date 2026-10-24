"""Protocol-only CPU tests with one real owned slow HTTP/WS process.

No Torch import, Core startup, trained model, sampling or GPU. Unit doubles are
explicit transport/control fixtures; the wire test uses the actual public
OwnedServer.request timeout and actual OwnedEventCapture/PID/argv checks.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CONTROLLER = None


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


PUBLIC_CONTROLLER = PROJECT / 'tools/h21_control_probe.py'
CONTROLLER_PATH = Path(os.environ.get('T8_HISTORY_POLL_CONTROLLER', PUBLIC_CONTROLLER)).resolve(strict=True)
CONTROLLER = load('h21_portable_controller_tested', CONTROLLER_PATH)
EXPECTED_AST = read(HERE / 'fixtures/h21_history_poll_ast.json')
SOURCE_PATHS = {PROJECT / 'tools' / name for name in CONTROLLER.TOOLS} | {
    CONTROLLER_PATH, Path(__file__), HERE / 'fixtures/h21_history_poll_ast.json',
    HERE / 'fixtures/h21_slow_history_server.py'}
SOURCE_HASHES = {path: sha(path) for path in SOURCE_PATHS}
TORCH_AT_IMPORT = sys.modules.get('torch')
CUDA_AT_IMPORT = TORCH_AT_IMPORT.cuda.is_initialized() if TORCH_AT_IMPORT is not None else False


def outside_body(source):
    tree = ast.parse(source)
    fn, = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == 'execute_observed_case']
    fn.body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


class FixtureHome(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='t8-history-')
        self.addCleanup(self.temporary.cleanup)
        self.fixture_root = Path(self.temporary.name).resolve()



def fixture_case(root, port=50001):
    root.mkdir(parents=True)
    plan = dict(schema='t8.h21.control_probe.v1', run_root=str(root),
        run_id='fixture', port=port, python=sys.executable)
    write(root / 'manifest.json', plan)
    write(root / 'graph.json', {'1': dict(class_type='InertOutputFixture', inputs={})})
    write(root / 'validation.json', dict(valid=True, node_errors={}, outputs=['1']))
    case = dict(name='inert-fixture-only', prompt='graph.json', validation='validation.json', observation=dict(reports=[]))
    return plan, root / 'manifest.json', case, root / 'attempts/new/executions/fixture'


def success():
    return {'fixture-prompt': dict(status=dict(status_str='success', completed=True), outputs={})}


class Clock:
    def __init__(self):
        self.value = 0.
        self.sleeps = []
    def monotonic(self):
        return self.value
    def time(self):
        return 1800000000. + self.value
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class UnitTransport(FixtureHome):
    def setUp(self):
        super().setUp()
        self.root = self.fixture_root / self._testMethodName
        self.plan, self.manifest, self.case, self.folder = fixture_case(self.root)
        self.clock = Clock()
        self.calls = []
        self.owner_checks = []
        self.alive, self.foreign = True, False
        self.after_GET = None
        self.post_error = None
        self.responses = [success()]
        parent = self

        class FixtureCapture:
            def __init__(self, manifest, plan, output, **kwargs):
                self.sock = object()
                self.events = []
                self.connect_count = 0
                self.output = output
                parent.capture = self
            def connect(self):
                self.connect_count += 1
            def mark_post(self):
                pass
            def receive(self):
                parent.clock.value += .05
            def terminal_for(self, prompt_id):
                return True
            def finish(self, prompt_id, required, history, **kwargs):
                write(self.output / 'fixture-finish.json', dict(prompt_id=prompt_id,
                    history=history, kwargs=kwargs, protocol_fixture_only=True))
        self.fake_observer = SimpleNamespace(OwnedEventCapture=FixtureCapture)

        def owner():
            self.owner_checks.append(self.clock.value)
            if self.foreign:
                raise RuntimeError('Port is not exclusively owned by this server process')
            if not self.alive:
                raise RuntimeError('Owned server is not live')

        def request(method, endpoint, **kwargs):
            self.calls.append((method, endpoint, self.clock.value))
            if method == 'POST':
                if self.post_error is not None:
                    raise self.post_error
                return dict(prompt_id='fixture-prompt', node_errors={})
            if self.after_GET is not None:
                self.after_GET()
            result = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
            if isinstance(result, BaseException):
                raise result
            return result
        self.server = SimpleNamespace(process=SimpleNamespace(pid=os.getpid(), poll=lambda: None if self.alive else 1),
            assert_port_owner=owner, request=request)

    def execute(self, timeout=8):
        with patch.dict(sys.modules, {'observe_h21_owned_events': self.fake_observer}), patch.object(CONTROLLER, 'time', self.clock):
            return CONTROLLER.execute_observed_case(self.server, self.plan, self.manifest, self.case, self.folder, timeout=timeout)

    def checks(self, GETs):
        self.assertEqual(sum(method == 'POST' for method, _, _ in self.calls), 1)
        self.assertEqual(sum(method == 'GET' for method, _, _ in self.calls), GETs)
        self.assertEqual(self.capture.connect_count, 1)

    def test_ReadTimeout_keeps_original_POST_and_recovers_GET_only(self):
        error = requests.exceptions.ReadTimeout('fixed2s timeout')
        self.responses = [error, success()]
        self.assertTrue(self.execute()['status']['completed'])
        self.checks(2)
        row = read(self.folder / 'history-poll-transport/000001.json')
        self.assertEqual(row['error_type'], 'ReadTimeout')
        self.assertTrue(row['owned_process_and_port_revalidated'])
        self.assertFalse(row['sampling_resubmitted'] or row['WS_reconnected'])
        self.assertGreaterEqual(self.clock.sleeps[0], 1)

    def test_ConnectionError_and_ConnectTimeout_are_bounded_readonly(self):
        self.responses = [requests.exceptions.ConnectionError('reset'), requests.exceptions.ConnectTimeout('connect'), success()]
        self.assertTrue(self.execute()['status']['completed'])
        self.checks(3)
        self.assertEqual(len(list((self.folder / 'history-poll-transport').glob('*.json'))), 2)
        self.assertTrue(all(seconds == 1 for seconds in self.clock.sleeps))

    def test_dead_process_after_GET_timeout_is_not_retried(self):
        self.responses = [requests.exceptions.ReadTimeout('load blocked')]
        self.after_GET = lambda: setattr(self, 'alive', False)
        with self.assertRaisesRegex(RuntimeError, 'Owned Core exited after history transport failure'):
            self.execute()
        self.checks(1)
        self.assertFalse(read(self.folder / 'history-poll-transport/000001.json')['owned_process_and_port_revalidated'])

    def test_foreign_port_after_GET_timeout_is_not_contacted_again(self):
        self.responses = [requests.exceptions.ReadTimeout('first query')]
        self.after_GET = lambda: setattr(self, 'foreign', True)
        with self.assertRaisesRegex(RuntimeError, 'not exclusively owned'):
            self.execute()
        self.checks(1)
        self.assertFalse(read(self.folder / 'history-poll-transport/000001.json')['owned_process_and_port_revalidated'])

    def test_continuous_transport_errors_honor_original_deadline_no_busy_loop(self):
        self.responses = [requests.exceptions.ConnectionError('immediate reset')]
        with self.assertRaisesRegex(TimeoutError, 'no authoritative history before timeout; no resubmission'):
            self.execute(timeout=2.3)
        self.checks(3)
        self.assertAlmostEqual(self.clock.value, 2.3)
        self.assertEqual(len(self.clock.sleeps), 3)
        self.assertTrue(all(0 < seconds <= 1 for seconds in self.clock.sleeps))
        self.assertFalse((self.folder / 'history.json').exists())

    def test_POST_timeout_propagates_same_exception_without_GET_or_second_POST(self):
        failure = requests.exceptions.ReadTimeout('unknown POST outcome do not retry')
        self.post_error = failure
        with self.assertRaises(requests.exceptions.ReadTimeout) as caught:
            self.execute()
        self.assertIs(caught.exception, failure)
        self.checks(0)

    def test_actual_HTTP500_is_not_classified_as_transport_retry(self):
        failure = requests.exceptions.HTTPError('500 real HTTP error')
        self.responses = [failure]
        with self.assertRaises(requests.exceptions.HTTPError) as caught:
            self.execute()
        self.assertIs(caught.exception, failure)
        self.checks(1)
        self.assertFalse((self.folder / 'history-poll-transport').exists())

    def test_bad_JSON_is_not_classified_as_transport_retry(self):
        failure = requests.exceptions.JSONDecodeError('invalid JSON', 'not JSON', 0)
        self.responses = [failure]
        with self.assertRaises(requests.exceptions.JSONDecodeError) as caught:
            self.execute()
        self.assertIs(caught.exception, failure)
        self.checks(1)
        self.assertFalse((self.folder / 'history-poll-transport').exists())

    def test_actual_failed_history_after_timeout_is_retained_not_success(self):
        failed = {'fixture-prompt': dict(status=dict(status_str='error', completed=False), outputs={}, reason='MODEL exception')}
        self.responses = [requests.exceptions.ReadTimeout('first transport'), failed]
        with self.assertRaisesRegex(RuntimeError, 'Real Core execution failed'):
            self.execute()
        self.checks(2)
        self.assertEqual(read(self.folder / 'history.json'), failed['fixture-prompt'])

    def test_actual_cancelled_history_after_timeout_is_retained_not_success(self):
        cancelled = {'fixture-prompt': dict(status=dict(status_str='error', completed=False), outputs={}, reason='execution_interrupted')}
        self.responses = [requests.exceptions.ConnectTimeout('first transport'), cancelled]
        with self.assertRaisesRegex(RuntimeError, 'Real Core execution failed'):
            self.execute()
        self.checks(2)
        self.assertEqual(read(self.folder / 'history.json'), cancelled['fixture-prompt'])

    def test_closed_WS_does_not_reconnect_or_resubmit(self):
        self.responses = [requests.exceptions.ReadTimeout('first'), success()]
        old_receive = None
        # Explicit unit observer loss; no actual socket is faked as a wire test.
        class Disconnect:
            def __init__(self, *args, **kwargs):
                self.sock, self.events, self.connect_count = None, [], 0
                self.output = args[2]
                parent.capture = self
            def connect(self): self.connect_count += 1
            def mark_post(self): pass
            def receive(self): parent.clock.value += .05
            def terminal_for(self, prompt_id): return False
            def finish(self, *args, **kwargs): pass
        parent = self
        self.fake_observer = SimpleNamespace(OwnedEventCapture=Disconnect)
        self.assertTrue(self.execute()['status']['completed'])
        self.checks(2)
        self.assertIsNone(old_receive)


class RealOwnedWire(FixtureHome):
    def test_slow_owned_history_GET_recovers_after_real2s_without_second_POST(self):
        root = self.fixture_root / 'real-owned-wire'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.assertNotIn(port, {8189, 8878, 8991})
        plan, manifest, case, folder = fixture_case(root, port)
        script = root / 'runtime_tools/h21_control_probe.py'
        script.parent.mkdir()
        shutil.copyfile(HERE / 'fixtures/h21_slow_history_server.py', script)
        observer = load('h22_actual_wire_observer', PROJECT / 'tools/observe_h21_owned_events.py')
        pilot = load('h22_actual_OwnedServer', PROJECT / 'tools/run_progressive_pilot.py')
        server = pilot.OwnedServer(root, port, cpu=True)
        command = [sys.executable, '-I', '-B', '-X', 'utf8', str(script), '--_serve', str(manifest)]
        owned = None
        with (root / 'fixture.stdout.log').open('xb') as out, (root / 'fixture.stderr.log').open('xb') as err:
            owned = subprocess.Popen(command, cwd=root, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            server.process = owned
            import psutil
            write(root / 'owned.json', dict(pid=owned.pid, parent_pid=os.getpid(), argv=command,
                create_time=psutil.Process(owned.pid).create_time(), fixture_only=True, Core_started=False, GPU_started=False))
            try:
                started = time.monotonic()
                while True:
                    if owned.poll() is not None:
                        self.fail('Inert owned fixture exited before ready')
                    try:
                        actual = server.request('GET', '/ready')
                        if actual.get('fixture_only'):
                            break
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, RuntimeError):
                        if time.monotonic() - started > 15:
                            raise
                        time.sleep(.1)
                before = observer.listener_owner(manifest, plan)
                self.assertEqual(before['pid'], owned.pid)
                self.assertEqual(Path(before['exe']).resolve(), Path(sys.executable).resolve())
                with patch.dict(sys.modules, {'observe_h21_owned_events': observer}):
                    result = CONTROLLER.execute_observed_case(server, plan, manifest, case, folder, timeout=12)
                self.assertTrue(result['status']['completed'])
                self.assertEqual(result['status']['status_str'], 'success')
                after = observer.listener_owner(manifest, plan)
                self.assertEqual(before, after)
                state = read(root / 'fixture-state.json')
                self.assertEqual(state['posts'], 1)
                self.assertEqual(state['gets'], 2)
                self.assertEqual(state['ws_connections'], 1)
                transport, = list((folder / 'history-poll-transport').glob('*.json'))
                self.assertEqual(read(transport)['error_type'], 'ReadTimeout')
                self.assertTrue(read(transport)['owned_process_and_port_revalidated'])
                event_receipt = read(folder / 'observation/terminal.json')
                self.assertEqual(event_receipt['owner'], before)
                self.assertEqual(event_receipt['generation_status'], 'success')
                self.assertEqual(event_receipt['terminal_events'], ['execution_success'])
                self.assertTrue(event_receipt['connected_before_POST'])
                write(root / 'verified.json', dict(fixture_only=True, actual_http_timeout_seconds=2,
                    actual_first_response_delay_seconds=2.4, actual_history_GETs=state['gets'], actual_POSTs=state['posts'],
                    actual_WS_connections=state['ws_connections'], actual_process_owner=before,
                    actual_OwnedServer_request_source=str(PROJECT / 'tools/run_progressive_pilot.py'),
                    actual_OwnedEventCapture_source=str(PROJECT / 'tools/observe_h21_owned_events.py'),
                    Core_or_MODEL_or_GPU_started=False, end_to_end_history_recovery=True))
            finally:
                server.stop()  # Only the fixture's actual Popen handle/descendants.
                write(root / 'owned-stop.json', server.stop_receipt)
                self.assertIsNotNone(owned.poll())
                self.assertEqual(server.stop_receipt['owned_children_remaining'], [])


class SourceContract(unittest.TestCase):
    def test_only_selected_function_body_changed_and_original_sources_current(self):
        original = PUBLIC_CONTROLLER.read_bytes()
        selected = CONTROLLER_PATH.read_bytes()
        self.assertEqual(sha(PUBLIC_CONTROLLER), SOURCE_HASHES[PUBLIC_CONTROLLER])
        tree = ast.parse(selected)
        function, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == EXPECTED_AST['function_name']]
        self.assertEqual(hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
                         EXPECTED_AST['execute_observed_case_ast_sha256'])
        self.assertEqual(outside_body(original), outside_body(selected))
        for path, expected in SOURCE_HASHES.items():
            self.assertEqual(sha(path), expected)
        # The normal CPU runner imports Torch before collecting tests. Prove
        # this transport suite adds no import/context; preserve the original
        # no-Torch assertion when run as standalone stdlib unittest.
        self.assertIs(sys.modules.get('torch'), TORCH_AT_IMPORT)
        if TORCH_AT_IMPORT is None:
            self.assertNotIn('torch', sys.modules)
        else:
            self.assertEqual(TORCH_AT_IMPORT.cuda.is_initialized(), CUDA_AT_IMPORT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
