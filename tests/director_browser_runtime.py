"""Optional pytest plugin for an explicitly chosen, already installed Chromium.

Usage: add tests to PYTHONPATH, -p director_browser_runtime --director-chromium=PATH.
This only selects a real browser executable, never mocks Playwright/browser APIs.
"""
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--director-chromium', help='Absolute path of an existing Chromium executable for Director UI tests')


@pytest.fixture(autouse=True)
def director_browser_executable(request, monkeypatch):
    binary = request.config.getoption('--director-chromium')
    if not binary or not request.node.path.name.startswith('test_director'):
        return
    path = Path(binary)
    if not path.is_absolute() or not path.is_file():
        pytest.fail('Explicit Director Chromium executable does not exist: '+binary)
    from playwright.sync_api import BrowserType

    launch = BrowserType.launch

    def selected_launch(self, *args, **kwargs):
        if self.name == 'chromium':
            kwargs.setdefault('executable_path', str(path))
        return launch(self, *args, **kwargs)

    monkeypatch.setattr(BrowserType, 'launch', selected_launch)
