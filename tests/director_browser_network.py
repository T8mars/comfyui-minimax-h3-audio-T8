"""Optional diagnostics only: record failed network requests without retries."""
import json

import pytest


@pytest.fixture(autouse=True)
def director_browser_network_evidence(monkeypatch):
    from playwright.sync_api import Browser

    original = Browser.new_page

    def observed_page(self, *args, **kwargs):
        page = original(self, *args, **kwargs)
        page.on('requestfailed', lambda request: print('Owned browser request failed: '+json.dumps(
            {'url': request.url, 'failure': request.failure}, ensure_ascii=False)))
        return page

    monkeypatch.setattr(Browser, 'new_page', observed_page)
