# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_console(monkeypatch):
    """Patch the console used by bruno_swarm.cli so prints don't hit stdout."""
    # We must access the real module object via sys.modules because
    # `import bruno_swarm.cli` resolves to the Click group (exported by __init__.py).
    import bruno_swarm.cli  # noqa: F401 — ensure the module is loaded

    cli_module = sys.modules["bruno_swarm.cli"]
    fake = MagicMock()
    monkeypatch.setattr(cli_module, "console", fake)
    return fake
