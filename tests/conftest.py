from __future__ import annotations

import pytest


@pytest.fixture
def tmp_dir(tmp_path) -> str:
    """Compatibility fixture for legacy script-style tests expecting a string path."""
    return str(tmp_path)
