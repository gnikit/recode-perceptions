from pathlib import Path

import pytest


@pytest.fixture
def test_data():
    return Path(__file__).parent / "test_images"


@pytest.fixture
def root_dir():
    return Path(__file__).parent.parent
