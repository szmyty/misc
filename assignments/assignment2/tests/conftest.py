from pathlib import Path

import pytest as pytest


@pytest.fixture(name="resources")
def fixture_resources() -> Path:
    return Path(__file__).resolve().parent.parent.joinpath("resources")
