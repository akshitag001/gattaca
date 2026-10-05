import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gattaca import synth  # noqa: E402


@pytest.fixture(scope="session")
def contact0():
    return synth.contact_impression(0)


@pytest.fixture(scope="session")
def photo0():
    img, truth = synth.contactless_photo(0, sample=1, pprp=10)
    return img, truth
