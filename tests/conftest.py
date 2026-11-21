import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core import GoalStore


@pytest.fixture
def store(tmp_path):
    return GoalStore(tmp_path / "synthetic.db")
