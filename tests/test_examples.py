from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = tuple(sorted((Path(__file__).parents[1] / "examples").glob("*.py")))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_shipped_example_runs(example: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=example.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
