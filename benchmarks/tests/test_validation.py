"""Regression tests for the executable benchmark validation boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks"
CASE_ID = "radial-crs-input-analytical-3dmu"
RESULT_NAME = f"{CASE_ID}.frontierguard.json"


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def valid_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output_directory = tmp_path_factory.mktemp("benchmark-results")
    completed = run_script(
        "benchmarks/scripts/run_frontierguard.py",
        "--case",
        CASE_ID,
        "--output-dir",
        str(output_directory),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((output_directory / RESULT_NAME).read_text(encoding="utf-8"))


def add_manifest_extra(cases_directory: Path) -> None:
    manifest_path = cases_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_forbidden_extra"] = True
    write_json(manifest_path, manifest)


def add_case_extra(cases_directory: Path) -> None:
    case_path = cases_directory / "radial_crs_input_analytical_3dmu.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["model"]["schema_forbidden_extra"] = True
    write_json(case_path, case)


@pytest.mark.parametrize(
    "mutate",
    [add_manifest_extra, add_case_extra],
    ids=["manifest-extra", "case-extra"],
)
def test_asset_validation_rejects_schema_forbidden_extras(
    tmp_path: Path,
    mutate: Callable[[Path], None],
) -> None:
    cases_directory = tmp_path / "cases"
    shutil.copytree(BENCHMARK_ROOT / "cases", cases_directory)
    mutate(cases_directory)

    completed = run_script(
        "benchmarks/scripts/validate_assets.py",
        "--manifest",
        str(cases_directory / "manifest.json"),
    )

    assert completed.returncode == 1
    assert "Draft 2020-12" in completed.stderr
    assert "Additional properties are not allowed" in completed.stderr


def test_result_validation_rejects_schema_forbidden_extra(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    document = deepcopy(valid_result)
    document["executions"][0]["schema_forbidden_extra"] = True
    write_json(tmp_path / RESULT_NAME, document)

    completed = run_script(
        "benchmarks/scripts/validate_results.py",
        "--case",
        CASE_ID,
        "--results-dir",
        str(tmp_path),
    )

    assert completed.returncode == 1
    assert "Draft 2020-12" in completed.stderr
    assert "Additional properties are not allowed" in completed.stderr


@pytest.mark.parametrize("variant_index", [0, 1], ids=["baseline", "metamorphic"])
def test_result_validation_rejects_nonoptimal_execution_status(
    tmp_path: Path,
    valid_result: dict[str, Any],
    variant_index: int,
) -> None:
    document = deepcopy(valid_result)
    document["executions"][variant_index]["statuses"][0] = "time_limit"
    write_json(tmp_path / RESULT_NAME, document)

    completed = run_script(
        "benchmarks/scripts/validate_results.py",
        "--case",
        CASE_ID,
        "--results-dir",
        str(tmp_path),
    )

    assert completed.returncode == 1
    assert "execution status" in completed.stderr
    assert "observed='time_limit' required='optimal'" in completed.stderr
