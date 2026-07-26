#!/usr/bin/env python3
"""Run the FrontierGuard source adapter for benchmark cases."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    DEFAULT_MANIFEST,
    REPOSITORY_ROOT,
    BenchmarkAssetError,
    load_suite,
    select_cases,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import frontierguard as fg

ADAPTER_VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the benchmark manifest.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one manifest case ID; repeat to run several.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated result JSON.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def transformed_data(
    baseline: dict[str, Any],
    transform: dict[str, Any] | None,
) -> dict[str, Any]:
    data = deepcopy(baseline)
    if transform is None:
        return data

    inputs = np.asarray(data["inputs"], dtype=float)
    outputs = np.asarray(data["outputs"], dtype=float)
    dmu_names = list(data["dmu_names"])
    input_names = list(data["input_names"])
    output_names = list(data["output_names"])
    kind = transform["kind"]

    if kind == "scale_columns":
        inputs = inputs * np.asarray(transform["input_factors"], dtype=float)
        outputs = outputs * np.asarray(transform["output_factors"], dtype=float)
    elif kind == "permute_dmus":
        order = np.asarray(transform["order"], dtype=int)
        inputs = inputs[order]
        outputs = outputs[order]
        dmu_names = [dmu_names[index] for index in order]
    elif kind == "permute_variables":
        input_order = np.asarray(transform["input_order"], dtype=int)
        output_order = np.asarray(transform["output_order"], dtype=int)
        inputs = inputs[:, input_order]
        outputs = outputs[:, output_order]
        input_names = [input_names[index] for index in input_order]
        output_names = [output_names[index] for index in output_order]
    elif kind == "duplicate_reference":
        source = int(transform["source_index"])
        inputs = np.vstack([inputs, inputs[source]])
        outputs = np.vstack([outputs, outputs[source]])
        dmu_names.append(str(transform["new_name"]))
    else:
        raise BenchmarkAssetError(f"Unsupported transform kind {kind!r}.")

    return {
        "inputs": inputs.tolist(),
        "outputs": outputs.tolist(),
        "dmu_names": dmu_names,
        "input_names": input_names,
        "output_names": output_names,
    }


def execute(
    case: dict[str, Any],
    *,
    variant_id: str,
    relation: str,
    transform: dict[str, Any] | None,
) -> dict[str, Any]:
    data_spec = transformed_data(case["data"], transform)
    data = fg.DEAData(
        inputs=data_spec["inputs"],
        outputs=data_spec["outputs"],
        dmu_names=data_spec["dmu_names"],
        input_names=data_spec["input_names"],
        output_names=data_spec["output_names"],
    )
    model = case["model"]
    result = fg.RadialDEA(
        orientation=model["orientation"],
        rts=model["returns_to_scale"],
        slacks=model["slacks"],
    ).fit(data, exclude_self=model["exclude_self"])
    return {
        "variant_id": variant_id,
        "relation": relation,
        "dmu_names": list(result.dmu_names),
        "reference_names": list(result.reference_names),
        "radial_factor": result.radial_factor.tolist(),
        "technical_efficiency": result.technical_efficiency.tolist(),
        "statuses": [status.value for status in result.statuses],
        "audit_residual": result.audit_residual.tolist(),
        "audit_tolerance": result.audit_tolerance,
        "data_fingerprint": result.data_fingerprint,
        "native_metrics": {
            "frontierguard_result_schema_version": result.to_dict()["schema_version"],
            "frontierguard_model": result.to_dict()["model"],
            "phase_two": result.phase_two,
            "solver": result.solver,
        },
    }


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, fg.DataValidationError):
        return "invalid_input"
    if isinstance(exc, fg.ModelValidationError):
        return "invalid_input"
    if isinstance(exc, fg.NumericalError):
        return "numerical_failure"
    if isinstance(exc, fg.SolverError):
        return "solver_error"
    if isinstance(exc, BenchmarkAssetError):
        return "schema_error"
    return "crash"


def implementation_metadata(solver_name: str) -> dict[str, Any]:
    try:
        import scipy

        solver_version = f"SciPy {scipy.__version__}; bundled HiGHS version not exposed"
    except ImportError:
        solver_version = "unknown"
    return {
        "name": "FrontierGuard DEA",
        "version": fg.__version__,
        "adapter_name": "frontierguard-source-adapter",
        "adapter_version": ADAPTER_VERSION,
        "language": "Python",
        "runtime_version": platform.python_version(),
        "package_source": "https://github.com/chindavamsi/frontierguard-dea",
        "package_license": "MIT",
        "solver_name": solver_name,
        "solver_version": solver_version,
        "source_revision": source_revision(),
        "normalizations": [],
    }


def run_case(case: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    started = utc_now()
    executions: list[dict[str, Any]] = []
    outcome = {"category": "success", "message": "All executions completed."}
    solver_name = "scipy-highs"
    failed = False

    try:
        baseline = execute(
            case,
            variant_id="baseline",
            relation="baseline",
            transform=None,
        )
        solver_name = str(baseline["native_metrics"]["solver"])
        executions.append(baseline)
        executions.extend(
            execute(
                case,
                variant_id=test["id"],
                relation=test["relation"],
                transform=test["transform"],
            )
            for test in case["metamorphic_tests"]
        )
    except Exception as exc:  # adapter boundary: classify and preserve evidence
        failed = True
        outcome = {
            "category": classify_exception(exc),
            "message": f"{type(exc).__name__}: {exc}",
        }

    document = {
        "schema_version": "0.1.0",
        "protocol_version": case["protocol_version"],
        "case_id": case["case_id"],
        "case_schema_version": case["schema_version"],
        "implementation": implementation_metadata(solver_name),
        "run": {
            "started_at": started,
            "finished_at": utc_now(),
            "operating_system": platform.platform(),
            "machine": platform.machine() or "unknown",
            "command": list(sys.argv),
        },
        "outcome": outcome,
        "model": deepcopy(case["model"]),
        "executions": executions,
        "notes": [
            "Generated from the source checkout; this is local adapter evidence.",
            "No external DEA implementation was invoked.",
        ],
    }
    return document, failed


def main() -> int:
    args = parse_args()
    try:
        _, all_cases = load_suite(args.manifest)
        selected = select_cases(all_cases, args.case)
    except BenchmarkAssetError as exc:
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case_id, (_, case) in selected.items():
        document, failed = run_case(case)
        output_path = args.output_dir / f"{case_id}.frontierguard.json"
        try:
            output_path.write_text(
                json.dumps(
                    document,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            print(f"RUN FAILED: cannot write {output_path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        failures += int(failed)
        print(
            f"WROTE {output_path} "
            f"outcome={document['outcome']['category']} "
            f"executions={len(document['executions'])}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
