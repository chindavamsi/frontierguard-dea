#!/usr/bin/env python3
"""Validate adapter results against oracles and declared relations."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from _common import (
    DEFAULT_MANIFEST,
    RESULT_SCHEMA,
    BenchmarkAssetError,
    load_json,
    load_suite,
    require_keys,
    require_version,
    select_cases,
    validate_json_schema,
)

PRIMARY_FIELDS = ("radial_factor", "technical_efficiency")


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
        help="Validate one manifest case ID; repeat to validate several.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing generated result JSON.",
    )
    parser.add_argument(
        "--result-suffix",
        default="frontierguard.json",
        help="Filename suffix after '<case-id>.' for one adapter.",
    )
    return parser.parse_args()


def close(
    observed: float,
    expected: float,
    *,
    absolute: float,
    relative: float,
) -> bool:
    return (
        math.isfinite(observed)
        and math.isfinite(expected)
        and abs(observed - expected) <= absolute + relative * abs(expected)
    )


def execution_map(
    execution: dict[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    require_keys(
        execution,
        (
            "variant_id",
            "relation",
            "dmu_names",
            "reference_names",
            "radial_factor",
            "technical_efficiency",
            "statuses",
            "audit_residual",
            "audit_tolerance",
            "data_fingerprint",
            "native_metrics",
        ),
        label=label,
    )
    names = execution["dmu_names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise BenchmarkAssetError(f"{label} has invalid DMU names.")
    for field in (*PRIMARY_FIELDS, "statuses", "audit_residual"):
        values = execution[field]
        if not isinstance(values, list) or len(values) != len(names):
            raise BenchmarkAssetError(
                f"{label} field {field} must contain one value per DMU."
            )
    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        result[name] = {
            field: execution[field][index]
            for field in (*PRIMARY_FIELDS, "statuses", "audit_residual")
        }
    return result


def validate_result_structure(
    document: dict[str, Any],
    *,
    case: dict[str, Any],
    source: Path,
) -> dict[str, dict[str, Any]]:
    label = f"result {source}"
    validate_json_schema(document, schema_path=RESULT_SCHEMA, label=label)
    require_version(document, label=label)
    require_keys(
        document,
        (
            "case_id",
            "case_schema_version",
            "implementation",
            "run",
            "outcome",
            "model",
            "executions",
            "notes",
        ),
        label=label,
    )
    if document["case_id"] != case["case_id"]:
        raise BenchmarkAssetError(
            f"{label} case_id does not match {case['case_id']!r}."
        )
    if document["case_schema_version"] != case["schema_version"]:
        raise BenchmarkAssetError(f"{label} case schema version does not match.")
    if document["model"] != case["model"]:
        raise BenchmarkAssetError(f"{label} model echo differs from the case.")
    outcome = document["outcome"]
    if not isinstance(outcome, dict) or outcome.get("category") != "success":
        category = (
            outcome.get("category", "missing")
            if isinstance(outcome, dict)
            else "missing"
        )
        message = outcome.get("message", "") if isinstance(outcome, dict) else ""
        raise BenchmarkAssetError(
            f"{label} has non-success outcome {category!r}: {message}"
        )
    implementation = document["implementation"]
    if not isinstance(implementation, dict):
        raise BenchmarkAssetError(f"{label} implementation must be an object.")
    require_keys(
        implementation,
        (
            "name",
            "version",
            "adapter_name",
            "adapter_version",
            "language",
            "runtime_version",
            "package_source",
            "package_license",
            "solver_name",
            "solver_version",
            "source_revision",
            "normalizations",
        ),
        label=f"{label} implementation",
    )
    executions = document["executions"]
    if not isinstance(executions, list) or not executions:
        raise BenchmarkAssetError(f"{label} executions must be a non-empty array.")
    by_variant: dict[str, dict[str, Any]] = {}
    for execution in executions:
        if not isinstance(execution, dict):
            raise BenchmarkAssetError(f"{label} execution must be an object.")
        variant_id = execution.get("variant_id")
        if not isinstance(variant_id, str) or variant_id in by_variant:
            raise BenchmarkAssetError(
                f"{label} has an invalid or repeated variant ID {variant_id!r}."
            )
        by_variant[variant_id] = execution
        execution_map(execution, label=f"{label} execution {variant_id!r}")
    expected_variants = {
        "baseline",
        *(test["id"] for test in case["metamorphic_tests"]),
    }
    if set(by_variant) != expected_variants:
        raise BenchmarkAssetError(
            f"{label} variants differ: expected {sorted(expected_variants)}, "
            f"received {sorted(by_variant)}."
        )
    if by_variant["baseline"]["relation"] != "baseline":
        raise BenchmarkAssetError(f"{label} baseline relation must be 'baseline'.")
    return by_variant


def compare_values(
    observed: list[Any],
    observed_names: list[str],
    expected: list[Any],
    expected_names: list[str],
    *,
    absolute: float,
    relative: float,
    label: str,
) -> list[str]:
    failures: list[str] = []
    observed_by_name = dict(zip(observed_names, observed, strict=True))
    for name, expected_value in zip(expected_names, expected, strict=True):
        if name not in observed_by_name:
            failures.append(f"{label}: missing DMU {name!r}")
            continue
        observed_value = observed_by_name[name]
        if not isinstance(observed_value, (int, float)) or isinstance(
            observed_value, bool
        ):
            failures.append(f"{label}: {name} is not numeric")
            continue
        if not close(
            float(observed_value),
            float(expected_value),
            absolute=absolute,
            relative=relative,
        ):
            failures.append(
                f"{label}: {name} observed={observed_value!r} "
                f"expected={expected_value!r}"
            )
    return failures


def validate_case_result(
    document: dict[str, Any],
    case: dict[str, Any],
    *,
    source: Path,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    try:
        variants = validate_result_structure(document, case=case, source=source)
    except BenchmarkAssetError as exc:
        return {}, [str(exc)]
    baseline = variants["baseline"]
    expected = case["oracle"]["expected"]
    tolerance = case["tolerances"]
    for field in PRIMARY_FIELDS:
        failures.extend(
            compare_values(
                baseline[field],
                baseline["dmu_names"],
                expected[field],
                expected["dmu_names"],
                absolute=float(tolerance["absolute"]),
                relative=float(tolerance["relative"]),
                label=f"{case['case_id']} baseline {field}",
            )
        )

    baseline_by_name = execution_map(baseline, label=f"{case['case_id']} baseline")
    expected_statuses = dict(
        zip(
            expected["dmu_names"],
            expected["statuses"],
            strict=True,
        )
    )
    for name, expected_status in expected_statuses.items():
        observed_status = baseline_by_name[name]["statuses"]
        if observed_status != expected_status:
            failures.append(
                f"{case['case_id']} baseline status: {name} "
                f"observed={observed_status!r} expected={expected_status!r}"
            )
    for variant_id, execution in variants.items():
        mapped = execution_map(
            execution,
            label=f"{case['case_id']} variant {variant_id}",
        )
        for name, metrics in mapped.items():
            status = metrics["statuses"]
            if status != "optimal":
                failures.append(
                    f"{case['case_id']} {variant_id} execution status: {name} "
                    f"observed={status!r} required='optimal'"
                )
            residual = metrics["audit_residual"]
            if (
                not isinstance(residual, (int, float))
                or isinstance(residual, bool)
                or not math.isfinite(float(residual))
                or float(residual) > float(tolerance["audit"])
            ):
                failures.append(
                    f"{case['case_id']} {variant_id} audit: {name} "
                    f"residual={residual!r} limit={tolerance['audit']!r}"
                )
        if variant_id == "baseline":
            continue
        if execution["relation"] != "name_aligned_primary_scores_equal":
            failures.append(
                f"{case['case_id']} {variant_id}: unexpected relation "
                f"{execution['relation']!r}"
            )
            continue
        for name, baseline_metrics in baseline_by_name.items():
            if name not in mapped:
                failures.append(
                    f"{case['case_id']} {variant_id}: missing baseline DMU {name!r}"
                )
                continue
            for field in PRIMARY_FIELDS:
                observed_value = mapped[name][field]
                expected_value = baseline_metrics[field]
                if (
                    not isinstance(observed_value, (int, float))
                    or isinstance(observed_value, bool)
                    or not close(
                        float(observed_value),
                        float(expected_value),
                        absolute=float(tolerance["absolute"]),
                        relative=float(tolerance["relative"]),
                    )
                ):
                    failures.append(
                        f"{case['case_id']} {variant_id} {field}: {name} "
                        f"observed={observed_value!r} "
                        f"baseline={expected_value!r}"
                    )
    return baseline, failures


def validate_cross_case_relations(
    manifest: dict[str, Any],
    selected_cases: dict[str, tuple[Path, dict[str, Any]]],
    baselines: dict[str, dict[str, Any]],
) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    for relation in manifest["cross_case_relations"]:
        left_id = relation["left_case"]
        right_id = relation["right_case"]
        if left_id not in selected_cases or right_id not in selected_cases:
            continue
        if left_id not in baselines or right_id not in baselines:
            continue
        checked += 1
        left = baselines[left_id]
        right = baselines[right_id]
        left_values = dict(
            zip(left["dmu_names"], left[relation["left_field"]], strict=True)
        )
        right_values = dict(
            zip(right["dmu_names"], right[relation["right_field"]], strict=True)
        )
        left_case = selected_cases[left_id][1]
        right_case = selected_cases[right_id][1]
        absolute = max(
            float(left_case["tolerances"]["absolute"]),
            float(right_case["tolerances"]["absolute"]),
        )
        relative = max(
            float(left_case["tolerances"]["relative"]),
            float(right_case["tolerances"]["relative"]),
        )
        if set(left_values) != set(right_values):
            failures.append(
                f"{relation['id']}: left and right baseline DMU sets differ"
            )
            continue
        for name, left_value in left_values.items():
            right_value = right_values[name]
            expected_right = (
                left_value
                if relation["relation"] == "name_aligned_equal"
                else 1.0 / left_value
            )
            if not close(
                float(right_value),
                float(expected_right),
                absolute=absolute,
                relative=relative,
            ):
                failures.append(
                    f"{relation['id']}: {name} left={left_value!r} "
                    f"right={right_value!r} expected_right={expected_right!r}"
                )
    return checked, failures


def main() -> int:
    args = parse_args()
    try:
        manifest, all_cases = load_suite(args.manifest)
        selected = select_cases(all_cases, args.case)
    except BenchmarkAssetError as exc:
        print(f"RESULT VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    baselines: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    metamorphic_count = 0
    for case_id, (_, case) in selected.items():
        source = args.results_dir / f"{case_id}.{args.result_suffix}"
        try:
            document = load_json(source)
        except BenchmarkAssetError as exc:
            failures.append(str(exc))
            continue
        baseline, case_failures = validate_case_result(
            document,
            case,
            source=source,
        )
        if baseline:
            baselines[case_id] = baseline
        failures.extend(case_failures)
        metamorphic_count += len(case["metamorphic_tests"])

    cross_count, cross_failures = validate_cross_case_relations(
        manifest,
        selected,
        baselines,
    )
    failures.extend(cross_failures)
    if failures:
        print(
            f"RESULT VALIDATION FAILED: {len(failures)} finding(s)",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        "RESULT VALIDATION PASSED: "
        f"cases={len(selected)} oracle_fields={2 * len(selected)} "
        f"metamorphic_relations={metamorphic_count} "
        f"cross_case_relations={cross_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
