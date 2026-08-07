"""Shared strict-loading and structural checks for benchmark scripts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
DEFAULT_MANIFEST = BENCHMARK_ROOT / "cases" / "manifest.json"
SCHEMA_DIRECTORY = BENCHMARK_ROOT / "schemas"
CASE_SCHEMA = SCHEMA_DIRECTORY / "case.schema.json"
MANIFEST_SCHEMA = SCHEMA_DIRECTORY / "manifest.schema.json"
RESULT_SCHEMA = SCHEMA_DIRECTORY / "result.schema.json"
SCHEMA_VERSION = "0.1.0"
PROTOCOL_VERSION = "0.1.0"


class BenchmarkAssetError(ValueError):
    """A benchmark asset is malformed or internally inconsistent."""


def _reject_constant(value: str) -> None:
    raise BenchmarkAssetError(f"Non-finite JSON constant {value!r} is forbidden.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkAssetError(f"Duplicate JSON key {key!r} is forbidden.")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    """Load one strict JSON object, rejecting duplicate and non-finite values."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkAssetError(f"Cannot read {path}: {exc}") from exc
    try:
        parsed = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as exc:
        raise BenchmarkAssetError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise BenchmarkAssetError(f"{path} must contain a top-level JSON object.")
    return parsed


@cache
def schema_validator(schema_path: Path) -> Draft202012Validator:
    """Load and verify one Draft 2020-12 schema before using it."""
    schema_path = schema_path.resolve()
    schema = load_json(schema_path)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise BenchmarkAssetError(
            f"{schema_path} must declare JSON Schema draft 2020-12."
        )
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise BenchmarkAssetError(
            f"Invalid Draft 2020-12 schema {schema_path}: {exc.message}"
        ) from exc
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_json_schema(
    document: Mapping[str, Any],
    *,
    schema_path: Path,
    label: str,
) -> None:
    """Validate a document against a verified Draft 2020-12 schema."""
    validator = schema_validator(schema_path)
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (error.json_path, error.message),
    )
    if not errors:
        return
    displayed = errors[:5]
    findings = "; ".join(f"{error.json_path}: {error.message}" for error in displayed)
    if len(errors) > len(displayed):
        findings += f"; {len(errors) - len(displayed)} additional violation(s)"
    raise BenchmarkAssetError(
        f"{label} violates {schema_path.name} (Draft 2020-12): {findings}"
    )


def require_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    *,
    label: str,
) -> None:
    """Require keys without attempting to replace full JSON Schema validation."""
    missing = sorted(set(required) - set(value))
    if missing:
        raise BenchmarkAssetError(f"{label} is missing: {', '.join(missing)}.")


def require_version(document: Mapping[str, Any], *, label: str) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkAssetError(f"{label} schema_version must be {SCHEMA_VERSION!r}.")
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise BenchmarkAssetError(
            f"{label} protocol_version must be {PROTOCOL_VERSION!r}."
        )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_names(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BenchmarkAssetError(f"{label} must be a non-empty array.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise BenchmarkAssetError(f"{label} must contain non-empty strings.")
    if len(set(value)) != len(value):
        raise BenchmarkAssetError(f"{label} must contain unique values.")
    return value


def _validate_matrix(
    value: object,
    *,
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> None:
    if not isinstance(value, list) or len(value) != expected_rows:
        raise BenchmarkAssetError(f"{label} must contain exactly {expected_rows} rows.")
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != expected_columns:
            raise BenchmarkAssetError(
                f"{label}[{row_index}] must contain exactly {expected_columns} columns."
            )
        if not all(_is_finite_number(item) and float(item) >= 0 for item in row):
            raise BenchmarkAssetError(
                f"{label}[{row_index}] must contain finite non-negative numbers."
            )
        if not any(float(item) > 0 for item in row):
            raise BenchmarkAssetError(
                f"{label}[{row_index}] must contain a positive component."
            )
    for column in range(expected_columns):
        if not any(float(row[column]) > 0 for row in value):
            raise BenchmarkAssetError(
                f"{label} column {column} must contain a positive component."
            )


def _validate_order(value: object, *, size: int, label: str) -> None:
    if not isinstance(value, list) or value != list(dict.fromkeys(value)):
        raise BenchmarkAssetError(f"{label} must be an array of unique indices.")
    if sorted(value) != list(range(size)):
        raise BenchmarkAssetError(
            f"{label} must be a permutation of indices 0 through {size - 1}."
        )


def validate_case(case: Mapping[str, Any], *, source: Path) -> None:
    """Validate the current case contract and cross-field dimensions."""
    label = f"case {source}"
    require_version(case, label=label)
    require_keys(
        case,
        (
            "case_id",
            "title",
            "status",
            "model",
            "data",
            "oracle",
            "tolerances",
            "metamorphic_tests",
            "license",
        ),
        label=label,
    )
    case_id = case["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise BenchmarkAssetError(f"{label} case_id must be a non-empty string.")
    if case["status"] not in {"active", "candidate", "retired"}:
        raise BenchmarkAssetError(f"{label} has an unknown status.")

    model = case["model"]
    if not isinstance(model, dict):
        raise BenchmarkAssetError(f"{label} model must be an object.")
    require_keys(
        model,
        (
            "family",
            "orientation",
            "returns_to_scale",
            "slacks",
            "exclude_self",
            "reference_scope",
            "disposability",
            "score_convention",
        ),
        label=f"{label} model",
    )
    if model["family"] != "radial_envelopment":
        raise BenchmarkAssetError(f"{label} currently supports radial cases only.")
    if model["orientation"] not in {"input", "output"}:
        raise BenchmarkAssetError(f"{label} has an invalid orientation.")
    if model["returns_to_scale"] not in {"crs", "vrs", "nirs", "ndrs"}:
        raise BenchmarkAssetError(f"{label} has an invalid returns-to-scale value.")
    if model["slacks"] not in {"none", "normalized", "max_sum"}:
        raise BenchmarkAssetError(f"{label} has an invalid slack strategy.")
    expected_convention = (
        "input_theta" if model["orientation"] == "input" else "output_reciprocal_phi"
    )
    if model["score_convention"] != expected_convention:
        raise BenchmarkAssetError(
            f"{label} score convention does not match orientation."
        )
    if model["reference_scope"] != "in_sample":
        raise BenchmarkAssetError(f"{label} currently supports in-sample cases only.")

    data = case["data"]
    if not isinstance(data, dict):
        raise BenchmarkAssetError(f"{label} data must be an object.")
    require_keys(
        data,
        ("dmu_names", "input_names", "output_names", "inputs", "outputs"),
        label=f"{label} data",
    )
    dmu_names = _validate_names(data["dmu_names"], label=f"{label} dmu_names")
    input_names = _validate_names(data["input_names"], label=f"{label} input_names")
    output_names = _validate_names(data["output_names"], label=f"{label} output_names")
    _validate_matrix(
        data["inputs"],
        label=f"{label} inputs",
        expected_rows=len(dmu_names),
        expected_columns=len(input_names),
    )
    _validate_matrix(
        data["outputs"],
        label=f"{label} outputs",
        expected_rows=len(dmu_names),
        expected_columns=len(output_names),
    )

    oracle = case["oracle"]
    if not isinstance(oracle, dict):
        raise BenchmarkAssetError(f"{label} oracle must be an object.")
    require_keys(
        oracle,
        ("provenance", "derivation", "expected"),
        label=f"{label} oracle",
    )
    provenance = oracle["provenance"]
    if not isinstance(provenance, dict):
        raise BenchmarkAssetError(f"{label} provenance must be an object.")
    require_keys(
        provenance,
        (
            "authority",
            "description",
            "repository_paths",
            "citation_keys",
            "independent_external_reproduction",
        ),
        label=f"{label} provenance",
    )
    if provenance["authority"] not in {
        "analytical",
        "primary_publication",
        "independent_formulation",
        "external_implementation",
    }:
        raise BenchmarkAssetError(f"{label} has an unknown oracle authority.")
    if not isinstance(oracle["derivation"], list) or not oracle["derivation"]:
        raise BenchmarkAssetError(f"{label} must document an oracle derivation.")

    expected = oracle["expected"]
    if not isinstance(expected, dict):
        raise BenchmarkAssetError(f"{label} expected oracle must be an object.")
    require_keys(
        expected,
        ("dmu_names", "radial_factor", "technical_efficiency", "statuses"),
        label=f"{label} expected oracle",
    )
    expected_names = _validate_names(
        expected["dmu_names"], label=f"{label} expected dmu_names"
    )
    if expected_names != dmu_names:
        raise BenchmarkAssetError(
            f"{label} expected DMU names must match case data in order."
        )
    for field in ("radial_factor", "technical_efficiency"):
        values = expected[field]
        if (
            not isinstance(values, list)
            or len(values) != len(dmu_names)
            or not all(_is_finite_number(item) and float(item) >= 0 for item in values)
        ):
            raise BenchmarkAssetError(
                f"{label} expected {field} must contain one finite "
                "non-negative value per DMU."
            )
    statuses = expected["statuses"]
    if (
        not isinstance(statuses, list)
        or len(statuses) != len(dmu_names)
        or not all(isinstance(item, str) for item in statuses)
    ):
        raise BenchmarkAssetError(
            f"{label} expected statuses must contain one string per DMU."
        )

    tolerances = case["tolerances"]
    if not isinstance(tolerances, dict):
        raise BenchmarkAssetError(f"{label} tolerances must be an object.")
    require_keys(
        tolerances,
        ("absolute", "relative", "audit"),
        label=f"{label} tolerances",
    )
    if not all(
        _is_finite_number(tolerances[key]) and float(tolerances[key]) > 0
        for key in ("absolute", "relative", "audit")
    ):
        raise BenchmarkAssetError(f"{label} tolerances must be finite and positive.")

    tests = case["metamorphic_tests"]
    if not isinstance(tests, list):
        raise BenchmarkAssetError(f"{label} metamorphic_tests must be an array.")
    seen_test_ids: set[str] = set()
    for test in tests:
        if not isinstance(test, dict):
            raise BenchmarkAssetError(f"{label} metamorphic test must be an object.")
        require_keys(
            test,
            ("id", "relation", "transform"),
            label=f"{label} metamorphic test",
        )
        if test["id"] in seen_test_ids:
            raise BenchmarkAssetError(
                f"{label} repeats metamorphic test ID {test['id']!r}."
            )
        seen_test_ids.add(test["id"])
        if test["relation"] != "name_aligned_primary_scores_equal":
            raise BenchmarkAssetError(f"{label} has an unsupported relation.")
        transform = test["transform"]
        if not isinstance(transform, dict) or "kind" not in transform:
            raise BenchmarkAssetError(f"{label} transform must declare kind.")
        kind = transform["kind"]
        if kind == "scale_columns":
            for field, size in (
                ("input_factors", len(input_names)),
                ("output_factors", len(output_names)),
            ):
                factors = transform.get(field)
                if (
                    not isinstance(factors, list)
                    or len(factors) != size
                    or not all(
                        _is_finite_number(item) and float(item) > 0 for item in factors
                    )
                ):
                    raise BenchmarkAssetError(
                        f"{label} {field} must contain {size} positive values."
                    )
        elif kind == "permute_dmus":
            _validate_order(
                transform.get("order"),
                size=len(dmu_names),
                label=f"{label} DMU order",
            )
        elif kind == "permute_variables":
            _validate_order(
                transform.get("input_order"),
                size=len(input_names),
                label=f"{label} input order",
            )
            _validate_order(
                transform.get("output_order"),
                size=len(output_names),
                label=f"{label} output order",
            )
        elif kind == "duplicate_reference":
            source_index = transform.get("source_index")
            new_name = transform.get("new_name")
            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or not 0 <= source_index < len(dmu_names)
            ):
                raise BenchmarkAssetError(
                    f"{label} duplicate source index is out of range."
                )
            if (
                not isinstance(new_name, str)
                or not new_name.strip()
                or new_name in dmu_names
            ):
                raise BenchmarkAssetError(
                    f"{label} duplicate reference name must be new and non-empty."
                )
        else:
            raise BenchmarkAssetError(
                f"{label} has unsupported transform kind {kind!r}."
            )


def load_suite(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    """Load and cross-check a manifest and all of its case documents."""
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    validate_json_schema(
        manifest,
        schema_path=MANIFEST_SCHEMA,
        label=f"manifest {manifest_path}",
    )
    require_version(manifest, label=f"manifest {manifest_path}")
    require_keys(
        manifest,
        (
            "suite_id",
            "title",
            "case_files",
            "cross_case_relations",
        ),
        label=f"manifest {manifest_path}",
    )
    case_files = manifest["case_files"]
    if not isinstance(case_files, list) or not case_files:
        raise BenchmarkAssetError("Manifest case_files must be a non-empty array.")
    if len(set(case_files)) != len(case_files):
        raise BenchmarkAssetError("Manifest case_files must be unique.")

    cases: dict[str, tuple[Path, dict[str, Any]]] = {}
    manifest_directory = manifest_path.parent
    for relative_name in case_files:
        if not isinstance(relative_name, str):
            raise BenchmarkAssetError("Manifest case file names must be strings.")
        case_path = (manifest_directory / relative_name).resolve()
        if case_path.parent != manifest_directory:
            raise BenchmarkAssetError(
                f"Case path {relative_name!r} must remain in {manifest_directory}."
            )
        case = load_json(case_path)
        validate_json_schema(
            case,
            schema_path=CASE_SCHEMA,
            label=f"case {case_path}",
        )
        validate_case(case, source=case_path)
        case_id = str(case["case_id"])
        if case_id in cases:
            raise BenchmarkAssetError(f"Duplicate case ID {case_id!r}.")
        cases[case_id] = (case_path, case)

    relations = manifest["cross_case_relations"]
    if not isinstance(relations, list):
        raise BenchmarkAssetError("Manifest cross_case_relations must be an array.")
    seen_relations: set[str] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise BenchmarkAssetError("Each cross-case relation must be an object.")
        require_keys(
            relation,
            (
                "id",
                "relation",
                "left_case",
                "right_case",
                "left_field",
                "right_field",
            ),
            label="cross-case relation",
        )
        relation_id = relation["id"]
        if not isinstance(relation_id, str) or relation_id in seen_relations:
            raise BenchmarkAssetError(
                f"Cross-case relation ID {relation_id!r} is invalid or repeated."
            )
        seen_relations.add(relation_id)
        if relation["left_case"] not in cases or relation["right_case"] not in cases:
            raise BenchmarkAssetError(
                f"Cross-case relation {relation_id!r} references an unknown case."
            )
        if relation["relation"] not in {
            "name_aligned_equal",
            "name_aligned_reciprocal",
        }:
            raise BenchmarkAssetError(
                f"Cross-case relation {relation_id!r} has an unknown relation."
            )
        if relation["left_field"] not in {
            "radial_factor",
            "technical_efficiency",
        } or relation["right_field"] not in {
            "radial_factor",
            "technical_efficiency",
        }:
            raise BenchmarkAssetError(
                f"Cross-case relation {relation_id!r} has an unknown field."
            )
    return manifest, cases


def select_cases(
    cases: Mapping[str, tuple[Path, dict[str, Any]]],
    requested: list[str],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Select requested cases or every active case, preserving manifest order."""
    if requested:
        unknown = sorted(set(requested) - set(cases))
        if unknown:
            raise BenchmarkAssetError("Unknown case ID(s): " + ", ".join(unknown) + ".")
        return {case_id: cases[case_id] for case_id in requested}
    return {
        case_id: item
        for case_id, item in cases.items()
        if item[1]["status"] == "active"
    }
