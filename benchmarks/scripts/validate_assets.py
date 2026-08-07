#!/usr/bin/env python3
"""Validate benchmark JSON assets and their cross-document invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    BENCHMARK_ROOT,
    DEFAULT_MANIFEST,
    BenchmarkAssetError,
    load_suite,
    schema_validator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the benchmark manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, cases = load_suite(args.manifest)
        schema_directory = BENCHMARK_ROOT / "schemas"
        schema_paths = sorted(schema_directory.glob("*.schema.json"))
        if not schema_paths:
            raise BenchmarkAssetError("No JSON Schema documents were found.")
        for schema_path in schema_paths:
            schema_validator(schema_path)
    except BenchmarkAssetError as exc:
        print(f"ASSET VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    active = sum(case["status"] == "active" for _, case in cases.values())
    tests = sum(len(case["metamorphic_tests"]) for _, case in cases.values())
    print(
        "ASSET VALIDATION PASSED: "
        f"suite={manifest['suite_id']} cases={len(cases)} active={active} "
        f"metamorphic_tests={tests} schemas={len(schema_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
