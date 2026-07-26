"""Small command-line entry point for discovery and environment diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from importlib import metadata
from typing import TypedDict

from frontierguard._version import __version__
from frontierguard.registry import (
    MethodStability,
    ProblemClass,
    list_methods,
)


class DoctorReport(TypedDict):
    """Machine-readable runtime diagnostic payload."""

    frontierguard: str
    python: str
    platform: str
    dependencies: dict[str, str | None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontierguard",
        description="Inspect FrontierGuard DEA methods and runtime capabilities.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("version", help="Print the package version.")
    methods = subcommands.add_parser(
        "methods",
        help="List implemented mathematical method contracts.",
    )
    methods.add_argument(
        "--stability",
        choices=[value.value for value in MethodStability],
    )
    methods.add_argument(
        "--problem-class",
        choices=[value.value for value in ProblemClass],
    )
    methods.add_argument("--json", action="store_true", dest="as_json")

    doctor = subcommands.add_parser(
        "doctor",
        help="Report package, Python, and numerical dependency versions.",
    )
    doctor.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _doctor() -> DoctorReport:
    return {
        "frontierguard": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": _installed_version("numpy"),
            "pandas": _installed_version("pandas"),
            "scipy": _installed_version("scipy"),
            "highspy": _installed_version("highspy"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FrontierGuard discovery CLI."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "version":
        print(__version__)
        return 0
    if arguments.command == "doctor":
        diagnostics = _doctor()
        if arguments.as_json:
            print(json.dumps(diagnostics, sort_keys=True))
        else:
            print(f"FrontierGuard DEA {diagnostics['frontierguard']}")
            print(f"Python {diagnostics['python']}")
            for name, version in diagnostics["dependencies"].items():
                print(f"{name}: {version if version is not None else 'not installed'}")
        return 0

    methods = list_methods(
        stability=arguments.stability,
        problem_class=arguments.problem_class,
    )
    if arguments.as_json:
        print(json.dumps([method.to_dict() for method in methods], sort_keys=True))
    else:
        for method in methods:
            print(
                f"{method.method_id}\t{method.stability.value}\t"
                f"{method.problem_class.value}\t{method.label}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
