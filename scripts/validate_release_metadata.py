"""Validate consistency across FrontierGuard release metadata files."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib
from render_whitepaper_pdf import DEFAULT_OUTPUT, DEFAULT_SOURCE, render

ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _read_json(relative_path: str) -> dict[str, Any]:
    value = json.loads(_read_text(relative_path))
    if not isinstance(value, dict):
        raise ValueError(f"{relative_path} must contain a JSON object")
    return value


def _match(pattern: str, text: str, source: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{source} does not match {pattern!r}")
    return match.group(1)


def _validate_whitepaper(version: str, release_date: str) -> list[str]:
    """Bind the report source and committed PDF to the release metadata."""
    errors: list[str] = []
    paper = _read_text("paper/whitepaper/README.md")
    paper_version = _match(
        r'^software-version:\s*["\']?([^"\'\s]+)',
        paper,
        "paper/whitepaper/README.md",
    )
    paper_date = _match(
        r'^date:\s*["\']?(\d{4}-\d{2}-\d{2})',
        paper,
        "paper/whitepaper/README.md",
    )
    if paper_version != version:
        errors.append("white-paper software version does not match project version")
    if paper_date != release_date:
        errors.append("white-paper date does not match CFF release date")

    expected_source = ROOT / "paper" / "whitepaper" / "README.md"
    expected_output = ROOT / "output" / "pdf" / "frontierguard-dea-white-paper-v0.1.pdf"
    if expected_source != DEFAULT_SOURCE:
        errors.append("white-paper renderer source path is not release-bound")
    if expected_output != DEFAULT_OUTPUT:
        errors.append("white-paper renderer output path is not release-bound")
    if not expected_output.is_file():
        errors.append("missing committed white-paper PDF")
        return errors

    with tempfile.TemporaryDirectory(prefix="frontierguard-whitepaper-") as temp_dir:
        rendered_output = Path(temp_dir) / expected_output.name
        render(expected_source, rendered_output)
        if rendered_output.read_bytes() != expected_output.read_bytes():
            errors.append(
                "committed white-paper PDF does not match the current source "
                "and renderer"
            )
    return errors


def validate() -> list[str]:
    """Return consistency errors; an empty list means the metadata is valid."""
    errors: list[str] = []
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    version = str(project["version"])

    version_source = _read_text("src/frontierguard/_version.py")
    source_version = _match(
        r'^__version__ = "([^"]+)"$', version_source, "src/frontierguard/_version.py"
    )
    if source_version != version:
        errors.append(
            f"source version {source_version!r} != project version {version!r}"
        )

    cff = _read_text("CITATION.cff")
    cff_version = _match(r"^version:\s*[\"']?([^\"'\s]+)", cff, "CITATION.cff")
    release_date = _match(
        r"^date-released:\s*[\"']?(\d{4}-\d{2}-\d{2})",
        cff,
        "CITATION.cff",
    )
    if cff_version != version:
        errors.append(f"CFF version {cff_version!r} != project version {version!r}")

    codemeta = _read_json("codemeta.json")
    if codemeta.get("version") != version:
        errors.append("codemeta.json version does not match project version")
    if codemeta.get("datePublished") != release_date:
        errors.append("codemeta.json datePublished does not match CFF release date")

    zenodo = _read_json(".zenodo.json")
    if zenodo.get("publication_date") != release_date:
        errors.append(".zenodo.json publication_date does not match CFF release date")
    if version not in str(zenodo.get("description", "")):
        errors.append(".zenodo.json description does not identify the release version")

    errors.extend(_validate_whitepaper(version, release_date))

    if project.get("license") != "MIT":
        errors.append("pyproject.toml must use the MIT license expression")
    if codemeta.get("license") != "https://spdx.org/licenses/MIT":
        errors.append("codemeta.json must use the MIT SPDX URI")
    if zenodo.get("license") != "MIT":
        errors.append(".zenodo.json must use the MIT license identifier")

    changelog = _read_text("CHANGELOG.md")
    if f"## [{version}] - {release_date}" not in changelog:
        errors.append("CHANGELOG.md has no matching version-and-date release heading")

    release_notes = ROOT / "docs" / "releases" / f"v{version}.md"
    if not release_notes.is_file():
        errors.append(f"missing release notes: {release_notes.relative_to(ROOT)}")

    return errors


def main() -> int:
    """Validate metadata and return a process-compatible status code."""
    try:
        errors = validate()
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"metadata validation failed: {error}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    print(f"validated release metadata for {project['name']} {project['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
