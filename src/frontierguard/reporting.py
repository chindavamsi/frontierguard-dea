"""Deterministic research evidence bundles for FrontierGuard results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from frontierguard._errors import ModelValidationError
from frontierguard._version import __version__
from frontierguard.registry import MethodDescriptor, get_method


class ReportableResult(Protocol):
    """Minimal interface implemented by public FrontierGuard results."""

    def to_dict(self) -> dict[str, Any]:
        """Return strict JSON-compatible evidence."""

    def to_frame(self) -> pd.DataFrame:
        """Return the primary tidy result table."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_card(
    evidence: Mapping[str, Any],
    *,
    descriptor: MethodDescriptor,
) -> str:
    maximum_audit = _maximum_audit(evidence)
    formatted_audit = (
        "not reported" if maximum_audit is None else f"{maximum_audit:.6e}"
    )
    audit_tolerance = evidence.get("audit_tolerance", "not reported")
    limitations = "\n".join(f"- {limitation}" for limitation in descriptor.limitations)
    citations = "\n".join(f"- {citation}" for citation in descriptor.citations)
    return (
        "# FrontierGuard evidence bundle\n\n"
        f"- Method: `{descriptor.method_id}` ({descriptor.label})\n"
        f"- Equation version: `{descriptor.equation_version}`\n"
        f"- Scientific stability: `{descriptor.stability.value}`\n"
        f"- Optimization class: `{descriptor.problem_class.value}`\n"
        f"- Score: `{descriptor.score_name}`; efficient value "
        f"`{descriptor.efficient_value}`\n"
        f"- Solver: `{evidence.get('solver', 'not reported')}`\n"
        f"- Data fingerprint: "
        f"`{evidence.get('data_fingerprint', 'not reported')}`\n"
        f"- Maximum finite audit residual: `{formatted_audit}`\n"
        f"- Audit tolerance: `{audit_tolerance}`\n"
        f"- FrontierGuard version: `{__version__}`\n\n"
        "## Citations\n\n"
        f"{citations or '- None recorded'}\n\n"
        "## Known limitations\n\n"
        f"{limitations or '- None recorded'}\n\n"
        "This bundle records a descriptive frontier estimate. It does not by "
        "itself establish causality, statistical uncertainty, or external "
        "validity.\n"
    )


def _maximum_audit(evidence: Mapping[str, Any]) -> float | None:
    declared_maximum = evidence.get("max_audit_residual")
    if isinstance(declared_maximum, (int, float)) and np.isfinite(declared_maximum):
        return float(declared_maximum)

    candidates: list[Any] = []
    if "audit_residual" in evidence:
        candidates.append(evidence["audit_residual"])
    nested = evidence.get("audit")
    if isinstance(nested, Mapping):
        candidates.extend(nested.values())

    finite_values: list[NDArray[np.float64]] = []
    for candidate in candidates:
        try:
            values = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return None
    return max(float(values.max()) for values in finite_values)


def write_evidence_bundle(
    result: ReportableResult,
    directory: str | Path,
    *,
    method_id: str,
    overwrite: bool = False,
) -> Path:
    """Write strict JSON, tidy CSV, a method card, and SHA-256 manifest.

    Existing non-empty directories fail by default. With ``overwrite=True``,
    only FrontierGuard's four known bundle files are replaced; unrelated files
    are preserved.
    """
    if not hasattr(result, "to_dict") or not hasattr(result, "to_frame"):
        raise ModelValidationError(
            "result must provide to_dict() and to_frame() methods."
        )
    descriptor = get_method(method_id)
    raw_evidence = result.to_dict()
    frame = result.to_frame()
    if not isinstance(raw_evidence, dict) or not isinstance(frame, pd.DataFrame):
        raise ModelValidationError(
            "result.to_dict() must return dict and to_frame() must return DataFrame."
        )
    evidence = dict(raw_evidence)
    result_model = evidence.get("model")
    if (
        not isinstance(result_model, str)
        or result_model not in descriptor.result_models
    ):
        expected = ", ".join(repr(model) for model in descriptor.result_models)
        raise ModelValidationError(
            f"Result model {result_model!r} is incompatible with method_id "
            f"{method_id!r}; expected one of: {expected}."
        )
    embedded_method = evidence.get("method_id")
    if embedded_method is not None and embedded_method != method_id:
        raise ModelValidationError(
            f"Result embeds method_id {embedded_method!r}, which does not match "
            f"requested method_id {method_id!r}."
        )
    evidence["method_id"] = method_id
    card = _method_card(evidence, descriptor=descriptor)

    target = Path(directory).expanduser().resolve()
    known_files = (
        target / "result.json",
        target / "scores.csv",
        target / "METHOD_CARD.md",
        target / "manifest.json",
    )
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Evidence directory {target} is not empty; set overwrite=True "
            "to replace FrontierGuard bundle files."
        )
    target.mkdir(parents=True, exist_ok=True)

    result_path, scores_path, card_path, manifest_path = known_files
    result_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    frame.to_csv(scores_path, lineterminator="\n")
    card_path.write_text(
        card,
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "0.1.0",
        "frontierguard_version": __version__,
        "method_id": method_id,
        "files": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (result_path, scores_path, card_path)
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
