"""Immutable result objects shared by model families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from frontierguard._enums import Orientation, ReturnsToScale, SolverStatus
from frontierguard._errors import DataValidationError


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class RadialResult:
    """Complete result of a radial envelopment fit."""

    dmu_names: tuple[str, ...]
    reference_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    orientation: Orientation
    returns_to_scale: ReturnsToScale
    radial_factor: NDArray[np.float64]
    technical_efficiency: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    input_slacks: NDArray[np.float64]
    output_slacks: NDArray[np.float64]
    input_targets: NDArray[np.float64]
    output_targets: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    solver: str
    phase_two: bool
    exclude_self: bool
    data_fingerprint: str
    reference_fingerprint: str
    audit_residual: NDArray[np.float64]
    audit_tolerance: float

    def __init__(
        self,
        dmu_names: Sequence[str],
        reference_names: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        orientation: Orientation,
        returns_to_scale: ReturnsToScale,
        radial_factor: ArrayLike,
        technical_efficiency: ArrayLike,
        lambdas: ArrayLike,
        input_slacks: ArrayLike,
        output_slacks: ArrayLike,
        input_targets: ArrayLike,
        output_targets: ArrayLike,
        statuses: Sequence[SolverStatus],
        solver: str,
        phase_two: bool,
        exclude_self: bool,
        data_fingerprint: str,
        reference_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        """Copy flexible result values into a canonical immutable representation."""
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "reference_names", tuple(reference_names))
        object.__setattr__(self, "input_names", tuple(input_names))
        object.__setattr__(self, "output_names", tuple(output_names))
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "returns_to_scale", returns_to_scale)
        object.__setattr__(self, "radial_factor", _readonly(radial_factor))
        object.__setattr__(
            self,
            "technical_efficiency",
            _readonly(technical_efficiency),
        )
        object.__setattr__(self, "lambdas", _readonly(lambdas))
        object.__setattr__(self, "input_slacks", _readonly(input_slacks))
        object.__setattr__(self, "output_slacks", _readonly(output_slacks))
        object.__setattr__(self, "input_targets", _readonly(input_targets))
        object.__setattr__(self, "output_targets", _readonly(output_targets))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "phase_two", phase_two)
        object.__setattr__(self, "exclude_self", exclude_self)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(
            self,
            "reference_fingerprint",
            reference_fingerprint,
        )
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    def to_frame(self) -> pd.DataFrame:
        """Return the primary scores and status as a tidy DataFrame."""
        frame = pd.DataFrame(
            {
                "dmu": self.dmu_names,
                "technical_efficiency": self.technical_efficiency,
                "radial_factor": self.radial_factor,
                "status": [status.value for status in self.statuses],
            }
        )
        return frame.set_index("dmu")

    def peers(
        self,
        dmu: str | int,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Return positive reference intensities for one evaluated DMU."""
        if isinstance(dmu, str):
            try:
                position = self.dmu_names.index(dmu)
            except ValueError as exc:
                raise DataValidationError(f"Unknown DMU {dmu!r}.") from exc
        else:
            position = dmu
        if not 0 <= position < len(self.dmu_names):
            raise DataValidationError(f"DMU position {position} is out of range.")
        return {
            name: float(weight)
            for name, weight in zip(
                self.reference_names,
                self.lambdas[position],
                strict=True,
            )
            if weight > tolerance
        }

    def targets(self, dmu: str | int) -> dict[str, dict[str, float]]:
        """Return input and output projection targets for one DMU."""
        if isinstance(dmu, str):
            try:
                position = self.dmu_names.index(dmu)
            except ValueError as exc:
                raise DataValidationError(f"Unknown DMU {dmu!r}.") from exc
        else:
            position = dmu
        if not 0 <= position < len(self.dmu_names):
            raise DataValidationError(f"DMU position {position} is out of range.")
        return {
            "inputs": {
                name: float(value)
                for name, value in zip(
                    self.input_names,
                    self.input_targets[position],
                    strict=True,
                )
            },
            "outputs": {
                name: float(value)
                for name, value in zip(
                    self.output_names,
                    self.output_targets[position],
                    strict=True,
                )
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible model result."""
        return {
            "schema_version": "0.1.0",
            "model": "radial_envelopment",
            "orientation": self.orientation.value,
            "returns_to_scale": self.returns_to_scale.value,
            "solver": self.solver,
            "phase_two": self.phase_two,
            "exclude_self": self.exclude_self,
            "data_fingerprint": self.data_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "reference_names": list(self.reference_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "radial_factor": self.radial_factor.tolist(),
            "technical_efficiency": self.technical_efficiency.tolist(),
            "lambdas": self.lambdas.tolist(),
            "input_slacks": self.input_slacks.tolist(),
            "output_slacks": self.output_slacks.tolist(),
            "input_targets": self.input_targets.tolist(),
            "output_targets": self.output_targets.tolist(),
            "statuses": [status.value for status in self.statuses],
        }


@dataclass(frozen=True, slots=True, init=False)
class SerialNetworkResult:
    """Relational efficiency result for a serial CRS network."""

    dmu_names: tuple[str, ...]
    stage_names: tuple[str, ...]
    overall_efficiency: NDArray[np.float64]
    stage_efficiency: NDArray[np.float64]
    input_weights: NDArray[np.float64]
    link_weights: tuple[NDArray[np.float64], ...]
    output_weights: NDArray[np.float64]
    closure_error: NDArray[np.float64]
    statuses: tuple[SolverStatus, ...]
    solver: str
    data_fingerprint: str
    audit_residual: NDArray[np.float64]
    audit_tolerance: float

    def __init__(
        self,
        dmu_names: Sequence[str],
        stage_names: Sequence[str],
        overall_efficiency: ArrayLike,
        stage_efficiency: ArrayLike,
        input_weights: ArrayLike,
        link_weights: Sequence[ArrayLike],
        output_weights: ArrayLike,
        closure_error: ArrayLike,
        statuses: Sequence[SolverStatus],
        solver: str,
        data_fingerprint: str,
        audit_residual: ArrayLike,
        audit_tolerance: float,
    ) -> None:
        """Copy network result values into a canonical immutable representation."""
        object.__setattr__(self, "dmu_names", tuple(dmu_names))
        object.__setattr__(self, "stage_names", tuple(stage_names))
        object.__setattr__(
            self,
            "overall_efficiency",
            _readonly(overall_efficiency),
        )
        object.__setattr__(self, "stage_efficiency", _readonly(stage_efficiency))
        object.__setattr__(self, "input_weights", _readonly(input_weights))
        object.__setattr__(
            self,
            "link_weights",
            tuple(_readonly(weights) for weights in link_weights),
        )
        object.__setattr__(self, "output_weights", _readonly(output_weights))
        object.__setattr__(self, "closure_error", _readonly(closure_error))
        object.__setattr__(self, "statuses", tuple(statuses))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "data_fingerprint", data_fingerprint)
        object.__setattr__(self, "audit_residual", _readonly(audit_residual))
        object.__setattr__(self, "audit_tolerance", float(audit_tolerance))

    def to_frame(self) -> pd.DataFrame:
        """Return overall and stage scores in one table."""
        values: dict[str, object] = {
            "dmu": self.dmu_names,
            "overall_efficiency": self.overall_efficiency,
        }
        for position, name in enumerate(self.stage_names):
            values[f"stage:{name}"] = self.stage_efficiency[:, position]
        values["closure_error"] = self.closure_error
        values["status"] = [status.value for status in self.statuses]
        return pd.DataFrame(values).set_index("dmu")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible network result."""
        stage_efficiency = [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in np.asarray(self.stage_efficiency)
        ]
        closure_error = [
            float(value) if np.isfinite(value) else None
            for value in np.asarray(self.closure_error)
        ]
        return {
            "schema_version": "0.1.0",
            "model": "serial_relational_network",
            "returns_to_scale": "crs",
            "solver": self.solver,
            "data_fingerprint": self.data_fingerprint,
            "audit_residual": self.audit_residual.tolist(),
            "audit_tolerance": self.audit_tolerance,
            "dmu_names": list(self.dmu_names),
            "stage_names": list(self.stage_names),
            "overall_efficiency": self.overall_efficiency.tolist(),
            "stage_efficiency": stage_efficiency,
            "input_weights": self.input_weights.tolist(),
            "link_weights": [weights.tolist() for weights in self.link_weights],
            "output_weights": self.output_weights.tolist(),
            "closure_error": closure_error,
            "statuses": [status.value for status in self.statuses],
        }
